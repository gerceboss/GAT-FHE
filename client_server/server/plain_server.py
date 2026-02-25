#!/usr/bin/env python3
"""
Plaintext GAT server: raw TCP transport (no FHE).

Mirrors the FHE server's two-command protocol:
  b'T'  ->  train   (recv training graph, run GAT training, return weights + metrics)
  b'I'  ->  infer   (recv node batch + weights, run forward pass, return predictions + metrics)

Per-epoch metrics are recorded during training.
Per-batch metrics are recorded during inference.
All server-side metrics are written to  server_metrics_<timestamp>.txt

Usage:
  python plain_server.py --host 0.0.0.0 --port 9998

In-process API:
  from plain_server import compute_plain_training, compute_plain_infer
"""

from __future__ import annotations

import os
import pickle
import socket
import struct
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import csv

def _write_metrics_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"[server] metrics written → {path}")

# ── RSS / energy helpers (same as metrics.py) ────────────────────────────────

def _rss_bytes() -> int:
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1]) * 1024
    except OSError:
        pass
    try:
        import resource
        return int(getattr(resource.getrusage(resource.RUSAGE_SELF), "ru_maxrss", 0)) * 1024
    except Exception:
        return 0


def _energy_uj() -> Optional[int]:
    try:
        base = "/sys/class/powercap"
        if not os.path.isdir(base):
            return None
        for entry in sorted(os.listdir(base)):
            path = os.path.join(base, entry, "energy_uj")
            if os.path.isfile(path):
                with open(path, "r", encoding="utf-8") as f:
                    return int(f.read().strip())
    except Exception:
        return None
    return None


@dataclass
class StepMetric:
    name: str
    seconds: float
    rss_delta_bytes: int
    rss_after_bytes: int
    energy_joules: float = 0.0
    power_watts: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)   # e.g. loss, acc


class _Timer:
    """Context manager that records one StepMetric."""
    def __init__(self, name: str, extra: Dict[str, Any] = None):
        self.name = name
        self.extra = extra or {}
        self.metric: Optional[StepMetric] = None
        self._t0 = self._rss0 = self._e0 = None

    def __enter__(self):
        self._t0 = time.perf_counter()
        self._rss0 = _rss_bytes()
        self._e0 = _energy_uj()
        return self

    def __exit__(self, *_):
        t1 = time.perf_counter()
        rss1 = _rss_bytes()
        e1 = _energy_uj()
        dt = float(t1 - (self._t0 or t1))
        energy_j = 0.0
        power_w = 0.0
        if self._e0 is not None and e1 is not None and dt > 0:
            energy_j = (e1 - self._e0) / 1e6
            power_w = energy_j / dt
        self.metric = StepMetric(
            name=self.name,
            seconds=dt,
            rss_delta_bytes=int(rss1 - (self._rss0 or rss1)),
            rss_after_bytes=int(rss1),
            energy_joules=energy_j,
            power_watts=power_w,
            extra=self.extra,
        )


# ── TCP framing ───────────────────────────────────────────────────────────────

def _recvall(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Connection closed mid-receive")
        buf.extend(chunk)
    return bytes(buf)


def _send_frame(sock: socket.socket, data: bytes) -> None:
    sock.sendall(struct.pack(">I", len(data)) + data)


def _recv_frame(sock: socket.socket) -> bytes:
    raw_len = _recvall(sock, 4)
    length = struct.unpack(">I", raw_len)[0]
    return _recvall(sock, length)


def _send_ok(sock: socket.socket) -> None:
    sock.sendall(b"\x00")


def _send_error(sock: socket.socket, exc: Exception) -> None:
    msg = str(exc).encode()
    sock.sendall(b"\x01" + struct.pack(">I", len(msg)) + msg)


# ── Plain GAT model ───────────────────────────────────────────────────────────

class PlainGATLayer(nn.Module):
    """
    Single-head GAT layer.
    Matches the FHE encoder's weight layout: W (out_channels x in_channels),
    attention vector a (2*out_channels,), LeakyReLU with negative_slope.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 negative_slope: float = 0.2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.negative_slope = negative_slope
        self.W = nn.Parameter(torch.empty(out_channels, in_channels))
        self.a = nn.Parameter(torch.empty(2 * out_channels))
        nn.init.xavier_uniform_(self.W.unsqueeze(0))
        nn.init.xavier_uniform_(self.a.unsqueeze(0))

    def load_weights(self, W_np: np.ndarray, a_np: np.ndarray) -> None:
        with torch.no_grad():
            self.W.copy_(torch.tensor(W_np, dtype=torch.float32))
            self.a.copy_(torch.tensor(a_np, dtype=torch.float32))

    def get_weights(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.W.detach().cpu().numpy(), self.a.detach().cpu().numpy()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # x: (N, in_channels)
        N = x.size(0)
        h = x @ self.W.t()                        # (N, out_channels)

        # Handle degenerate batches with no edges or wrong edge_index shape
        if edge_index.dim() != 2 or edge_index.numel() == 0 or edge_index.size(0) < 2:
            # No message passing possible — return zeros (no aggregated neighbours)
            return torch.zeros(N, self.out_channels, dtype=h.dtype, device=h.device)

        src, dst = edge_index[0], edge_index[1]   # both shape (E,)

        # Attention scores
        a_src = self.a[:self.out_channels]         # (out_channels,)
        a_dst = self.a[self.out_channels:]         # (out_channels,)

        e_src = (h * a_src).sum(dim=-1)            # (N,)
        e_dst = (h * a_dst).sum(dim=-1)            # (N,)
        e = F.leaky_relu(e_src[src] + e_dst[dst],
                         negative_slope=self.negative_slope)  # (E,)

        # Softmax per destination node
        # Numerically stable: subtract per-dst max
        alpha = torch.zeros(N, dtype=e.dtype, device=e.device)  # temp
        e_max = torch.full((N,), float("-inf"), dtype=e.dtype, device=e.device)
        e_max.scatter_reduce_(0, dst, e, reduce="amax", include_self=True)
        e_exp = torch.exp(e - e_max[dst])
        e_sum = torch.zeros(N, dtype=e.dtype, device=e.device)
        e_sum.scatter_add_(0, dst, e_exp)
        alpha_edge = e_exp / (e_sum[dst] + 1e-16)  # (E,)

        # Aggregate
        out = torch.zeros(N, self.out_channels, dtype=h.dtype, device=h.device)
        out.scatter_add_(0, dst.unsqueeze(1).expand(-1, self.out_channels),
                         alpha_edge.unsqueeze(1) * h[src])
        return out


class PlainGATModel(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 negative_slope: float = 0.2):
        super().__init__()
        self.gat = PlainGATLayer(in_channels, out_channels, negative_slope)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.gat(x, edge_index)

    def load_weights(self, W_np: np.ndarray, a_np: np.ndarray) -> None:
        self.gat.load_weights(W_np, a_np)

    def get_weights(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.gat.get_weights()



# ── Core compute functions ────────────────────────────────────────────────────

def compute_plain_training_batch(
    *,
    x_batch,
    edge_index_batch,
    y_batch,
    in_channels,
    out_channels,
    W,
    a,
    num_epochs,
    negative_slope=0.2,
    lr=0.01,
):
    """
    Perform num_epochs gradient steps on a subgraph batch.
    Returns updated weights and per-epoch metric rows.
    """
    model = PlainGATModel(in_channels, out_channels, negative_slope)
    model.load_weights(W, a)

    x_t = torch.tensor(x_batch, dtype=torch.float32)
    ei_t = torch.tensor(edge_index_batch, dtype=torch.long)
    y_t = torch.tensor(y_batch, dtype=torch.float32)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Initialise so we always have valid weights to return even if num_epochs==0
    W_new, a_new = W, a

    epoch_metrics: List[Dict] = []

    for epoch in range(1, num_epochs + 1):
        extra: Dict[str, Any] = {}
        with _Timer(f"epoch_{epoch:03d}", extra) as timer:
            model.train()
            optimizer.zero_grad()

            out = model(x_t, ei_t)
            logits = out[:, 0]

            pos_weight = torch.tensor(
                [(len(y_batch) - np.sum(y_batch)) / (np.sum(y_batch) + 1e-6)],
                dtype=torch.float32,
            )

            loss = F.binary_cross_entropy_with_logits(
                logits,
                y_t,
                pos_weight=pos_weight,
            )
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                preds = (torch.sigmoid(logits) > 0.5).long()
                correct = (preds == y_t.long()).float().mean().item()
            extra["loss"] = float(loss.item())
            extra["train_acc"] = float(correct)

        m = timer.metric
        epoch_metrics.append({
            "epoch": epoch,
            "seconds": m.seconds,
            "rss_delta_bytes": m.rss_delta_bytes,
            "rss_after_bytes": m.rss_after_bytes,
            "energy_joules": m.energy_joules,
            "power_watts": m.power_watts,
            "loss": extra.get("loss", 0.0),
            "train_acc": extra.get("train_acc", 0.0),
        })

        W_new, a_new = model.get_weights()

    # Build metrics_rows list — accumulate ALL epochs (was erroneously overwriting)
    metrics_rows: List[Dict] = []
    for m_dict in epoch_metrics:
        metrics_rows.append({
            "phase": "train_batch",
            "batch": 0,
            "epoch": m_dict["epoch"],
            "seconds": m_dict["seconds"],
            "rss_delta_bytes": m_dict["rss_delta_bytes"],
            "rss_after_bytes": m_dict["rss_after_bytes"],
            "energy_joules": m_dict["energy_joules"],
            "power_watts": m_dict["power_watts"],
            "loss": m_dict.get("loss", 0.0),
            "train_acc": m_dict.get("train_acc", 0.0),
        })

    return W_new, a_new, metrics_rows

def compute_plain_training(
    *,
    x_train: np.ndarray,          # (N_train, F_in)
    edge_index_train: np.ndarray, # (2, E_train)
    y_train: np.ndarray,          # (N_train,)
    train_mask: np.ndarray,       # (N_train,) bool
    in_channels: int,
    out_channels: int,
    W_init: np.ndarray,
    a_init: np.ndarray,
    negative_slope: float = 0.2,
    num_epochs: int = 3,
    lr: float = 0.01,
    print_metrics: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """
    Train a 1-layer plaintext GAT for num_epochs.

    Returns:
        W_trained (np.ndarray), a_trained (np.ndarray),
        epoch_metrics (list of dicts, one per epoch)
    """
    model = PlainGATModel(in_channels, out_channels, negative_slope)
    model.load_weights(W_init, a_init)

    x_t = torch.tensor(x_train, dtype=torch.float32)
    ei_t = torch.tensor(edge_index_train, dtype=torch.long)
    y_t = torch.tensor(y_train, dtype=torch.float32)
    mask = torch.tensor(train_mask, dtype=torch.bool)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    epoch_metrics: List[Dict] = []

    for epoch in range(1, num_epochs + 1):
        extra: Dict[str, Any] = {}
        with _Timer(f"epoch_{epoch:03d}", extra) as timer:
            model.train()
            optimizer.zero_grad()
            out = model(x_t, ei_t)          # (N, out_channels)
            logits = out[mask, 0]           # binary: single output channel
            loss = F.binary_cross_entropy_with_logits(logits, y_t[mask])
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                preds = (torch.sigmoid(logits) > 0.5).long()
                correct = (preds == y_t[mask].long()).float().mean().item()
            extra["loss"] = float(loss.item())
            extra["train_acc"] = float(correct)

        m = timer.metric
        epoch_metrics.append({
            "epoch": epoch,
            "seconds": m.seconds,
            "rss_delta_bytes": m.rss_delta_bytes,
            "rss_after_bytes": m.rss_after_bytes,
            "energy_joules": m.energy_joules,
            "power_watts": m.power_watts,
            "loss": extra["loss"],
            "train_acc": extra["train_acc"],
        })

        if print_metrics:
            mb_after = m.rss_after_bytes / (1024 * 1024)
            dmb = m.rss_delta_bytes / (1024 * 1024)
            print(
                f"  [server] epoch {epoch:3d}/{num_epochs}  "
                f"loss={extra['loss']:.4f}  acc={extra['train_acc']:.4f}  "
                f"t={m.seconds:.4f}s  "
                f"RSS Δ{dmb:+.2f}MB  RSS {mb_after:.2f}MB  "
                f"power={m.power_watts:.4f}W"
            )
    metrics_rows = []

    for m in epoch_metrics:
        metrics_rows.append({
            "phase": "train",
            "batch": "",
            "epoch": m["epoch"],
            "seconds": m["seconds"],
            "rss_delta_bytes": m["rss_delta_bytes"],
            "rss_after_bytes": m["rss_after_bytes"],
            "energy_joules": m["energy_joules"],
            "power_watts": m["power_watts"],
        })

    W_trained, a_trained = model.get_weights()
    return W_trained, a_trained, metrics_rows


def compute_plain_infer_batch(
    *,
    x_batch: np.ndarray,           # (B, F_in)  — full sub-graph features
    edge_index_batch: np.ndarray,  # (2, E_batch)
    node_indices: np.ndarray,      # which rows of x_batch to return predictions for
    in_channels: int,
    out_channels: int,
    W: np.ndarray,
    a: np.ndarray,
    negative_slope: float = 0.2,
    batch_id: int = 0,
) -> Tuple[np.ndarray, Dict]:
    """
    Run one forward pass on a batch of nodes.

    Returns:
        logits (np.ndarray, shape (len(node_indices),)),
        batch_metrics (dict)
    """
    model = PlainGATModel(in_channels, out_channels, negative_slope)
    model.load_weights(W, a)
    model.eval()

    x_t = torch.tensor(x_batch, dtype=torch.float32)
    ei_t = torch.tensor(edge_index_batch, dtype=torch.long)

    with _Timer(f"batch_{batch_id:04d}") as timer, torch.no_grad():
        out = model(x_t, ei_t)
        logits = out[node_indices, 0].cpu().numpy()

    m = timer.metric
    # Return flat dict — easier for client to access .get("seconds", ...) directly
    batch_metrics = {
        "batch": batch_id,
        "seconds": m.seconds,
        "rss_delta_bytes": m.rss_delta_bytes,
        "rss_after_bytes": m.rss_after_bytes,
        "energy_joules": m.energy_joules,
        "power_watts": m.power_watts,
        "encrypted": False,
    }
    return logits, batch_metrics


# ── Connection handler ────────────────────────────────────────────────────────

# Thread-local storage so each connection accumulates its own metrics
_server_epoch_metrics: List[Dict] = []
_server_batch_metrics: List[Dict] = []
_metrics_lock = threading.Lock()


def _handle_connection(conn: socket.socket, addr: tuple) -> None:
    print(f"[server] connection from {addr[0]}:{addr[1]}")

    try:
        with conn:
            cmd = conn.recv(1)

            # ── TRAIN ──────────────────────────────────────────────────────
            if cmd == b"T":
                raw = _recv_frame(conn)
                payload: Dict = pickle.loads(raw)

                try:
                    W_trained, a_trained, metrics_rows = compute_plain_training(
                        x_train=np.asarray(payload["x_train"], dtype=np.float64),
                        edge_index_train=np.asarray(payload["edge_index_train"], dtype=np.int64),
                        y_train=np.asarray(payload["y_train"], dtype=np.float64),
                        train_mask=np.asarray(payload["train_mask"], dtype=bool),
                        in_channels=int(payload["in_channels"]),
                        out_channels=int(payload["out_channels"]),
                        W_init=np.asarray(payload["W_init"], dtype=np.float64),
                        a_init=np.asarray(payload["a_init"], dtype=np.float64),
                        negative_slope=float(payload.get("negative_slope", 0.2)),
                        num_epochs=int(payload.get("epochs", 5)),
                        lr=float(payload.get("lr", 0.01)),
                        print_metrics=bool(payload.get("print_metrics", True)),
                    )
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    _write_metrics_csv(f"server_train_metrics_{ts}.csv", metrics_rows)
                    result = {"W_trained": W_trained, "a_trained": a_trained,
                              "metrics": metrics_rows, "ok": True}
                    _send_ok(conn)
                    _send_frame(conn, pickle.dumps(result))

                except Exception as exc:
                    traceback.print_exc()
                    _send_error(conn, exc)

            # ── INFER (single batch) ───────────────────────────────────────
            elif cmd == b"I":
                raw = _recv_frame(conn)
                payload: Dict = pickle.loads(raw)

                try:
                    logits, batch_metrics = compute_plain_infer_batch(
                        x_batch=np.asarray(payload["x_batch"], dtype=np.float64),
                        edge_index_batch=np.asarray(payload["edge_index_batch"], dtype=np.int64),
                        node_indices=np.asarray(payload["node_indices"], dtype=np.int64),
                        in_channels=int(payload["in_channels"]),
                        out_channels=int(payload["out_channels"]),
                        W=np.asarray(payload["W"], dtype=np.float64),
                        a=np.asarray(payload["a"], dtype=np.float64),
                        negative_slope=float(payload.get("negative_slope", 0.2)),
                        batch_id=int(payload.get("batch_id", 0)),
                    )

                    # Build metrics_rows from the returned flat batch_metrics dict
                    infer_metrics_rows = [{
                        "phase": "infer",
                        "batch": batch_metrics.get("batch", 0),
                        "epoch": "",
                        "seconds": batch_metrics.get("seconds", 0.0),
                        "rss_delta_bytes": batch_metrics.get("rss_delta_bytes", 0),
                        "rss_after_bytes": batch_metrics.get("rss_after_bytes", 0),
                        "energy_joules": batch_metrics.get("energy_joules", 0.0),
                        "power_watts": batch_metrics.get("power_watts", 0.0),
                    }]
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    _write_metrics_csv(f"server_infer_metrics_{ts}.csv", infer_metrics_rows)

                    result = {
                        "logits": logits,
                        "metrics": batch_metrics,
                        "ok": True
                    }
                    _send_ok(conn)
                    _send_frame(conn, pickle.dumps(result))

                except Exception as exc:
                    traceback.print_exc()
                    _send_error(conn, exc)
            elif cmd == b"G":
                raw = _recv_frame(conn)
                payload = pickle.loads(raw)

                W_new, a_new, metrics_rows = compute_plain_training_batch(
                    x_batch=payload["x_batch"],
                    edge_index_batch=payload["edge_index_batch"],
                    y_batch=payload["y_batch"],
                    in_channels=payload["in_channels"],
                    out_channels=payload["out_channels"],
                    W=payload["W"],
                    a=payload["a"],
                    num_epochs=payload.get("num_epochs", payload.get("epochs", 1)),
                    negative_slope=payload["negative_slope"],
                    lr=payload["lr"],
                )

                result = {"W": W_new, "a": a_new, "metrics": metrics_rows}
                ts = time.strftime("%Y%m%d_%H%M%S")
                _write_metrics_csv(f"server_train_metrics_{ts}.csv", metrics_rows)
                _send_ok(conn)
                _send_frame(conn, pickle.dumps(result))
            else:
                print(f"[server] unknown command {cmd!r}", file=sys.stderr)

    except Exception as exc:
        print(f"[server] connection error: {exc}", file=sys.stderr)
        traceback.print_exc()
    finally:
        print(f"[server] closed {addr[0]}:{addr[1]}")


# ── TCP server ────────────────────────────────────────────────────────────────

def serve(host: str = "127.0.0.1", port: int = 9998) -> None:
    """Start TCP server; each client connection handled in a daemon thread."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        srv.bind((host, port))
        srv.listen(8)
        print(f"[server] Plaintext GAT server listening on {host}:{port}")
        print("[server]  command b'T' -> train  |  b'I' -> infer (per batch)")
        try:
            while True:
                conn, addr = srv.accept()
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                t = threading.Thread(
                    target=_handle_connection, args=(conn, addr), daemon=True
                )
                t.start()
        except KeyboardInterrupt:
            print("\n[server] shutting down.")


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Plaintext GAT TCP server")
    ap.add_argument("--host", default="127.0.0.1",
                    help="Bind address (use 0.0.0.0 for LAN/Raspberry Pi)")
    ap.add_argument("--port", type=int, default=9998, help="TCP port")
    args = ap.parse_args()
    serve(args.host, args.port)


if __name__ == "__main__":
    main()