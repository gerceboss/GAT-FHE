#!/usr/bin/env python3
"""
Plaintext GAT client: raw TCP transport (no FHE). Line-graph (dual) formulation.

  We build the line graph: each node = one original edge; features on nodes.
  Phase 1 (train): Send line-graph node batches (with train_mask) → server trains
                   node-level GAT → receive W, a + metrics.
  Phase 2 (infer): Batch test line-graph nodes → server returns logits for subgraph;
                   we take logits[target_indices] as predictions for original edges.

Arguments:
  --batch_size       INT   Edges per batch (default 60)
  --test_ratio       FLOAT Fraction of edges held out for testing (default 0.2)
  --epochs           INT   Training epochs per batch (default 3)
  --lr               FLOAT Learning rate (default 0.01)
  --host             STR   Server host (omit for in-process mode)
  --port             INT   Server port (default 9998)
  --data             STR   Path to iot.csv
  --max_rows_dataset INT   Cap CSV rows (edges) for quick runs (optional)

Usage:
  python plain_client.py --batch_size 60 --epochs 3
  python plain_client.py --batch_size 60 --host 192.168.1.20 --port 9998
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import socket
import struct
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# ── Optional sklearn metrics ──────────────────────────────────────────────────
try:
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        classification_report,
    )
    _SKLEARN = True
except ImportError:
    _SKLEARN = False

# Shared data loading, batching, metrics (see client_server.client.utils)
from client_server.client.utils import (
    build_line_graph,
    build_line_graph_batch,
    compute_classification_metrics,
    load_iot_edge_train_test,
    make_connected_batches,
)


def _resolve_plain_load_weights_dir(path_like: str) -> str:
    """
    Resolve checkpoint directory for --load_weights (parity with FHE client):
      - directory that already contains plain_weights.pt, or
      - parent with last_successful_checkpoint.txt → batch_XXXX/, or
      - parent with batch_* subdirs → newest batch_* containing plain_weights.pt.
    """
    p = Path(path_like)
    if (p / "plain_weights.pt").exists():
        return str(p.resolve())

    pointer = p / "last_successful_checkpoint.txt"
    if pointer.exists():
        rel = pointer.read_text(encoding="utf-8").strip()
        if rel:
            cand = (p / rel).resolve()
            if (cand / "plain_weights.pt").exists():
                return str(cand)

    candidates: List[Tuple[int, Path]] = []
    for d in p.glob("batch_*"):
        if d.is_dir() and (d / "plain_weights.pt").exists():
            try:
                idx = int(d.name.split("_")[-1])
            except Exception:
                idx = -1
            candidates.append((idx, d))
    if candidates:
        candidates.sort(key=lambda t: t[0])
        return str(candidates[-1][1].resolve())
    return str(p.resolve())


def _save_plain_checkpoint(
    save_root: str,
    batch_index: int,
    W: np.ndarray,
    a: np.ndarray,
    f_in: int,
    f_out: int,
) -> str:
    """Write batch_XXXX/plain_weights.pt + meta.json and update last_successful_checkpoint.txt."""
    root = Path(save_root)
    ckpt_dir = root / f"batch_{batch_index:04d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"W": torch.tensor(W), "a": torch.tensor(a)},
        ckpt_dir / "plain_weights.pt",
    )
    meta = {"F_in": f_in, "F_out": f_out, "mode": "plain"}
    (ckpt_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    root.mkdir(parents=True, exist_ok=True)
    (root / "last_successful_checkpoint.txt").write_text(
        f"batch_{batch_index:04d}\n",
        encoding="utf-8",
    )
    return str(ckpt_dir)


def tcp_gradient_step(host, port, payload):
    with _open_tcp(host, port) as sock:
        sock.sendall(b"G")
        _send_frame(sock, pickle.dumps(payload))
        _recv_status(sock)
        raw = _recv_frame(sock)
    result = pickle.loads(raw)
    return result["W"], result["a"], result.get("metrics", [])

# ── RSS / energy helpers ──────────────────────────────────────────────────────

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


# ── Metrics recorder ──────────────────────────────────────────────────────────

class _Step:
    """Context-manager for one named measurement."""
    def __init__(self, name: str):
        self.name = name
        self.seconds: float = 0.0
        self.rss_delta_bytes: int = 0
        self.rss_after_bytes: int = 0
        self.energy_joules: float = 0.0
        self.power_watts: float = 0.0

    def __enter__(self):
        self._t0 = time.perf_counter()
        self._rss0 = _rss_bytes()
        self._e0 = _energy_uj()
        return self

    def __exit__(self, *_):
        t1 = time.perf_counter()
        rss1 = _rss_bytes()
        e1 = _energy_uj()
        dt = float(t1 - self._t0)
        self.seconds = dt
        self.rss_delta_bytes = int(rss1 - self._rss0)
        self.rss_after_bytes = int(rss1)
        if self._e0 is not None and e1 is not None and dt > 0:
            self.energy_joules = (e1 - self._e0) / 1e6
            self.power_watts = self.energy_joules / dt



# ── TCP helpers ───────────────────────────────────────────────────────────────

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
    length = struct.unpack(">I", _recvall(sock, 4))[0]
    return _recvall(sock, length)


def _recv_status(sock: socket.socket) -> None:
    status = _recvall(sock, 1)
    if status == b"\x01":
        n = struct.unpack(">I", _recvall(sock, 4))[0]
        msg = _recvall(sock, n).decode(errors="replace")
        raise RuntimeError(f"Server error: {msg}")


def _open_tcp(host: str, port: int) -> socket.socket:
    sock = socket.create_connection((host, port), timeout=600)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    return sock


# ── TCP calls ─────────────────────────────────────────────────────────────────

def tcp_infer_batch(
    host: str, port: int, payload: dict,
) -> Tuple[np.ndarray, Dict]:
    with _open_tcp(host, port) as sock:
        sock.sendall(b"I")
        _send_frame(sock, pickle.dumps(payload))
        _recv_status(sock)
        raw = _recv_frame(sock)
    result = pickle.loads(raw)
    # Server sends result["metrics"] (flat dict), not result["batch_metrics"]
    return result["logits"], result.get("metrics", {})


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="IoT Malicious Node Prediction — Plaintext GAT baseline"
    )
    parser.add_argument("--batch_size", type=int, default=60,
                        help="Graph nodes per inference batch (default 60)")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Training epochs (default 3)")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Learning rate (default 0.01)")
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.2,
        help="Fraction of nodes for test set (default 0.2)",
    )
    parser.add_argument("--host", type=str, default="",
                        help="Server host. Omit for in-process mode.")
    parser.add_argument("--port", type=int, default=9998,
                        help="Server port (default 9998)")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to iot.csv")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for train/test split (default 42)")
    parser.add_argument(
        "--save_weights",
        type=str,
        default=None,
        metavar="DIR",
        help="Directory for per-batch checkpoints (batch_XXXX/plain_weights.pt + meta.json) "
        "and last_successful_checkpoint.txt; also writes top-level plain_weights.pt after training.",
    )

    parser.add_argument(
        "--load_weights",
        type=str,
        default=None,
        metavar="DIR",
        help="Checkpoint dir (plain_weights.pt), or parent dir with batch_*/ "
        "and optional last_successful_checkpoint.txt (same resolution as FHE). "
        "Skips training.",
    )

    parser.add_argument(
        "--train_only",
        action="store_true",
        help="Run training only and exit (no inference)."
    )

    parser.add_argument(
        "--infer_only",
        action="store_true",
        help="Run inference only (requires --load_weights)."
    )
    parser.add_argument(
        "--max_rows_dataset",
        type=int,
        default=None,
        help="Cap number of CSV rows (edges) for quick runs."
    )
    parser.add_argument(
        "--max_degree_batch",
        type=int,
        default=10,
        metavar="K",
        help="Cap in-degree per node in each batch (max edges = batch_size*K). Default 10 for parity with FHE; use 8--15 for 8GB RAM.",
    )
    args = parser.parse_args()
    batch_csv_rows = []
    ts = time.strftime("%Y%m%d_%H%M%S")
    plain_batch_csv_path = f"plain_batch_metrics_{ts}.csv"
    plain_batch_fieldnames = ["step", "server_time", "edges_in_batch", "phase", "batch"]
    plain_train_csv_path = f"server_train_metrics_{ts}.csv"
    plain_infer_csv_path = f"server_infer_metrics_{ts}.csv"
    from client_server.server.utils import append_dict_rows as _append_dict_rows  # local alias

    # Line-graph node dimensions (features = edge features: src_bytes, dst_bytes, duration)
    IN_CHANNELS = 3
    OUT_CHANNELS = 1   # one logit per node = per original edge
    negative_slope = 0.2

    print("=" * 70)
    print("Plaintext GAT — IoT Malicious Edge (Link) via Line Graph")
    print(
        f"  batch_size={args.batch_size} line-graph nodes  epochs={args.epochs}  "
        f"lr={args.lr}  test_ratio={args.test_ratio}  max_rows_dataset={args.max_rows_dataset or 'ALL'}"
    )
    if args.host:
        print(f"  Transport: TCP  {args.host}:{args.port}")
    else:
        print("  Transport: in-process (no network)")
    print("=" * 70)

    # ── 1. Load edge data and build line graph ───────────────────────────────────
    print("\n1. Loading IoT edge train + test...")
    (
        x_nodes,
        edge_index_full,
        edge_feats,
        edge_labels,
        train_edge_ids,
        test_edge_ids,
        N,
    ) = load_iot_edge_train_test(
        path=args.data,
        test_ratio=args.test_ratio,
        seed=args.seed,
        max_rows_dataset=args.max_rows_dataset,
    )
    n_train_edges = len(train_edge_ids)
    n_test_edges = len(test_edge_ids)
    E_total = edge_index_full.shape[1]
    print(f"   Original: Nodes={N}, Edges={E_total}  Train edges={n_train_edges}, Test edges={n_test_edges}")

    print("   Building line graph (one node per edge)...")
    x_line, edge_index_line, y_line = build_line_graph(edge_index_full, edge_feats, edge_labels)
    n_line = x_line.shape[0]
    # Train/test IDs are edge indices = line-graph node IDs
    train_line_ids = train_edge_ids
    test_line_ids = test_edge_ids
    # Connected batches: BFS-grown subgraphs (fewer edges per batch than contiguous chunks)
    train_batches = make_connected_batches(train_line_ids, edge_index_line, args.batch_size)
    test_batches = make_connected_batches(test_line_ids, edge_index_line, args.batch_size)
    n_batches_train = len(train_batches)
    n_batches_test = len(test_batches)
    print(f"   Line graph: nodes={n_line}  Batches train={n_batches_train}, test={n_batches_test} (batch_size={args.batch_size}, connected)")

    # ── 2. Initialise weights (node-level GAT only; no edge head) ───────────────
    rng = np.random.default_rng(args.seed)
    W = rng.standard_normal((OUT_CHANNELS, IN_CHANNELS)).astype(np.float64) * 0.1
    a = rng.standard_normal((2 * OUT_CHANNELS,)).astype(np.float64) * 0.1

    if args.load_weights:
        resolved_w = _resolve_plain_load_weights_dir(args.load_weights)
        print(f"\n[weights] Loading plaintext weights from: {resolved_w}")
        weights_path = os.path.join(resolved_w, "plain_weights.pt")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"No weights found at {weights_path}")
        ckpt = torch.load(weights_path)
        W = ckpt["W"].numpy() if torch.is_tensor(ckpt["W"]) else ckpt["W"]
        a = ckpt["a"].numpy() if torch.is_tensor(ckpt["a"]) else ckpt["a"]
        print("   Weights loaded. Skipping training.")

    # ── 3. Training (line-graph node batches) ───────────────────────────────────
    train_time = 0.0
    all_train_metrics_rows: List[Dict] = []  # for in-process: write server_train_metrics_*.csv
    if not args.load_weights and not args.infer_only:
        print(f"\n3. Mini-batch training ({args.epochs} epochs per batch, line-graph nodes)...")
        _train_t0 = time.perf_counter()
        train_batches_shuffled = [list(b) for b in train_batches]
        rng.shuffle(train_batches_shuffled)

        for b_idx in range(n_batches_train):
            batch_line_ids = np.asarray(train_batches_shuffled[b_idx], dtype=np.int64)

            x_batch, edge_index_batch, y_batch, target_indices = build_line_graph_batch(
                batch_line_ids, edge_index_line, x_line, y_line,
                max_degree_per_node=args.max_degree_batch,
            )

            if edge_index_batch.shape[1] == 0 and x_batch.shape[0] == 0:
                continue

            train_mask = np.zeros(x_batch.shape[0], dtype=bool)
            train_mask[target_indices] = True

            print(f"\n   Batch {b_idx+1}/{n_batches_train} (line-graph nodes={len(batch_line_ids)}, edges in line-graph nodes={edge_index_batch.shape[1]})")

            train_payload = {
                "x_batch": x_batch,
                "edge_index_batch": edge_index_batch,
                "y_batch": y_batch.astype(np.float64),
                "train_mask": train_mask,
                "in_channels": IN_CHANNELS,
                "out_channels": OUT_CHANNELS,
                "W": W,
                "a": a,
                "negative_slope": negative_slope,
                "lr": args.lr,
                "num_epochs": args.epochs,
            }

            if args.host:
                with _open_tcp(args.host, args.port) as sock:
                    sock.sendall(b"G")
                    _send_frame(sock, pickle.dumps(train_payload))
                    _recv_status(sock)
                    data = pickle.loads(_recv_frame(sock))
                W = np.asarray(data["W"], dtype=np.float64)
                a = np.asarray(data["a"], dtype=np.float64)
                batch_metrics_rows = data.get("metrics", [])
            else:
                from client_server.server.plain_server import compute_plain_training_batch
                W, a, batch_metrics_rows = compute_plain_training_batch(
                    x_batch=x_batch,
                    edge_index_batch=edge_index_batch,
                    y_batch=y_batch.astype(np.float64),
                    train_mask=train_mask,
                    in_channels=IN_CHANNELS,
                    out_channels=OUT_CHANNELS,
                    W=W,
                    a=a,
                    num_epochs=args.epochs,
                    negative_slope=negative_slope,
                    lr=args.lr,
                )
                for r in batch_metrics_rows:
                    all_train_metrics_rows.append({**r, "batch": b_idx})

            server_t = sum(r.get("seconds", 0.0) for r in batch_metrics_rows)
            _train_batch_row = {
                "step": f"train_batch_{b_idx:04d}",
                "phase": "train",
                "batch": b_idx,
                "edges_in_batch": len(batch_line_ids),
                "server_time": server_t,
            }
            batch_csv_rows.append(_train_batch_row)
            print(f"      Completed {args.epochs} epochs  server_t={server_t:.4f}s")

            try:
                _append_dict_rows(plain_batch_csv_path, [_train_batch_row], plain_batch_fieldnames)
            except Exception as _csv_exc:
                print(f"[client] WARN: failed to append plain_batch_metrics row ({b_idx}): {_csv_exc}")

            if batch_metrics_rows:
                try:
                    _train_server_rows = [
                        {**r, "phase": f"train_batch_{b_idx:04d}_epoch_{i}"} for i, r in enumerate(batch_metrics_rows)
                    ]
                    from client_server.server.utils import append_metrics_rows as _append_metrics_rows
                    _append_metrics_rows(plain_train_csv_path, _train_server_rows, time_side="server")
                except Exception as _csv_exc:
                    print(f"[client] WARN: failed to append per-batch train server metrics ({b_idx}): {_csv_exc}")

            if args.save_weights:
                ckpt_dir = _save_plain_checkpoint(
                    str(args.save_weights),
                    b_idx,
                    W,
                    a,
                    IN_CHANNELS,
                    OUT_CHANNELS,
                )
                print(f"[weights] Checkpoint saved → {ckpt_dir}")

        train_time = time.perf_counter() - _train_t0
        if args.save_weights:
            os.makedirs(args.save_weights, exist_ok=True)
            torch.save({
                "W": torch.tensor(W), "a": torch.tensor(a),
            }, os.path.join(args.save_weights, "plain_weights.pt"))
            print(f"[weights] Final weights saved → {os.path.join(args.save_weights, 'plain_weights.pt')}")
        if args.train_only:
            print("\n✓ Training complete (train_only). Exiting.")
            return

    # ── 4. Inference (line-graph node batches) ───────────────────────────────────
    if args.infer_only and not args.load_weights:
        raise ValueError("--infer_only requires --load_weights")

    if not args.train_only:
        print(f"\n4. Inference on {n_test_edges} test edges (line-graph nodes)...")
        all_logits = []
        all_y_true = []
        server_batch_metrics_list = []
        # Note: do NOT reset batch_csv_rows here so the final summary still
        # accounts for any train rows; per-batch CSV is already incremental.

        for b_idx in range(n_batches_test):
            batch_line_ids = np.asarray(test_batches[b_idx], dtype=np.int64)

            x_batch, edge_index_batch, y_batch, target_indices = build_line_graph_batch(
                batch_line_ids, edge_index_line, x_line, y_line,
                max_degree_per_node=args.max_degree_batch,
            )
            if x_batch.shape[0] == 0:
                continue

            infer_payload = {
                "x_batch": x_batch,
                "edge_index_batch": edge_index_batch,
                "node_indices": target_indices,
                "in_channels": IN_CHANNELS,
                "out_channels": OUT_CHANNELS,
                "W": W,
                "a": a,
                "negative_slope": negative_slope,
                "batch_id": b_idx,
            }

            if args.host:
                with _open_tcp(args.host, args.port) as sock:
                    sock.sendall(b"I")
                    _send_frame(sock, pickle.dumps(infer_payload))
                    _recv_status(sock)
                    result = pickle.loads(_recv_frame(sock))
                logits = result["logits"]
                batch_metrics = result.get("metrics", {})
            else:
                from client_server.server.plain_server import compute_plain_infer_batch
                logits_all, batch_metrics = compute_plain_infer_batch(
                    x_batch=x_batch,
                    edge_index_batch=edge_index_batch,
                    node_indices=target_indices,
                    in_channels=IN_CHANNELS,
                    out_channels=OUT_CHANNELS,
                    W=W,
                    a=a,
                    negative_slope=negative_slope,
                    batch_id=b_idx,
                )
                logits = np.asarray(logits_all)[target_indices]

            all_logits.extend(np.asarray(logits).ravel().tolist())
            all_y_true.extend(y_batch[target_indices].tolist())
            server_batch_metrics_list.append(batch_metrics)
            _b_seconds = batch_metrics.get("seconds", 0.0)
            print(f"   Batch {b_idx+1}/{n_batches_test} server_t={_b_seconds:.4f}s")
            _infer_batch_row = {
                "step": f"infer_batch_{b_idx:04d}",
                "phase": "infer",
                "batch": b_idx,
                "server_time": _b_seconds,
                "edges_in_batch": len(batch_line_ids),
            }
            batch_csv_rows.append(_infer_batch_row)

            try:
                _append_dict_rows(plain_batch_csv_path, [_infer_batch_row], plain_batch_fieldnames)
            except Exception as _csv_exc:
                print(f"[client] WARN: failed to append plain_batch_metrics row ({b_idx}): {_csv_exc}")

            if batch_metrics:
                try:
                    _row = dict(batch_metrics)
                    _row["phase"] = f"infer_batch_{b_idx:04d}"
                    from client_server.server.utils import append_metrics_rows as _append_metrics_rows
                    _append_metrics_rows(plain_infer_csv_path, [_row], time_side="server")
                except Exception as _csv_exc:
                    print(f"[client] WARN: failed to append per-batch infer server metrics ({b_idx}): {_csv_exc}")

    csv_path = plain_batch_csv_path
    if batch_csv_rows:
        print(f"[client] Metrics written → {csv_path}")
    if not args.host and server_batch_metrics_list:
        from client_server.server.utils import write_metrics_csv
        infer_metrics_rows = [
            {
                "phase": "infer",
                "batch": i,
                "epoch": "",
                "seconds": m.get("seconds", 0.0),
                "rss_delta_bytes": m.get("rss_delta_bytes", 0),
                "rss_after_bytes": m.get("rss_after_bytes", 0),
                "energy_joules": m.get("energy_joules", 0.0),
                "power_watts": m.get("power_watts", 0.0),
            }
            for i, m in enumerate(server_batch_metrics_list)
        ]
        write_metrics_csv(f"server_infer_metrics_{ts}.csv", infer_metrics_rows)

    if not args.train_only and all_logits:
        import csv as _csv
        n_test = len(all_y_true)
        Tserver = sum(r.get("server_time", 0.0) for r in batch_csv_rows)
        summary_path = f"plain_summary_{ts}.csv"
        with open(summary_path, "w", newline="") as f:
            writer = _csv.writer(f)
            writer.writerow(["Tserver", Tserver])
            writer.writerow(["n_test_edges", n_test])
        print(f"[client] Summary → {summary_path}")

        print(f"\n5. Evaluating on {n_test} test edges...")
        y_scores = np.array(all_logits, dtype=np.float64)
        y_pred = (1 / (1 + np.exp(-y_scores)) > 0.5).astype(np.int64)
        y_true = np.array(all_y_true, dtype=np.int64)

        cls_metrics = compute_classification_metrics(y_true, y_pred, y_scores)
        print("\n" + "=" * 70)
        print("=== Test-set Classification Results (edges) ===")
        print("=" * 70)
        print(f"  Accuracy  : {cls_metrics['accuracy']:.4f}")
        print(f"  Precision : {cls_metrics['precision']:.4f}")
        print(f"  Recall    : {cls_metrics['recall']:.4f}")
        print(f"  F1 Score  : {cls_metrics['f1']:.4f}")
        print(f"  Test edges: {n_test}  |  Batches: {n_batches_test}")
        if _SKLEARN:
            print("\n  Per-class report:")
            print(classification_report(y_true, y_pred, labels=[0, 1], target_names=["BENIGN", "ATTACKER"], zero_division=0))
        print("=" * 70)
        print(f"\n  Training time: {train_time:.4f}s  Inference time: {Tserver:.4f}s")

    print("\n✓ Done. Plaintext line-graph (node-level GAT) training + inference.")


if __name__ == "__main__":
    main()