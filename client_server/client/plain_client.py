#!/usr/bin/env python3
"""
Plaintext GAT client: raw TCP transport (no FHE).

Mirrors the FHE client's two-phase flow but uses plaintext numpy arrays:
  Phase 1 (train): Send train graph → server trains → receive W_trained, a_trained
                   + per-epoch server metrics.
  Phase 2 (infer): Batch test nodes by --batch_size; send each batch → server
                   runs forward pass → receive logits + per-batch server metrics.
                   Compute accuracy / precision / recall / F1 over ALL test batches.

Arguments:
  --batch_size  INT   Nodes per inference batch (default 60)
  --test_ratio  FLOAT Fraction of nodes held out for testing (default 0.2)
  --epochs      INT   Training epochs (default 3)
  --lr          FLOAT Learning rate (default 0.01)
  --host        STR   Server host (omit for in-process mode)
  --port        INT   Server port (default 9998)
  --f_in        INT   Number of input features to keep (default 5)
  --data        STR   Path to iot.csv

Output files (written to current directory):
  client_metrics_<timestamp>.txt  — client-side timing, RSS, energy per step
  server_metrics_train_<timestamp>.txt — server-side per-epoch metrics
  server_metrics_infer_<timestamp>.txt — server-side per-batch inference metrics

Usage:
  # In-process (no network):
  python plain_client.py --batch_size 60 --epochs 3

  # Remote server:
  python plain_client.py --batch_size 60 --host 192.168.1.20 --port 9998
"""

from __future__ import annotations

import argparse
import os
import pickle
import socket
import struct
import sys
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

# ── Optional sklearn metrics ──────────────────────────────────────────────────
try:
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        classification_report,
    )
    _SKLEARN = True
except ImportError:
    _SKLEARN = False

# Helpers

def make_connected_batches(
    train_node_ids: np.ndarray,
    edge_index_global: np.ndarray,
    batch_size: int,
) -> list[list[int]]:
    """
    Partition *train_node_ids* into batches of exactly *batch_size* nodes
    (last batch may be smaller) where each batch has as many edges as possible.

    Algorithm
    ---------
    1. Build adjacency restricted to train↔train edges from *edge_index_global*.
    2. Produce a single BFS traversal order over ALL train nodes, seeding from
       the highest-degree unvisited node whenever the queue empties (handles
       disconnected components).  Because BFS visits spatially close nodes
       consecutively, slicing this ordering into chunks of *batch_size* puts
       graph-neighbours into the same batch — guaranteeing edges in every
       batch that lives inside a connected component.
    3. Slice the BFS ordering into chunks of *batch_size*.

    This always produces exactly ceil(N / batch_size) batches and never
    creates a batch with more than *batch_size* nodes.

    Returns
    -------
    List of node-ID lists, each len ≤ batch_size.
    """
    from collections import deque

    train_list = [int(n) for n in train_node_ids]

    # ── Build train↔train adjacency ──────────────────────────────────────────
    adj: dict[int, list[int]] = {n: [] for n in train_list}
    if edge_index_global.ndim == 2 and edge_index_global.shape[1] > 0:
        for s, t in edge_index_global.T:
            s, t = int(s), int(t)
            if s in adj and t in adj:
                adj[s].append(t)
                adj[t].append(s)

    degrees = {n: len(adj[n]) for n in train_list}

    # ── BFS ordering: seed from highest-degree unvisited node ────────────────
    # Using a priority pool (sorted by degree desc) ensures we start each new
    # component from its hub node, pulling in well-connected neighbours first.
    unvisited = sorted(train_list, key=lambda n: degrees[n], reverse=True)
    unvisited_set = set(unvisited)

    bfs_order: list[int] = []
    queue: deque[int] = deque()

    for seed in unvisited:
        if seed not in unvisited_set:
            continue  # already visited via BFS
        unvisited_set.discard(seed)
        queue.append(seed)
        while queue:
            node = queue.popleft()
            bfs_order.append(node)
            # Expand neighbours degree-desc so high-degree nodes are batched
            # together and contribute more edges to the same batch.
            for nb in sorted(adj[node], key=lambda n: degrees[n], reverse=True):
                if nb in unvisited_set:
                    unvisited_set.discard(nb)
                    queue.append(nb)

    # ── Slice into batch_size chunks ─────────────────────────────────────────
    batches = [
        bfs_order[i : i + batch_size] for i in range(0, len(bfs_order), batch_size)
    ]

    return batches

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



# ── Dataset loading ───────────────────────────────────────────────────────────

def _default_iot_csv_path() -> str:
    here = Path(__file__).resolve().parent
    for candidate in [
        here / "iot.csv",
        here.parent.parent / "examples" / "dataset" / "iot.csv",
        here.parent.parent / "iot.csv",
    ]:
        if candidate.exists():
            return str(candidate)
    return str(here / "iot.csv")


def load_and_preprocess_iot_csv(
    path: Optional[str] = None,
    f_in: int = 5,
    min_nodes: Optional[int] = None,
):
    """
    Load IoT CSV, build graph (edge_index), extract node features (up to f_in cols)
    and binary labels. Returns (x, edge_index, y, node_order).
    """
    if path is None:
        path = _default_iot_csv_path()
    df = pd.read_csv(path, encoding="latin1")
    df.rename(columns={"ÿsrc_ip": "src_ip"}, inplace=True)

    # If a minimum node count is requested, keep the smallest prefix of rows
    # that contains at least min_nodes distinct src_ip values.
    if min_nodes is not None:
        min_nodes = int(min_nodes)
        if min_nodes <= 0:
            raise ValueError("min_nodes must be positive when provided")
        seen: set[str] = set()
        cutoff = None
        for idx, ip in enumerate(df["src_ip"]):
            seen.add(ip)
            if len(seen) >= min_nodes:
                cutoff = idx
                break
        if cutoff is not None:
            df = df.iloc[: cutoff + 1].copy()

    all_ips = pd.concat([df["src_ip"], df["dst_ip"]]).unique()
    ip_to_idx = {ip: idx for idx, ip in enumerate(all_ips)}
    df["src_idx"] = df["src_ip"].map(ip_to_idx)
    df["dst_idx"] = df["dst_ip"].map(ip_to_idx)

    edge_index = np.array(df[["src_idx", "dst_idx"]].values.T, dtype=np.int64)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in ["src_idx", "dst_idx", "label"]]

    node_features = (
        df.groupby("src_idx")[numeric_cols]
        .mean()
        .reindex(range(len(all_ips)), fill_value=0)
    )
    x = StandardScaler().fit_transform(node_features.values)
    f_in = min(f_in, x.shape[1])
    x = x[:, :f_in].astype(np.float64)

    node_labels = (
        df.groupby("src_idx")["label"]
        .agg(lambda lbl: lbl.value_counts().index[0])
        .reindex(range(len(all_ips)), fill_value=0)
    )
    y = (node_labels.values > 0).astype(np.int64)

    return x, edge_index, y


def build_train_test_split(
    x: np.ndarray,
    edge_index: np.ndarray,
    y: np.ndarray,
    test_ratio: float = 0.2,
    seed: int = 42,
    total_nodes: Optional[int] = None,
):
    """
    Stratified split preserving attacker/benign ratio.
    """

    N_full = x.shape[0]
    # Drop isolated nodes (degree 0) before splitting
    if edge_index.size > 0:
        deg = np.bincount(edge_index.reshape(-1), minlength=N_full)
        keep_nodes = np.where(deg > 0)[0]
        if keep_nodes.size > 0 and keep_nodes.size < N_full:
            mapping = -np.ones(N_full, dtype=np.int64)
            mapping[keep_nodes] = np.arange(keep_nodes.size, dtype=np.int64)
            x = x[keep_nodes]
            y = y[keep_nodes]
            edge_index = mapping[edge_index]
            N_full = x.shape[0]
    rng = np.random.default_rng(seed)

    if total_nodes is not None:
        # When a cap is requested, use a simple random split on a subset
        total_nodes = int(total_nodes)
        if total_nodes <= 0:
            raise ValueError("total_nodes must be positive when provided")
        N_eff = min(total_nodes, N_full)

        n_test = max(1, int(N_eff * test_ratio))
        n_train = max(1, N_eff - n_test)

        perm = rng.permutation(N_full)[:N_eff]
        train_ids = perm[:n_train]
        test_ids = perm[n_train:]
    else:
        # Default: stratified split preserving attacker/benign ratio
        idx_benign = np.where(y == 0)[0]
        idx_attack = np.where(y == 1)[0]

        rng.shuffle(idx_benign)
        rng.shuffle(idx_attack)

        n_test_benign = max(1, int(len(idx_benign) * test_ratio))
        n_test_attack = max(1, int(len(idx_attack) * test_ratio))

        test_ids = np.concatenate(
            [
                idx_benign[:n_test_benign],
                idx_attack[:n_test_attack],
            ]
        )

        train_ids = np.concatenate(
            [
                idx_benign[n_test_benign:],
                idx_attack[n_test_attack:],
            ]
        )

        rng.shuffle(train_ids)
        rng.shuffle(test_ids)

    # Remap train first, then test
    node_order = np.concatenate([train_ids, test_ids])
    old_to_new = {int(old): new for new, old in enumerate(node_order)}

    new_edge_list = [
        [old_to_new[int(s)], old_to_new[int(t)]]
        for s, t in edge_index.T
        if int(s) in old_to_new and int(t) in old_to_new
    ]
    new_edge_list = list({tuple(e) for e in new_edge_list})

    edge_index_full_remap = (
        np.array(new_edge_list, dtype=np.int64).T
        if new_edge_list else np.zeros((2, 0), dtype=np.int64)
    )

    x_full = x[node_order]
    y_full = y[node_order]

    n_train = len(train_ids)

    x_train = x_full[:n_train]
    y_train = y_full[:n_train]
    x_test = x_full[n_train:]
    y_test = y_full[n_train:]

    train_nodes = set(range(n_train))
    train_edge_list = [
        [s, t] for s, t in edge_index_full_remap.T
        if s in train_nodes and t in train_nodes
    ]
    train_edge_list = list({tuple(e) for e in train_edge_list})

    edge_index_train = (
        np.array(train_edge_list, dtype=np.int64).T
        if train_edge_list else np.zeros((2, 0), dtype=np.int64)
    )

    test_node_global_ids = np.arange(n_train, len(node_order))

    return (
        x_train, edge_index_train, y_train,
        x_test, edge_index_full_remap, y_test,
        test_node_global_ids, x_full,
    )

def build_batch(
    batch_idx: int,
    batch_node_ids: np.ndarray,   # local IDs within x_test
    x_test: np.ndarray,
    edge_index_full: np.ndarray,  # edges in the remapped full graph
    n_train: int,                 # offset: test nodes start at n_train in full graph
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a mini sub-graph for inference on a batch of test nodes.

    Includes 1-hop neighbours from the training set so attention can
    flow, but returns predictions only for the requested test nodes.

    Returns:
        x_batch (B+neighbours, F_in),
        edge_index_batch (2, E_sub),
        node_indices_in_batch  — which rows of x_batch are the test targets
    """
    # Global IDs of the test nodes in this batch (within the full remapped graph)
    global_batch = set(int(n_train + nid) for nid in batch_node_ids)

    # Add 1-hop neighbours (from training or other test nodes)
    neighbours = set()
    for s, t in edge_index_full.T:
        if s in global_batch:
            neighbours.add(int(t))
        if t in global_batch:
            neighbours.add(int(s))

    all_nodes_global = sorted(global_batch | neighbours)
    old_to_local = {g: l for l, g in enumerate(all_nodes_global)}

    # Build local x:  we need x for ALL nodes (train + test combined)
    # edge_index_full references the full graph (0..N-1 remapped)
    # We'll pass in the full stacked x (train + test)
    # The caller provides x_full; here we only have x_test.
    # So return the index set for the caller to slice from x_full.
    return all_nodes_global, old_to_local

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


# ── In-process compute (no network) ──────────────────────────────────────────

def _inprocess_train(
    x_train, edge_index_train, y_train, train_mask,
    in_channels, out_channels, W_init, a_init,
    negative_slope, num_epochs, lr, metrics_path,
):
    # Import server functions from the client_server package
    from client_server.server.plain_server import (
        compute_plain_training
    )
    W_trained, a_trained, epoch_metrics = compute_plain_training(
        x_train=x_train, edge_index_train=edge_index_train,
        y_train=y_train, train_mask=train_mask,
        in_channels=in_channels, out_channels=out_channels,
        W_init=W_init, a_init=a_init,
        negative_slope=negative_slope, num_epochs=num_epochs,
        lr=lr, print_metrics=True,
    )
    return W_trained, a_trained, epoch_metrics


def _inprocess_infer_batch(
    x_batch, edge_index_batch, node_indices,
    in_channels, out_channels, W, a, negative_slope, batch_id,
):
    from client_server.server.plain_server import compute_plain_infer_batch
    return compute_plain_infer_batch(
        x_batch=x_batch, edge_index_batch=edge_index_batch,
        node_indices=node_indices, in_channels=in_channels,
        out_channels=out_channels, W=W, a=a,
        negative_slope=negative_slope, batch_id=batch_id,
    )


# ── TCP calls ─────────────────────────────────────────────────────────────────

def tcp_train(
    host: str, port: int, payload: dict,
) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    print(f"   [tcp] connecting to {host}:{port} for training...")
    with _open_tcp(host, port) as sock:
        sock.sendall(b"T")
        _send_frame(sock, pickle.dumps(payload))
        _recv_status(sock)
        raw = _recv_frame(sock)
    result = pickle.loads(raw)
    return result["W_trained"], result["a_trained"], result.get("metrics", [])


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


# ── Classification metrics ────────────────────────────────────────────────────

def compute_classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_scores: np.ndarray
) -> Dict:
    """Compute accuracy, precision, recall, F1. Falls back to manual if no sklearn."""
    if _SKLEARN:
        acc = float(accuracy_score(y_true, y_pred))
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
    else:
        tp = int(np.sum((y_pred == 1) & (y_true == 1)))
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))
        fn = int(np.sum((y_pred == 0) & (y_true == 1)))
        tn = int(np.sum((y_pred == 0) & (y_true == 0)))
        acc = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


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
    parser.add_argument(
        "--total_nodes",
        type=int,
        default=None,
        help="Total number of nodes to use (cap dataset; default = all nodes)",
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
        help="Directory to save trained plaintext weights after training."
    )

    parser.add_argument(
        "--load_weights",
        type=str,
        default=None,
        metavar="DIR",
        help="Directory to load previously saved plaintext weights. "
            "Skips training."
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
    args = parser.parse_args()
    batch_csv_rows = []
    ts = time.strftime("%Y%m%d_%H%M%S")

    F_in ,F_out = 5, 1        # binary classification: 1 output channel, sigmoid threshold 0.5
    negative_slope = 0.2

    print("=" * 70)
    print("Plaintext GAT Baseline — IoT Malicious Node Prediction")
    print(
        f"  batch_size={args.batch_size}  epochs={args.epochs}  "
        f"lr={args.lr}  test_ratio={args.test_ratio}  "
        f"total_nodes={args.total_nodes or 'ALL'}"
    )
    if args.host:
        print(f"  Transport: TCP  {args.host}:{args.port}")
    else:
        print("  Transport: in-process (no network)")
    print("=" * 70)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("\n1. Loading IoT data...")

    x, edge_index, y = load_and_preprocess_iot_csv(
        args.data,
        f_in=F_in,
        min_nodes=args.total_nodes,
    )
    N_total = x.shape[0]
    print(f"   Total nodes={N_total}  features={x.shape[1]}  "
          f"classes={len(np.unique(y))}")

    # ── 2. Train/test split ───────────────────────────────────────────────────
    print("\n2. Splitting into train / test...")
    x_train,edge_index_train,y_train, x_test, edge_index_full,y_test,test_node_global_ids,x_full = build_train_test_split(
        x,
        edge_index,
        y,
        test_ratio=args.test_ratio,
        seed=args.seed,
        total_nodes=args.total_nodes,
    )
    n_train = x_train.shape[0]
    n_test = x_test.shape[0]
    n_batches_train = max(1, (n_train + args.batch_size - 1) // args.batch_size)
    n_batches_test = max(1, (n_test + args.batch_size - 1) // args.batch_size)
    print(f"   Train nodes={n_train}  Test nodes={n_test}  "
          f"Batches for the training={n_batches_train} (batch_size={args.batch_size})"
          f"Batches for inference={n_batches_test} (batch_size={args.batch_size})")

    # ── 3. Initialise weights ─────────────────────────────────────────────────
    rng = np.random.default_rng(args.seed)

    if args.load_weights:
        print(f"\n[weights] Loading plaintext weights from {args.load_weights}")

        weights_path = os.path.join(args.load_weights, "plain_weights.pt")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"No weights found at {weights_path}")

        ckpt = torch.load(weights_path)
        W = ckpt["W"]
        a = ckpt["a"]

        print("   Weights loaded. Skipping training.")

    else:
        W = rng.standard_normal((F_out, F_in)).astype(np.float64) * 0.1
        a = rng.standard_normal((2 * F_out,)).astype(np.float64) * 0.1

    # ── 4. Training ───────────────────────────────────────────────────────────
    train_time = 0.0  # always defined so summary section can reference it
    if not args.load_weights and not args.infer_only:

        print(f"\n3. Mini-batch training ({args.epochs} epochs per batch)...")

        train_batches = make_connected_batches(
            train_node_ids=np.arange(n_train),
            edge_index_global=edge_index_full,
            batch_size=args.batch_size,
        )

        n_batches_train = len(train_batches)
        _train_t0 = time.perf_counter()

        for b_idx, batch_node_ids in enumerate(train_batches):
            batch_nodes = np.array(batch_node_ids, dtype=np.int64)
            all_nodes = sorted(batch_nodes)
            old_to_local = {g: i for i, g in enumerate(all_nodes)}

            x_batch = x_full[all_nodes]

            edge_index_batch = (
                np.array(
                    [
                        [old_to_local[int(s)], old_to_local[int(t)]]
                        for s, t in edge_index_full.T
                        if int(s) in old_to_local and int(t) in old_to_local
                    ],
                    dtype=np.int64,
                ).T
                if len(all_nodes) > 0
                else np.zeros((2, 0), dtype=np.int64)
            )

            if edge_index_batch.shape[1] == 0:
                print(f"   Batch {b_idx+1}/{n_batches_train} has no edges; skipping.")
                continue

            y_batch = np.concatenate([y_train, y_test])[all_nodes]

            print(
                f"\n   Batch {b_idx+1}/{n_batches_train} "
                f"(nodes={len(batch_nodes)}, edges={edge_index_batch.shape[1]})"
            )

            # Send num_epochs so ALL epochs run server-side in one round-trip
            train_payload = {
                "x_batch": x_batch,
                "edge_index_batch": edge_index_batch,
                "y_batch": y_batch,
                "in_channels": F_in,
                "out_channels": F_out,
                "W": W,
                "a": a,
                "negative_slope": negative_slope,
                "lr": args.lr,
                "num_epochs": args.epochs,
            }

            if args.host:
                W, a, batch_metrics_rows = tcp_gradient_step(args.host, args.port, train_payload)
            else:
                from client_server.server.plain_server import compute_plain_training_batch

                W, a, batch_metrics_rows = compute_plain_training_batch(
                    x_batch=train_payload["x_batch"],
                    edge_index_batch=train_payload["edge_index_batch"],
                    y_batch=train_payload["y_batch"],
                    in_channels=train_payload["in_channels"],
                    out_channels=train_payload["out_channels"],
                    W=train_payload["W"],
                    a=train_payload["a"],
                    negative_slope=train_payload["negative_slope"],
                    lr=train_payload["lr"],
                    num_epochs=train_payload["num_epochs"],
                )

            # Aggregate per-epoch rows into per-batch summary
            server_time_seconds = sum(r.get("seconds", 0.0) for r in batch_metrics_rows)
            server_energy_joules = sum(r.get("energy_joules", 0.0) for r in batch_metrics_rows)
            server_power_watts = (
                server_energy_joules / server_time_seconds if server_time_seconds > 0 else 0.0
            )
            server_rss_after_mb = max(
                (r.get("rss_after_bytes", 0) for r in batch_metrics_rows), default=0
            ) / (1024 * 1024)

            batch_csv_rows.append({
                "phase": "train",
                "batch": b_idx,
                "nodes_in_batch": len(batch_nodes),
                "client_encryption_time": 0.0,
                "client_decryption_time": 0.0,
                "payload_size_bytes": len(pickle.dumps(train_payload)),
                "server_time_seconds": server_time_seconds,
                "server_rss_after_mb": server_rss_after_mb,
                "server_energy_joules": server_energy_joules,
                "server_power_watts": server_power_watts,
            })
            print(f"      Completed {args.epochs} epochs  server_t={server_time_seconds:.4f}s")

        train_time = time.perf_counter() - _train_t0

        # Save weights
        if args.save_weights:
            os.makedirs(args.save_weights, exist_ok=True)
            torch.save({"W": W, "a": a},
                    os.path.join(args.save_weights, "plain_weights.pt"))
            print(f"[weights] Saved plaintext weights → {args.save_weights}")

        if args.train_only:
            print("\n✓ Training complete (train_only mode). Exiting.")
            return

    # ── 5. Inference ───────────────────────────────────────────────────────────
    if args.infer_only and not args.load_weights:
        raise ValueError("--infer_only requires --load_weights")

    if not args.train_only:

        print(f"\n4. Inference on {n_test} test nodes...")

        all_logits = []
        all_y_true = []
        server_batch_metrics_list = []
        batch_csv_rows = []

        n_batches_test = max(1, (n_test + args.batch_size - 1) // args.batch_size)

        for b_idx in range(n_batches_test):

            start = b_idx * args.batch_size
            end = min(start + args.batch_size, n_test)

            local_test_ids = np.arange(start, end)
            global_test_ids = n_train + local_test_ids

            all_nodes = list(global_test_ids)
            old_to_local = {g: i for i, g in enumerate(all_nodes)}

            x_batch = x_full[all_nodes]

            edge_index_batch = (
                np.array(
                    [
                        [old_to_local[int(s)], old_to_local[int(t)]]
                        for s, t in edge_index_full.T
                        if int(s) in old_to_local and int(t) in old_to_local
                    ],
                    dtype=np.int64,
                ).T
                if len(all_nodes) > 0
                else np.zeros((2, 0), dtype=np.int64)
            )

            if edge_index_batch.shape[1] == 0:
                print(f"   Batch {b_idx+1} has no edges; skipping.")
                continue

            node_indices_in_batch = np.array(
                [old_to_local[int(g)] for g in global_test_ids],
                dtype=np.int64,
            )

            infer_payload = {
                "x_batch": x_batch,
                "edge_index_batch": edge_index_batch,
                "node_indices": node_indices_in_batch,
                "in_channels": F_in,
                "out_channels": F_out,
                "W": W,
                "a": a,
                "negative_slope": negative_slope,
                "batch_id": b_idx,
            }

            if args.host:
                logits, batch_metrics = tcp_infer_batch(
                    args.host, args.port, infer_payload
                )
            else:
                logits, batch_metrics = _inprocess_infer_batch(
                    x_batch=x_batch,
                    edge_index_batch=edge_index_batch,
                    node_indices=node_indices_in_batch,
                    in_channels=F_in,
                    out_channels=F_out,
                    W=W,
                    a=a,
                    negative_slope=negative_slope,
                    batch_id=b_idx,
                )

            all_logits.extend(logits.tolist())
            all_y_true.extend(y_test[start:end].tolist())
            server_batch_metrics_list.append(batch_metrics)

            _b_seconds = batch_metrics.get("seconds", 0.0)
            _b_rss_mb = batch_metrics.get("rss_after_bytes", 0) / (1024 * 1024)
            _b_energy = batch_metrics.get("energy_joules", 0.0)
            _b_power = batch_metrics.get("power_watts", 0.0)

            print(
                f"   Batch {b_idx+1}/{n_batches_test} "
                f"server_t={_b_seconds:.4f}s"
            )

            # Append per-batch row INSIDE the loop so every batch is recorded
            batch_csv_rows.append({
                "phase": "infer",
                "batch": b_idx,
                "nodes_in_batch": len(local_test_ids),
                "client_encryption_time": 0.0,
                "client_decryption_time": 0.0,
                "payload_size_bytes": len(pickle.dumps(infer_payload)),
                "server_time_seconds": _b_seconds,
                "server_rss_after_mb": _b_rss_mb,
                "server_energy_joules": _b_energy,
                "server_power_watts": _b_power,
            })

    import csv
    csv_path = f"plain_batch_metrics_{ts}.csv"

    if batch_csv_rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=batch_csv_rows[0].keys())
            writer.writeheader()
            writer.writerows(batch_csv_rows)
        print(f"[client] Plain per-batch metrics written → {csv_path}")

    if not args.train_only and all_logits:
        Tserver = sum(r["server_time_seconds"] for r in batch_csv_rows if r["phase"] == "infer")
        Energy_total = sum(r["server_energy_joules"] for r in batch_csv_rows if r["phase"] == "infer")
        Ttotal = Tserver

        Energy_per_batch = Energy_total / max(n_batches_test, 1)
        Energy_per_node = Energy_total / n_test if n_test > 0 else 0.0

        summary_path = f"plain_summary_{ts}.csv"
        with open(summary_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Tserver", Tserver])
            writer.writerow(["Ttotal", Ttotal])
            writer.writerow(["Energy_total", Energy_total])
            writer.writerow(["Energy_per_batch", Energy_per_batch])
            writer.writerow(["Energy_per_node", Energy_per_node])

        print(f"[client] Plain summary written → {summary_path}")

        # ── 6. Evaluate ───────────────────────────────────────────────────────────
        print(f"\n5. Evaluating on {n_test} test nodes...")
        y_scores = np.array(all_logits, dtype=np.float64)
        y_pred = (1 / (1 + np.exp(-y_scores)) > 0.5).astype(np.int64)
        y_true = np.array(all_y_true, dtype=np.int64)

        cls_metrics = compute_classification_metrics(y_true, y_pred, y_scores)

        print("\n" + "=" * 70)
        print("=== Test-set Classification Results ===")
        print("=" * 70)
        print(f"  Accuracy  : {cls_metrics['accuracy']:.4f}")
        print(f"  Precision : {cls_metrics['precision']:.4f}")
        print(f"  Recall    : {cls_metrics['recall']:.4f}")
        print(f"  F1 Score  : {cls_metrics['f1']:.4f}")
        print(f"  Test nodes: {n_test}  |  Batches: {n_batches_test}")
        if _SKLEARN:
            print("\n  Per-class report:")
            print(classification_report(y_true, y_pred,
                                        target_names=["BENIGN", "ATTACKER"]))
        print("=" * 70)

        # ── 7. Summary timing (energy/power in micro units) ───────────────────────
        total_infer_t = sum(m.get("seconds", 0.0) for m in server_batch_metrics_list)
        total_infer_e_j = sum(m.get("energy_joules", 0.0) for m in server_batch_metrics_list)
        total_infer_e_uj = total_infer_e_j * 1e6
        avg_infer_power_w = (
            total_infer_e_j / total_infer_t if total_infer_t > 0 else 0.0
        )
        avg_infer_power_uw = avg_infer_power_w * 1e6
        print("\n=== End-to-End Latency Summary (energy in µJ, power in µW) ===")
        print(f"  Training time (client total)        : {train_time:.4f}s")
        print(f"  Inference time (server, all batches): {total_infer_t:.4f}s")
        print(f"  Inference energy total              : {total_infer_e_uj:.2f} µJ")
        if n_batches_test:
            print(f"  Inference energy per batch          : {(total_infer_e_uj / n_batches_test):.2f} µJ")
        if n_test:
            print(f"  Inference energy per node           : {(total_infer_e_uj / n_test):.2f} µJ")
        print(f"  Average inference power             : {avg_infer_power_uw:.2f} µW")

    print("\n✓ Done. Plaintext training + inference (server with raw data).")


if __name__ == "__main__":
    main()