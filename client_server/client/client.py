#!/usr/bin/env python3
"""
CKKS-only FHE GAT client: raw TCP transport.

Data flow (train then infer; no test data during training):
  Phase 1 (train): Client sends TRAIN graph only. Server returns
    (out_cts_train, metrics, ct_W_trained).
  Phase 2 (infer): Client sends FULL graph + ct_W_trained.
    Server returns (out_cts, metrics). Client decrypts.

Usage:
  # In-process (no network):
  python -m client_server.client.client --epochs 3

  # TCP server on same machine or LAN:
  # Terminal 1:  python -m client_server.server.server --host 0.0.0.0 --port 9999
  # Terminal 2:  python -m client_server.client.client --epochs 3 --host 192.168.x.x --port 9999

Bootstrap nullptr fix:
  client_keys.py calls cc.EvalBootstrapSetup(levelBudget=[4,4], slots=slots).
  The precomputed tables this builds are NOT serialized with the CryptoContext.
  After deserialization the server must replay the EXACT same call — same
  levelBudget AND same slots value — or EvalBootstrap() crashes with
  "KeySwitchDown(): Input ciphertext is nullptr".
  We send bootstrap_level_budget=[4,4] in the train payload so the serializer
  can do this replay automatically in recv_train_payload().
"""

from __future__ import annotations

import argparse
import socket
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from client_server.openfhe_serializer import (
    send_train_payload,
    recv_train_result,
    recv_status,
)


def decrypt_weights(cc, secret_key, ct_W_trained, a):
    W_trained = []
    for ct in ct_W_trained:
        pt = cc.Decrypt(secret_key, ct)
        W_trained.append(pt.GetCKKSPackedValue())
    a_trained = cc.Decrypt(secret_key, a)
    return W_trained, a_trained


ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from client_server.client.client_keys import (
    create_client_context,
    openfhe_available,
)
from client_server.client.metrics import MetricsRecorder
from client_server.client.plain_client import compute_classification_metrics

from client_server.openfhe_serializer import (
    send_gradient_step_payload,
    recv_gradient_step_result,
    recv_status,
)


def run_tcp_gradient_step(host, port, payload):
    with _open_tcp(host, port) as sock:
        sock.sendall(b"G")
        send_gradient_step_payload(sock, payload)
        recv_status(sock)
        return recv_gradient_step_result(sock)


# ── Metrics printer ───────────────────────────────────────────────────────

import openfhe


def get_ct_size_bytes(ct):
    of = openfhe
    s = of.Serialize(ct, of.BINARY)
    return len(s if isinstance(s, bytes) else s.encode("latin-1"))


def print_server_metrics(metrics_dict: dict, title: str = "Server metrics") -> None:
    if not metrics_dict:
        return
    print("\n" + "=" * 80)
    print(f"=== {title} (time + RSS delta, RSS after) ===")
    print("=" * 80)


def write_server_metrics_txt(path: str, metrics_dict: dict, title: str) -> None:
    """
    Write FHE server metrics (training or inference) to a text file.

    TOTAL RSS Δ is computed as:
        peak_rss_after - initial_rss_after
    NOT as sum of per-step deltas.
    """

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n")
        f.write(
            "step,seconds,rss_delta_bytes,rss_after_bytes,"
            "encrypted,energy_joules,power_watts\n"
        )
        for name, m in metrics_dict.items():
            f.write(
                f"{name},{m.get('seconds', 0.0):.6f},"
                f"{int(m.get('rss_delta_bytes', 0))},"
                f"{int(m.get('rss_after_bytes', 0))},"
                f"{int(bool(m.get('encrypted', True)))},"
                f"{float(m.get('energy_joules', 0.0)):.6f},"
                f"{float(m.get('power_watts', 0.0)):.6f}\n"
            )

    print("\n" + "=" * 80)
    print(f"=== {title} (time + RSS delta, RSS after) ===")
    print("=" * 80)

    total_time = 0.0
    enc_time = 0.0

    rss_values = []

    for name, m in metrics_dict.items():
        sec = m.get("seconds", 0.0)
        rss_delta = m.get("rss_delta_bytes", 0)
        rss_after = m.get("rss_after_bytes", 0)
        enc = m.get("encrypted", True)

        mode = "ENC" if enc else "DEC"
        mb = rss_after / (1024 * 1024) if rss_after else 0.0
        dmb = rss_delta / (1024 * 1024)

        print(
            f"  {name:<36} {mode:<6} {sec:>8.4f}s  "
            f"RSS Δ {dmb:>+8.2f} MB  RSS {mb:>8.2f} MB"
        )

        total_time += sec
        if enc:
            enc_time += sec

        if rss_after > 0:
            rss_values.append(rss_after)

    # ---- Correct TOTAL RSS computation ----
    if rss_values:
        initial_rss = rss_values[0]
        peak_rss = max(rss_values)
        total_rss_delta_mb = (peak_rss - initial_rss) / (1024 * 1024)
    else:
        total_rss_delta_mb = 0.0

    print("-" * 80)
    print(
        f"  {'TOTAL':<36} {'':6} {total_time:>8.4f}s  "
        f"RSS Δ {total_rss_delta_mb:>+8.2f} MB"
    )

    if total_time > 0:
        dec_time = total_time - enc_time
        print()
        print(
            f"  Encrypted ops: {enc_time:>8.4f}s "
            f"({enc_time / total_time * 100:>5.1f}%)"
        )
        print(
            f"  Plaintext ops: {dec_time:>8.4f}s "
            f"({dec_time / total_time * 100:>5.1f}%)"
        )

    print("=" * 80)


# ── Dataset loading (unchanged) ───────────────────────────────────────────


def _default_iot_csv_path():
    client_dir = Path(__file__).resolve().parent
    for candidate in [
        client_dir / "iot.csv",
        ROOT / "examples" / "dataset" / "iot.csv",
        ROOT / "iot.csv",
    ]:
        if candidate.exists():
            return str(candidate)
    return str(client_dir / "iot.csv")


def load_and_preprocess_iot_csv(
    path=None,
    return_labels: bool = False,
    min_nodes: (
        int | None
    ) = None,  # kept for backward compat; ignored — always loads full CSV
):
    """
    Always loads the ENTIRE iot.csv and builds the full graph.
    Subsetting to a desired number of nodes is done AFTER loading via
    extract_connected_subgraph() so the resulting subgraph is guaranteed
    to be well-connected.  The `min_nodes` parameter is accepted but ignored.
    """
    if path is None:
        path = _default_iot_csv_path()
    df = pd.read_csv(path, encoding="latin1")
    df.rename(columns={"ÿsrc_ip": "src_ip"}, inplace=True)

    all_ips = pd.concat([df["src_ip"], df["dst_ip"]]).unique()
    ip_to_idx = {ip: idx for idx, ip in enumerate(all_ips)}
    df["src_idx"] = df["src_ip"].map(ip_to_idx)
    df["dst_idx"] = df["dst_ip"].map(ip_to_idx)

    edge_index = torch.tensor(df[["src_idx", "dst_idx"]].values.T, dtype=torch.long)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in ["src_idx", "dst_idx", "label"]]

    node_features = (
        df.groupby("src_idx")[numeric_cols]
        .mean()
        .reindex(range(len(all_ips)), fill_value=0)
    )
    node_features = StandardScaler().fit_transform(node_features)
    x = torch.tensor(node_features, dtype=torch.float32)

    node_labels = (
        df.groupby("src_idx")["label"]
        .agg(lambda x: x.value_counts().index[0])
        .reindex(range(len(all_ips)), fill_value=0)
    )
    y = torch.tensor(node_labels.values, dtype=torch.long)

    N = x.shape[0]
    F_in = x.shape[1]
    F_out = len(torch.unique(y))

    if return_labels:
        return x, edge_index, N, F_in, F_out, y
    return x, edge_index, N, F_in, F_out


def extract_connected_subgraph(
    x: np.ndarray,
    edge_index: np.ndarray,
    y: np.ndarray,
    n_nodes: int,
    seed: int = 42,
) -> tuple:
    """
    BFS from the highest-degree node to extract a densely connected subgraph
    of exactly *n_nodes* (or fewer if the graph itself is smaller).

    Strategy
    --------
    • Start from the node with the highest degree so the BFS frontier is wide
      from the very first step.
    • Expand the BFS queue in degree-descending order at each step so we keep
      pulling in well-connected nodes rather than low-degree leaf nodes.
    • If the whole graph is disconnected and BFS stalls before reaching
      *n_nodes*, continue from the next highest-degree unvisited node until
      the target is met.

    Returns
    -------
    (x_sub, edge_index_sub, y_sub) all remapped to node IDs [0, n_nodes).
    """
    from collections import deque

    N = x.shape[0]
    n_nodes = min(int(n_nodes), N)

    # Build adjacency list
    adj: list[list[int]] = [[] for _ in range(N)]
    if edge_index.size > 0:
        for s, t in edge_index.T:
            s, t = int(s), int(t)
            if 0 <= s < N and 0 <= t < N:
                adj[s].append(t)
                adj[t].append(s)

    degrees = np.array([len(a) for a in adj], dtype=np.int64)

    visited_order: list[int] = []
    visited_set: set[int] = set()

    # Walk through components in degree-descending order until we have enough nodes
    candidate_seeds = list(np.argsort(degrees)[::-1])  # highest degree first
    rng = np.random.default_rng(seed)

    for start in candidate_seeds:
        if len(visited_order) >= n_nodes:
            break
        if start in visited_set:
            continue

        # BFS from this seed
        queue: deque[int] = deque([start])
        visited_set.add(start)
        while queue and len(visited_order) < n_nodes:
            node = queue.popleft()
            visited_order.append(node)
            # Expand neighbors sorted by degree desc for density
            nbrs = sorted(adj[node], key=lambda nb: degrees[nb], reverse=True)
            for nb in nbrs:
                if nb not in visited_set:
                    visited_set.add(nb)
                    queue.append(nb)

    sub_nodes = np.array(visited_order[:n_nodes], dtype=np.int64)
    sub_set = set(int(v) for v in sub_nodes)
    old_to_new = {int(old): new for new, old in enumerate(sub_nodes)}

    x_sub = x[sub_nodes]
    y_sub = y[sub_nodes]

    edge_list = list(
        {
            (old_to_new[int(s)], old_to_new[int(t)])
            for s, t in edge_index.T
            if int(s) in sub_set and int(t) in sub_set
        }
    )
    edge_index_sub = (
        np.array(edge_list, dtype=np.int64).T
        if edge_list
        else np.zeros((2, 0), dtype=np.int64)
    )

    n_edges = edge_index_sub.shape[1] if edge_index_sub.ndim == 2 else 0
    print(
        f"   [subgraph] extracted {len(sub_nodes)} nodes, {n_edges} edges "
        f"from full graph of {N} nodes"
    )
    return x_sub, edge_index_sub, y_sub


def load_iot_train_test(
    test_ratio: float = 0.2,
    F_in: int = 5,
    path=None,
    seed: int = 42,
    total_nodes=None,
):
    """
    Load IoT data and split using randomized train/test split.
    Matches plaintext client behaviour.

    Returns:
        x_train, edge_index_train, y_train,
        x_test, edge_index_full_remap,
        test_node_global_ids, y_test, x_full,
        edge_index_global   <-- NEW: all edges over all N_eff nodes (remapped),
                                used for graph-aware batch construction so that
                                training batches are guaranteed to have edges.
    """
    x, edge_index, N_full, F_in_raw, F_out, y_full = load_and_preprocess_iot_csv(
        path=path,
        return_labels=True,
        # Always loads the full CSV; total_nodes subsetting happens below via BFS
    )

    x = x.numpy()
    edge_index = edge_index.numpy()
    y_full = y_full.numpy()

    F_in = min(F_in, x.shape[1])
    x = x[:, :F_in].astype(np.float64)

    # ---- Drop globally isolated nodes (degree 0 in the full graph) ----
    N_full = x.shape[0]
    if edge_index.size > 0:
        deg = np.bincount(edge_index.reshape(-1), minlength=N_full)
        keep_nodes = np.where(deg > 0)[0]
        if keep_nodes.size > 0 and keep_nodes.size < N_full:
            mapping = -np.ones(N_full, dtype=np.int64)
            mapping[keep_nodes] = np.arange(keep_nodes.size, dtype=np.int64)
            x = x[keep_nodes]
            y_full = y_full[keep_nodes]
            edge_index = mapping[edge_index]
            valid = (edge_index[0] >= 0) & (edge_index[1] >= 0)
            edge_index = edge_index[:, valid]
            N_full = x.shape[0]

    # ---- If total_nodes requested: BFS-extract a connected subgraph --------
    # This happens on the FULL graph (not a row-truncated CSV), so the
    # extracted subgraph inherits the genuine connectivity of the dataset.
    if total_nodes is not None:
        x, edge_index, y_full = extract_connected_subgraph(
            x, edge_index, y_full, int(total_nodes), seed
        )
        N_full = x.shape[0]

    # ---- Random train/test split on the (subgraph-capped) nodes ----
    # N_full is now the final node count (either total_nodes or the full graph size).
    rng = np.random.default_rng(seed)
    N_eff = N_full

    n_test = max(1, int(N_eff * test_ratio))
    n_train = max(1, N_eff - n_test)

    perm = rng.permutation(N_full)[:N_eff]

    train_ids = perm[:n_train]
    test_ids = perm[n_train:]

    # Remap nodes so train first, then test
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
        if new_edge_list
        else np.zeros((2, 0), dtype=np.int64)
    )

    x_full = x[node_order]
    y_full = y_full[node_order]

    x_train = x_full[:n_train]
    y_train = y_full[:n_train]
    x_test = x_full[n_train:]
    y_test = y_full[n_train:]

    # Train-only edges
    train_nodes = set(range(n_train))
    train_edge_list = [
        [s, t]
        for s, t in edge_index_full_remap.T
        if s in train_nodes and t in train_nodes
    ]
    train_edge_list = list({tuple(e) for e in train_edge_list})

    edge_index_train = (
        np.array(train_edge_list, dtype=np.int64).T
        if train_edge_list
        else np.zeros((2, 0), dtype=np.int64)
    )

    test_node_global_ids = np.arange(n_train, n_train + n_test)

    return (
        x_train,
        edge_index_train,
        y_train,
        x_test,
        edge_index_full_remap,
        test_node_global_ids,
        y_test,
        x_full,
        edge_index_full_remap,  # edge_index_global: full graph over all N_eff nodes
    )


# ── Graph-aware batch builder ─────────────────────────────────────────────


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
    train_set = set(train_list)

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


# ── In-process compute (no network) ──────────────────────────────────────


def run_train_inprocess(
    client_ctx,
    F_in,
    F_out,
    slots,
    num_train,
    edge_index_train,
    ct_x_train,
    ct_labels_train,
    ct_W_list,
    a,
    num_epochs,
    lr,
    bootstrap_weights=True,
):
    from client_server.server.server import compute_fhe_training

    return compute_fhe_training(
        crypto_context=client_ctx.crypto_context,
        public_key=client_ctx.keys.publicKey,
        in_channels=F_in,
        out_channels=F_out,
        slots=slots,
        ct_W_list=ct_W_list,
        a=a,
        negative_slope=0.2,
        num_nodes=num_train,
        edge_index=edge_index_train,
        node_features_enc=ct_x_train,
        ct_labels=ct_labels_train,
        train_mask=np.ones(num_train, dtype=bool),
        num_epochs=num_epochs,
        lr=lr,
        print_metrics=True,
        bootstrap_weights=bootstrap_weights,
    )


def run_infer_inprocess(
    client_ctx, F_in, F_out, slots, N, edge_index_full, ct_x_full, ct_W_trained, a
):
    from client_server.server.server import compute_forward_only

    return compute_forward_only(
        crypto_context=client_ctx.crypto_context,
        public_key=client_ctx.keys.publicKey,
        in_channels=F_in,
        out_channels=F_out,
        slots=slots,
        ct_W_list=ct_W_trained,
        a=a,
        negative_slope=0.2,
        num_nodes=N,
        edge_index=edge_index_full,
        node_features_enc=ct_x_full,
        print_metrics=False,
    )


# ── TCP transport helpers ─────────────────────────────────────────────────


def _open_tcp(host: str, port: int) -> socket.socket:
    """Open a TCP socket with TCP_NODELAY (no Nagle delay on large blobs).

    We use a bounded timeout only for the initial connect; subsequent
    send/recv calls run with blocking I/O (no overall deadline) to allow
    long-running FHE operations.
    """
    # Timeout applies only to connect here
    sock = socket.create_connection((host, port), timeout=30)
    # Disable per-op timeouts so long FHE steps don't hit client-side timeouts
    sock.settimeout(None)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    return sock


def run_tcp_train(host: str, port: int, payload: dict):
    """
    Send train payload over TCP; receive and return
    (out_cts_train, metrics_dict, ct_W_trained).
    """
    from client_server.openfhe_serializer import (
        send_train_payload,
        recv_train_result,
        recv_status,
    )

    print(f"   [tcp] connecting to {host}:{port} for training...")
    with _open_tcp(host, port) as sock:
        sock.sendall(b"T")
        send_train_payload(sock, payload)
        recv_status(sock)  # raises on server error
        return recv_train_result(sock)


def run_tcp_infer(host: str, port: int, payload: dict):
    """
    Send infer payload over TCP; receive and return (out_cts, metrics_dict).
    """
    from client_server.openfhe_serializer import (
        send_infer_payload,
        recv_infer_result,
        recv_status,
    )

    print(f"   [tcp] connecting to {host}:{port} for inference...")
    with _open_tcp(host, port) as sock:
        sock.sendall(b"I")
        send_infer_payload(sock, payload)
        recv_status(sock)
        return recv_infer_result(sock)


# ── Restored context (used when loading pre-trained weights) ─────────────


class _RestoredClientCtx:
    """
    Drop-in replacement for the object returned by create_client_context()
    when loading pre-trained weights from disk.

    Provides the same interface used in main():
        .crypto_context          OpenFHE CryptoContext
        .keys.publicKey          OpenFHE PublicKey
        .keys.secretKey          OpenFHE SecretKey
        .slots                   CKKS slot count
        .encrypt_node_features(x, F_in)   → list[Ciphertext]
        .decrypt_node_features(cts, F_out) → np.ndarray [N, F_out]
        .encrypt_weight_matrix(W, F_in)    → list[Ciphertext]
    """

    class _Keys:
        def __init__(self, pk, sk):
            self.publicKey = pk
            self.secretKey = sk

    def __init__(self, cc, pk, sk, slots: int):
        self.crypto_context = cc
        self.keys = self._Keys(pk, sk)
        self.slots = slots

    # --- encryption ---------------------------------------------------------

    def encrypt_node_features(self, x: np.ndarray, F_in: int):
        """Encrypt each row of x as one CKKS ciphertext (zero-padded to slots)."""
        cc = self.crypto_context
        pk = self.keys.publicKey
        cts = []
        pad = self.slots - F_in
        for row in x:
            vals = [float(v) for v in row[:F_in]] + [0.0] * pad
            pt = cc.MakeCKKSPackedPlaintext(vals)
            cts.append(cc.Encrypt(pk, pt))
        return cts

    def encrypt_weight_matrix(self, W: np.ndarray, F_in: int):
        """Encrypt each row of W (one per output channel) as one ciphertext."""
        return self.encrypt_node_features(W, F_in)

    # --- decryption ---------------------------------------------------------

    def decrypt_node_features(self, out_cts: list, F_out: int) -> np.ndarray:
        """Decrypt output ciphertexts; return array of shape [N, F_out]."""
        cc = self.crypto_context
        sk = self.keys.secretKey
        rows = []
        for ct in out_cts:
            pt = cc.Decrypt(sk, ct)
            vals = pt.GetRealPackedValue()
            rows.append([float(vals[k]) for k in range(F_out)])
        return np.array(rows, dtype=np.float64)


# ── Main ──────────────────────────────────────────────────────────────────


def main() -> None:
    if not openfhe_available():
        print("ERROR: OpenFHE not installed. pip install openfhe")
        sys.exit(1)

    import numpy as np

    parser = argparse.ArgumentParser(
        description="IoT Malicious Node Prediction (FHE client, CKKS-only)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=60,
        help="Graph nodes per training batch to send to the FHE server. ",
    )
    parser.add_argument(
        "--epochs", type=int, default=3, help="Training epochs (default 3)"
    )
    parser.add_argument(
        "--lr", type=float, default=0.01, help="Learning rate (default 0.01)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="",
        help="TCP server host (e.g. 192.168.1.10). " "Omit for in-process mode.",
    )
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
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/test split and batching (default 42)",
    )
    parser.add_argument(
        "--port", type=int, default=9999, help="TCP server port (default 9999)"
    )
    parser.add_argument("--return_encrypted_only", action="store_true")
    parser.add_argument(
        "--mult_depth", type=int, default=25, help="CKKS levels per epoch (default 25)"
    )
    parser.add_argument(
        "--ring_dim",
        type=int,
        default=16384,
        help="CKKS ring dimension (default 16384)",
    )
    parser.add_argument(
        "--no_bootstrap", action="store_true", help="Disable weight bootstrapping"
    )
    parser.add_argument("--data", type=str, default=None, help="Path to iot.csv")
    parser.add_argument(
        "--save_weights",
        type=str,
        default=None,
        metavar="DIR",
        help="Directory to save trained weight ciphertexts + key material after "
        "training.  Created if absent.  Defaults to fhe_weights_<timestamp>.",
    )
    parser.add_argument(
        "--load_weights",
        type=str,
        default=None,
        metavar="DIR",
        help="Directory from which to load previously saved weight ciphertexts + "
        "key material.  When given, steps 2-4 (key-gen, encrypt, training) "
        "are skipped entirely and the saved weights are used for inference.",
    )
    parser.add_argument(
        "--train_only",
        action="store_true",
        help="Run training only and exit (no inference). Weights are saved to --save_weights.",
    )
    parser.add_argument(
        "--infer_only",
        action="store_true",
        help="Run inference only (requires --load_weights). Skip all training steps.",
    )
    args = parser.parse_args()

    F_in, F_out, slots = 5, 1, 8

    ts = time.strftime("%Y%m%d_%H%M%S")
    client_metrics_path = f"client_fhe_metrics_{ts}.txt"
    server_train_metrics_path = f"server_fhe_metrics_train_{ts}.txt"
    server_infer_metrics_path = f"server_fhe_metrics_infer_{ts}.txt"

    print("=" * 60)
    print("IoT Malicious Node Prediction (FHE GAT, CKKS-only)")
    print(
        f"  batch_size={args.batch_size}  epochs={args.epochs}  "
        f"lr={args.lr}  test_ratio={args.test_ratio}  "
        f"total_nodes={args.total_nodes or 'ALL'}"
    )
    if args.host:
        print(f"  Transport: TCP  {args.host}:{args.port}")
    else:
        print("  Transport: in-process (no network)")
    print("=" * 60)

    # 1. Load data (same randomized split as plaintext baseline)
    print("\n1. Loading IoT train + test...")
    (
        x_train,
        edge_index_train,
        y_train,
        x_test,
        edge_index_full,
        test_node_global_ids,
        y_test,
        x_full,
        edge_index_global,  # full graph edges over all N_eff nodes
    ) = load_iot_train_test(
        test_ratio=args.test_ratio,
        F_in=F_in,
        path=args.data,
        seed=args.seed,
        total_nodes=args.total_nodes,
    )
    num_train = x_train.shape[0]
    num_test = x_test.shape[0]
    N_total = x_full.shape[0]
    n_test = x_test.shape[0]

    # ---- Build graph-aware training batches --------------------------------
    # Use the FULL graph (edge_index_global) so neighbour look-ups are not
    # restricted to train-only edges.  BFS grouping guarantees every batch
    # has ≥1 edge — no more silent skips.
    print("\n   Building graph-aware training batches from full IoT graph...")
    train_batches = make_connected_batches(
        train_node_ids=np.arange(num_train),
        edge_index_global=edge_index_global,
        batch_size=args.batch_size,
    )
    n_batches_train = len(train_batches)
    # -----------------------------------------------------------------------

    n_batches_test = max(1, (n_test + args.batch_size - 1) // args.batch_size)
    print(
        f"   Train nodes={num_train}, Number of test nodes={n_test}, Total nodes={N_total}"
    )
    print(
        f"   Batches for train={n_batches_train} (batch_size={args.batch_size}, graph-aware)"
    )
    print(f"   Batches for inference={n_batches_test} (batch_size={args.batch_size})")

    client_metrics = MetricsRecorder()

    # ── Resolve weight-save path ──────────────────────────────────────────────
    weights_save_dir = args.save_weights or f"fhe_weights_{ts}"

    # ══════════════════════════════════════════════════════════════════════════
    #  BRANCH A: load pre-trained weights →  key-gen + training entirely
    # ══════════════════════════════════════════════════════════════════════════
    if args.load_weights:
        print(f"\n[weights] Loading pre-trained weights from: {args.load_weights}")

        from client_server.openfhe_serializer import load_trained_weights

        w = load_trained_weights(args.load_weights)

        # Create fresh crypto context (new keys!)
        client_ctx = create_client_context(
            in_channels=w["F_in"],
            out_channels=w["F_out"],
            slots=w["slots"],
            mult_depth=args.mult_depth,
            scale_mod_size=50,
            ring_dim=args.ring_dim,
            bootstrap=not args.no_bootstrap,
        )

        # Re-encrypt plaintext weights
        W_loaded = np.asarray(w["W_list"], dtype=np.float64)
        ct_W_trained = client_ctx.encrypt_weight_matrix(W_loaded, w["F_in"])

        a = np.asarray(w["a"], dtype=np.float64)

        F_in = w["F_in"]
        F_out = w["F_out"]

        server_train_metrics_list = []

        print(
            f"   Loaded {len(ct_W_trained)} weight ciphertext(s). " "Skipping training."
        )

    # ══════════════════════════════════════════════════════════════════════════
    #  BRANCH B: normal path — generate keys, encrypt, run mini-batch training
    # ══════════════════════════════════════════════════════════════════════════
    else:
        # 2. Key generation
        print("\n2. Client: generating keys...")
        with client_metrics.step("client_context_keygen", encrypted=True):
            client_ctx = create_client_context(
                in_channels=F_in,
                out_channels=F_out,
                slots=slots,
                mult_depth=args.mult_depth,
                scale_mod_size=50,
                ring_dim=args.ring_dim,
                bootstrap=not args.no_bootstrap,
            )
        print("   Done. Client keeps secret key.")

        # 3. Encrypt train data
        print("\n3. Client: encrypting initial weights + train features/labels...")

        np.random.seed(42)
        W_init = np.random.randn(F_out, F_in).astype(np.float64) * 0.1
        concat_dim = 2 * F_out
        a = np.random.randn(concat_dim).astype(np.float64) * 0.1

        with client_metrics.step("client_encrypt_train", encrypted=True):
            ct_W_list = client_ctx.encrypt_weight_matrix(W_init, F_in)
        print("   Done. (Train data only; test data not sent for training.)")

        # 4. Mini-batch FHE training
        print("\n4. Mini-batch FHE training...")

        ct_W_trained = ct_W_list
        server_train_metrics_list = []

        for b_idx, batch_node_ids in enumerate(train_batches):

            batch_nodes = np.array(batch_node_ids, dtype=np.int64)

            # ---- Build subgraph , already we make sure in make_connected using BFS
            all_nodes = sorted(batch_nodes)
            old_to_local = {g: i for i, g in enumerate(all_nodes)}

            # Clamp feature/label look-ups to valid indices
            # (neighbours may include test-set nodes, so clamp to x_full)
            x_batch = x_full[all_nodes]

            # Build induced subgraph edges from full graph
            edge_index_batch = (
                np.array(
                    [
                        [old_to_local[int(s)], old_to_local[int(t)]]
                        for s, t in edge_index_global.T
                        if int(s) in old_to_local and int(t) in old_to_local
                    ],
                    dtype=np.int64,
                ).T
                if len(all_nodes) > 0
                else np.zeros((2, 0), dtype=np.int64)
            )

            # Sanity-check: the BFS batching guarantees edges, but guard anyway.
            if edge_index_batch.ndim != 2 or edge_index_batch.shape[1] == 0:
                print(
                    f"   [WARNING] Batch {b_idx+1}/{n_batches_train} unexpectedly "
                    "has no edges after subgraph construction — skipping."
                )
                continue

            # Labels: use x_full / y labels, clamped to available indices
            import numpy as _np

            y_batch = _np.concatenate([y_train, y_test])[all_nodes]

            print(
                f"\n   Batch {b_idx+1}/{n_batches_train} "
                f"(core nodes={len(batch_nodes)}, "
                f"subgraph nodes={len(all_nodes)}, "
                f"edges={edge_index_batch.shape[1]})"
            )

            # ---- Encrypt batch ----
            with client_metrics.step(f"encrypt_train_batch_{b_idx}", encrypted=True):
                ct_x_batch = client_ctx.encrypt_node_features(x_batch, F_in)
                ct_labels_batch = [
                    client_ctx.crypto_context.Encrypt(
                        client_ctx.keys.publicKey,
                        client_ctx.crypto_context.MakeCKKSPackedPlaintext(
                            [float(y)] * slots
                        ),
                    )
                    for y in y_batch
                ]

            gradient_step_payload = {
                "crypto_context": client_ctx.crypto_context,
                "public_key": client_ctx.keys.publicKey,
                "in_channels": F_in,
                "out_channels": F_out,
                "slots": slots,
                "ct_W_list": ct_W_trained,
                "a": a,
                "negative_slope": 0.2,
                "num_nodes": len(all_nodes),
                "edge_index": edge_index_batch,
                "node_features_enc": ct_x_batch,
                "ct_labels": ct_labels_batch,
                "lr": args.lr,
                "num_epochs": args.epochs,
            }

            # ---- TCP OR IN-PROCESS ----
            if args.host:
                ct_W_trained, batch_metrics = run_tcp_gradient_step(
                    args.host, args.port, gradient_step_payload
                )
            else:
                from client_server.server.server import compute_fhe_training_batch

                ct_W_trained, batch_metrics = compute_fhe_training_batch(
                    crypto_context=gradient_step_payload["crypto_context"],
                    public_key=gradient_step_payload["public_key"],
                    in_channels=gradient_step_payload["in_channels"],
                    out_channels=gradient_step_payload["out_channels"],
                    slots=gradient_step_payload["slots"],
                    ct_W_list=gradient_step_payload["ct_W_list"],
                    a=gradient_step_payload["a"],
                    negative_slope=gradient_step_payload["negative_slope"],
                    num_nodes=gradient_step_payload["num_nodes"],
                    edge_index=gradient_step_payload["edge_index"],
                    node_features_enc=gradient_step_payload["node_features_enc"],
                    ct_labels=gradient_step_payload["ct_labels"],
                    lr=gradient_step_payload["lr"],
                    num_epochs=gradient_step_payload["num_epochs"],
                )

            server_train_metrics_list.append(batch_metrics)

            batch_time = sum(
                step.get("seconds", 0.0) for step in batch_metrics.values()
            )

            print(
                f"      Local epochs={args.epochs}  " f"server_time={batch_time:.4f}s"
            )
            # ── End of BRANCH B (training loop) ──────────────────────────────────
        # Close the `else:` block that started at "BRANCH B".

    # ---- Aggregate training metrics (both branches) ----
    aggregated_train_metrics = {}
    for idx, batch_m in enumerate(server_train_metrics_list):
        for name, step in batch_m.items():
            key = f"batch_{idx}_{name}"
            aggregated_train_metrics[key] = step

    if aggregated_train_metrics:
        write_server_metrics_txt(
            server_train_metrics_path,
            aggregated_train_metrics,
            "FHE Server metrics (mini-batch training)",
        )
    # Decrypt weight ciphertexts → numpy matrix (F_out, F_in)
    W_rows = []
    for ct in ct_W_trained:
        pt = client_ctx.crypto_context.Decrypt(client_ctx.keys.secretKey, ct)
        vals = pt.GetRealPackedValue()
        W_rows.append([float(vals[i]) for i in range(F_in)])

    W_trained = np.array(W_rows, dtype=np.float64)
    a_trained = a

    # ── Save trained weights (only after fresh training, not when loading) ───
    if not args.load_weights:
        print(f"\n[weights] Saving trained weights → {weights_save_dir} ...")
        from client_server.openfhe_serializer import save_trained_weights

        save_trained_weights(
            weights_save_dir,
            W_list=W_trained,
            a=a_trained,
            slots=slots,
            F_in=F_in,
            F_out=F_out,
        )
        print(
            f"   Weights saved.  Re-run with --load_weights {weights_save_dir} "
            "to skip training next time."
        )

    # ── train_only mode: exit after training ──────────────────────────────────
    if args.train_only:
        print("\n✓ Training complete (train_only mode). Exiting.")
        return

    # ── infer_only mode: require --load_weights ───────────────────────────────
    if args.infer_only and not args.load_weights:
        raise ValueError(
            "--infer_only requires --load_weights to provide weight ciphertexts."
        )

    # 5. Encrypt graph
    print("\n5. Client: encrypting  graph (test) for inference...")
    with client_metrics.step("client_encrypt_infer", encrypted=True):
        ct_x_full = client_ctx.encrypt_node_features(x_test, F_in)
    print("   Done. Test data only; train data not sent for inference.")

    print(f"\n6. Server: batched FHE inference over {n_test} test nodes...")
    all_scores: list[float] = []
    all_labels: list[int] = []
    server_infer_metrics_list: list[dict] = []
    batch_csv_rows = []

    for b_idx in range(n_batches_test):
        start = b_idx * args.batch_size
        end = min(start + args.batch_size, n_test)

        local_test_ids = np.arange(start, end)
        global_test_ids = num_train + local_test_ids

        # ---- Build batch subgraph ----
        all_nodes_global = list(global_test_ids)
        old_to_local = {g: l for l, g in enumerate(all_nodes_global)}

        x_batch = x_full[all_nodes_global]

        # Build induced subgraph edges; ensure shape is (2, E) or (2, 0)
        if len(all_nodes_global) > 0:
            edges = [
                [old_to_local[int(s)], old_to_local[int(t)]]
                for s, t in edge_index_full.T
                if int(s) in old_to_local and int(t) in old_to_local
            ]
            if edges:
                edge_index_batch = np.array(edges, dtype=np.int64).T
            else:
                edge_index_batch = np.zeros((2, 0), dtype=np.int64)
        else:
            edge_index_batch = np.zeros((2, 0), dtype=np.int64)

        # If this inference batch has no edges, skip it to avoid invalid FHEGraph.
        if (
            edge_index_batch.ndim != 2
            or edge_index_batch.shape[0] != 2
            or edge_index_batch.shape[1] == 0
        ):
            print(
                f"   Inference batch {b_idx+1}/{n_batches_test} has no edges; "
                "skipping FHE inference for this batch."
            )
            continue

        # ---- Encrypt batch only ----
        with client_metrics.step(f"encrypt_batch_{b_idx}", encrypted=True):
            _t0 = time.perf_counter()
            ct_x_batch = client_ctx.encrypt_node_features(x_batch, F_in)
            encryption_time = time.perf_counter() - _t0

        infer_payload = {
            "crypto_context": client_ctx.crypto_context,
            "public_key": client_ctx.keys.publicKey,
            "in_channels": F_in,
            "out_channels": F_out,
            "slots": slots,
            "ct_W_list": ct_W_trained,
            "a": a,
            "negative_slope": 0.2,
            "num_nodes": len(all_nodes_global),
            "edge_index": edge_index_batch,
            "node_features_enc": ct_x_batch,
            "print_metrics": False,
        }

        # Per-batch FHE payload sizes (ciphertexts only)
        ct_x_batch_size_bytes = sum(get_ct_size_bytes(ct) for ct in ct_x_batch)
        ct_W_list_size_bytes = sum(get_ct_size_bytes(ct) for ct in ct_W_trained)
        total_batch_size_bytes = ct_x_batch_size_bytes + ct_W_list_size_bytes
        if args.host:
            out_cts, batch_metrics = run_tcp_infer(args.host, args.port, infer_payload)
        else:
            out_cts, batch_metrics = run_infer_inprocess(
                client_ctx,
                F_in,
                F_out,
                slots,
                len(all_nodes_global),
                edge_index_batch,
                ct_x_batch,
                ct_W_trained,
                a,
            )

        # ----7.  Decrypt batch ----
        with client_metrics.step(f"decrypt_batch_{b_idx}", encrypted=False):
            _t0 = time.perf_counter()
            output = client_ctx.decrypt_node_features(out_cts, F_out)
            decryption_time = time.perf_counter() - _t0

        for i, g in enumerate(global_test_ids):
            local_idx = old_to_local[g]
            score = float(output[local_idx, 0])
            all_scores.append(score)
            all_labels.append(int(y_test[start + i]))

        # Save RAM
        del out_cts
        del ct_x_batch

        # Aggregate server metrics for this batch from per-step metrics
        if batch_metrics:
            server_time_seconds = sum(
                m.get("seconds", 0.0) for m in batch_metrics.values()
            )
            max_rss_after_bytes = max(
                (m.get("rss_after_bytes", 0) for m in batch_metrics.values()),
                default=0,
            )
            server_rss_after_mb = max_rss_after_bytes / (1024 * 1024)
            server_energy_joules = sum(
                m.get("energy_joules", 0.0) for m in batch_metrics.values()
            )
            # For power, prefer time-weighted average if possible; fall back to max
            if server_time_seconds > 0:
                server_power_watts = server_energy_joules / server_time_seconds
            else:
                server_power_watts = max(
                    (m.get("power_watts", 0.0) for m in batch_metrics.values()),
                    default=0.0,
                )
        else:
            server_time_seconds = 0.0
            server_rss_after_mb = 0.0
            server_energy_joules = 0.0
            server_power_watts = 0.0

        batch_csv_rows.append(
            {
                "batch": b_idx,
                "nodes_in_batch": len(local_test_ids),
                "client_encryption_time": encryption_time,
                "client_decryption_time": decryption_time,
                "ct_x_batch_size_bytes": ct_x_batch_size_bytes,
                "ct_W_list_size_bytes": ct_W_list_size_bytes,
                "total_ciphertext_batch_size_bytes": total_batch_size_bytes,
                "server_time_seconds": server_time_seconds,
                "server_rss_after_mb": server_rss_after_mb,
                "server_energy_joules": server_energy_joules,
                "server_power_watts": server_power_watts,
            }
        )

        server_infer_metrics_list.append(batch_metrics)

    # ---- Write batch metrics to CSV ----
    import csv

    ts = time.strftime("%Y%m%d_%H%M%S")
    csv_path = f"fhe_batch_metrics_{ts}.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=batch_csv_rows[0].keys())
        writer.writeheader()
        writer.writerows(batch_csv_rows)

    print(f"[client] FHE per-batch metrics written → {csv_path}")

    Tenc = sum(r["client_encryption_time"] for r in batch_csv_rows)
    Tserver = sum(r["server_time_seconds"] for r in batch_csv_rows)
    Tdec = sum(r["client_decryption_time"] for r in batch_csv_rows)

    Ttotal = Tenc + Tserver + Tdec
    Energy_total = sum(r["server_energy_joules"] for r in batch_csv_rows)

    Energy_per_batch = Energy_total / len(batch_csv_rows) if batch_csv_rows else 0.0
    Energy_per_node = Energy_total / len(all_labels) if all_labels else 0.0

    summary_path = f"fhe_summary_{ts}.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Tenc", Tenc])
        writer.writerow(["Tserver", Tserver])
        writer.writerow(["Tdec", Tdec])
        writer.writerow(["Ttotal", Ttotal])
        writer.writerow(["Energy_total", Energy_total])
        writer.writerow(["Energy_per_batch", Energy_per_batch])
        writer.writerow(["Energy_per_node", Energy_per_node])

    import numpy as np

    y_scores = np.asarray(all_scores, dtype=np.float64)
    y_true = np.asarray(all_labels, dtype=np.int64)
    y_pred = (1.0 / (1.0 + np.exp(-y_scores)) > 0.5).astype(np.int64)

    # 8. Test nodes prediction (only for nodes that received scores)
    print(f"\n--- Test nodes prediction ---")
    for i in range(len(y_true)):
        score = float(y_scores[i])
        pred = "ATTACKER" if score > 0.5 else "BENIGN"
        actual_label = int(y_true[i])
        actual_str = "ATTACKER" if actual_label == 1 else "BENIGN"
        print(
            f"  Node {i}:  actual={actual_str} ({actual_label})  "
            f"predicted={pred} (score={score:.4f})"
        )

    cls_metrics = compute_classification_metrics(y_true, y_pred, y_scores)

    print("\n=== FHE Test-set Classification Results ===")
    print(f"Accuracy  : {cls_metrics['accuracy']:.4f}")
    print(f"Precision : {cls_metrics['precision']:.4f}")
    print(f"Recall    : {cls_metrics['recall']:.4f}")
    print(f"F1 Score  : {cls_metrics['f1']:.4f}")
    print(f"Test nodes: {len(y_true)}  |  Batches: {n_batches_test}")

    # 9. End-to-end latency and energy summary (encrypted path)
    client_m = client_metrics.to_dict()

    def _sum_client(names):
        return sum(client_m[n].get("seconds", 0.0) for n in names if n in client_m)

    def _sum_client_energy(names):
        return sum(
            client_m[n].get("energy_joules", 0.0) for n in names if n in client_m
        )

    # Encryption: initial train encryption + per-batch test encryption
    encrypt_batch_steps = [
        name for name in client_m if name.startswith("encrypt_batch_")
    ]
    decrypt_batch_steps = [
        name for name in client_m if name.startswith("decrypt_batch_")
    ]

    t_enc = _sum_client(
        ["client_encrypt_train", "client_encrypt_infer"] + encrypt_batch_steps
    )
    e_enc = _sum_client_energy(
        ["client_encrypt_train", "client_encrypt_infer"] + encrypt_batch_steps
    )
    t_dec = _sum_client(decrypt_batch_steps)
    e_dec = _sum_client_energy(decrypt_batch_steps)

    t_server_train = sum(
        step.get("seconds", 0.0)
        for batch_m in server_train_metrics_list
        for step in batch_m.values()
    )

    e_server_train = sum(
        step.get("energy_joules", 0.0)
        for batch_m in server_train_metrics_list
        for step in batch_m.values()
    )

    t_server_infer = sum(
        step.get("seconds", 0.0)
        for batch_m in server_infer_metrics_list
        for step in batch_m.values()
    )
    e_server_infer = sum(
        step.get("energy_joules", 0.0)
        for batch_m in server_infer_metrics_list
        for step in batch_m.values()
    )

    t_server = t_server_train + t_server_infer
    e_server = e_server_train + e_server_infer
    t_total = t_enc + t_server + t_dec
    e_total = e_enc + e_server + e_dec

    print("\n=== FHE End-to-End Latency / Energy Summary ===")
    print(f"  Encryption time (client)        : {t_enc:.4f}s")
    print(f"  Training time   (server)        : {t_server_train:.4f}s")
    print(f"  Inference time  (server)        : {t_server_infer:.4f}s")
    print(f"  Decryption time (client)        : {t_dec:.4f}s")
    print(f"  T_total (enc+server+dec)        : {t_total:.4f}s")
    if e_total > 0.0:
        print(f"  Encryption energy (client)      : {e_enc:.6f} J")
        print(f"  Server energy (train+infer)     : {e_server:.6f} J")
        print(f"  Decryption energy (client)      : {e_dec:.6f} J")
        print(f"  Total energy (1000-node equiv.) : {e_total:.6f} J")
        if N_total > 0:
            print(f"  Energy per node                 : {e_total / N_total:.6f} J")

    # 10. Print client metrics and summary
    client_metrics.print_report()

    print("\n✓ Done. FHE training + inference (server never had secret key).")


if __name__ == "__main__":
    main()
