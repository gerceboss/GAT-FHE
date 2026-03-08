"""
Shared utilities for plain and FHE GAT clients: data loading, batching, metrics.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

try:
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
    )
    _SKLEARN = True
except ImportError:
    _SKLEARN = False


# ── Dataset paths ─────────────────────────────────────────────────────────────

def default_iot_csv_path() -> str:
    """Return path to iot.csv (client dir, examples/dataset, or repo root)."""
    here = Path(__file__).resolve().parent
    for candidate in [
        here / "iot.csv",
        here.parent.parent / "examples" / "dataset" / "iot.csv",
        here.parent.parent / "iot.csv",
    ]:
        if candidate.exists():
            return str(candidate)
    return str(here / "iot.csv")


# ── Edge-based loading (one row = one edge, label per edge) ────────────────────

def load_iot_edge_train_test(
    path: Optional[str] = None,
    test_ratio: float = 0.2,
    seed: int = 42,
    max_rows_dataset: Optional[int] = None,
) -> Tuple[
    np.ndarray,  # x_nodes (N, node_dim)
    np.ndarray,  # edge_index_full (2, E)
    np.ndarray,  # edge_feats (E, 3)
    np.ndarray,  # edge_labels (E,)
    np.ndarray,  # train_edge_ids
    np.ndarray,  # test_edge_ids
    int,         # num_nodes
]:
    """
    Load IoT CSV for edge classification: each row is one edge (communication).
    Nodes = unique IPs; node features = structural (in_degree, out_degree).
    Edge features = src_bytes, dst_bytes, duration per row.
    Edge labels = label column per row.
    Returns train/test split by edge index (not by node).
    """
    if path is None:
        path = default_iot_csv_path()
    df = pd.read_csv(path, encoding="latin1")
    df.rename(columns={"ÿsrc_ip": "src_ip"}, inplace=True)

    # All 6 columns used: src_ip, dst_ip → graph topology; src_bytes, dst_bytes, duration → features; label → target
    required = ["src_ip", "dst_ip", "src_bytes", "dst_bytes", "duration", "label"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"CSV missing columns: {missing}")

    if max_rows_dataset is not None:
        df = df.head(int(max_rows_dataset))

    all_ips = pd.concat([df["src_ip"], df["dst_ip"]]).unique()
    ip_to_idx = {ip: idx for idx, ip in enumerate(all_ips)}
    N = len(all_ips)
    src = df["src_ip"].map(ip_to_idx).to_numpy(dtype=np.int64)
    dst = df["dst_ip"].map(ip_to_idx).to_numpy(dtype=np.int64)
    E = len(src)
    edge_index_full = np.stack([src, dst], axis=0)  # (2, E)

    # Node features: in-degree, out-degree (structural)
    deg_in = np.bincount(dst, minlength=N)
    deg_out = np.bincount(src, minlength=N)
    x_nodes_raw = np.stack([deg_in, deg_out], axis=1).astype(np.float64)
    x_nodes = StandardScaler().fit_transform(x_nodes_raw).astype(np.float64)

    # Edge features (sent as line-graph node features): src_bytes, dst_bytes, duration only
    edge_feats_raw = df[["src_bytes", "dst_bytes", "duration"]].to_numpy(dtype=np.float64)
    edge_feats = StandardScaler().fit_transform(edge_feats_raw).astype(np.float64)
    # Target per link (used as line-graph node labels for training/eval)
    edge_labels = (df["label"].to_numpy() > 0).astype(np.int64)

    # Train/test split by edges
    rng = np.random.default_rng(seed)
    perm = rng.permutation(E)
    n_test = max(1, int(E * test_ratio))
    n_train = E - n_test
    train_edge_ids = perm[:n_train]
    test_edge_ids = perm[n_train:]

    return (
        x_nodes,
        edge_index_full,
        edge_feats,
        edge_labels,
        train_edge_ids,
        test_edge_ids,
        N,
    )


def build_edge_batch(
    edge_ids: np.ndarray,
    edge_index_full: np.ndarray,
    edge_feats: np.ndarray,
    edge_labels: np.ndarray,
    x_nodes: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build one edge-batch subgraph: nodes involved in the given edges,
    local edge_index, edge features and labels for those edges.
    Returns (x_nodes_batch, edge_index_local, edge_feats_batch, y_edges_batch, node_ids_global).
    """
    edge_index_batch = edge_index_full[:, edge_ids]  # (2, batch_size)
    nodes_global = np.unique(edge_index_batch.ravel())
    old_to_local = {int(n): i for i, n in enumerate(nodes_global)}
    x_batch = x_nodes[nodes_global]
    edge_feats_batch = edge_feats[edge_ids]
    y_batch = edge_labels[edge_ids]
    # Remap edge indices to local
    src_local = np.array([old_to_local[int(s)] for s in edge_index_batch[0]], dtype=np.int64)
    dst_local = np.array([old_to_local[int(d)] for d in edge_index_batch[1]], dtype=np.int64)
    edge_index_local = np.stack([src_local, dst_local], axis=0)
    return x_batch, edge_index_local, edge_feats_batch, y_batch, nodes_global


def derive_node_labels_from_edges(
    edge_index_local: np.ndarray,
    y_edges_batch: np.ndarray,
    num_nodes: int,
) -> np.ndarray:
    """
    Derive one label per node from edge labels (for FHE GAT training on server).
    Node label = 1 if any incident edge in the batch has label 1, else 0.
    """
    node_labels = np.zeros(num_nodes, dtype=np.int64)
    for e in range(edge_index_local.shape[1]):
        s = int(edge_index_local[0, e])
        t = int(edge_index_local[1, e])
        if y_edges_batch[e] > 0:
            node_labels[s] = 1
            node_labels[t] = 1
    return node_labels


# ── Line graph (dual): edges become nodes ───────────────────────────────────────

def build_line_graph(
    edge_index: np.ndarray,
    edge_feats: np.ndarray,
    edge_labels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the line graph (dual) of the original graph.

    In the line graph, each NODE = one EDGE of the original graph. Two nodes are
    adjacent iff the corresponding original edges share a vertex. So GAT runs
    on nodes (with features on nodes); one node = one original edge, so predictions
    are naturally per edge. No separate edge head needed.

    Parameters
    ----------
    edge_index : (2, E) int64
        Original graph edges (src, dst) per column.
    edge_feats : (E, d) float
        Feature vector per original edge (e.g. src_bytes, dst_bytes, duration).
    edge_labels : (E,) int
        Label per original edge.

    Returns
    -------
    x_line : (E, d) float
        Node features of the line graph (= edge_feats).
    edge_index_line : (2, E_line) int64
        Edges of the line graph (adjacency: two original edges connected if they share a vertex).
    y_line : (E,) int64
        Node labels of the line graph (= edge_labels).
    """
    E = edge_index.shape[1]
    edge_index = np.asarray(edge_index, dtype=np.int64)
    # For each original vertex v, collect edge indices incident to v
    # Then add an edge in the line graph between every pair of distinct edges incident to v
    from collections import defaultdict
    v_to_edges: Dict[int, list] = defaultdict(list)
    for e in range(E):
        s, t = int(edge_index[0, e]), int(edge_index[1, e])
        v_to_edges[s].append(e)
        v_to_edges[t].append(e)
    line_edges = []
    seen = set()
    for v, edge_list in v_to_edges.items():
        for i in range(len(edge_list)):
            for j in range(i + 1, len(edge_list)):
                ei, ej = edge_list[i], edge_list[j]
                if ei > ej:
                    ei, ej = ej, ei
                if (ei, ej) not in seen:
                    seen.add((ei, ej))
                    line_edges.append([ei, ej])
                    line_edges.append([ej, ei])  # both directions for GAT
    x_line = np.asarray(edge_feats, dtype=np.float64)
    y_line = np.asarray(edge_labels, dtype=np.int64)
    edge_index_line = (
        np.array(line_edges, dtype=np.int64).T
        if line_edges
        else np.zeros((2, 0), dtype=np.int64)
    )
    return x_line, edge_index_line, y_line


def build_line_graph_batch(
    batch_line_node_ids: np.ndarray,
    edge_index_line: np.ndarray,
    x_line: np.ndarray,
    y_line: np.ndarray,
    max_degree_per_node: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build one batch subgraph of the line graph for training or inference.

    Uses only the batch nodes (no 1-hop neighbours). Edges in the subgraph are
    those between batch nodes. If max_degree_per_node is set, each node keeps at
    most that many in-edges (limits memory for FHE; use e.g. 8–15 for 8 GB RAM).

    Returns
    -------
    x_batch, edge_index_batch, y_batch, target_indices
    """
    batch_set = set(int(n) for n in batch_line_node_ids)
    all_nodes = sorted(batch_set)
    old_to_local = {n: i for i, n in enumerate(all_nodes)}
    x_batch = x_line[all_nodes]
    y_batch = y_line[all_nodes]
    edge_list = []
    for e in range(edge_index_line.shape[1]):
        s, t = int(edge_index_line[0, e]), int(edge_index_line[1, e])
        if s in old_to_local and t in old_to_local:
            edge_list.append([old_to_local[s], old_to_local[t]])

    if max_degree_per_node is not None and max_degree_per_node >= 1 and edge_list:
        by_dst: Dict[int, list] = defaultdict(list)
        for s, t in edge_list:
            by_dst[t].append((s, t))
        edge_list = []
        for t, pairs in by_dst.items():
            keep = pairs[:max_degree_per_node]
            edge_list.extend(keep)

    edge_index_batch = (
        np.array(edge_list, dtype=np.int64).T
        if edge_list
        else np.zeros((2, 0), dtype=np.int64)
    )
    target_indices = np.array([old_to_local[n] for n in batch_line_node_ids], dtype=np.int64)
    return x_batch, edge_index_batch, y_batch, target_indices


# ── Classification metrics ─────────────────────────────────────────────────────

def compute_classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_scores: np.ndarray
) -> Dict[str, float]:
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


def make_connected_batches(train_node_ids, edge_index_global, batch_size):

    train_nodes = set(int(n) for n in train_node_ids)

    # Build adjacency
    adj = {n: [] for n in train_nodes}
    for s, t in edge_index_global.T:
        s, t = int(s), int(t)
        if s in adj and t in adj:
            adj[s].append(t)
            adj[t].append(s)

    from collections import deque

    unvisited = set(train_nodes)
    batches = []

    while unvisited:

        seed = unvisited.pop()
        queue = deque([seed])
        batch = [seed]

        while queue and len(batch) < batch_size:

            node = queue.popleft()

            for nb in adj[node]:
                if nb in unvisited:
                    unvisited.remove(nb)
                    queue.append(nb)
                    batch.append(nb)

                if len(batch) >= batch_size:
                    break

        batches.append(batch)

    return batches