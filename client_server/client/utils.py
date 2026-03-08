"""
Shared utilities for plain and FHE GAT clients: data loading, batching, metrics.
"""

from __future__ import annotations

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
    max_edges: Optional[int] = None,
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

    required = ["src_ip", "dst_ip", "src_bytes", "dst_bytes", "duration", "label"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"CSV missing columns: {missing}")

    if max_edges is not None:
        df = df.head(int(max_edges))

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

    # Edge features and labels (one per row)
    edge_feats_raw = df[["src_bytes", "dst_bytes", "duration"]].to_numpy(dtype=np.float64)
    edge_feats = StandardScaler().fit_transform(edge_feats_raw).astype(np.float64)
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
