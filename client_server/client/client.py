#!/usr/bin/env python3
"""
CKKS-only FHE GAT client: raw TCP transport. Line-graph (dual graph) formulation.

We build the line graph once: each node = one original edge; features on nodes.
  Phase 1 (train): For each batch of train edges, client builds a line-graph
    subgraph (build_line_graph_batch), encrypts node features and labels, sends
    gradient-step payload (with train_mask on target nodes). Server runs node-level
    GAT and returns updated encrypted weights. No test data during training.
  Phase 2 (infer): For each batch of test edges, client builds line-graph subgraph,
    encrypts, sends infer payload. Server returns encrypted node logits; client
    decrypts and takes logits[target_indices] as predictions per original edge.
  Metrics (precision, recall, F1) are computed over original edges.

Usage:
  # In-process (no network):
  python -m client_server.client.client --epochs 3

  # TCP server on same machine or LAN:
  # Terminal 1:  python -m client_server.server.server --host 0.0.0.0 --port 9999
  # Terminal 2:  python -m client_server.client.client --epochs 3 --host 192.168.x.x --port 9999
"""

from __future__ import annotations

import argparse
import os
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
    decrypt_node_features_rows_with_retry,
    openfhe_available,
)
from client_server.client.metrics import MetricsRecorder
from client_server.client.utils import (
    build_line_graph,
    build_line_graph_batch,
    compute_classification_metrics,
    load_iot_edge_train_test,
    make_connected_batches,
)
from client_server.server.utils import write_metrics_csv as write_server_metrics_csv

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
    """Print server metrics summary to console (no file output)."""
    if not metrics_dict:
        return
    print("\n" + "=" * 80)
    print(f"=== {title} (time + RSS delta, RSS after) ===")
    print("=" * 80)
    total_time = 0.0
    for name, m in metrics_dict.items():
        sec = m.get("seconds", 0.0)
        rss_delta = m.get("rss_delta_bytes", 0)
        rss_after = m.get("rss_after_bytes", 0)
        mb = rss_after / (1024 * 1024) if rss_after else 0.0
        dmb = rss_delta / (1024 * 1024)
        print(f"  {name:<36} {sec:>8.4f}s  RSS Δ {dmb:>+8.2f} MB  RSS {mb:>8.2f} MB")
        total_time += sec
    print("-" * 80)
    print(f"  {'TOTAL':<36} {total_time:>8.4f}s")
    print("=" * 80)


# Dataset loading and graph-aware batching: see plain_client.load_iot_train_test,
# plain_client.make_connected_batches (shared for consistent train/test split and batching).


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

    def decrypt_node_features_with_bootstrap_retry(self, out_cts: list, F_out: int):
        return decrypt_node_features_rows_with_retry(
            self.crypto_context, self.keys.secretKey, out_cts, F_out
        )


# ── Main ──────────────────────────────────────────────────────────────────


def main() -> None:
    if not openfhe_available():
        print("ERROR: OpenFHE not installed. pip install openfhe")
        sys.exit(1)

    import numpy as np

    parser = argparse.ArgumentParser(
        description="IoT Malicious Edge (Link) Prediction (FHE GAT, CKKS-only, line-graph)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=60,
        help="Line-graph nodes (original edges) per training/inference batch.",
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
        help="Fraction of edges for test set (default 0.2)",
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
        "--scale_mod_size",
        type=int,
        default=50,
        help="CKKS scaling modulus size in bits (default 50). Increase to improve decode precision (slower/more RAM).",
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
    parser.add_argument(
        "--reuse_encrypted_weights_infer",
        action="store_true",
        help="Reuse the same weight ciphertext handles for every inference batch (less client CPU). "
        "Default is off: re-encrypt from plaintext W each batch so weights start at a full CKKS level "
        "and stay consistent with server-side weight bootstrap (avoids stale handles / depth drift).",
    )
    parser.add_argument(
        "--max_rows_dataset",
        type=int,
        default=None,
        help="Cap number of CSV rows (edges) for quick runs.",
    )
    parser.add_argument(
        "--max_degree_batch",
        type=int,
        default=3,
        metavar="K",
        help="Cap in-degree per node in each batch (max edges = batch_size*K). Default 5 for FHE memory; use 8--15 for 8GB RAM.",
    )
    args = parser.parse_args()

    # Line-graph: node features = edge features (3), one logit per node = per original edge
    F_in, F_out, slots = 3, 1, 8

    ts = time.strftime("%Y%m%d_%H%M%S")
    client_metrics_path = f"client_fhe_metrics_{ts}.csv"
    server_train_metrics_path = f"server_fhe_metrics_train_{ts}.csv"
    server_infer_metrics_path = f"server_fhe_metrics_infer_{ts}.csv"

    print("=" * 60)
    print("IoT Malicious Edge (Link) Prediction (FHE GAT, CKKS-only)")
    print(
        f"  batch_size={args.batch_size}  epochs={args.epochs}  "
        f"lr={args.lr}  test_ratio={args.test_ratio}  max_rows_dataset={args.max_rows_dataset or 'ALL'}"
    )
    if args.max_degree_batch is not None:
        print(f"  max_degree_batch={args.max_degree_batch} (cap edges per batch for memory)")
    if args.host:
        print(f"  Transport: TCP  {args.host}:{args.port}")
    else:
        print("  Transport: in-process (no network)")
    print("=" * 60)

    # 1. Load edge-based data and build line graph (one node per edge)
    print("\n1. Loading IoT edge train + test...")
    (
        x_nodes,
        edge_index_full,
        edge_feats,
        edge_labels,
        train_edge_ids,
        test_edge_ids,
        N_total,
    ) = load_iot_edge_train_test(
        path=args.data,
        test_ratio=args.test_ratio,
        seed=args.seed,
        max_rows_dataset=args.max_rows_dataset,
    )
    n_train_edges = len(train_edge_ids)
    n_test_edges = len(test_edge_ids)
    print("   Building line graph (one node per edge)...")
    x_line, edge_index_line, y_line = build_line_graph(
        edge_index_full, edge_feats, edge_labels
    )
    train_line_ids = train_edge_ids
    test_line_ids = test_edge_ids
    # Connected batches: BFS-grown subgraphs (fewer edges per batch than contiguous chunks)
    train_batches = make_connected_batches(train_line_ids, edge_index_line, args.batch_size)
    test_batches = make_connected_batches(test_line_ids, edge_index_line, args.batch_size)
    n_batches_train = len(train_batches)
    n_batches_test = len(test_batches)
    print(
        f"   Line graph nodes={x_line.shape[0]}, Train={n_train_edges}, Test={n_test_edges}"
    )
    print(
        f"   Batches train={n_batches_train}, test={n_batches_test} "
        f"(batch_size={args.batch_size} line-graph nodes, connected)"
    )

    client_metrics = MetricsRecorder()

    # ── Resolve weight-save path ──────────────────────────────────────────────
    weights_save_dir = args.save_weights or f"fhe_weights_{ts}"

    def _resolve_load_weights_dir(path_like: str) -> str:
        """
        Accept either:
          - exact checkpoint dir (meta.json/W.npy/a.npy), or
          - parent dir containing batch_* checkpoints.
        """
        p = Path(path_like)
        if (p / "meta.json").exists() and (p / "W.npy").exists() and (p / "a.npy").exists():
            return str(p)

        pointer = p / "last_successful_checkpoint.txt"
        if pointer.exists():
            rel = pointer.read_text(encoding="utf-8").strip()
            if rel:
                cand = (p / rel).resolve()
                if (cand / "meta.json").exists() and (cand / "W.npy").exists() and (cand / "a.npy").exists():
                    return str(cand)

        # fallback: highest batch index with valid files
        candidates = []
        for d in p.glob("batch_*"):
            if d.is_dir() and (d / "meta.json").exists() and (d / "W.npy").exists() and (d / "a.npy").exists():
                try:
                    idx = int(d.name.split("_")[-1])
                except Exception:
                    idx = -1
                candidates.append((idx, d))
        if candidates:
            candidates.sort(key=lambda t: t[0])
            return str(candidates[-1][1])
        return str(p)

    # Plaintext W for per-batch re-encryption during inference (stable Decode vs reusing ct handles).
    W_plain_for_infer: np.ndarray | None = None

    # ══════════════════════════════════════════════════════════════════════════
    #  BRANCH A: load pre-trained weights →  key-gen + training entirely
    # ══════════════════════════════════════════════════════════════════════════
    if args.load_weights:
        resolved_load_dir = _resolve_load_weights_dir(args.load_weights)
        print(f"\n[weights] Loading pre-trained weights from: {resolved_load_dir}")

        from client_server.openfhe_serializer import load_trained_weights

        w = load_trained_weights(resolved_load_dir)

        # Create fresh crypto context (new keys!)
        client_ctx = create_client_context(
            in_channels=w["F_in"],
            out_channels=w["F_out"],
            slots=w["slots"],
            mult_depth=args.mult_depth,
            scale_mod_size=args.scale_mod_size,
            ring_dim=args.ring_dim,
            bootstrap=not args.no_bootstrap,
        )

        # Re-encrypt plaintext weights
        W_loaded = np.asarray(w["W_list"], dtype=np.float64)
        W_plain_for_infer = W_loaded.copy()
        ct_W_trained = client_ctx.encrypt_weight_matrix(W_loaded, w["F_in"])

        a = np.asarray(w["a"], dtype=np.float64)

        F_in = w["F_in"]
        F_out = w["F_out"]

        server_train_metrics_list = []

        print(
            f"   Loaded {len(ct_W_trained)} weight ciphertext(s). Skipping training."
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
                scale_mod_size=args.scale_mod_size,
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

        rng = np.random.default_rng(args.seed)
        train_batches_shuffled = [list(b) for b in train_batches]
        rng.shuffle(train_batches_shuffled)
        for b_idx in range(n_batches_train):
            batch_line_ids = np.asarray(train_batches_shuffled[b_idx], dtype=np.int64)

            x_batch, edge_index_batch, y_batch, target_indices = build_line_graph_batch(
                batch_line_ids, edge_index_line, x_line, y_line,
                max_degree_per_node=args.max_degree_batch,
            )
            if x_batch.shape[0] == 0:
                continue
            num_nodes_batch = x_batch.shape[0]
            train_mask = np.zeros(num_nodes_batch, dtype=bool)
            train_mask[target_indices] = True

            print(
                f"\n   Batch {b_idx+1}/{n_batches_train} "
                f"(line-graph nodes={len(batch_line_ids)}, edges in line-graph nodes={edge_index_batch.shape[1]})"
            )
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
                "num_nodes": num_nodes_batch,
                "edge_index": edge_index_batch,
                "node_features_enc": ct_x_batch,
                "ct_labels": ct_labels_batch,
                "train_mask": train_mask,
                "lr": args.lr,
                "num_epochs": args.epochs,
                "bootstrap_level_budget": [4, 4],
            }
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
                    train_mask=gradient_step_payload["train_mask"],
                    lr=gradient_step_payload["lr"],
                    num_epochs=gradient_step_payload["num_epochs"],
                )
            server_train_metrics_list.append(batch_metrics)
            batch_time = sum(step.get("seconds", 0.0) for step in batch_metrics.values())
            print(f"      Local epochs={args.epochs}  server_time={batch_time:.4f}s")

            # Optional per-batch checkpoint: try to decrypt refreshed weights and save
            # them as plaintext for later --load_weights runs.
            if args.save_weights:
                try:
                    from client_server.openfhe_serializer import save_trained_weights

                    try:
                        W_ckpt = client_ctx.decrypt_weight_matrix(ct_W_trained, F_in)
                    except RuntimeError:
                        # If decode fails, try a client-side bootstrap refresh and retry.
                        # Client has bootstrap keys, so this is safe and often restores
                        # decryptability for checkpointing.
                        cc = client_ctx.crypto_context
                        ct_W_refreshed = [cc.EvalBootstrap(ct) for ct in ct_W_trained]
                        W_ckpt = client_ctx.decrypt_weight_matrix(ct_W_refreshed, F_in)
                        ct_W_trained = ct_W_refreshed
                    ckpt_dir = os.path.join(str(args.save_weights), f"batch_{b_idx:04d}")
                    save_trained_weights(
                        ckpt_dir,
                        W_list=W_ckpt,
                        a=a,
                        slots=slots,
                        F_in=F_in,
                        F_out=F_out,
                    )
                    print(f"[weights] Checkpoint saved → {ckpt_dir}")
                    # Update pointer so --load_weights train_weights auto-resolves.
                    Path(str(args.save_weights)).mkdir(parents=True, exist_ok=True)
                    (Path(str(args.save_weights)) / "last_successful_checkpoint.txt").write_text(
                        f"batch_{b_idx:04d}",
                        encoding="utf-8",
                    )
                except RuntimeError as exc:
                    print(f"[weights] Checkpoint decrypt failed (batch {b_idx}): {exc}")

    # ---- Write server training metrics (in-process only; when TCP, server writes to its CWD) ----
    aggregated_train_metrics = {}
    for idx, batch_m in enumerate(server_train_metrics_list):
        for name, step in batch_m.items():
            key = f"batch_{idx}_{name}"
            aggregated_train_metrics[key] = step

    if not args.host and aggregated_train_metrics:
        write_server_metrics_csv(server_train_metrics_path, aggregated_train_metrics)

    # Decrypt weight ciphertexts → numpy matrix (F_out, F_in) for saving.
    # Note: decoding can fail when the final ciphertext noise/level is still
    # too high. Training/inference can still proceed using `ct_W_trained`,
    # so we skip saving if Decode() fails.
    decrypt_ok = True
    W_trained = None
    try:
        W_rows = []
        for ct in ct_W_trained:
            print("Level:", ct.GetLevel())
            # print("Scale:", ct.GetScalingFactor())
            pt = client_ctx.crypto_context.Decrypt(client_ctx.keys.secretKey, ct)
            vals = pt.GetRealPackedValue()
            W_rows.append([float(vals[i]) for i in range(F_in)])
        W_trained = np.array(W_rows, dtype=np.float64)
    except RuntimeError as exc:
        decrypt_ok = False
        print(
            f"\n[weights] WARNING: decrypting trained weights failed; "
            f"skipping plaintext weight saving. Error: {exc}"
        )

    if not args.load_weights and decrypt_ok and W_trained is not None:
        W_plain_for_infer = np.asarray(W_trained, dtype=np.float64).copy()

    a_trained = a

    # ── Save trained weights (only after fresh training, not when loading) ───
    if not args.load_weights and decrypt_ok and W_trained is not None:
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
            f"   Weights saved.  Re-run with --load_weights {weights_save_dir} to skip training next time."
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

    all_scores: list[float] = []
    all_labels: list[int] = []
    server_infer_metrics_list: list[dict] = []
    batch_csv_rows = []

    # Line-graph inference: batch test line-graph nodes, decrypt; logits at target_indices = edge predictions
    print("\n5. Client: encrypting test line-graph batches for inference...")
    infer_fresh_weights = (
        W_plain_for_infer is not None and not args.reuse_encrypted_weights_infer
    )
    if infer_fresh_weights:
        print(
            "   [client] Inference weights: re-encrypt from plaintext each batch "
            "(full CKKS level; avoids stale handles after server bootstrap)."
        )
    elif n_batches_test > 0 and W_plain_for_infer is None:
        print(
            "   [client] WARNING: no plaintext W for inference — reusing encrypted "
            "weight handles across batches (may worsen Decode / depth)."
        )
    print(f"\n6. Server: batched FHE inference over {n_test_edges} test edges (line-graph nodes)...")
    # Same protocol as training: one round-trip per batch — encrypt payload → server
    # compute_forward_only → return out_cts → client decrypt — then next batch (no
    # server-side buffering of multiple test batches; TCP opens one connection per batch).
    for b_idx in range(n_batches_test):
        batch_line_ids = np.asarray(test_batches[b_idx], dtype=np.int64)
        x_batch, edge_index_batch, y_batch, target_indices = build_line_graph_batch(
            batch_line_ids, edge_index_line, x_line, y_line,
            max_degree_per_node=args.max_degree_batch,
        )
        if x_batch.shape[0] == 0:
            continue
        with client_metrics.step(f"encrypt_batch_{b_idx}", encrypted=True):
            _t0 = time.perf_counter()
            ct_x_batch = client_ctx.encrypt_node_features(x_batch, F_in)
            if infer_fresh_weights:
                ct_W_infer = client_ctx.encrypt_weight_matrix(W_plain_for_infer, F_in)
            else:
                ct_W_infer = ct_W_trained
            encryption_time = time.perf_counter() - _t0
        infer_payload = {
            "crypto_context": client_ctx.crypto_context,
            "public_key": client_ctx.keys.publicKey,
            "in_channels": F_in,
            "out_channels": F_out,
            "slots": slots,
            "ct_W_list": ct_W_infer,
            "a": a_trained,
            "negative_slope": 0.2,
            "num_nodes": x_batch.shape[0],
            "edge_index": edge_index_batch,
            "node_features_enc": ct_x_batch,
            "print_metrics": False,
        }
        ct_x_batch_size_bytes = sum(get_ct_size_bytes(ct) for ct in ct_x_batch)
        ct_W_list_size_bytes = sum(get_ct_size_bytes(ct) for ct in ct_W_infer)
        total_batch_size_bytes = ct_x_batch_size_bytes + ct_W_list_size_bytes
        if args.host:
            out_cts, batch_metrics = run_tcp_infer(args.host, args.port, infer_payload)
        else:
            out_cts, batch_metrics = run_infer_inprocess(
                client_ctx, F_in, F_out, slots,
                x_batch.shape[0], edge_index_batch, ct_x_batch,
                ct_W_infer, a_trained,
            )
        with client_metrics.step(f"decrypt_batch_{b_idx}", encrypted=False):
            _t0 = time.perf_counter()
            output = decrypt_node_features_rows_with_retry(
                client_ctx.crypto_context,
                client_ctx.keys.secretKey,
                out_cts,
                F_out,
            )
            decryption_time = time.perf_counter() - _t0
        # One logit per line-graph node; take only target nodes (batch of original edges)
        logits_batch = output[target_indices, 0] if output.ndim > 1 else output[target_indices]
        for i in range(len(target_indices)):
            all_scores.append(float(logits_batch[i]))
            all_labels.append(int(y_batch[target_indices[i]]))
        del out_cts, ct_x_batch
        if infer_fresh_weights:
            del ct_W_infer
        server_time_seconds = sum(m.get("seconds", 0.0) for m in (batch_metrics or {}).values())
        max_rss = max((m.get("rss_after_bytes", 0) for m in (batch_metrics or {}).values()), default=0)
        server_energy_joules = sum(m.get("energy_joules", 0.0) for m in (batch_metrics or {}).values())
        server_rss_delta = sum(m.get("rss_delta_bytes", 0) for m in (batch_metrics or {}).values())
        server_power = server_energy_joules / server_time_seconds if server_time_seconds > 0 else 0.0
        client_time_batch = encryption_time + decryption_time
        throughput = (len(batch_line_ids) / (server_time_seconds + client_time_batch)) if (server_time_seconds + client_time_batch) > 0 else 0.0
        batch_csv_rows.append({
            "step": f"fhe_batch_{b_idx}",
            "server_time": server_time_seconds,
            "client_time": client_time_batch,
            "rss_after_bytes": int(max_rss),
            "rss_delta_bytes": server_rss_delta,
            "power_watts": server_power,
            "energy_joules": server_energy_joules,
            "throughput": round(throughput, 6),
            "batch": b_idx,
            "nodes_in_batch": x_batch.shape[0],
            "edges_in_batch": len(batch_line_ids),
            "client_encryption_time": encryption_time,
            "client_decryption_time": decryption_time,
            "ciphertext_size_bytes": total_batch_size_bytes,
            "ct_x_batch_size_bytes": ct_x_batch_size_bytes,
            "ct_W_list_size_bytes": ct_W_list_size_bytes,
            "total_ciphertext_batch_size_bytes": total_batch_size_bytes,
        })
        server_infer_metrics_list.append(batch_metrics or {})

    # ---- Write server infer metrics (in-process only; when TCP, server writes to its CWD) ----
    if not args.host and server_infer_metrics_list:
        infer_rows = []
        for i, batch_m in enumerate(server_infer_metrics_list):
            for name, m in batch_m.items():
                infer_rows.append({"phase": f"infer_batch_{i}_{name}", **m})
        write_server_metrics_csv(server_infer_metrics_path, infer_rows)

    # ---- Write batch metrics to CSV ----
    import csv

    ts = time.strftime("%Y%m%d_%H%M%S")
    csv_path = f"fhe_batch_metrics_{ts}.csv"

    batch_fieldnames = ["step", "server_time", "client_time", "rss_after_bytes", "rss_delta_bytes", "power_watts", "energy_joules", "throughput", "batch", "nodes_in_batch", "edges_in_batch", "client_encryption_time", "client_decryption_time", "ciphertext_size_bytes", "ct_x_batch_size_bytes", "ct_W_list_size_bytes", "total_ciphertext_batch_size_bytes"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=batch_fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(batch_csv_rows)

    print(f"[client] FHE per-batch metrics written → {csv_path}")

    Tenc = sum(r["client_encryption_time"] for r in batch_csv_rows)
    Tserver = sum(r["server_time"] for r in batch_csv_rows)
    Tdec = sum(r["client_decryption_time"] for r in batch_csv_rows)

    Ttotal = Tenc + Tserver + Tdec
    Energy_total = sum(
        r.get("energy_joules", r.get("server_energy_joules", 0.0))
        for r in batch_csv_rows
    )

    Energy_per_batch = Energy_total / len(batch_csv_rows) if batch_csv_rows else 0.0
    Energy_per_node = Energy_total / len(all_labels) if all_labels else 0.0

    summary_path = f"fhe_summary_{ts}.csv"
    throughput_nodes_per_sec = (len(all_labels) / Ttotal) if Ttotal > 0 and all_labels else 0.0
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Tenc", Tenc])
        writer.writerow(["Tserver", Tserver])
        writer.writerow(["Tdec", Tdec])
        writer.writerow(["Ttotal", Ttotal])
        writer.writerow(["Energy_total_J", Energy_total])
        writer.writerow(["Energy_per_batch_J", Energy_per_batch])
        writer.writerow(["Energy_per_node_J", Energy_per_node])
        writer.writerow(["throughput_nodes_per_sec", throughput_nodes_per_sec])

    import numpy as np

    y_scores = np.asarray(all_scores, dtype=np.float64)
    y_true = np.asarray(all_labels, dtype=np.int64)
    y_pred = (1.0 / (1.0 + np.exp(-y_scores)) > 0.5).astype(np.int64)

    target_name = "edges"
    print(f"\n--- Test {target_name} prediction ---")
    n_show = min(10, len(y_true))
    for i in range(n_show):
        score = float(y_scores[i])
        pred = "ATTACKER" if score > 0.5 else "BENIGN"
        actual_label = int(y_true[i])
        actual_str = "ATTACKER" if actual_label == 1 else "BENIGN"
        print(
            f"  {target_name.capitalize()[:-1]} {i}:  actual={actual_str} ({actual_label})  "
            f"predicted={pred} (score={score:.4f})"
        )
    if len(y_true) > n_show:
        print(f"  ... and {len(y_true) - n_show} more")

    cls_metrics = compute_classification_metrics(y_true, y_pred, y_scores)

    print("\n=== FHE Test-set Classification Results ===")
    print(f"Accuracy  : {cls_metrics['accuracy']:.4f}")
    print(f"Precision : {cls_metrics['precision']:.4f}")
    print(f"Recall    : {cls_metrics['recall']:.4f}")
    print(f"F1 Score  : {cls_metrics['f1']:.4f}")
    print(f"Test {target_name}: {len(y_true)}  |  Batches: {n_batches_test}")

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

    # 10. Write client metrics to CSV (client_time per phase: keygen, encrypt, decrypt) and print summary
    write_server_metrics_csv(client_metrics_path, client_metrics.to_dict(), time_side="client")
    print(f"[client] Client metrics written → {client_metrics_path}")
    client_metrics.print_report()

    print("\n✓ Done. FHE training + inference (server never had secret key).")


if __name__ == "__main__":
    main()
