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
from client_server.client.utils import (
    build_edge_batch,
    compute_classification_metrics,
    derive_node_labels_from_edges,
    load_iot_edge_train_test,
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
    parser.add_argument(
        "--max_edges",
        type=int,
        default=None,
        help="Cap number of edges for quick runs (edge mode only).",
    )
    args = parser.parse_args()

    # Edge-based: node_in_dim=2 (in/out degree), hidden_dim=8
    F_in, F_out, slots = 2, 8, 8
    EDGE_FEAT_DIM = 3

    ts = time.strftime("%Y%m%d_%H%M%S")
    client_metrics_path = f"client_fhe_metrics_{ts}.csv"
    server_train_metrics_path = f"server_fhe_metrics_train_{ts}.csv"
    server_infer_metrics_path = f"server_fhe_metrics_infer_{ts}.csv"

    print("=" * 60)
    print("IoT Malicious Edge (Link) Prediction (FHE GAT, CKKS-only)")
    print(
        f"  batch_size={args.batch_size}  epochs={args.epochs}  "
        f"lr={args.lr}  test_ratio={args.test_ratio}  max_edges={args.max_edges or 'ALL'}"
    )
    if args.host:
        print(f"  Transport: TCP  {args.host}:{args.port}")
    else:
        print("  Transport: in-process (no network)")
    print("=" * 60)

    # 1. Load edge-based data (one row = one edge, label per edge)
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
        max_edges=args.max_edges,
    )
    n_train_edges = len(train_edge_ids)
    n_test_edges = len(test_edge_ids)
    n_batches_train = max(1, (n_train_edges + args.batch_size - 1) // args.batch_size)
    n_batches_test = max(1, (n_test_edges + args.batch_size - 1) // args.batch_size)
    print(
        f"   Nodes={N_total}, Train edges={n_train_edges}, Test edges={n_test_edges}"
    )
    print(
        f"   Batches train={n_batches_train}, test={n_batches_test} "
        f"(batch_size={args.batch_size} edges)"
    )

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

        edge_head_weight = np.asarray(w["edge_head_weight"], dtype=np.float64) if w.get("edge_head_weight") is not None else None
        edge_head_bias = np.asarray(w["edge_head_bias"], dtype=np.float64) if w.get("edge_head_bias") is not None else None

        server_train_metrics_list = []

        print(
            f"   Loaded {len(ct_W_trained)} weight ciphertext(s). Skipping training."
        )
        if edge_head_weight is not None:
            print("   Loaded edge head weights for inference.")

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
        edge_head_weight, edge_head_bias = None, None

        # Batches are disjoint: shuffle train edge indices once, then slice [0:batch_size], ...
        rng = np.random.default_rng(args.seed)
        shuffled_train_ids = rng.permutation(train_edge_ids)
        for b_idx in range(n_batches_train):
                start = b_idx * args.batch_size
                end = min(start + args.batch_size, n_train_edges)
                batch_edge_ids = shuffled_train_ids[start:end]  # disjoint from other batches
                # Each batch gets a disjoint slice of shuffled train edges (no overlap)
                edge_ids_min, edge_ids_max = int(batch_edge_ids.min()), int(batch_edge_ids.max())
                x_batch, edge_index_batch, edge_feats_batch, y_edges_batch, _ = build_edge_batch(
                    batch_edge_ids, edge_index_full, edge_feats, edge_labels, x_nodes,
                )
                if edge_index_batch.shape[1] == 0:
                    continue
                num_nodes_batch = x_batch.shape[0]
                node_labels_batch = derive_node_labels_from_edges(
                    edge_index_batch, y_edges_batch, num_nodes_batch,
                )
                print(
                    f"\n   Batch {b_idx+1}/{n_batches_train} "
                    f"(edges={len(batch_edge_ids)}, nodes={num_nodes_batch}) "
                    f"edge_ids=[{edge_ids_min}..{edge_ids_max}]"
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
                        for y in node_labels_batch
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
                    "lr": args.lr,
                    "num_epochs": args.epochs,
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
                        lr=gradient_step_payload["lr"],
                        num_epochs=gradient_step_payload["num_epochs"],
                    )
                server_train_metrics_list.append(batch_metrics)
                batch_time = sum(step.get("seconds", 0.0) for step in batch_metrics.values())
                print(f"      Local epochs={args.epochs}  server_time={batch_time:.4f}s")

    # ---- Aggregate training metrics (both branches) ----
    aggregated_train_metrics = {}
    for idx, batch_m in enumerate(server_train_metrics_list):
        for name, step in batch_m.items():
            key = f"batch_{idx}_{name}"
            aggregated_train_metrics[key] = step

    if aggregated_train_metrics:
        write_server_metrics_csv(server_train_metrics_path, aggregated_train_metrics)
    # Decrypt weight ciphertexts → numpy matrix (F_out, F_in)
    W_rows = []
    for ct in ct_W_trained:
        pt = client_ctx.crypto_context.Decrypt(client_ctx.keys.secretKey, ct)
        vals = pt.GetRealPackedValue()
        W_rows.append([float(vals[i]) for i in range(F_in)])

    W_trained = np.array(W_rows, dtype=np.float64)
    a_trained = a

    # ── Edge head: train on client from FHE embeddings (fresh training only)
    if not args.load_weights:
        print("\n   Training edge head on client (FHE GAT embeddings)...")
        rng_eh = np.random.default_rng(args.seed)
        shuffled_train_ids_eh = rng_eh.permutation(train_edge_ids)
        X_emb_list, y_emb_list = [], []
        for b_idx in range(n_batches_train):
            start = b_idx * args.batch_size
            end = min(start + args.batch_size, n_train_edges)
            batch_edge_ids = shuffled_train_ids_eh[start:end]
            x_batch, edge_index_batch, edge_feats_batch, y_edges_batch, _ = build_edge_batch(
                batch_edge_ids, edge_index_full, edge_feats, edge_labels, x_nodes,
            )
            if edge_index_batch.shape[1] == 0:
                continue
            ct_x_batch = client_ctx.encrypt_node_features(x_batch, F_in)
            infer_payload = {
                "crypto_context": client_ctx.crypto_context,
                "public_key": client_ctx.keys.publicKey,
                "in_channels": F_in,
                "out_channels": F_out,
                "slots": slots,
                "ct_W_list": ct_W_trained,
                "a": a_trained,
                "negative_slope": 0.2,
                "num_nodes": x_batch.shape[0],
                "edge_index": edge_index_batch,
                "node_features_enc": ct_x_batch,
                "print_metrics": False,
            }
            if args.host:
                out_cts, _ = run_tcp_infer(args.host, args.port, infer_payload)
            else:
                out_cts, _ = run_infer_inprocess(
                    client_ctx, F_in, F_out, slots,
                    x_batch.shape[0], edge_index_batch, ct_x_batch,
                    ct_W_trained, a_trained,
                )
            output = client_ctx.decrypt_node_features(out_cts, F_out)
            for e in range(edge_index_batch.shape[1]):
                src, dst = int(edge_index_batch[0, e]), int(edge_index_batch[1, e])
                row = np.concatenate([
                    output[src], output[dst], edge_feats_batch[e].ravel(),
                ])
                X_emb_list.append(row)
                y_emb_list.append(y_edges_batch[e])
        X_emb = np.array(X_emb_list, dtype=np.float64)
        y_emb = np.array(y_emb_list, dtype=np.int64)
        try:
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(max_iter=500, solver="lbfgs")
            clf.fit(X_emb, y_emb)
            edge_head_weight = np.asarray(clf.coef_, dtype=np.float64)
            edge_head_bias = np.asarray(clf.intercept_, dtype=np.float64)
        except Exception as e:
            print(f"   [WARNING] Edge head fit failed ({e}); using zero head.")
            edge_head_weight = np.zeros((1, 2 * F_out + EDGE_FEAT_DIM), dtype=np.float64)
            edge_head_bias = np.zeros(1, dtype=np.float64)
        print(f"   Edge head fitted on {len(y_emb)} train edges.")

    # Fallback zero edge head when loading old weights without edge head
    if edge_head_weight is None and args.load_weights:
        edge_head_weight = np.zeros((1, 2 * F_out + EDGE_FEAT_DIM), dtype=np.float64)
        edge_head_bias = np.zeros(1, dtype=np.float64)

    # ── Save trained weights (only after fresh training, not when loading) ───
    if not args.load_weights:
        print(f"\n[weights] Saving trained weights → {weights_save_dir} ...")
        from client_server.openfhe_serializer import save_trained_weights

        save_kw = dict(
            W_list=W_trained,
            a=a_trained,
            slots=slots,
            F_in=F_in,
            F_out=F_out,
        )
        if edge_head_weight is not None:
            save_kw["edge_head_weight"] = edge_head_weight
            save_kw["edge_head_bias"] = edge_head_bias
        save_trained_weights(weights_save_dir, **save_kw)
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

    all_scores: list[float] = []
    all_labels: list[int] = []
    server_infer_metrics_list: list[dict] = []
    batch_csv_rows = []

    # Edge-based inference: batches of test edges, decrypt node embeddings, apply edge head
    print("\n5. Client: encrypting test edge batches for inference...")
    print(f"\n6. Server: batched FHE inference over {n_test_edges} test edges...")
    for b_idx in range(n_batches_test):
        start = b_idx * args.batch_size
        end = min(start + args.batch_size, n_test_edges)
        batch_edge_ids = test_edge_ids[start:end]
        x_batch, edge_index_batch, edge_feats_batch, y_edges_batch, _ = build_edge_batch(
            batch_edge_ids, edge_index_full, edge_feats, edge_labels, x_nodes,
        )
        if edge_index_batch.shape[1] == 0:
            continue
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
            "a": a_trained,
            "negative_slope": 0.2,
            "num_nodes": x_batch.shape[0],
            "edge_index": edge_index_batch,
            "node_features_enc": ct_x_batch,
            "print_metrics": False,
        }
        ct_x_batch_size_bytes = sum(get_ct_size_bytes(ct) for ct in ct_x_batch)
        ct_W_list_size_bytes = sum(get_ct_size_bytes(ct) for ct in ct_W_trained)
        total_batch_size_bytes = ct_x_batch_size_bytes + ct_W_list_size_bytes
        if args.host:
            out_cts, batch_metrics = run_tcp_infer(args.host, args.port, infer_payload)
        else:
            out_cts, batch_metrics = run_infer_inprocess(
                client_ctx, F_in, F_out, slots,
                x_batch.shape[0], edge_index_batch, ct_x_batch,
                ct_W_trained, a_trained,
            )
        with client_metrics.step(f"decrypt_batch_{b_idx}", encrypted=False):
            _t0 = time.perf_counter()
            output = client_ctx.decrypt_node_features(out_cts, F_out)
            decryption_time = time.perf_counter() - _t0
        # Edge logits: for each edge (src, dst), logit = edge_head @ [h_src; h_dst; edge_feats]
        for e in range(edge_index_batch.shape[1]):
            src, dst = int(edge_index_batch[0, e]), int(edge_index_batch[1, e])
            feat = np.concatenate([
                output[src], output[dst], edge_feats_batch[e].ravel(),
            ])
            logit = float(np.dot(edge_head_weight.ravel(), feat) + edge_head_bias.ravel()[0])
            all_scores.append(logit)
            all_labels.append(int(y_edges_batch[e]))
        del out_cts, ct_x_batch
        server_time_seconds = sum(m.get("seconds", 0.0) for m in (batch_metrics or {}).values())
        max_rss = max((m.get("rss_after_bytes", 0) for m in (batch_metrics or {}).values()), default=0)
        server_energy_joules = sum(m.get("energy_joules", 0.0) for m in (batch_metrics or {}).values())
        server_rss_delta = sum(m.get("rss_delta_bytes", 0) for m in (batch_metrics or {}).values())
        server_power = server_energy_joules / server_time_seconds if server_time_seconds > 0 else 0.0
        client_time_batch = encryption_time + decryption_time
        throughput = (len(batch_edge_ids) / (server_time_seconds + client_time_batch)) if (server_time_seconds + client_time_batch) > 0 else 0.0
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
            "edges_in_batch": len(batch_edge_ids),
            "client_encryption_time": encryption_time,
            "client_decryption_time": decryption_time,
            "ciphertext_size_bytes": total_batch_size_bytes,
            "ct_x_batch_size_bytes": ct_x_batch_size_bytes,
            "ct_W_list_size_bytes": ct_W_list_size_bytes,
            "total_ciphertext_batch_size_bytes": total_batch_size_bytes,
        })
        server_infer_metrics_list.append(batch_metrics or {})

    # ---- Write server infer metrics to CSV (standard: rss_after_bytes, rss_delta_bytes, power_watts, energy_joules, throughput) ----
    if server_infer_metrics_list:
        infer_rows = []
        for i, batch_m in enumerate(server_infer_metrics_list):
            for name, m in batch_m.items():
                infer_rows.append({"phase": f"infer_batch_{i}_{name}", **m})
        write_server_metrics_csv(server_infer_metrics_path, infer_rows)
        print(f"[client] Server infer metrics written → {server_infer_metrics_path}")

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
    Energy_total = sum(r["server_energy_joules"] for r in batch_csv_rows)

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
