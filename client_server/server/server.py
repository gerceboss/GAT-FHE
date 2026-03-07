#!/usr/bin/env python3
"""
CKKS-only FHE GAT server: raw TCP transport.

No secret key on server. Client sends encrypted weights, features, labels
over a persistent TCP connection using OpenFHE BINARY serialization.

Protocol (batched-friendly):
  Client connects, sends 1-byte command:
    b'T' -> full-train         (send_train_payload / recv_train_result)
    b'I' -> infer (per call)   (send_infer_payload / recv_infer_result)
    b'G' -> gradient-step      (batched FHE training on one subgraph)

  Server sends status frame (b'\x00' ok, b'\x01' + error string on error),
  then the result frames.

In-process API (unchanged):
  from client_server.server import compute_fhe_training, compute_forward_only

Run TCP server:
  python -m client_server.server.server --host 0.0.0.0 --port 9999
"""

from __future__ import annotations

import socket
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from .encoder_ckks import GATEncoderCKKS
from .fhe_graph import FHEGraph

# from .runner_ckks import run_gat_pipeline_fhe_training, run_gat_forward_only
from .gpt_runner import run_gat_pipeline_fhe_training, run_gat_forward_only
# from .metrics import MetricsRecorder
from .metrics_pi import MetricsRecorder


import csv


def _write_metrics_csv(path: str, metrics_dict: dict) -> None:
    rows = []

    for name, m in metrics_dict.items():
        rows.append(
            {
                "phase": name,
                "seconds": m.get("seconds", 0.0),
                "rss_delta_bytes": m.get("rss_delta_bytes", 0),
                "rss_after_bytes": m.get("rss_after_bytes", 0),
                "energy_joules": m.get("energy_joules", 0.0),
                "power_watts": m.get("power_watts", 0.0),
            }
        )

    if rows:
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        print(f"[server] metrics written → {path}")


# ── Bootstrap helper (public-key only, usable on server) ─────────────────


def _bootstrap_output_cts(
    crypto_context,
    out_cts,
    metrics_dict,
    bootstrap_level_threshold: int = 4,
):
    """
    Conditionally bootstrap output ciphertexts based on remaining level.
    Now records real RSS delta.
    """

    import time
    import sys as _sys

    # ---- Real memory measurement ----
    def _rss_bytes() -> int:
        """Return resident set size in bytes (Linux)."""
        try:
            with open("/proc/self/status", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        parts = line.split()
                        return int(parts[1]) * 1024  # kB → bytes
        except Exception:
            return 0
        return 0

    rss_before = _rss_bytes()
    t0 = time.perf_counter()

    refreshed = []
    n_bootstrapped = 0
    n_skipped = 0
    n_failed = 0

    for ct in out_cts:
        try:
            current_level = ct.GetLevel()

            if current_level >= bootstrap_level_threshold:
                ct = crypto_context.EvalBootstrap(ct)
                n_bootstrapped += 1
            else:
                n_skipped += 1

        except Exception as exc:
            print(
                f"[server] WARNING: bootstrap failed ({exc}); "
                "keeping original ciphertext.",
                file=_sys.stderr,
            )
            n_failed += 1

        refreshed.append(ct)

    elapsed = time.perf_counter() - t0
    rss_after = _rss_bytes()

    rss_delta = rss_after - rss_before

    print(
        f"[server] bootstrapped {n_bootstrapped}/{len(out_cts)} outputs "
        f"(skipped {n_skipped}, failed {n_failed}) "
        f"in {elapsed:.3f}s  "
        f"RSS Δ {rss_delta / (1024*1024):+.2f} MB"
    )

    metrics_dict = dict(metrics_dict)
    metrics_dict["fhe_output_bootstrap"] = {
        "seconds": elapsed,
        "rss_delta_bytes": rss_delta,
        "rss_after_bytes": rss_after,
        "encrypted": True,
        "energy_joules": 0.0,
        "power_watts": 0.0,
        "n_bootstrapped": n_bootstrapped,
        "n_skipped": n_skipped,
        "n_failed": n_failed,
    }

    return refreshed, metrics_dict


# ── Core compute functions ────────────────────────────────────────────────


def compute_fhe_training_batch(
    *,
    crypto_context,
    public_key,
    in_channels,
    out_channels,
    slots,
    ct_W_list,
    a,
    negative_slope,
    num_nodes,
    edge_index,
    node_features_enc,
    ct_labels,
    lr,
    num_epochs,
):
    """
    Perform mini-batch encrypted training.
    Runs `num_epochs` encrypted epochs on this subgraph.
    Returns:
        ct_W_list_new, metrics_dict
    """
    import numpy as np

    a_np = np.asarray(a, dtype=np.float64)
    edge_index_np = np.asarray(edge_index, dtype=np.int64)
    if edge_index_np.shape[0] != 2:
        edge_index_np = edge_index_np.T

    graph = FHEGraph.from_encrypted(
        num_nodes=num_nodes,
        in_channels=in_channels,
        edge_index=edge_index_np,
        node_features_enc=node_features_enc,
    )

    encoder = GATEncoderCKKS.from_client_keys_with_encrypted_weights(
        crypto_context=crypto_context,
        public_key=public_key,
        in_channels=in_channels,
        out_channels=out_channels,
        slots=slots,
        ct_W_list=ct_W_list,
        a=a_np,
        negative_slope=negative_slope,
    )

    #  Run multiple encrypted epochs per batch
    _, metrics_dict, ct_W_list_new = run_gat_pipeline_fhe_training(
        encoder=encoder,
        graph=graph,
        ct_labels=ct_labels,
        train_mask=np.ones(num_nodes, dtype=bool),
        num_epochs=num_epochs,
        lr=lr,
        print_metrics=False,
        bootstrap_weights=True,
    )

    # Bootstrap the updated weight ciphertexts so the next batch starts with
    # fresh levels.  run_gat_pipeline_fhe_training with bootstrap_weights=True
    # already bootstraps during training, but we do a final check here in case
    # the last epoch left ct_W_list_new at a low level.
    ct_W_list_new, metrics_dict = _bootstrap_output_cts(
        crypto_context=crypto_context,
        out_cts=ct_W_list_new,
        metrics_dict=metrics_dict,
        bootstrap_level_threshold=4,
    )

    return ct_W_list_new, metrics_dict


def compute_fhe_training(
    *,
    crypto_context: Any,
    public_key: Any,
    in_channels: int,
    out_channels: int,
    slots: int,
    ct_W_list: list,
    a: Any,
    negative_slope: float,
    num_nodes: int,
    edge_index: Any,
    node_features_enc: list,
    ct_labels: list,
    train_mask: Any,
    num_epochs: int = 3,
    lr: float = 0.01,
    print_metrics: bool = True,
    bootstrap_weights: bool = True,
    bootstrap_level_budget: list = None,  # consumed by serializer; accepted here so **payload unpacking works
) -> tuple:
    """
    Run FHE training on server (train data only). Server never holds secret key.

    EvalBootstrapSetup() has already been replayed by recv_train_payload() in
    openfhe_serializer.py before this function is called, so crypto_context is
    ready to bootstrap without any further setup here.

    Returns:
        (out_cts_train, metrics_dict, ct_W_list_trained)
    """
    import numpy as np

    a_np = np.asarray(a, dtype=np.float64)
    train_mask_np = np.asarray(train_mask, dtype=bool)
    edge_index_np = np.asarray(edge_index, dtype=np.int64)
    if edge_index_np.shape[0] != 2:
        edge_index_np = edge_index_np.T

    graph = FHEGraph.from_encrypted(
        num_nodes=num_nodes,
        in_channels=in_channels,
        edge_index=edge_index_np,
        node_features_enc=node_features_enc,
    )
    encoder = GATEncoderCKKS.from_client_keys_with_encrypted_weights(
        crypto_context=crypto_context,
        public_key=public_key,
        in_channels=in_channels,
        out_channels=out_channels,
        slots=slots,
        ct_W_list=ct_W_list,
        a=a_np,
        negative_slope=negative_slope,
    )
    out_cts, metrics_dict, ct_W_list_trained = run_gat_pipeline_fhe_training(
        encoder=encoder,
        graph=graph,
        ct_labels=ct_labels,
        train_mask=train_mask_np,
        num_epochs=num_epochs,
        lr=lr,
        print_metrics=print_metrics,
        bootstrap_weights=bootstrap_weights,
    )

    # Also record an outer "epoch_total" style metric summarizing the full training call.
    # This uses the same format as other FHE/server metrics so the client can
    # write them out alongside inner pipeline timings if desired.
    rec = MetricsRecorder()
    with rec.step("fhe_training_total", encrypted=True):
        pass  # just capture RSS/energy snapshot around the end of training
    outer = rec.to_dict()
    # Merge outer summary into inner metrics_dict under a distinct key
    metrics_dict = {**metrics_dict, **outer}

    return out_cts, metrics_dict, ct_W_list_trained


def compute_forward_only(
    *,
    crypto_context: Any,
    public_key: Any,
    in_channels: int,
    out_channels: int,
    ct_W_list: list,
    a: Any,
    negative_slope: float,
    num_nodes: int,
    edge_index: Any,
    node_features_enc: list,
    print_metrics: bool = False,
    slots: int | None = None,
    bootstrap_output: bool = True,
    bootstrap_level_threshold: int = 4,
) -> tuple:
    """
    Run one forward pass (inference only). Returns (out_cts, metrics_dict).

    After the forward pass the output ciphertexts are refreshed via
    EvalBootstrap() whenever their remaining multiplicative depth is at or
    below *bootstrap_level_threshold* (default 4).  This prevents the
    "approximation error is too high" decryption failure that occurs when the
    GAT forward pass exhausts the CKKS level budget before the client can
    decrypt.

    Set bootstrap_output=False to skip bootstrapping (e.g. for unit tests with
    a shallow parameter set that has no bootstrap keys).
    """
    import numpy as np

    a_np = np.asarray(a, dtype=np.float64)
    edge_index_np = np.asarray(edge_index, dtype=np.int64)
    if edge_index_np.shape[0] != 2:
        edge_index_np = edge_index_np.T

    graph = FHEGraph.from_encrypted(
        num_nodes=num_nodes,
        in_channels=in_channels,
        edge_index=edge_index_np,
        node_features_enc=node_features_enc,
    )
    encoder = GATEncoderCKKS.from_client_keys_with_encrypted_weights(
        crypto_context=crypto_context,
        public_key=public_key,
        in_channels=in_channels,
        out_channels=out_channels,
        slots=slots,
        ct_W_list=ct_W_list,
        a=a_np,
        negative_slope=negative_slope,
    )

    rec = MetricsRecorder()
    with rec.step("fhe_infer_forward", encrypted=True):
        out_cts, metrics_dict = run_gat_forward_only(
            encoder=encoder,
            graph=graph,
            print_metrics=print_metrics,
        )

    # ── Bootstrap output ciphertexts to restore decryptable level ────────────
    # The GAT forward pass (linear projection + LeakyReLU poly approx +
    # softmax approx + aggregation) typically consumes 10-20 multiplicative
    # levels.  If the remaining level is too low, client Decode() throws
    # "approximation error is too high".  EvalBootstrap() is public-key-only
    # and refreshes the level without the secret key.
    if bootstrap_output:
        with rec.step("fhe_infer_bootstrap_output", encrypted=True):
            out_cts, metrics_dict = _bootstrap_output_cts(
                crypto_context=crypto_context,
                out_cts=out_cts,
                metrics_dict=metrics_dict,
                bootstrap_level_threshold=bootstrap_level_threshold,
            )

    outer = rec.to_dict()
    metrics_dict = {**metrics_dict, **outer}

    return out_cts, metrics_dict


# ── TCP connection handler ────────────────────────────────────────────────


def _replay_bootstrap_setup(payload: dict):
    """
    Replay EvalBootstrapSetup in this worker process.
    Required because bootstrap precompute is NOT serialized.
    """
    cc = payload.get("crypto_context", None)
    slots = payload.get("slots", None)

    if cc is None or slots is None:
        return  # nothing to do

    try:
        # Must match client EXACTLY
        cc.EvalBootstrapSetup(levelBudget=[4, 4], slots=slots)
    except Exception as e:
        print(f"[server] WARNING: EvalBootstrapSetup replay failed: {e}")


def _handle_connection(conn: socket.socket, addr: tuple) -> None:
    from client_server.openfhe_serializer import (
        recv_train_payload,
        send_train_result,
        recv_infer_payload,
        send_infer_result,
        send_ok,
        send_error,
    )

    print(f"[server] connection from {addr[0]}:{addr[1]}")

    try:
        with conn:
            cmd = conn.recv(1)

            if cmd == b"T":
                payload = recv_train_payload(conn)

                _replay_bootstrap_setup(payload)

                try:
                    out_cts, metrics, ct_W_trained = compute_fhe_training(**payload)
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    _write_metrics_csv(f"server_fhe_train_metrics_{ts}.csv", metrics)
                    send_ok(conn)
                    send_train_result(conn, out_cts, metrics, ct_W_trained)
                except Exception as exc:
                    print(f"[server] train error: {exc}", file=sys.stderr)
                    traceback.print_exc()
                    send_error(conn, exc)

            elif cmd == b"I":
                payload = recv_infer_payload(conn)

                _replay_bootstrap_setup(payload)

                try:
                    out_cts, metrics = compute_forward_only(**payload)
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    _write_metrics_csv(f"server_fhe_infer_metrics_{ts}.csv", metrics)
                    send_ok(conn)
                    send_infer_result(conn, out_cts, metrics)
                except Exception as exc:
                    print(f"[server] infer error: {exc}", file=sys.stderr)
                    traceback.print_exc()
                    send_error(conn, exc)

            elif cmd == b"G":
                from client_server.openfhe_serializer import (
                    recv_gradient_step_payload,
                    send_gradient_step_result,
                )

                payload = recv_gradient_step_payload(conn)

                _replay_bootstrap_setup(payload)

                try:
                    ct_W_new, metrics = compute_fhe_training_batch(**payload)
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    _write_metrics_csv(f"server_fhe_grad_metrics_{ts}.csv", metrics)
                    send_ok(conn)
                    send_gradient_step_result(conn, ct_W_new, metrics)
                except Exception as exc:
                    print(f"[server] gradient step error: {exc}", file=sys.stderr)
                    traceback.print_exc()
                    send_error(conn, exc)

    except Exception as exc:
        print(f"[server] connection error from {addr}: {exc}", file=sys.stderr)
        traceback.print_exc()

    finally:
        print(f"[server] closed {addr[0]}:{addr[1]}")


# ── TCP server main ───────────────────────────────────────────────────────


import multiprocessing


def _worker_entry(conn, addr):
    """
    Worker process entry.
    Handles exactly one client connection, then exits.
    """
    try:
        _handle_connection(conn, addr)
    finally:
        try:
            conn.close()
        except Exception:
            pass


def serve(host: str = "127.0.0.1", port: int = 9999) -> None:
    """
    Start TCP server.
    Each accepted connection is handled in a separate process.
    When the process exits, all OpenFHE memory is released.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        srv.bind((host, port))
        srv.listen(8)

        print(f"[server] CKKS-only FHE GAT server listening on {host}:{port}")
        print(
            "[server]  command b'T' -> full-train  |  "
            "b'I' -> infer  |  b'G' -> gradient-step train (batched)"
        )
        print("[server]  mode: stateless (one worker process per request)")

        try:
            while True:
                conn, addr = srv.accept()
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

                print(f"[server] spawning worker for {addr[0]}:{addr[1]}")

                # IMPORTANT: do NOT use daemon=True here
                p = multiprocessing.Process(
                    target=_worker_entry,
                    args=(conn, addr),
                )
                p.start()

                # Parent must close its copy of the socket
                conn.close()

                # Optional: wait for worker to finish (sequential processing)
                p.join()

        except KeyboardInterrupt:
            print("\n[server] shutting down.")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="CKKS-only FHE GAT TCP server")
    ap.add_argument(
        "--host", default="127.0.0.1", help="Bind address (use 0.0.0.0 for LAN)"
    )
    ap.add_argument("--port", type=int, default=9999, help="TCP port")
    args = ap.parse_args()
    serve(args.host, args.port)


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.set_start_method("spawn", force=True)
    main()
