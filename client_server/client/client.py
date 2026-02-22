#!/usr/bin/env python3
"""
CKKS-only FHE GAT client: same flow as examples/run_iot_malicious_fhe.py.
Client generates keys, encrypts weights/features/labels, calls server for compute,
decrypts result. Server never has the secret key.

Data flow (train then infer; no test data during training):
- Phase 1 (train): Client sends TRAIN graph only (num_train nodes, edge_index_train,
  encrypted train features, encrypted train labels). Server returns (out_cts_train,
  metrics, ct_W_trained). Test data is never sent during training.
- Phase 2 (infer): Client sends FULL graph (train+test) with encrypted features and
  ct_W_trained. Server runs forward only; client decrypts and uses output[test_node_idx].

Run from repo root:
  source venv/bin/activate
  python -m client_server.client.client --epochs 3

Optional: start HTTP server and use --url http://127.0.0.1:8080
  Terminal 1: python -m client_server.server.server
  Terminal 2: python -m client_server.client.client --epochs 3 --url http://127.0.0.1:8080
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from client_server.client.client_keys import (
    create_client_context,
    openfhe_available,
)
from client_server.client.metrics import MetricsRecorder


def print_server_metrics(metrics_dict: dict, title: str = "Server metrics") -> None:
    """Print server metrics dict (same shape as MetricsRecorder.to_dict()) in a readable format."""
    if not metrics_dict:
        return
    print("\n" + "=" * 80)
    print(f"=== {title} (time + RSS delta, RSS after) ===")
    print("=" * 80)
    total_time = 0.0
    total_rss_delta = 0.0
    enc_time = 0.0
    for name, m in metrics_dict.items():
        sec = m.get("seconds", 0.0)
        rss_delta = m.get("rss_delta_bytes", 0)
        rss_after = m.get("rss_after_bytes", 0)
        enc = m.get("encrypted", True)
        mode = "ENC" if enc else "DEC"
        mb = rss_after / (1024 * 1024) if rss_after else 0.0
        dmb = rss_delta / (1024 * 1024)
        print(f"  {name:<36} {mode:<6} {sec:>8.4f}s  "
              f"RSS Δ {dmb:>+8.2f} MB  RSS {mb:>8.2f} MB")
        total_time += sec
        total_rss_delta += rss_delta / (1024 * 1024)
        if enc:
            enc_time += sec
    print("-" * 80)
    print(f"  {'TOTAL':<36} {'':6} {total_time:>8.4f}s  "
          f"RSS Δ {total_rss_delta:>+8.2f} MB")
    if total_time > 0:
        dec_time = total_time - enc_time
        print()
        print(f"  Encrypted ops: {enc_time:>8.4f}s ({enc_time / total_time * 100:>5.1f}%)")
        print(f"  Plaintext ops: {dec_time:>8.4f}s ({dec_time / total_time * 100:>5.1f}%)")
    print("=" * 80)
from client_server.server.server import compute_fhe_training, compute_forward_only

# Dataset: same format as run_iot_malicious_fhe.py


def _default_iot_csv_path():
    """Resolve iot.csv: client_server/client/ or examples/dataset/ or repo root."""
    client_dir = Path(__file__).resolve().parent
    for candidate in [client_dir / "iot.csv", ROOT / "examples" / "dataset" / "iot.csv", ROOT / "iot.csv"]:
        if candidate.exists():
            return str(candidate)
    return str(client_dir / "iot.csv")  # fallback for clearer error if missing


def load_and_preprocess_iot_csv(path=None, return_labels: bool = False):
    if path is None:
        path = _default_iot_csv_path()
    # 1. Load CSV (fix BOM in column name)
    df = pd.read_csv(path, encoding="latin1")
    df.rename(columns={"ÿsrc_ip": "src_ip"}, inplace=True)

    # 2. Create node index mapping
    all_ips = pd.concat([df["src_ip"], df["dst_ip"]]).unique()
    ip_to_idx = {ip: idx for idx, ip in enumerate(all_ips)}

    df["src_idx"] = df["src_ip"].map(ip_to_idx)
    df["dst_idx"] = df["dst_ip"].map(ip_to_idx)

    # 3. Build edge_index
    edge_index = torch.tensor(df[["src_idx", "dst_idx"]].values.T, dtype=torch.long)

    # 4. Select numeric flow features
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Remove index columns and label from node features
    numeric_cols = [c for c in numeric_cols if c not in ["src_idx", "dst_idx", "label"]]

    # 5. Aggregate features per source node (mean aggregation)
    node_features = (
        df.groupby("src_idx")[numeric_cols]
        .mean()
        .reindex(range(len(all_ips)), fill_value=0)
    )

    # Normalize features
    scaler = StandardScaler()
    node_features = scaler.fit_transform(node_features)

    x = torch.tensor(node_features, dtype=torch.float32)

    # 6. Labels (node-level)
    # Assign majority label per node
    node_labels = (
        df.groupby("src_idx")["label"]
        .agg(lambda x: x.value_counts().index[0])
        .reindex(range(len(all_ips)), fill_value=0)
    )

    y = torch.tensor(node_labels.values, dtype=torch.long)

    # 7. Metadata
    N = x.shape[0]
    F_in = x.shape[1]
    F_out = len(torch.unique(y))

    if return_labels:
        return x, edge_index, N, F_in, F_out, y
    return x, edge_index, N, F_in, F_out


def load_iot_subgraph(num_nodes: int = 10, path=None):
    """
    Load IoT data and return a connected subgraph of num_nodes for malicious/benign prediction.
    Returns x, edge_index, y, N, F_in.
    y: binary labels (0=benign, 1=malicious) for each node.
    """
    x, edge_index, N_full, F_in, F_out, y_full = load_and_preprocess_iot_csv(
        path=path, return_labels=True
    )
    x = x.numpy()
    edge_index_np = edge_index.numpy()
    y_full = y_full.numpy()

    # BFS from node 0 to get connected component, then take first num_nodes
    from collections import deque

    adj = {}
    for i in range(edge_index_np.shape[1]):
        s, t = int(edge_index_np[0, i]), int(edge_index_np[1, i])
        adj.setdefault(s, []).append(t)
        adj.setdefault(t, []).append(s)

    visited = set()
    q = deque([0])
    while q and len(visited) < num_nodes * 2:  # get enough for subgraph
        u = q.popleft()
        if u in visited:
            continue
        visited.add(u)
        for v in adj.get(u, []):
            if v not in visited:
                q.append(v)

    # Take first num_nodes from BFS order
    node_order = list(visited)[:num_nodes]
    if len(node_order) < num_nodes:
        # Pad with extra nodes if graph is small
        all_nodes = list(range(min(N_full, num_nodes)))
        node_order = (node_order + [n for n in all_nodes if n not in node_order])[
            :num_nodes
        ]

    old_to_new = {old: new for new, old in enumerate(node_order)}
    new_edge_list = []
    for i in range(edge_index_np.shape[1]):
        s, t = int(edge_index_np[0, i]), int(edge_index_np[1, i])
        if s in old_to_new and t in old_to_new:
            new_edge_list.append([old_to_new[s], old_to_new[t]])

    if not new_edge_list:
        # Ensure at least self-loops or single edge for connectivity
        new_edge_list = [[0, 0]] if num_nodes > 0 else []

    # Deduplicate edges
    edge_set = {tuple(e) for e in new_edge_list}
    new_edge_list = [list(e) for e in edge_set]
    edge_index_sub = (
        np.array(new_edge_list).T if new_edge_list else np.zeros((2, 0), dtype=np.int64)
    )
    x_sub = x[node_order]
    y_sub = y_full[node_order]

    # Binary: 0 = benign, 1 = malicious (map any non-zero to 1)
    y_binary = (y_sub > 0).astype(np.int64)

    return (
        torch.tensor(x_sub, dtype=torch.float32),
        torch.tensor(edge_index_sub, dtype=torch.long),
        torch.tensor(y_binary, dtype=torch.long),
        num_nodes,
        F_in,
    )


def load_iot_train_test(
    num_train: int = 10, num_test: int = 1, F_in: int = 5, path=None
):
    """
    Load IoT data: train (10 nodes, 5 features) + test (1 node, 5 features).
    Returns x_train, edge_index_train, y_train, x_test, edge_index_full, test_node_idx, y_test.
    - edge_index_train: edges among train nodes only
    - edge_index_full: edges for full graph (train+test), test node connected to train nodes
    - test_node_idx: index of test node in full graph (num_train)
    - y_test: true labels for test node(s), shape (num_test,) binary 0/1
    """
    x, edge_index, N_full, F_in_raw, F_out, y_full = load_and_preprocess_iot_csv(
        path=path, return_labels=True
    )
    x = x.numpy()
    edge_index_np = edge_index.numpy()
    y_full = y_full.numpy()

    if F_in_raw < F_in:
        F_in = F_in_raw
    x = x[:, :F_in]

    from collections import deque

    adj = {}
    for i in range(edge_index_np.shape[1]):
        s, t = int(edge_index_np[0, i]), int(edge_index_np[1, i])
        adj.setdefault(s, []).append(t)
        adj.setdefault(t, []).append(s)

    visited = set()
    q = deque([0])
    while q and len(visited) < (num_train + num_test) * 2:
        u = q.popleft()
        if u in visited:
            continue
        visited.add(u)
        for v in adj.get(u, []):
            if v not in visited:
                q.append(v)

    node_order = list(visited)[: num_train + num_test]
    if len(node_order) < num_train + num_test:
        all_nodes = list(range(min(N_full, num_train + num_test)))
        node_order = (node_order + [n for n in all_nodes if n not in node_order])[
            : num_train + num_test
        ]

    old_to_new = {old: new for new, old in enumerate(node_order)}
    new_edge_list = []
    for i in range(edge_index_np.shape[1]):
        s, t = int(edge_index_np[0, i]), int(edge_index_np[1, i])
        if s in old_to_new and t in old_to_new:
            new_edge_list.append([old_to_new[s], old_to_new[t]])

    edge_set = {tuple(e) for e in new_edge_list}
    new_edge_list = [list(e) for e in edge_set]
    edge_index_full = (
        np.array(new_edge_list).T if new_edge_list else np.zeros((2, 0), dtype=np.int64)
    )

    x_full = x[node_order]
    y_full_sub = y_full[node_order]
    y_binary = (y_full_sub > 0).astype(np.int64)

    x_train = x_full[:num_train]
    y_train = y_binary[:num_train]
    x_test = x_full[num_train : num_train + num_test]
    y_test = y_binary[num_train : num_train + num_test]

    train_nodes = set(range(num_train))
    train_edge_list = [
        [s, t] for s, t in edge_index_full.T if s in train_nodes and t in train_nodes
    ]
    edge_index_train = (
        np.array(train_edge_list).T
        if train_edge_list
        else np.zeros((2, 0), dtype=np.int64)
    )

    test_node_idx = num_train
    return (
        x_train.astype(np.float64),
        edge_index_train,
        y_train,
        x_test.astype(np.float64),
        edge_index_full,
        test_node_idx,
        y_test,
    )


def run_train_inprocess(
    client_ctx,
    F_in: int,
    F_out: int,
    slots: int,
    num_train: int,
    edge_index_train: np.ndarray,
    ct_x_train: list,
    ct_labels_train: list,
    ct_W_list: list,
    a: np.ndarray,
    num_epochs: int,
    lr: float,
    bootstrap_weights: bool = True,
):
    """Train on train-only graph (in-process). Returns (out_cts_train, metrics, ct_W_trained)."""
    train_mask = np.ones(num_train, dtype=bool)
    payload = {
        "crypto_context": client_ctx.crypto_context,
        "public_key": client_ctx.keys.publicKey,
        "in_channels": F_in,
        "out_channels": F_out,
        "batch_size": slots,
        "ct_W_list": ct_W_list,
        "a": a,
        "negative_slope": 0.2,
        "num_nodes": num_train,
        "edge_index": edge_index_train,
        "node_features_enc": ct_x_train,
        "ct_labels": ct_labels_train,
        "train_mask": train_mask,
        "num_epochs": num_epochs,
        "lr": lr,
        "print_metrics": True,
        "bootstrap_weights": bootstrap_weights,
    }
    return compute_fhe_training(**payload)


def run_infer_inprocess(
    client_ctx,
    F_in: int,
    F_out: int,
    slots: int,
    N: int,
    edge_index_full: np.ndarray,
    ct_x_full: list,
    ct_W_trained: list,
    a: np.ndarray,
):
    """Inference on full graph (train+test) with trained weights. Returns (out_cts, metrics)."""
    payload = {
        "crypto_context": client_ctx.crypto_context,
        "public_key": client_ctx.keys.publicKey,
        "in_channels": F_in,
        "out_channels": F_out,
        "batch_size": slots,
        "ct_W_list": ct_W_trained,
        "a": a,
        "negative_slope": 0.2,
        "num_nodes": N,
        "edge_index": edge_index_full,
        "node_features_enc": ct_x_full,
        "print_metrics": False,
    }
    return compute_forward_only(**payload)


def run_http_train(url: str, payload: dict):
    """POST /train; returns (out_cts_train, metrics_dict, ct_W_trained). Uses OpenFHE serializer."""
    import urllib.request
    import urllib.error
    from client_server.openfhe_serializer import serialize_train_payload, deserialize_train_result

    serialized = serialize_train_payload(payload)
    data = json.dumps({"train_payload": serialized}).encode("utf-8")
    req = urllib.request.Request(
        url.rstrip("/") + "/train",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            out = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8") if e.fp else ""
        try:
            err_json = json.loads(err_body)
            msg = err_json.get("error", err_body or str(e))
            tb = err_json.get("traceback", "")
            if tb:
                print("Server traceback:\n" + tb, file=sys.stderr)
        except Exception:
            msg = err_body or str(e)
        raise RuntimeError(f"Server error ({e.code}): {msg}") from e
    if "error" in out:
        raise RuntimeError(out["error"])
    return deserialize_train_result(out["result"])


def run_http_infer(url: str, payload: dict):
    """POST /infer; returns (out_cts, metrics_dict). Uses OpenFHE serializer."""
    import urllib.request
    import urllib.error
    from client_server.openfhe_serializer import serialize_infer_payload, deserialize_infer_result

    serialized = serialize_infer_payload(payload)
    data = json.dumps({"infer_payload": serialized}).encode("utf-8")
    req = urllib.request.Request(
        url.rstrip("/") + "/infer",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            out = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8") if e.fp else ""
        try:
            err_json = json.loads(err_body)
            msg = err_json.get("error", err_body or str(e))
            tb = err_json.get("traceback", "")
            if tb:
                print("Server traceback:\n" + tb, file=sys.stderr)
        except Exception:
            msg = err_body or str(e)
        raise RuntimeError(f"Server error ({e.code}): {msg}") from e
    if "error" in out:
        raise RuntimeError(out["error"])
    return deserialize_infer_result(out["result"])


def main() -> None:
    if not openfhe_available():
        print("ERROR: OpenFHE not installed. pip install openfhe")
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="IoT Malicious Node Prediction (FHE client, CKKS-only)"
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument(
        "--num_train",
        type=int,
        default=5,
        help="Number of training nodes",
    )
    parser.add_argument(
        "--num_test",
        type=int,
        default=1,
        help="Number of test nodes",
    )
    parser.add_argument(
        "--url",
        type=str,
        default="",
        help="If set, POST to this URL (e.g. http://127.0.0.1:8080)",
    )
    parser.add_argument("--return_encrypted_only", action="store_true")
    parser.add_argument(
        "--mult_depth",
        type=int,
        default=25,
        help="CKKS levels per epoch when using bootstrap (default 25); total depth = mult_depth + bootstrap overhead",
    )
    parser.add_argument(
        "--ring_dim",
        type=int,
        default=16384,
        help="CKKS ring dimension (default 16384, 8GB-safe)",
    )
    parser.add_argument(
        "--no_bootstrap",
        action="store_true",
        help="Disable bootstrapping (longer modulus chain, more RAM); default is bootstrap weights after each epoch",
    )
    args = parser.parse_args()

    F_in = 5
    F_out = 1
    slots = 8

    print("=" * 60)
    print("IoT Malicious Node Prediction (FHE GAT, CKKS-only client/server)")
    print(f"  Train: {args.num_train} nodes, {F_in} features | Test: {args.num_test} node(s) | Epochs: {args.epochs}")
    print("=" * 60)

    print("\n1. Loading IoT train + test...")
    x_train, edge_index_train, y_train, x_test, edge_index_full, test_node_idx, y_test = (
        load_iot_train_test(num_train=args.num_train, num_test=args.num_test, F_in=F_in)
    )
    num_train = x_train.shape[0]
    x_full = np.vstack([x_train, x_test]).astype(np.float64)
    N_full = x_full.shape[0]
    print(f"   Train nodes={num_train}, test node index={test_node_idx}, full N={N_full}")

    client_metrics = MetricsRecorder()

    print("\n2. Client: generating keys...")
    with client_metrics.step("client_context_keygen", encrypted=True):
        client_ctx = create_client_context(
            in_channels=F_in,
            out_channels=F_out,
            batch_size=slots,
            mult_depth=args.mult_depth,
            scale_mod_size=50,
            ring_dim=args.ring_dim,
            bootstrap=not args.no_bootstrap,
        )
    print("   Done. Client keeps secret key.")

    print("\n3. Client: encrypting initial weights + train-only features/labels...")
    np.random.seed(42)
    W_init = np.random.randn(F_out, F_in).astype(np.float64) * 0.1
    concat_dim = 2 * F_out
    a = np.random.randn(concat_dim).astype(np.float64) * 0.1

    with client_metrics.step("client_encrypt_train", encrypted=True):
        ct_W_list = client_ctx.encrypt_weight_matrix(W_init, F_in)
        ct_x_train = client_ctx.encrypt_node_features(x_train, F_in)
        ct_labels_train = []
        for i in range(num_train):
            val = float(y_train[i])
            pt = client_ctx.crypto_context.MakeCKKSPackedPlaintext([val] * slots)
            ct_labels_train.append(
                client_ctx.crypto_context.Encrypt(client_ctx.keys.publicKey, pt)
            )
    print("   Done. (Train data only; test data not sent for training.)")

    print("\n4. Server: FHE training on TRAIN graph only...")
    if args.url:
        train_payload = {
            "crypto_context": client_ctx.crypto_context,
            "public_key": client_ctx.keys.publicKey,
            "in_channels": F_in,
            "out_channels": F_out,
            "batch_size": slots,
            "ct_W_list": ct_W_list,
            "a": a,
            "negative_slope": 0.2,
            "num_nodes": num_train,
            "edge_index": edge_index_train,
            "node_features_enc": ct_x_train,
            "ct_labels": ct_labels_train,
            "train_mask": np.ones(num_train, dtype=bool),
            "num_epochs": args.epochs,
            "lr": args.lr,
            "print_metrics": True,
            "bootstrap_weights": not args.no_bootstrap,
        }
        out_cts_train, metrics_dict, ct_W_trained = run_http_train(args.url, train_payload)
    else:
        out_cts_train, metrics_dict, ct_W_trained = run_train_inprocess(
            client_ctx, F_in, F_out, slots,
            num_train, edge_index_train, ct_x_train, ct_labels_train,
            ct_W_list, a, args.epochs, args.lr,
            bootstrap_weights=not args.no_bootstrap,
        )
    print("   Done. Server returned trained encrypted weights.")
    print_server_metrics(metrics_dict, title="Server metrics (training)")

    if args.return_encrypted_only:
        client_metrics.print_report()
        print("\n✓ Training done. Use trained weights for inference (not run in this mode).")
        return

    print("\n5. Client: encrypting FULL graph (train+test) for inference...")
    with client_metrics.step("client_encrypt_infer", encrypted=True):
        ct_x_full = client_ctx.encrypt_node_features(x_full, F_in)
    print("   Done.")

    print("\n6. Server: forward-only inference on full graph with trained weights...")
    if args.url:
        infer_payload = {
            "crypto_context": client_ctx.crypto_context,
            "public_key": client_ctx.keys.publicKey,
            "in_channels": F_in,
            "out_channels": F_out,
            "batch_size": slots,
            "ct_W_list": ct_W_trained,
            "a": a,
            "negative_slope": 0.2,
            "num_nodes": N_full,
            "edge_index": edge_index_full,
            "node_features_enc": ct_x_full,
            "print_metrics": False,
        }
        out_cts, metrics_infer = run_http_infer(args.url, infer_payload)
    else:
        out_cts, metrics_infer = run_infer_inprocess(
            client_ctx, F_in, F_out, slots,
            N_full, edge_index_full, ct_x_full, ct_W_trained, a,
        )
    print("   Done. Encrypted predictions for all nodes (including test) returned.")
    print_server_metrics(metrics_infer, title="Server metrics (inference)")

    print("\n7. Client: decrypting prediction for test node...")
    try:
        with client_metrics.step("client_decrypt", encrypted=False):
            output = client_ctx.decrypt_node_features(out_cts, F_out)
    except RuntimeError as e:
        if "approximation error" in str(e) or "Decode" in str(e):
            print("   Decryption failed. Use --return_encrypted_only.")
            return
        raise

    print(f"\n--- Test node(s) prediction ---")
    num_test = y_test.size
    for i in range(num_test):
        idx = test_node_idx + i
        score = float(output[idx, 0])
        pred = "ATTACKER" if score > 0.5 else "BENIGN"
        actual_label = int(y_test[i])
        actual_str = "ATTACKER" if actual_label == 1 else "BENIGN"
        print(f"  Node {idx}:  actual={actual_str} ({actual_label})  predicted={pred} (score={score:.4f})")

    client_metrics.print_report()
    print("\n✓ Done. FHE training + inference (server never had secret key).")


if __name__ == "__main__":
    main()
