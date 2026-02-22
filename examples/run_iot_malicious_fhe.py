#!/usr/bin/env python3
"""
IoT Malicious Node Prediction with FHE GAT (NO secret key on server).

Flow:
1. Client generates keys, encrypts initial weights W, features, labels
2. Client sends: crypto context, public key, encrypted W, encrypted data to server
3. Server runs FHE training (weights stay encrypted; ct_W_new = ct_W_old - lr * ct_grad_W)
4. Server returns encrypted predictions
5. Client decrypts for attacker/benign

Server NEVER has the secret key.

Run from repo root:
  source venv/bin/activate
  python examples/run_iot_malicious_fhe.py --epochs 3
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXAMPLES_DIR))

import argparse
import numpy as np

from gat_encoder_fhe import (
    GATEncoderFHE,
    FHEGraph,
    run_gat_pipeline_fhe_training,
    create_client_context,
    MetricsRecorder,
    openfhe_available,
)
from dataset_retreival import load_iot_train_test


def main() -> None:
    if not openfhe_available():
        print("ERROR: OpenFHE not installed. pip install openfhe")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="IoT Malicious Node Prediction (FHE training, no SK on server)")
    parser.add_argument("--epochs", type=int, default=3, help="Number of FHE training epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--return_encrypted_only", action="store_true")
    args = parser.parse_args()

    F_in = 5
    F_out = 1
    slots = 8 

    print("=" * 60)
    print("IoT Malicious Node Prediction (FHE GAT, NO secret key on server)")
    print(f"  Train: 10 nodes, 5 features | Test: 1 node | Epochs: {args.epochs}")
    print("=" * 60)

    # --- Load train + test ---
    print("\n1. Loading IoT train + test...")
    x_train, edge_index_train, y_train, x_test, edge_index_full, test_node_idx = load_iot_train_test(
        num_train=5, num_test=1, F_in=F_in
    )
    x_full = np.vstack([x_train, x_test]).astype(np.float64)
    y_full = np.concatenate([y_train, [0]])  # placeholder label for test node
    N = x_full.shape[0]
    train_mask = np.array([i < 10 for i in range(N)])  # first 10 = train
    print(f"   N={N}, train nodes=5, test node index={test_node_idx}")

    # --- Client: generate keys (keeps secret key) ---
    print("\n2. Client: generating keys...")
    client_ctx = create_client_context(
        in_channels=F_in,
        out_channels=F_out,
        batch_size=slots,
        mult_depth=50,
        scale_mod_size=50,
    )
    print("   Done. Client keeps secret key.")

    # --- Client: create initial W and a, encrypt W ---
    print("\n3. Client: encrypting initial weights + features + labels...")
    np.random.seed(42)
    W_init = np.random.randn(F_out, F_in).astype(np.float64) * 0.1
    concat_dim = 2 * F_out
    a = np.random.randn(concat_dim).astype(np.float64) * 0.1

    client_metrics = MetricsRecorder()
    with client_metrics.step("client_encrypt", encrypted=True):
        ct_W_list = client_ctx.encrypt_weight_matrix(W_init, F_in)
        ct_x_list = client_ctx.encrypt_node_features(x_full, F_in)
        ct_labels = []
        for i in range(N):
            val = float(y_full[i])
            pt = client_ctx.crypto_context.MakeCKKSPackedPlaintext([val] * slots)
            ct_labels.append(client_ctx.crypto_context.Encrypt(client_ctx.keys.publicKey, pt))
    graph = FHEGraph.from_encrypted(
        num_nodes=N,
        in_channels=F_in,
        edge_index=edge_index_full,
        node_features_enc=ct_x_list,
    )
    print("   Done.")

    # --- Server: create encoder (NO secret key, encrypted weights only) ---
    print("\n4. Server: creating encoder from client's public key + encrypted weights...")
    encoder = GATEncoderFHE.from_client_keys_with_encrypted_weights(
        crypto_context=client_ctx.crypto_context,
        public_key=client_ctx.keys.publicKey,
        in_channels=F_in,
        out_channels=F_out,
        batch_size=slots,
        ct_W_list=ct_W_list,
        a=a,
        negative_slope=0.2,
    )
    print("   Done. Server has NO secret key.")

    # --- Server: FHE training loop (weights stay encrypted) ---
    print(f"\n5. Server: FHE training ({args.epochs} epochs, lr={args.lr})...")
    out_cts, metrics_dict = run_gat_pipeline_fhe_training(
        encoder=encoder,
        graph=graph,
        ct_labels=ct_labels,
        train_mask=train_mask,
        num_epochs=args.epochs,
        lr=args.lr,
        print_metrics=True,
    )
    print("   Done. Encrypted predictions returned.")

    if args.return_encrypted_only:
        print("\n✓ Encrypted predictions returned (client can decrypt).")
        return

    # --- Client: decrypt (only client has secret key) ---
    print("\n6. Client: decrypting prediction for test node...")
    try:
        with client_metrics.step("decrypt", encrypted=False):
            output = client_ctx.decrypt_node_features(out_cts, F_out)
    except RuntimeError as e:
        if "approximation error" in str(e) or "Decode" in str(e):
            print("   Decryption failed. Use --return_encrypted_only.")
            return
        raise

    score = float(output[test_node_idx, 0])
    pred = "ATTACKER" if score > 0.5 else "BENIGN"
    print(f"\n--- Test Node {test_node_idx} Prediction ---")
    print(f"  Score: {score:.4f}  =>  {pred}")

    print("\n=== Client Metrics ===")
    client_metrics.print_report()
    print("\n✓ Done. FHE training + inference complete (server never had secret key).")


if __name__ == "__main__":
    main()
