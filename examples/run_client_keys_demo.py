#!/usr/bin/env python3
"""
Client-key architecture demo for GAT FHE inference.

Flow:
1. Client generates CKKS keypair and evaluation keys
2. Client encrypts node features with public key
3. Client sends: crypto context, public key, encrypted inputs to server
4. Server runs full GAT pipeline in CKKS (never has secret key)
5. Server returns encrypted result
6. Client decrypts with secret key

Run from repo root:
  source venv/bin/activate
  python examples/run_client_keys_demo.py
"""

import sys
from pathlib import Path
import argparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from gat_encoder_fhe import (
    GATEncoderFHE,
    FHEGraph,
    create_client_context,
    run_gat_pipeline_client_keys,
    MetricsRecorder,
    openfhe_available,
)


def main() -> None:
    if not openfhe_available():
        print("ERROR: OpenFHE not installed. pip install openfhe")
        sys.exit(1)

    from generate_graph import generate_random_graph

    print("=" * 60)
    print("Client-Key Architecture Demo (CKKS-only) ")
    print("=" * 60)

    parser = argparse.ArgumentParser(description="Client-Key Architecture Demo (CKKS-only)")
    parser.add_argument("--dataset", type=str, default="random", choices=["cora", "iot", "random"], help="Dataset to use")
    parser.add_argument("--num_nodes", type=int, default=10, help="Number of nodes in the graph")
    parser.add_argument("--num_edges", type=int, default=10, help="Number of edges in the graph")
    parser.add_argument("--in_channels", type=int, default=5, help="Number of input features")
    parser.add_argument("--out_channels", type=int, default=5, help="Number of output features")
    args = parser.parse_args()

    if args.dataset == "cora":
        x, edge_index, N, F_in, F_out = load_cora_from_csv()
    elif args.dataset == "iot":
        x, edge_index, N, F_in, F_out = load_and_preprocess_iot_csv()
    else:
        print(f"DEFAULT: Generating random graph "
              f"{args.num_nodes} nodes, "
              f"{args.num_edges} edges, "
              f"{args.in_channels} input features, "
              f"{args.out_channels} output features")

        x, edge_index, N, F_in, F_out = generate_random_graph(
            num_nodes=args.num_nodes,
            num_edges=args.num_edges,
            in_channels=args.in_channels,
            out_channels=args.out_channels
        )

    # --- Client: generate keys ---
    print("\n1. Client: generating keys...")
    client_metrics = MetricsRecorder()
    with client_metrics.step("client_keygen", encrypted=True):
        client_ctx = create_client_context(
            in_channels=F_in,
            out_channels=F_out,
            batch_size=32,
            mult_depth=25,
            scale_mod_size=40,
        )
    print("   Done. Client keeps secret key.")

    # --- Reference encoder for model weights (server has these) ---
    np.random.seed(42)
    ref_enc = GATEncoderFHE(
        in_channels=F_in,
        out_channels=F_out,
        batch_size=32,
        mult_depth=25,
        scale_mod_size=40,
        use_cggi=False,
        negative_slope=0.2,
    )

    # --- Server: encoder with ONLY public key (cannot decrypt) ---
    print("\n2. Server: creating encoder from client's public key only...")
    server_enc = GATEncoderFHE.from_client_keys(
        crypto_context=client_ctx.crypto_context,
        public_key=client_ctx.keys.publicKey,
        in_channels=F_in,
        out_channels=F_out,
        batch_size=32,
        W=ref_enc._W,
        a=ref_enc._a,
        negative_slope=0.2,
    )
    print("   Done. Server has NO secret key.")

    # --- Client: encrypt inputs ---
    print("\n3. Client: encrypting node features...")
    with client_metrics.step("client_encrypt", encrypted=True):
        ct_x_list = client_ctx.encrypt_node_features(x, F_in)
    graph = FHEGraph.from_encrypted(
        num_nodes=N,
        in_channels=F_in,
        edge_index=edge_index,
        node_features_enc=ct_x_list,
    )
    print("   Done. Plaintext discarded.")

    # --- Server: run pipeline (all encrypted, no decryption) ---
    print("\n4. Server: running GAT pipeline (CKKS-only)...")
    out_cts, metrics_dict = run_gat_pipeline_client_keys(
        encoder=server_enc,
        graph=graph,
        print_metrics=True,
    )
    print("   Done. Output is encrypted.")

    # --- Client: decrypt result ---
    print("\n5. Client: decrypting result...")
    with client_metrics.step("client_decrypt", encrypted=False):
        output = client_ctx.decrypt_node_features(out_cts, F_out)
    print("   Done.")

    # --- Print client-side metrics ---
    print("\n" + "=" * 80)
    print("=== Client-Side Metrics (keygen, encrypt, decrypt) ===")
    print("=" * 80)
    client_metrics.print_report()

    print(f"\nOutput shape: {output.shape}")
    print(f"Output mean: {output.mean():.6f}")
    print("\n✓ Client-key architecture demo complete.")
    print("  Server never had access to secret key or plaintext.")


if __name__ == "__main__":
    main()
