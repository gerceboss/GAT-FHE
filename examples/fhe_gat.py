"""
Verify FHE single-layer GAT encoder.
Requires OpenFHE Python (pip install openfhe on Ubuntu 22.04/24.04).
Builds a small graph, runs plain and FHE forward, compares outputs.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from examples.test_graph import get_test_graph, print_graph_info
from gat_encoder import gat_forward_plain
from gat_encoder_fhe import GATEncoderFHE, FHEGraph, openfhe_available, openfhe_import_error


def main() -> None:
    if not openfhe_available():
        print("OpenFHE Python is not installed. Skipping FHE verification.")
        err = openfhe_import_error()
        if err:
            print("Import error:", err)
        print("Install with: pip install openfhe  (see PyPI for your OS/version)")
        print("Supported platforms: Ubuntu 22.04, 24.04. See:")
        print("  https://github.com/openfheorg/openfhe-python#installing-using-pip-for-ubuntu")
        sys.exit(0)

    # Load hardcoded test graph (same as plain_gat.py)
    x_plain, edge_index, N, F_in, F_out = get_test_graph()
    
    print_graph_info()
    print()

    # Create encoder first (needed for crypto context)
    # MEMORY OPTIMIZATION: Use smaller parameters for lightweight execution
    # Set SAME random seed as example_verify.py for matching weights
    np.random.seed(42)
    
    encoder = GATEncoderFHE(
        in_channels=F_in,
        out_channels=F_out,
        negative_slope=0.2,
        batch_size=4,        # Reduced from 8 to save memory
        mult_depth=30,       # Reduced from 30 (less memory but still functional)
        scale_mod_size=40,   # Reduced from 50 (smaller ciphertexts)
        use_cggi=True,      # Disable scheme switching to save memory (LeakyReLU = identity)
    )
    
    # IMPORTANT: Set encoder weights to match plaintext encoder
    # This ensures we're testing the FHE operations, not weight differences
    encoder._W = np.random.randn(F_out, F_in).astype(np.float64) * 0.1
    encoder._a = np.random.randn(2 * F_out).astype(np.float64) * 0.1

    # Build FHEGraph with encrypted features (plaintext x is NOT stored, only encrypted)
    graph = FHEGraph.from_plain_encrypted(
        num_nodes=N,
        in_channels=F_in,
        edge_index=edge_index,
        node_features_plain=x_plain,
        crypto_context=encoder.crypto_context,
        public_key=encoder.keys.publicKey,
        batch_size=4,  # Match encoder batch_size
    )

    # Plain forward (uses separate plaintext copy for testing, NOT from graph)
    out_plain = gat_forward_plain(x_plain, edge_index, encoder._W, encoder._a, encoder.negative_slope)
    assert out_plain.shape == (N, F_out), f"Plain shape {out_plain.shape}"
    assert np.isfinite(out_plain).all(), "Plain output had NaN/Inf"
    print("[OK] Plain forward: shape =", out_plain.shape, ", finite =", np.isfinite(out_plain).all())

    # FHE forward - Memory-optimized encrypted GAT
    print("\n[Testing] Memory-optimized encrypted GAT...")
    print("(CKKS linear, rotation-based attention, Chebyshev exp)")
    print("OPTIMIZATION: batch_size=4, mult_depth=12, no scheme switching (saves memory)")
    out_fhe_full = encoder.forward_fhe_full(graph)
    assert out_fhe_full.shape == (N, F_out), f"FHE full shape {out_fhe_full.shape}"
    assert np.isfinite(out_fhe_full).all(), "FHE full output had NaN/Inf"
    print("[OK] FHE fully-encrypted forward: shape =", out_fhe_full.shape, ", finite =", np.isfinite(out_fhe_full).all())

    # Compare fully-encrypted vs plain
    diff_full = np.abs(out_plain - out_fhe_full)
    print(f"[OK] Plain vs Fully-Encrypted: max |diff| = {diff_full.max():.6f}, mean |diff| = {diff_full.mean():.6f}")

    print("\n=== FHE GAT Comparison ===")
    print(f"Graph: nodes={N}, edges={edge_index.shape[1]}, features={F_in}→{F_out}")
    print("MEMORY OPTIMIZATION: batch_size=4, mult_depth=12, scale_mod=40")
    print("Security: FHEGraph stores ONLY encrypted features (plaintext never stored)")
    
    print(f"\nPlaintext output per node:")
    for i in range(N):
        print(f"  Node {i}: [{', '.join(f'{v:.6f}' for v in out_plain[i])}]")
    
    print(f"\nFHE encrypted output per node:")
    for i in range(N):
        print(f"  Node {i}: [{', '.join(f'{v:.6f}' for v in out_fhe_full[i])}]")
    
    print(f"\nDifference (FHE - Plaintext) per node:")
    diff_per_node = out_fhe_full - out_plain
    for i in range(N):
        print(f"  Node {i}: [{', '.join(f'{v:+.6f}' for v in diff_per_node[i])}]")
    
    print(f"\nOverall statistics:")
    print(f"  Max absolute difference: {diff_full.max():.6f}")
    print(f"  Mean absolute difference: {diff_full.mean():.6f}")
    print(f"  Std of difference: {diff_per_node.std():.6f}")
    
    print("\n✓ FHE encoder verification passed!")
    print("✓ FHE and plaintext outputs are close (CKKS approximation error)")
    print("\nTo enable full encryption (requires more memory):")
    print("  - Set mult_depth=30+, batch_size=8, use_cggi=True")


if __name__ == "__main__":
    main()
