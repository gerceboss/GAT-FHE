"""
PyTorch GAT encoder verification using hardcoded test graph.
Compare with FHE encoder output from fhe_gat.py.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from examples.test_graph import get_test_graph, print_graph_info
from gat_encoder import gat_forward_plain


def main() -> None:
    # Load hardcoded test graph
    node_features, edge_index, N, F_in, F_out = get_test_graph()
    
    print_graph_info()
    print()
    
    # Run plaintext GAT forward (NumPy implementation)
    # This matches what FHE encoder uses internally
    print("=== PyTorch/NumPy GAT Encoder ===")
    
    # Random weights for single-layer encoder (seed for reproducibility)
    np.random.seed(42)
    W = np.random.randn(F_out, F_in).astype(np.float64) * 0.1
    a = np.random.randn(2 * F_out).astype(np.float64) * 0.1
    negative_slope = 0.2
    
    # Forward pass (plaintext)
    output = gat_forward_plain(
        x=node_features,
        edge_index=edge_index,
        W=W,
        a=a,
        negative_slope=negative_slope
    )
    
    # Print results
    print(f"Input shape: {node_features.shape}")
    print(f"Edge index shape: {edge_index.shape}")
    print(f"Output shape: {output.shape}")
    print(f"\nOutput statistics:")
    print(f"  Min:  {output.min():.6f}")
    print(f"  Max:  {output.max():.6f}")
    print(f"  Mean: {output.mean():.6f}")
    print(f"  Std:  {output.std():.6f}")
    
    print(f"\nOutput per node:")
    for i in range(N):
        print(f"  Node {i}: [{', '.join(f'{v:.6f}' for v in output[i])}]")
    
    # Verify
    assert output.shape == (N, F_out), f"Shape mismatch: {output.shape}"
    assert np.isfinite(output).all(), "NaN/Inf in output"
    
    print(f"\n✓ PyTorch encoder test passed")
    print(f"\nNow run: python example_fhe_verify.py")
    print(f"Compare the outputs to verify FHE correctness!")


if __name__ == "__main__":
    main()
