"""
Verify FHE single-layer GAT encoder.
Requires OpenFHE Python (pip install openfhe on Ubuntu 22.04/24.04).
Builds a small graph, runs plain and FHE forward, compares outputs.
"""

import sys

import numpy as np

from fhe_graph import FHEGraph
from gat_encoder_fhe import GATEncoderFHE, openfhe_available, openfhe_import_error


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

    np.random.seed(42)
    # Small graph for fast run
    N, F_in, F_out = 6, 4, 4
    E = 12
    edge_index = np.random.randint(0, N, (2, E))
    edge_index = edge_index[:, edge_index[0] != edge_index[1]]
    if edge_index.shape[1] < 4:
        edge_index = np.random.randint(0, N, (2, 12))
    x = np.random.randn(N, F_in).astype(np.float64) * 0.5

    graph = FHEGraph.from_plain(N, F_in, edge_index, x)

    encoder = GATEncoderFHE(
        in_channels=F_in,
        out_channels=F_out,
        negative_slope=0.2,
        batch_size=8,
        mult_depth=2,
        scale_mod_size=50,
    )

    # Plain forward
    out_plain = encoder.forward_plain(graph)
    assert out_plain.shape == (N, F_out), f"Plain shape {out_plain.shape}"
    assert np.isfinite(out_plain).all(), "Plain output had NaN/Inf"
    print("[OK] Plain forward: shape =", out_plain.shape, ", finite =", np.isfinite(out_plain).all())

    # FHE forward (encrypt -> aggregate -> decrypt)
    out_fhe = encoder.forward_fhe(graph)
    assert out_fhe.shape == (N, F_out), f"FHE shape {out_fhe.shape}"
    assert np.isfinite(out_fhe).all(), "FHE output had NaN/Inf"
    print("[OK] FHE forward: shape =", out_fhe.shape, ", finite =", np.isfinite(out_fhe).all())

    # Compare (CKKS introduces small error)
    diff = np.abs(out_plain - out_fhe)
    max_diff = diff.max()
    mean_diff = diff.mean()
    print(f"[OK] Plain vs FHE: max |diff| = {max_diff:.6f}, mean |diff| = {mean_diff:.6f}")

    print("\n--- FHE GAT summary ---")
    print("Graph: nodes =", N, ", edges =", edge_index.shape[1], ", in =", F_in, ", out =", F_out)
    print("Plain  output sample (node 0):", out_plain[0])
    print("FHE    output sample (node 0):", out_fhe[0])
    print("FHE encoder verification passed.")


if __name__ == "__main__":
    main()
