#!/usr/bin/env python3
"""
FHE GAT Encoder Pipeline Demo with Per-Step Control.

This demonstrates the configurable pipeline architecture with 3 runs:
1) Fully encrypted WITH scheme switching (CKKS -> FHEW -> EvalSign -> CKKS)
2) NO scheme switching baseline (decrypt CKKS -> BinFHE EvalFunc LUT sign -> back to CKKS)
3) Fully plaintext computation

Run from GAT-FHE directory:
  cd GAT-FHE
  source venv312/bin/activate
  python examples/pipeline_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from gat_encoder_fhe import (
    GATEncoderFHE,
    FHEGraph,
    GATRunConfig,
    run_gat_pipeline,
    openfhe_available,
    openfhe_import_error,
)
from examples.test_graph import get_test_graph, print_graph_info
import argparse


def run_fully_encrypted() -> None:
    """Run 1: All steps encrypted + scheme switching (maximum security).

    We call the pipeline TWICE with the SAME encoder instance:
    - First call: includes scheme-switch precompute setup.
    - Second call: reuses cached precompute (shows *_cached metrics).
    """
    print("\n" + "="*80)
    print("Run 1: Fully Encrypted + Scheme Switching (All Steps)")
    print("="*80)
    
    x, edge_index, N, F_in, F_out = get_test_graph()
    print_graph_info()
    
    # Initialize encoder
    np.random.seed(42)
    encoder = GATEncoderFHE(
        in_channels=F_in,
        out_channels=F_out,
        batch_size=8,
        mult_depth=25,
        scale_mod_size=40,
        use_cggi=True,
        negative_slope=0.2,
    )
    
    # Create encrypted graph
    graph = FHEGraph.from_plain_encrypted(
        num_nodes=N,
        in_channels=F_in,
        edge_index=edge_index,
        node_features_plain=x,
        crypto_context=encoder.crypto_context,
        public_key=encoder.keys.publicKey,
        batch_size=8,
    )
    
    # Configure: all steps encrypted
    cfg = GATRunConfig(
        step1_linear="enc",
        step2_attention="enc",
        step3_leakyrelu="enc",
        step4_softmax="enc",
        step5_aggregation="enc",
        step3_evalsign_mode="schemeswitch",
        use_cggi=True,
        print_metrics=True,
        print_shapes=False,
    )
    
    print("\nPipeline Configuration: ALL ENCRYPTED 🔒 (with scheme switching)")
    print("Step-3 EvalSign mode: schemeswitch (CKKS↔FHEW)")
    print("-"*80)
    
    # First run: includes schemeswitch precompute
    print("\n[Run 1A] First call (includes schemeswitch precompute)")
    output1 = run_gat_pipeline(encoder=encoder, graph=graph, cfg=cfg)
    print(f"  Output shape: {output1.shape}, mean={output1.mean():.6f}")

    # Second run: precompute is cached
    print("\n[Run 1B] Second call (schemeswitch precompute cached)")
    output2 = run_gat_pipeline(encoder=encoder, graph=graph, cfg=cfg)
    print(f"  Output shape: {output2.shape}, mean={output2.mean():.6f}")


def run_no_schemeswitch_baseline() -> None:
    """Run 2: No scheme switching baseline for Step-3 sign (decrypt->BinFHE EvalFunc LUT->back to CKKS).

    We call the pipeline TWICE with the SAME encoder instance:
    - First call: includes BinFHE self-context setup + LUT generation.
    - Second call: reuses cached BinFHE context/LUT (shows *_cached metrics).
    """
    print("\n" + "="*80)
    print("Run 2: NO Scheme Switching Baseline (Step-3 via BinFHE EvalFunc LUT)")
    print("="*80)
    
    x, edge_index, N, F_in, F_out = get_test_graph()
    
    # Initialize encoder
    np.random.seed(42)
    encoder = GATEncoderFHE(
        in_channels=F_in,
        out_channels=F_out,
        batch_size=8,
        mult_depth=25,
        scale_mod_size=40,
        use_cggi=False,  # important: no scheme switching
        negative_slope=0.2,
    )
    
    # Create encrypted graph
    graph = FHEGraph.from_plain_encrypted(
        num_nodes=N,
        in_channels=F_in,
        edge_index=edge_index,
        node_features_plain=x,
        crypto_context=encoder.crypto_context,
        public_key=encoder.keys.publicKey,
        batch_size=8,
    )
    
    # Configure: keep pipeline encrypted, but Step-3 uses no-scheme-switch baseline
    cfg = GATRunConfig(
        step1_linear="enc",
        step2_attention="enc",
        step3_leakyrelu="enc",
        step4_softmax="enc",
        step5_aggregation="enc",
        step3_evalsign_mode="decrypt_encrypt_fhew_evalfunc",
        use_cggi=False,
        print_metrics=True,
        print_shapes=False,
    )
    
    print("\nPipeline Configuration: ENCRYPTED PIPELINE + Step-3 NO-SWITCH BASELINE 🔒")
    print("Step-3 EvalSign mode: decrypt->BinFHE EvalFunc LUT->decrypt bit->CKKS")
    print("-"*80)
    
    # First run: includes BinFHE selfctx setup + LUT
    print("\n[Run 2A] First call (includes BinFHE selfctx setup + LUT)")
    output1 = run_gat_pipeline(encoder=encoder, graph=graph, cfg=cfg)
    print(f"  Output shape: {output1.shape}, mean={output1.mean():.6f}")

    # Second run: selfctx is cached on encoder
    print("\n[Run 2B] Second call (BinFHE selfctx cached)")
    output2 = run_gat_pipeline(encoder=encoder, graph=graph, cfg=cfg)
    print(f"  Output shape: {output2.shape}, mean={output2.mean():.6f}")


def run_plaintext_baseline() -> None:
    """Run 3: All plaintext (for comparison)."""
    print("\n" + "="*80)
    print("Run 3: Fully Plaintext Baseline (No Encryption)")
    print("="*80)
    
    x, edge_index, N, F_in, F_out = get_test_graph()
    
    # Initialize encoder
    np.random.seed(42)
    encoder = GATEncoderFHE(
        in_channels=F_in,
        out_channels=F_out,
        batch_size=8,
        mult_depth=12,
        scale_mod_size=40,
        use_cggi=False,
        negative_slope=0.2,
    )
    
    # Create encrypted graph (still needed for initial encryption)
    graph = FHEGraph.from_plain_encrypted(
        num_nodes=N,
        in_channels=F_in,
        edge_index=edge_index,
        node_features_plain=x,
        crypto_context=encoder.crypto_context,
        public_key=encoder.keys.publicKey,
        batch_size=8,
    )
    
    # Configure: all steps plaintext
    cfg = GATRunConfig(
        step1_linear="dec",
        step2_attention="dec",
        step3_leakyrelu="dec",
        step4_softmax="dec",
        step5_aggregation="dec",
        print_metrics=True,
        print_shapes=False,
    )
    
    print("\nPipeline Configuration: ALL PLAINTEXT 🔓")
    print("-"*80)
    
    # Run pipeline
    output = run_gat_pipeline(encoder=encoder, graph=graph, cfg=cfg)
    
    print("\nOutput:")
    print(f"  Shape: {output.shape}")
    print(f"  Min:   {output.min():.6f}")
    print(f"  Max:   {output.max():.6f}")
    print(f"  Mean:  {output.mean():.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FHE GAT Encoder - Pipeline Demo (3 Runs / selectable modes)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["all", "schemeswitch", "noswitch", "plain"],
        default="all",
        help=(
            "Which demo run(s) to execute:\n"
            "  all         : run all 3 (default)\n"
            "  schemeswitch: only fully encrypted + scheme switching (Run 1A/1B)\n"
            "  noswitch    : only no-scheme-switch BinFHE baseline (Run 2A/2B)\n"
            "  plain       : only fully plaintext baseline (Run 3)"
        ),
    )
    args = parser.parse_args()

    if not openfhe_available():
        print("ERROR: OpenFHE Python is not installed.")
        err = openfhe_import_error()
        if err:
            print(f"Import error: {err}")
        print("\nInstall with: pip install openfhe")
        sys.exit(1)
    
    print("\n" + "="*80)
    print("FHE GAT Encoder - Pipeline Demo (3 Runs / selectable modes)")
    print("="*80)
    print("\nThis demo shows:")
    print("  1. Fully encrypted pipeline with scheme switching (EvalSign)")
    print("  2. No-scheme-switch baseline for Step-3 sign (decrypt->BinFHE EvalFunc LUT->back to CKKS)")
    print("  3. Fully plaintext baseline")
    print("  4. Automatic memory (RSS) and timing metrics for each step/substep")
    print(f"\nSelected mode: {args.mode}")
    
    # Run selected modes
    if args.mode in ("all", "schemeswitch"):
        run_fully_encrypted()
    if args.mode in ("all", "noswitch"):
        run_no_schemeswitch_baseline()
    if args.mode in ("all", "plain"):
        run_plaintext_baseline()
    
    print("\n" + "="*80)
    print("Demo Complete!")
    print("="*80)
    print("\nKey Insights:")
    print("  ✓ Scheme switching has one-time setup + per-inference costs (reported separately)")
    print("  ✓ No-scheme-switch baseline isolates alternative sign costs (decrypt + BinFHE EvalFunc + re-encrypt)")
    print("  ✓ Use metrics to identify bottlenecks in your use case")
    print()


if __name__ == "__main__":
    main()
