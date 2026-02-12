"""
GAT Encoder - Plaintext Graph Attention Network implementation.
Includes PyTorch multi-layer encoder and NumPy utilities for testing.
"""

from .core import (
    GATEncoder,
    GATLayer,
    matmul_plain,
    linear_plain,
    attention_plain,
    gat_forward_plain,
)

__all__ = [
    "GATEncoder",
    "GATLayer",
    "matmul_plain",
    "linear_plain",
    "attention_plain",
    "gat_forward_plain",
]
