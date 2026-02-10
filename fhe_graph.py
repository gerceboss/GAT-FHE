"""
FHE-friendly graph structure for GAT encoder.
Holds graph topology (plaintext) and node features either in plaintext or as encrypted ciphertexts.
See PLAN.md §4 and §5.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

import numpy as np


# OpenFHE ciphertext type (any when openfhe not installed)
Ciphertext = Any


@dataclass
class FHEGraph:
    """
    Graph representation for FHE GAT: plaintext structure, optional encrypted node features.

    - num_nodes: number of nodes N
    - in_channels: input feature dimension F_in
    - edge_index: (2, E) in COO form; row 0 = source, row 1 = target (edge source -> target)
    - node_features_plain: optional (N, F_in) plaintext features
    - node_features_enc: optional list of N CKKS ciphertexts (one per node, slots = F_in)
    Exactly one of node_features_plain or node_features_enc should be set for forward pass.
    """

    num_nodes: int
    in_channels: int
    edge_index: np.ndarray  # (2, E), dtype int
    node_features_plain: Optional[np.ndarray] = None  # (N, F_in)
    node_features_enc: Optional[List[Ciphertext]] = None  # length N

    def __post_init__(self) -> None:
        if self.edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape (2, E)")
        if self.node_features_plain is not None:
            if self.node_features_plain.shape != (self.num_nodes, self.in_channels):
                raise ValueError(
                    f"node_features_plain must have shape ({self.num_nodes}, {self.in_channels})"
                )
        if self.node_features_enc is not None:
            if len(self.node_features_enc) != self.num_nodes:
                raise ValueError(
                    f"node_features_enc must have length {self.num_nodes}"
                )

    @property
    def num_edges(self) -> int:
        return self.edge_index.shape[1]

    @property
    def is_encrypted(self) -> bool:
        return self.node_features_enc is not None

    def to_plain(self) -> np.ndarray:
        """Return plaintext node features; raises if only encrypted features exist."""
        if self.node_features_plain is not None:
            return self.node_features_plain
        raise ValueError("No plaintext node features; graph has encrypted features only.")

    @classmethod
    def from_plain(
        cls,
        num_nodes: int,
        in_channels: int,
        edge_index: np.ndarray,
        node_features: np.ndarray,
    ) -> "FHEGraph":
        """Build FHEGraph with plaintext node features."""
        return cls(
            num_nodes=num_nodes,
            in_channels=in_channels,
            edge_index=np.asarray(edge_index, dtype=np.int64),
            node_features_plain=np.asarray(node_features, dtype=np.float64),
            node_features_enc=None,
        )
