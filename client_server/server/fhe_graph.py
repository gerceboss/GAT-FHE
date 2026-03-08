"""
FHE graph for CKKS-only server: topology + encrypted node features.
Optional edge features (plaintext) for edge-based prediction alignment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np


@dataclass
class FHEGraph:
    """Graph with encrypted node features (server never sees plaintext)."""
    num_nodes: int
    in_channels: int
    edge_index: np.ndarray  # (2, E)
    node_features_enc: List[Any]  # N CKKS ciphertexts
    edge_features: Optional[np.ndarray] = None  # (E, edge_feat_dim) optional, plaintext

    def __post_init__(self) -> None:
        if self.edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape (2, E)")
        if len(self.node_features_enc) != self.num_nodes:
            raise ValueError(f"node_features_enc length must be {self.num_nodes}")
        if self.edge_features is not None and self.edge_features.shape[0] != self.edge_index.shape[1]:
            raise ValueError(
                f"edge_features rows ({self.edge_features.shape[0]}) must match "
                f"edge_index cols ({self.edge_index.shape[1]})"
            )

    @classmethod
    def from_encrypted(
        cls,
        num_nodes: int,
        in_channels: int,
        edge_index: np.ndarray,
        node_features_enc: List[Any],
        edge_features: Optional[np.ndarray] = None,
    ) -> "FHEGraph":
        return cls(
            num_nodes=num_nodes,
            in_channels=in_channels,
            edge_index=np.asarray(edge_index, dtype=np.int64),
            node_features_enc=node_features_enc,
            edge_features=edge_features,
        )
