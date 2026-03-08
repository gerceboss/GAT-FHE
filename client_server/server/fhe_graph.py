"""
FHE graph for CKKS-only server: topology + encrypted node features.

Dual (line) graph, node-based GAT only. This represents a (sub)graph of the
line graph: each node = one original edge; node features are encrypted
(e.g. src_bytes, dst_bytes, duration from the original edge). One output
per node = one logit per original edge. No separate edge features.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

import numpy as np


@dataclass
class FHEGraph:
    """Graph with encrypted node features (server never sees plaintext). Line-graph pipeline: nodes = original edges."""
    num_nodes: int
    in_channels: int
    edge_index: np.ndarray  # (2, E) adjacency of the (line-graph) subgraph
    node_features_enc: List[Any]  # N CKKS ciphertexts

    def __post_init__(self) -> None:
        if self.edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape (2, E)")
        if len(self.node_features_enc) != self.num_nodes:
            raise ValueError(f"node_features_enc length must be {self.num_nodes}")

    @classmethod
    def from_encrypted(
        cls,
        num_nodes: int,
        in_channels: int,
        edge_index: np.ndarray,
        node_features_enc: List[Any],
    ) -> "FHEGraph":
        return cls(
            num_nodes=num_nodes,
            in_channels=in_channels,
            edge_index=np.asarray(edge_index, dtype=np.int64),
            node_features_enc=node_features_enc,
        )
