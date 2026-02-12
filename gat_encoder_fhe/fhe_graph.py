"""
FHE-friendly graph structure for GAT encoder.
Holds graph topology (plaintext) and ONLY encrypted node features (never stores plaintext).
Security principle: Plaintext features are encrypted immediately and discarded.
See PLAN.md §4 and §5.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

import numpy as np


# OpenFHE ciphertext type (any when openfhe not installed)
Ciphertext = Any


@dataclass
class FHEGraph:
    """
    Secure graph representation for FHE GAT: plaintext structure, ENCRYPTED node features only.

    - num_nodes: number of nodes N
    - in_channels: input feature dimension F_in
    - edge_index: (2, E) in COO form; row 0 = source, row 1 = target (edge source -> target)
    - node_features_enc: list of N CKKS ciphertexts (one per node, slots = F_in)
    
    SECURITY: Plaintext features are NEVER stored; only encrypted ciphertexts.
    Use `from_plain_encrypted()` to build from plaintext inputs (encrypts immediately).
    """

    num_nodes: int
    in_channels: int
    edge_index: np.ndarray  # (2, E), dtype int
    node_features_enc: List[Ciphertext]  # length N, CKKS ciphertexts

    def __post_init__(self) -> None:
        if self.edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape (2, E)")
        if len(self.node_features_enc) != self.num_nodes:
            raise ValueError(
                f"node_features_enc must have length {self.num_nodes}"
            )

    @property
    def num_edges(self) -> int:
        return self.edge_index.shape[1]

    @property
    def is_encrypted(self) -> bool:
        """Always True; FHEGraph only stores encrypted features."""
        return True

    @classmethod
    def from_encrypted(
        cls,
        num_nodes: int,
        in_channels: int,
        edge_index: np.ndarray,
        node_features_enc: List[Ciphertext],
    ) -> "FHEGraph":
        """Build FHEGraph with pre-encrypted node features."""
        return cls(
            num_nodes=num_nodes,
            in_channels=in_channels,
            edge_index=np.asarray(edge_index, dtype=np.int64),
            node_features_enc=node_features_enc,
        )

    @classmethod
    def from_plain_encrypted(
        cls,
        num_nodes: int,
        in_channels: int,
        edge_index: np.ndarray,
        node_features_plain: np.ndarray,
        crypto_context: Any,
        public_key: Any,
        batch_size: int = 8,
    ) -> "FHEGraph":
        """
        Build FHEGraph from plaintext features by encrypting them immediately.
        Plaintext features are NOT stored (security).
        
        node_features_plain: (N, F_in) plaintext array
        crypto_context: CKKS CryptoContext from OpenFHE
        public_key: CKKS public key
        batch_size: CKKS batch size (must be >= in_channels)
        
        Returns: FHEGraph with encrypted features only
        """
        node_features_plain = np.asarray(node_features_plain, dtype=np.float64)
        if node_features_plain.shape != (num_nodes, in_channels):
            raise ValueError(
                f"node_features_plain must have shape ({num_nodes}, {in_channels})"
            )
        if in_channels > batch_size:
            raise ValueError(f"in_channels ({in_channels}) > batch_size ({batch_size})")
        
        # Encrypt features (one ciphertext per node, packed slots)
        ct_list = []
        for i in range(num_nodes):
            row = np.zeros(batch_size, dtype=np.float64)
            row[:in_channels] = node_features_plain[i]
            pt_x = crypto_context.MakeCKKSPackedPlaintext(row.tolist())
            ct_x = crypto_context.Encrypt(public_key, pt_x)
            ct_list.append(ct_x)
        
        # Plaintext features are discarded after encryption (not stored)
        return cls(
            num_nodes=num_nodes,
            in_channels=in_channels,
            edge_index=np.asarray(edge_index, dtype=np.int64),
            node_features_enc=ct_list,
        )
