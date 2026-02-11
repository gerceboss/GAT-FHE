"""
GAT Encoder — Graph Attention Network encoder only (no classification head).
Based on: Veličković et al., "Graph Attention Networks", ICLR 2018.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class GATLayer(nn.Module):
    """Single head of a GAT layer. Computes attention over neighbors and aggregates."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        negative_slope: float = 0.2,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.negative_slope = negative_slope

        self.W = nn.Linear(in_channels, out_channels, bias=False)
        self.a = nn.Parameter(torch.empty(2 * out_channels, 1))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
    ) -> Tensor:
        # x: (N, F_in), edge_index: (2, E) with [source, target]
        N = x.size(0)
        row, col = edge_index[1], edge_index[0]  # target, source per edge

        h = self.W(x)  # (N, F_out)
        h_row = h[row]  # (E, F_out)
        h_col = h[col]  # (E, F_out)
        h_cat = torch.cat([h_row, h_col], dim=-1)  # (E, 2*F_out)

        e = (h_cat @ self.a).squeeze(-1)  # (E,)
        e = F.leaky_relu(e, negative_slope=self.negative_slope)

        # Softmax over edges grouped by target (row)
        alpha = self._edge_softmax(e, row, N)

        # Aggregate: out_i = sum_j alpha_ij * h_j  (j = col, i = row)
        out = torch.zeros(N, self.out_channels, device=x.device, dtype=x.dtype)
        alpha_exp = alpha.unsqueeze(-1)  # (E, 1)
        out.index_add_(0, row, alpha_exp * h_col)
        return out

    def _edge_softmax(self, e: Tensor, index: Tensor, num_nodes: int) -> Tensor:
        """Softmax over edges that share the same target node (index)."""
        e_max = torch.zeros(num_nodes, device=e.device, dtype=e.dtype)
        e_max.scatter_reduce_(0, index, e, reduce="amax", include_self=False)
        e_max = e_max[index]  # (E,)
        e_exp = torch.exp(e - e_max)
        e_sum = torch.zeros(num_nodes, device=e.device, dtype=e.dtype)
        e_sum.index_add_(0, index, e_exp)
        e_sum = e_sum[index].clamp(min=1e-16)
        return e_exp / e_sum


class GATEncoder(nn.Module):
    """
    Multi-layer GAT encoder. Hidden layers use multi-head concat;
    final layer uses mean over heads so output dim is out_channels.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        negative_slope: float = 0.2,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.layers = nn.ModuleList()
        if num_layers == 1:
            # Single layer: in -> out, mean over heads
            self.layers.append(
                nn.ModuleList(
                    [
                        GATLayer(in_channels, out_channels, negative_slope)
                        for _ in range(num_heads)
                    ]
                )
            )
        else:
            # First layer: in -> hidden * num_heads (concat)
            self.layers.append(
                nn.ModuleList(
                    [
                        GATLayer(in_channels, hidden_channels, negative_slope)
                        for _ in range(num_heads)
                    ]
                )
            )
            # Middle layers: hidden*num_heads -> hidden*num_heads
            for _ in range(num_layers - 2):
                self.layers.append(
                    nn.ModuleList(
                        [
                            GATLayer(
                                hidden_channels * num_heads,
                                hidden_channels,
                                negative_slope,
                            )
                            for _ in range(num_heads)
                        ]
                    )
                )
            # Last layer: hidden*num_heads -> out (mean over heads)
            self.layers.append(
                nn.ModuleList(
                    [
                        GATLayer(
                            hidden_channels * num_heads,
                            out_channels,
                            negative_slope,
                        )
                        for _ in range(num_heads)
                    ]
                )
            )

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
    ) -> Tensor:
        if self.num_layers == 1:
            last_heads = self.layers[0]
            h = torch.stack([head(x, edge_index) for head in last_heads], dim=0).mean(dim=0)
            return h

        # Input layer
        layer_heads = self.layers[0]
        h = torch.cat([head(x, edge_index) for head in layer_heads], dim=-1)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # Middle layers
        for layer_heads in self.layers[1 : -1]:
            h_next = torch.cat([head(h, edge_index) for head in layer_heads], dim=-1)
            h = F.elu(h_next)
            h = F.dropout(h, p=self.dropout, training=self.training)

        # Last layer: mean over heads
        last_heads = self.layers[-1]
        h = torch.stack([head(h, edge_index) for head in last_heads], dim=0).mean(dim=0)
        return h


# ============================================================================
# NumPy-based plaintext utilities for FHE testing and comparison
# ============================================================================

import numpy as np
from typing import Tuple


def matmul_plain(x: np.ndarray, W: np.ndarray) -> np.ndarray:
    """
    Plaintext matrix multiplication helper.

    - x: shape (N, F_in)
    - W: shape (F_out, F_in)
    Returns: (N, F_out) = x @ W.T
    """
    return x @ W.T


def linear_plain(x: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Apply linear transformation h' = W @ x in plaintext."""
    return matmul_plain(x, W)


def attention_plain(
    edge_index: np.ndarray,
    h: np.ndarray,
    a: np.ndarray,
    num_nodes: int,
    negative_slope: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute attention coefficients alpha over edges (plaintext).
    
    edge_index: (2, E) with row 0 = source, row 1 = target
    h: (N, F_out) transformed node features
    a: (2*F_out,) attention vector
    num_nodes: N
    negative_slope: LeakyReLU slope
    
    Returns: (e_scores, alpha) both shape (E,)
    """
    row = edge_index[1]  # target
    col = edge_index[0]  # source
    h_row = h[row]   # (E, F_out)
    h_col = h[col]   # (E, F_out)
    h_cat = np.concatenate([h_row, h_col], axis=1)  # (E, 2*F_out)
    e_raw = h_cat @ a
    
    # LeakyReLU
    e = np.where(e_raw >= 0, e_raw, negative_slope * e_raw)
    
    # Softmax over edges by target
    e_max = np.full(num_nodes, -np.inf, dtype=np.float64)
    np.maximum.at(e_max, row, e)
    e_max = e_max[row]
    e_exp = np.exp(np.clip(e - e_max, -50, 50))
    e_sum = np.zeros(num_nodes, dtype=np.float64)
    np.add.at(e_sum, row, e_exp)
    e_sum = e_sum[row]
    e_sum = np.maximum(e_sum, 1e-16)
    alpha = e_exp / e_sum
    return e, alpha


def gat_forward_plain(
    x: np.ndarray,
    edge_index: np.ndarray,
    W: np.ndarray,
    a: np.ndarray,
    negative_slope: float = 0.2,
) -> np.ndarray:
    """
    Single-layer GAT forward pass in plaintext (NumPy).
    
    x: (N, F_in) node features
    edge_index: (2, E) graph structure
    W: (F_out, F_in) linear weight
    a: (2*F_out,) attention vector
    negative_slope: LeakyReLU slope
    
    Returns: (N, F_out) output embeddings
    """
    num_nodes = x.shape[0]
    out_channels = W.shape[0]
    
    # Linear transform
    h = linear_plain(x, W)
    
    # Attention
    _, alpha = attention_plain(edge_index, h, a, num_nodes, negative_slope)
    
    # Aggregate
    out = np.zeros((num_nodes, out_channels), dtype=np.float64)
    row = edge_index[1]
    col = edge_index[0]
    for e in range(len(alpha)):
        i = row[e]
        j = col[e]
        out[i] += alpha[e] * h[j]
    
    return out
