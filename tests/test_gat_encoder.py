"""Unit tests for plaintext GAT encoder."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from gat_encoder import matmul_plain, linear_plain, attention_plain, gat_forward_plain


class TestMatmulPlain:
    def test_matmul_shape(self):
        x = np.random.randn(5, 3).astype(np.float64)
        W = np.random.randn(4, 3).astype(np.float64)
        result = matmul_plain(x, W)
        assert result.shape == (5, 4)
    
    def test_matmul_values(self):
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        W = np.array([[1.0, 0.0], [0.0, 1.0]])
        result = matmul_plain(x, W)
        expected = np.array([[1.0, 2.0], [3.0, 4.0]])
        np.testing.assert_allclose(result, expected, rtol=1e-10)


class TestLinearPlain:
    def test_linear_shape(self):
        x = np.random.randn(6, 4).astype(np.float64)
        W = np.random.randn(8, 4).astype(np.float64)
        result = linear_plain(x, W)
        assert result.shape == (6, 8)


class TestAttentionPlain:
    def test_attention_shape(self):
        edge_index = np.array([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=np.int64)
        h = np.random.randn(4, 3).astype(np.float64)
        a = np.random.randn(6).astype(np.float64)
        e, alpha = attention_plain(edge_index, h, a, num_nodes=4, negative_slope=0.2)
        assert e.shape == (6,)
        assert alpha.shape == (6,)
    
    def test_attention_normalization(self):
        edge_index = np.array([[0, 1, 2], [0, 0, 1]], dtype=np.int64)
        h = np.random.randn(3, 2).astype(np.float64)
        a = np.random.randn(4).astype(np.float64)
        _, alpha = attention_plain(edge_index, h, a, num_nodes=3, negative_slope=0.2)
        alpha_node0 = alpha[[0, 1]]
        assert np.abs(alpha_node0.sum() - 1.0) < 1e-6


class TestGATForwardPlain:
    def test_forward_shape(self):
        np.random.seed(42)
        N, F_in, F_out = 5, 3, 4
        x = np.random.randn(N, F_in).astype(np.float64)
        edge_index = np.array([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int64)
        W = np.random.randn(F_out, F_in).astype(np.float64) * 0.1
        a = np.random.randn(2 * F_out).astype(np.float64) * 0.1
        output = gat_forward_plain(x, edge_index, W, a, negative_slope=0.2)
        assert output.shape == (N, F_out)
    
    def test_forward_finite(self):
        np.random.seed(42)
        N = 6
        x = np.random.randn(N, 4).astype(np.float64) * 0.5
        edge_index = np.array([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=np.int64)
        W = np.random.randn(4, 4).astype(np.float64) * 0.1
        a = np.random.randn(8).astype(np.float64) * 0.1
        output = gat_forward_plain(x, edge_index, W, a)
        assert np.isfinite(output).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
