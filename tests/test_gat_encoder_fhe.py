"""
Unit tests for FHE GAT encoder.
Tests individual FHE functions and full encrypted pipeline.
Skips tests gracefully if OpenFHE is not installed.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from gat_encoder_fhe import (
    GATEncoderFHE,
    FHEGraph,
    openfhe_available,
    encrypted_reciprocal_newton_raphson,
)


# Skip all tests if OpenFHE not available
pytestmark = pytest.mark.skipif(
    not openfhe_available(),
    reason="OpenFHE not installed (pip install openfhe)"
)


class TestFHEEncryptDecrypt:
    """Test basic encryption/decryption roundtrip."""
    
    def test_encrypt_decrypt_roundtrip(self):
        """Test encrypting and decrypting node features."""
        np.random.seed(42)
        N, F = 4, 3
        x = np.random.randn(N, F).astype(np.float64) * 0.5
        
        encoder = GATEncoderFHE(
            in_channels=F,
            out_channels=F,
            batch_size=4,
            mult_depth=5,
        )
        
        # Encrypt
        ct_list = encoder.encrypt_node_features(x)
        assert len(ct_list) == N
        
        # Decrypt
        x_dec = encoder.decrypt_node_features(ct_list, F)
        
        # Should match within CKKS precision
        np.testing.assert_allclose(x, x_dec, rtol=1e-3, atol=1e-6)


class TestCKKSLinearLayer:
    """Test CKKS linear layer (matmul_ckks)."""
    
    def test_matmul_ckks_shape(self):
        """Test matmul_ckks output shape."""
        np.random.seed(42)
        F_in, F_out = 3, 4
        x = np.random.randn(F_in).astype(np.float64) * 0.5
        W = np.random.randn(F_out, F_in).astype(np.float64) * 0.1
        
        encoder = GATEncoderFHE(
            in_channels=F_in,
            out_channels=F_out,
            batch_size=4,
            mult_depth=8,
        )
        
        # Encrypt single node
        row = np.zeros(4, dtype=np.float64)
        row[:F_in] = x
        pt = encoder.crypto_context.MakeCKKSPackedPlaintext(row.tolist())
        ct_x = encoder.crypto_context.Encrypt(encoder.keys.publicKey, pt)
        
        # CKKS matmul
        ct_h_list = encoder.matmul_ckks(ct_x, W)
        
        assert len(ct_h_list) == F_out, f"Expected {F_out} ciphertexts, got {len(ct_h_list)}"
    
    def test_matmul_ckks_correctness(self):
        """Test matmul_ckks matches plaintext multiplication."""
        np.random.seed(42)
        F_in, F_out = 3, 3
        x = np.random.randn(F_in).astype(np.float64) * 0.5
        W = np.random.randn(F_out, F_in).astype(np.float64) * 0.1
        
        # Plaintext reference
        h_plain = W @ x
        
        # CKKS encrypted
        encoder = GATEncoderFHE(
            in_channels=F_in,
            out_channels=F_out,
            batch_size=4,
            mult_depth=8,
        )
        
        row = np.zeros(4, dtype=np.float64)
        row[:F_in] = x
        pt = encoder.crypto_context.MakeCKKSPackedPlaintext(row.tolist())
        ct_x = encoder.crypto_context.Encrypt(encoder.keys.publicKey, pt)
        
        ct_h_list = encoder.matmul_ckks(ct_x, W)
        
        # Decrypt each output dimension
        h_ckks = np.zeros(F_out, dtype=np.float64)
        for k, ct_h_k in enumerate(ct_h_list):
            pt_dec = encoder.crypto_context.Decrypt(encoder.keys.secretKey, ct_h_k)
            pt_dec.SetLength(1)
            values = pt_dec.GetCKKSPackedValue()
            h_ckks[k] = np.real(complex(values[0]).real)
        
        # Should match within CKKS precision
        np.testing.assert_allclose(h_plain, h_ckks, rtol=1e-3, atol=1e-6)


class TestFHEGraph:
    """Test FHEGraph construction and encryption."""
    
    def test_from_plain_encrypted(self):
        """Test building FHEGraph from plaintext (encrypts immediately)."""
        np.random.seed(42)
        N, F = 4, 3
        x = np.random.randn(N, F).astype(np.float64) * 0.5
        edge_index = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int64)
        
        encoder = GATEncoderFHE(in_channels=F, out_channels=F, batch_size=4, mult_depth=5)
        
        graph = FHEGraph.from_plain_encrypted(
            num_nodes=N,
            in_channels=F,
            edge_index=edge_index,
            node_features_plain=x,
            crypto_context=encoder.crypto_context,
            public_key=encoder.keys.publicKey,
            batch_size=4,
        )
        
        assert graph.num_nodes == N
        assert graph.in_channels == F
        assert len(graph.node_features_enc) == N
        assert graph.is_encrypted is True
    
    def test_graph_no_plaintext_storage(self):
        """Verify FHEGraph does NOT store plaintext features."""
        np.random.seed(42)
        N, F = 3, 2
        x = np.random.randn(N, F).astype(np.float64)
        edge_index = np.array([[0, 1], [1, 2]], dtype=np.int64)
        
        encoder = GATEncoderFHE(in_channels=F, out_channels=F, batch_size=4, mult_depth=5)
        
        graph = FHEGraph.from_plain_encrypted(
            num_nodes=N,
            in_channels=F,
            edge_index=edge_index,
            node_features_plain=x,
            crypto_context=encoder.crypto_context,
            public_key=encoder.keys.publicKey,
            batch_size=4,
        )
        
        # Should NOT have plaintext attribute
        assert not hasattr(graph, 'node_features_plain') or graph.node_features_plain is None


class TestFHEForward:
    """Test full FHE forward pass."""
    
    def test_forward_fhe_full_shape(self):
        """Test forward_fhe_full output shape."""
        np.random.seed(42)
        N, F_in, F_out = 4, 3, 3
        x = np.random.randn(N, F_in).astype(np.float64) * 0.5
        edge_index = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int64)
        
        encoder = GATEncoderFHE(
            in_channels=F_in,
            out_channels=F_out,
            batch_size=4,
            mult_depth=12,
            use_cggi=False,
        )
        
        graph = FHEGraph.from_plain_encrypted(
            num_nodes=N,
            in_channels=F_in,
            edge_index=edge_index,
            node_features_plain=x,
            crypto_context=encoder.crypto_context,
            public_key=encoder.keys.publicKey,
            batch_size=4,
        )
        
        output = encoder.forward_fhe_full(graph)
        
        assert output.shape == (N, F_out), f"Expected ({N}, {F_out}), got {output.shape}"
        assert np.isfinite(output).all(), "Output contains NaN or Inf"
    
    def test_forward_fhe_vs_plain(self):
        """Test FHE output is close to plaintext output."""
        from examples.test_graph import get_test_graph
        from gat_encoder import gat_forward_plain
        
        x, edge_index, N, F_in, F_out = get_test_graph()
        
        np.random.seed(42)
        encoder = GATEncoderFHE(
            in_channels=F_in,
            out_channels=F_out,
            batch_size=4,
            mult_depth=12,
            use_cggi=False,
        )
        
        # Set same weights
        encoder._W = np.random.randn(F_out, F_in).astype(np.float64) * 0.1
        encoder._a = np.random.randn(2 * F_out).astype(np.float64) * 0.1
        
        # Plaintext reference
        out_plain = gat_forward_plain(x, edge_index, encoder._W, encoder._a, 0.2)
        
        # FHE encrypted
        graph = FHEGraph.from_plain_encrypted(
            num_nodes=N,
            in_channels=F_in,
            edge_index=edge_index,
            node_features_plain=x,
            crypto_context=encoder.crypto_context,
            public_key=encoder.keys.publicKey,
            batch_size=4,
        )
        
        out_fhe = encoder.forward_fhe_full(graph)
        
        # Should match within CKKS precision (allow larger tolerance due to approximations)
        diff = np.abs(out_fhe - out_plain)
        assert diff.max() < 1.0, f"Max difference {diff.max()} too large"
        assert diff.mean() < 0.3, f"Mean difference {diff.mean()} too large"


class TestHomomorphicDivision:
    """Test encrypted division utilities."""
    
    def test_newton_raphson_reciprocal(self):
        """Test Newton-Raphson reciprocal approximation."""
        np.random.seed(42)
        
        encoder = GATEncoderFHE(
            in_channels=2,
            out_channels=2,
            batch_size=4,
            mult_depth=15,  # Need depth for iterations
        )
        
        # Test value
        d = 2.0
        vec = [d, 0, 0, 0]
        pt_d = encoder.crypto_context.MakeCKKSPackedPlaintext(vec)
        ct_d = encoder.crypto_context.Encrypt(encoder.keys.publicKey, pt_d)
        
        # Compute 1/d
        ct_recip = encrypted_reciprocal_newton_raphson(
            encoder.crypto_context,
            ct_d,
            num_iterations=2,
            initial_guess=0.5,  # Close to 1/2
            batch_size=4,
        )
        
        # Decrypt
        pt_recip = encoder.crypto_context.Decrypt(encoder.keys.secretKey, ct_recip)
        pt_recip.SetLength(1)
        values = pt_recip.GetCKKSPackedValue()
        recip = np.real(complex(values[0]).real)
        
        # Should be close to 0.5
        expected = 1.0 / d
        assert abs(recip - expected) < 0.1, f"Expected {expected}, got {recip}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
