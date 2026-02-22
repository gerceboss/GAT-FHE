"""
Client-side key architecture for CKKS-only GAT inference.

Flow:
1. Client generates CKKS keypair and evaluation keys
2. Client encrypts node features with public key
3. Client sends: crypto context (with eval keys), public key, encrypted inputs to server
4. Server runs full GAT pipeline in CKKS (no decryption - server never has secret key)
5. Server returns encrypted result
6. Client decrypts with secret key

For same-process / programmatic use: client creates keys, passes (cc, publicKey) by reference.
For cross-process: serialization of cc, publicKey, ciphertexts would be needed (OpenFHE serialization).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .encoder import GATEncoderFHE
from .fhe_graph import FHEGraph


@dataclass
class ClientKeysContext:
    """
    Client-owned keypair and crypto context.
    Client keeps this; sends only cc + publicKey to server.
    """

    crypto_context: Any
    keys: Any  # OpenFHE KeyPair (has secretKey - CLIENT ONLY)
    batch_size: int

    def encrypt_node_features(self, x: np.ndarray, in_channels: int) -> list[Any]:
        """Encrypt (N, in_channels) plaintext to N ciphertexts."""
        cc = self.crypto_context
        pk = self.keys.publicKey
        N, F = x.shape
        if F != in_channels:
            raise ValueError(f"x columns {F} != in_channels {in_channels}")
        ct_list = []
        for i in range(N):
            row = np.zeros(self.batch_size, dtype=np.float64)
            row[:in_channels] = x[i]
            pt = cc.MakeCKKSPackedPlaintext(row.tolist())
            ct_list.append(cc.Encrypt(pk, pt))
        return ct_list

    def encrypt_weight_matrix(self, W: np.ndarray, in_channels: int) -> list[Any]:
        """Encrypt (F_out, F_in) weight matrix to list of F_out ciphertexts (one per row)."""
        cc = self.crypto_context
        pk = self.keys.publicKey
        F_out, F_in = W.shape
        if F_in != in_channels:
            raise ValueError(f"W columns {F_in} != in_channels {in_channels}")
        ct_list = []
        for k in range(F_out):
            row = np.zeros(self.batch_size, dtype=np.float64)
            row[:in_channels] = W[k]
            pt = cc.MakeCKKSPackedPlaintext(row.tolist())
            ct_list.append(cc.Encrypt(pk, pt))
        return ct_list

    def decrypt_node_features(self, ct_list: list[Any], feature_dim: int) -> np.ndarray:
        """Decrypt list of ciphertexts to (N, feature_dim) plaintext."""
        cc = self.crypto_context
        sk = self.keys.secretKey
        N = len(ct_list)
        x = np.zeros((N, feature_dim), dtype=np.float64)
        for i, ct in enumerate(ct_list):
            pt = cc.Decrypt(sk, ct)
            pt.SetLength(feature_dim)
            vals = pt.GetCKKSPackedValue()
            x[i] = np.real([complex(v).real for v in vals[:feature_dim]])
        return x


class _PublicKeyOnly:
    """Wrapper exposing only publicKey; raises if secretKey is accessed."""

    def __init__(self, public_key: Any) -> None:
        self.publicKey = public_key

    @property
    def secretKey(self) -> None:
        raise RuntimeError(
            "Server does not have secret key (client-key mode). "
            "Decryption must be done by the client."
        )


def create_client_context(
    in_channels: int,
    out_channels: int,
    batch_size: int = 32,
    mult_depth: int = 25,
    scale_mod_size: int = 40,
) -> ClientKeysContext:
    """
    Create client-side crypto context and keys.
    Client runs this, keeps the result, and sends cc + keys.publicKey to server.
    """
    from .encoder import openfhe_available

    if not openfhe_available():
        raise ImportError("OpenFHE not available")

    from .encoder import _OPENFHE_SYMBOLS

    CC = _OPENFHE_SYMBOLS.CCParamsCKKSRNS
    Gen = _OPENFHE_SYMBOLS.GenCryptoContext
    PKEScheme = _OPENFHE_SYMBOLS.PKESchemeFeature
    SecretDist = _OPENFHE_SYMBOLS.SecretKeyDist
    KeySwitch = _OPENFHE_SYMBOLS.KeySwitchTechnique
    Scaling = _OPENFHE_SYMBOLS.ScalingTechnique
    SecurityLevel = _OPENFHE_SYMBOLS.HEStd_128_classic

    params = CC()
    params.SetMultiplicativeDepth(mult_depth)
    params.SetScalingModSize(scale_mod_size)
    params.SetFirstModSize(60)
    params.SetBatchSize(batch_size)
    params.SetSecretKeyDist(SecretDist.UNIFORM_TERNARY)
    params.SetSecurityLevel(SecurityLevel)
    params.SetScalingTechnique(Scaling.FLEXIBLEAUTO)
    params.SetKeySwitchTechnique(KeySwitch.HYBRID)

    cc = Gen(params)
    cc.Enable(PKEScheme.PKE)
    cc.Enable(PKEScheme.KEYSWITCH)
    cc.Enable(PKEScheme.LEVELEDSHE)
    cc.Enable(PKEScheme.ADVANCEDSHE)

    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey)
    max_rot = min(2 * max(in_channels, out_channels), 2 * batch_size)
    rotation_indices = list(range(1, max_rot + 1))
    cc.EvalRotateKeyGen(keys.secretKey, rotation_indices)

    return ClientKeysContext(crypto_context=cc, keys=keys, batch_size=batch_size)

