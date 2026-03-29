"""
CKKS-only client keys: keygen, encrypt, decrypt. Client keeps secret key.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# Client-side EvalBootstrap retries per output row after CKKS Decode() failure.
_INFER_DECRYPT_MAX_BOOTSTRAP_ROUNDS = 16


def _is_ckks_decode_error(exc: BaseException) -> bool:
    s = str(exc).lower()
    return "approximation error" in s or "decode()" in s or "decode:" in s


def decrypt_node_features_rows_with_retry(
    crypto_context: Any,
    secret_key: Any,
    ct_list: list[Any],
    feature_dim: int,
) -> np.ndarray:
    """
    Decrypt packed CKKS ciphertexts (one row per ct). On Decode() failure,
    call EvalBootstrap on that row and retry up to _INFER_DECRYPT_MAX_BOOTSTRAP_ROUNDS times.
    """
    max_bootstrap_rounds = _INFER_DECRYPT_MAX_BOOTSTRAP_ROUNDS
    cc = crypto_context
    sk = secret_key
    N = len(ct_list)
    x = np.zeros((N, feature_dim), dtype=np.float64)
    for i in range(N):
        ct = ct_list[i]
        last_err: BaseException | None = None
        for _round in range(max_bootstrap_rounds + 1):
            try:
                pt = cc.Decrypt(sk, ct)
                pt.SetLength(feature_dim)
                vals = pt.GetCKKSPackedValue()
                x[i] = np.real([complex(v).real for v in vals[:feature_dim]])
                last_err = None
                break
            except RuntimeError as e:
                last_err = e
                if not _is_ckks_decode_error(e):
                    raise
                if _round >= max_bootstrap_rounds:
                    break
                ct = cc.EvalBootstrap(ct)
        if last_err is not None:
            raise last_err
    return x


def _openfhe_symbols():
    try:
        import openfhe
        s = type("OpenFHESymbols", (), {})()
        s.CCParamsCKKSRNS = openfhe.CCParamsCKKSRNS
        s.GenCryptoContext = openfhe.GenCryptoContext
        s.PKESchemeFeature = openfhe.PKESchemeFeature
        s.SecretKeyDist = openfhe.SecretKeyDist
        s.KeySwitchTechnique = openfhe.KeySwitchTechnique
        s.ScalingTechnique = openfhe.ScalingTechnique
        s.HEStd_128_classic = openfhe.HEStd_128_classic
        s.openfhe = openfhe
        return s
    except ImportError:
        return None


def openfhe_available() -> bool:
    return _openfhe_symbols() is not None


@dataclass
class ClientKeysContext:
    """Client-owned keypair and crypto context. Only client has secret key."""
    crypto_context: Any
    keys: Any
    slots: int

    def encrypt_node_features(self, x: np.ndarray, in_channels: int) -> list[Any]:
        cc = self.crypto_context
        pk = self.keys.publicKey
        N, F = x.shape
        if F != in_channels:
            raise ValueError(f"x columns {F} != in_channels {in_channels}")
        ct_list = []
        for i in range(N):
            row = np.zeros(self.slots, dtype=np.float64)
            row[:in_channels] = x[i]
            pt = cc.MakeCKKSPackedPlaintext(row.tolist())
            ct_list.append(cc.Encrypt(pk, pt))
        return ct_list

    def encrypt_weight_matrix(self, W: np.ndarray, in_channels: int) -> list[Any]:
        cc = self.crypto_context
        pk = self.keys.publicKey
        F_out, F_in = W.shape
        if F_in != in_channels:
            raise ValueError(f"W columns {F_in} != in_channels {in_channels}")
        ct_list = []
        for k in range(F_out):
            row = np.zeros(self.slots, dtype=np.float64)
            row[:in_channels] = W[k]
            pt = cc.MakeCKKSPackedPlaintext(row.tolist())
            ct_list.append(cc.Encrypt(pk, pt))
        return ct_list

    def decrypt_node_features(self, ct_list: list[Any], feature_dim: int) -> np.ndarray:
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

    def decrypt_node_features_with_bootstrap_retry(
        self,
        ct_list: list[Any],
        feature_dim: int,
    ) -> np.ndarray:
        return decrypt_node_features_rows_with_retry(
            self.crypto_context,
            self.keys.secretKey,
            ct_list,
            feature_dim,
        )

    def decrypt_weight_matrix(self, ct_list: list[Any], in_channels: int) -> np.ndarray:
        cc = self.crypto_context
        sk = self.keys.secretKey
        N = len(ct_list)
        W = np.zeros((N, in_channels), dtype=np.float64)
        for i, ct in enumerate(ct_list):
            pt = cc.Decrypt(sk, ct)
            pt.SetLength(in_channels)
            vals = pt.GetCKKSPackedValue()
            W[i] = np.real([complex(v).real for v in vals[:in_channels]])
        return W

def create_client_context(
    in_channels: int,
    out_channels: int,
    slots: int = 32,
    mult_depth: int = 25,
    scale_mod_size: int = 50,
    ring_dim: int = 16384,
    bootstrap: bool = True,
) -> ClientKeysContext:
    """
    Create client CKKS context and keys. With bootstrap=True (default), we use a short
    modulus chain (depth 20-25 for one epoch) and bootstrap weights after each epoch.
    Recommended: RingDim 16384, ScalingModSize 50, FirstModSize 60, HEStd_128_classic.
    """
    sym = _openfhe_symbols()
    if sym is None:
        raise ImportError("OpenFHE not available")
    openfhe = sym.openfhe
    CC = sym.CCParamsCKKSRNS
    Gen = sym.GenCryptoContext
    PKEScheme = sym.PKESchemeFeature
    SecretDist = sym.SecretKeyDist
    KeySwitch = sym.KeySwitchTechnique
    Scaling = sym.ScalingTechnique
    # HEStd_128_classic requires ring dimension 131072; for smaller ring (e.g. 16384) use HEStd_NotSet
    RING_128_STD = 131072
    HEStd_NotSet = getattr(openfhe, "HEStd_NotSet", None)
    if ring_dim >= RING_128_STD:
        params = CC()
        params.SetSecurityLevel(sym.HEStd_128_classic)
        params.SetRingDim(ring_dim)
    elif HEStd_NotSet is not None:
        params = CC()
        params.SetRingDim(ring_dim)
        params.SetSecurityLevel(HEStd_NotSet)
    else:
        raise RuntimeError(
            "OpenFHE build does not expose HEStd_NotSet; use --ring_dim 131072 for 128-bit standard (more RAM)"
        )

    params.SetScalingModSize(scale_mod_size)
    # OpenFHE requires FirstModSize >= ScalingModSize.
    # Keep a small cushion so users can raise --scale_mod_size safely.
    params.SetFirstModSize(max(60, int(scale_mod_size)))
    params.SetBatchSize(slots)
    params.SetSecretKeyDist(SecretDist.UNIFORM_TERNARY)
    params.SetScalingTechnique(Scaling.FLEXIBLEAUTO)
    params.SetKeySwitchTechnique(KeySwitch.HYBRID)

    if bootstrap:
        # When bootstrapping is enabled we must allocate enough depth for:
        # - the intended compute (mult_depth)
        # - bootstrap overhead (depends on level_budget and secret distribution)
        #
        # So the *actual* CKKS chain depth is mult_depth + bootstrap_depth.
        level_budget = [4, 4]
        try:
            bootstrap_depth = openfhe.FHECKKSRNS.GetBootstrapDepth(level_budget, SecretDist.UNIFORM_TERNARY)
        except Exception:
            bootstrap_depth = 10
        total_depth = int(mult_depth) + int(bootstrap_depth)
        params.SetMultiplicativeDepth(total_depth)
    else:
        params.SetMultiplicativeDepth(mult_depth)

    cc = Gen(params)
    cc.Enable(PKEScheme.PKE)
    cc.Enable(PKEScheme.KEYSWITCH)
    cc.Enable(PKEScheme.LEVELEDSHE)
    cc.Enable(PKEScheme.ADVANCEDSHE)
    if bootstrap:
        cc.Enable(PKEScheme.FHE)

    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey)
    max_rot = min(2 * max(in_channels, out_channels), 2 * slots)
    rotation_indices = list(range(1, max_rot + 1))
    cc.EvalRotateKeyGen(keys.secretKey, rotation_indices)

    if bootstrap:
        # Use slots so precomputations match EvalBootstrapKeyGen (else "Precomputations for N slots not found")
        cc.EvalBootstrapSetup(levelBudget=level_budget, slots=slots)
        cc.EvalBootstrapKeyGen(keys.secretKey, slots)

    return ClientKeysContext(crypto_context=cc, keys=keys, slots=slots)
