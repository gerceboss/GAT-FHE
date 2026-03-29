"""
CKKS-only GAT encoder for server: encrypted weights, no secret key.
Only from_client_keys_with_encrypted_weights and forward path used in FHE training.
"""

from __future__ import annotations

import gc
from typing import Any, List

import numpy as np

from .fhe_utils_ckks import (
    early_bootstrap_enabled,
    early_bootstrap_threshold_for_path,
    encrypted_reciprocal_newton_raphson,
)


def _try_import_openfhe():
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
        return True, None, s
    except ImportError as e:
        return False, str(e), None


_OPENFHE_AVAILABLE, _OPENFHE_ERROR, _OPENFHE_SYMBOLS = _try_import_openfhe()


class _PublicKeyOnly:
    def __init__(self, public_key: Any) -> None:
        self.publicKey = public_key

    @property
    def secretKey(self) -> None:
        raise RuntimeError("Server does not have secret key.")


class GATEncoderCKKS:
    """CKKS-only GAT encoder: client's crypto context + public key + encrypted weights."""

    @classmethod
    def from_client_keys_with_encrypted_weights(
        cls,
        crypto_context: Any,
        public_key: Any,
        in_channels: int,
        out_channels: int,
        slots: int,
        ct_W_list: List[Any],
        a: np.ndarray,
        negative_slope: float = 0.2,
    ) -> "GATEncoderCKKS":
        if not _OPENFHE_AVAILABLE:
            raise ImportError(f"OpenFHE not available: {_OPENFHE_ERROR}")
        self = cls.__new__(cls)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.negative_slope = negative_slope
        self._slots = slots
        self._cc = crypto_context
        self._keys = _PublicKeyOnly(public_key)
        self._a = np.asarray(a, dtype=np.float64)
        self._ct_W_list = list(ct_W_list)
        return self

    @property
    def crypto_context(self) -> Any:
        return self._cc

    @property
    def slots(self) -> int:
        return self._slots

    @property
    def keys(self) -> Any:
        return self._keys

    def _sum_slots_via_rotations(self, ct: Any, num_slots: int) -> Any:
        acc = ct
        shift = 1
        while shift < num_slots:
            rotated = self._cc.EvalRotate(acc, shift)
            acc = self._cc.EvalAdd(acc, rotated)
            shift *= 2
        return acc

    def _matmul_ckks_dispatch(self, ct_x: Any) -> List[Any]:
        return self._matmul_ckks_encrypted_W(ct_x, self._ct_W_list)

    def _matmul_ckks_encrypted_W(self, ct_x: Any, ct_W_list: List[Any]) -> List[Any]:
        F_out = len(ct_W_list)
        ct_h_list = []
        for k in range(F_out):
            ct_prod = self._cc.EvalMult(ct_x, ct_W_list[k])
            ct_sum = self._sum_slots_via_rotations(ct_prod, self.in_channels)
            del ct_prod
            ct_h_list.append(ct_sum)
        return ct_h_list

    def attention_scores_ckks(
        self,
        ct_h_list: List[List[Any]],
        edge_index: np.ndarray,
        num_nodes: int,
        *,
        training: bool = True,
    ) -> List[Any]:
        E = edge_index.shape[1]
        F_out = len(ct_h_list[0])
        ct_e_list = []
        a_concat = self._a
        _gc_interval = max(1, E // 20)
        _thr = early_bootstrap_threshold_for_path(training)
        for e in range(E):
            i, j = edge_index[0, e], edge_index[1, e]
            ct_h_i_packed = None
            for k in range(F_out):
                if k == 0:
                    ct_h_i_packed = ct_h_list[i][0]
                else:
                    ct_rotated = self._cc.EvalRotate(ct_h_list[i][k], k)
                    ct_h_i_packed = self._cc.EvalAdd(ct_h_i_packed, ct_rotated)
                    del ct_rotated
            ct_h_j_packed = None
            for k in range(F_out):
                ct_rotated = self._cc.EvalRotate(ct_h_list[j][k], F_out + k)
                if ct_h_j_packed is None:
                    ct_h_j_packed = ct_rotated
                else:
                    ct_h_j_packed = self._cc.EvalAdd(ct_h_j_packed, ct_rotated)
                    del ct_rotated
            ct_concat = self._cc.EvalAdd(ct_h_i_packed, ct_h_j_packed)
            del ct_h_i_packed, ct_h_j_packed
            a_padded = np.zeros(self._slots, dtype=np.float64)
            a_padded[: min(2 * F_out, self._slots)] = a_concat[: min(2 * F_out, self._slots)]
            pt_a = self._cc.MakeCKKSPackedPlaintext(a_padded.tolist())
            ct_weighted = self._cc.EvalMult(ct_concat, pt_a)
            del ct_concat, pt_a, a_padded
            ct_e_ij = self._sum_slots_via_rotations(ct_weighted, 2 * F_out)
            del ct_weighted
            if early_bootstrap_enabled():
                try:
                    if int(ct_e_ij.GetLevel()) >= _thr:
                        ct_e_ij = self._cc.EvalBootstrap(ct_e_ij)
                except Exception:
                    pass
            ct_e_list.append(ct_e_ij)
            if (e + 1) % _gc_interval == 0:
                gc.collect()
        return ct_e_list

    def softmax_ckks_chebyshev(
        self,
        ct_e_list: List[Any],
        edge_index: np.ndarray,
        num_nodes: int,
    ) -> List[Any]:
        E = edge_index.shape[1]
        # degree-2 exp approx (1 + x + x^2/2) to save multiplicative depth
        coeffs = [1.0, 1.0, 0.5]
        ct_exp_list = []
        for ct_e in ct_e_list:
            try:
                ct_exp = self._cc.EvalChebyshevSeries(ct_e, coeffs, -1.0, 1.0)
                ct_exp_list.append(ct_exp)
            except Exception:
                ct_exp_list.append(ct_e)
        ct_alpha_list = [None] * E
        for t in range(num_nodes):
            mask = edge_index[1] == t
            edge_indices = np.where(mask)[0]
            if len(edge_indices) == 0:
                continue
            ct_sum = ct_exp_list[edge_indices[0]]
            for idx in edge_indices[1:]:
                ct_sum = self._cc.EvalAdd(ct_sum, ct_exp_list[idx])
            num_incoming = len(edge_indices)
            initial_guess = 1.0 / max(1.0, num_incoming * 0.5)
            # 1 Newton iteration to save depth (2 iters was pushing over 50 levels)
            ct_reciprocal = encrypted_reciprocal_newton_raphson(
                self._cc, ct_sum, num_iterations=1, initial_guess=initial_guess, slots=self._slots
            )
            del ct_sum
            for idx in edge_indices:
                ct_alpha_list[idx] = self._cc.EvalMult(ct_exp_list[idx], ct_reciprocal)
            del ct_reciprocal
            if (t + 1) % 10 == 0:
                gc.collect()
        return ct_alpha_list

    def aggregate_fhe(
        self,
        ct_list: List[Any],
        edge_index: np.ndarray,
        alpha: Any,
        num_nodes: int,
    ) -> List[Any]:
        E = edge_index.shape[1]
        alpha_is_encrypted = isinstance(alpha, list)
        out_cts = []
        for t in range(num_nodes):
            mask = edge_index[1] == t
            edge_indices = np.where(mask)[0]
            if len(edge_indices) == 0:
                zeros = [0.0] * self._slots
                pt_zero = self._cc.MakeCKKSPackedPlaintext(zeros)
                ct_zero = self._cc.Encrypt(self._keys.publicKey, pt_zero)
                out_cts.append(ct_zero)
                continue
            cols_j = edge_index[0, mask]
            acc = None
            for k, edge_idx in enumerate(edge_indices):
                if alpha_is_encrypted:
                    term = self._cc.EvalMult(ct_list[cols_j[k]], alpha[edge_idx])
                else:
                    w = alpha[edge_idx]
                    pt_scale = self._cc.MakeCKKSPackedPlaintext([w] * self._slots)
                    term = self._cc.EvalMult(ct_list[cols_j[k]], pt_scale)
                    del pt_scale
                if acc is None:
                    acc = term
                else:
                    acc = self._cc.EvalAdd(acc, term)
                    del term
            out_cts.append(acc)
            if (t + 1) % 10 == 0:
                gc.collect()
        return out_cts
