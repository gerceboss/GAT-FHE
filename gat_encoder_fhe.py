"""
Single-layer GAT encoder under FHE: CKKS for arithmetic, CGGI (BinFHE) for boolean if-else.
Graph structure and attention weights are plaintext; node features can be encrypted.
- CKKS: linear, aggregate (weighted sums).
- CGGI: LeakyReLU branch (sign bit) and other boolean ops; see cggi_helpers.py and OpenFHE binfhe examples.
"""

from __future__ import annotations

import math
from typing import Any, List, Optional, Tuple

import numpy as np

from fhe_graph import FHEGraph

# Optional CGGI for boolean branches (LeakyReLU sign)
try:
    from cggi_helpers import CGGIContext, leaky_relu_with_cggi, cggi_available
except ImportError:
    CGGIContext = None  # type: ignore
    leaky_relu_with_cggi = None  # type: ignore
    def cggi_available() -> bool:
        return False


def _try_import_openfhe() -> Tuple[bool, Optional[str], Any]:
    """
    Attempt to import OpenFHE Python.

    Returns: (ok, error_message, openfhe_module_or_None)
    Notes:
    - It's common for `pip install openfhe` to succeed but `import openfhe` to fail
      on unsupported Python versions (e.g. missing compiled extension).
    """
    try:
        import openfhe

        return True, None, openfhe
    except Exception as e:  # ImportError, ModuleNotFoundError, etc.
        return False, f"{type(e).__name__}: {e}", None


def _leaky_relu(x: float, negative_slope: float = 0.2) -> float:
    return x if x >= 0 else negative_slope * x


class GATEncoderFHE:
    """
    Single-head GAT layer with FHE support.
    - CKKS: linear (plaintext), aggregate out_i = sum_j alpha_ij * h'_j (ciphertext).
    - CGGI: LeakyReLU branch (encrypted sign bit) when cggi_context is provided.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        negative_slope: float = 0.2,
        batch_size: Optional[int] = None,
        mult_depth: int = 2,
        scale_mod_size: int = 50,
        use_cggi: bool = True,
        cggi_context: Optional[Any] = None,
    ):
        ok, err, openfhe = _try_import_openfhe()
        if not ok:
            raise RuntimeError(
                "OpenFHE Python is not usable in this environment.\n"
                f"Import error: {err}\n"
                "This usually means your Python version does not match the available OpenFHE wheels.\n"
                "Try Python 3.10–3.12 on Ubuntu 22.04/24.04, then reinstall openfhe."
            )
        # Import symbols the same way as official examples (they rely on openfhe exporting these names).
        from openfhe import (  # type: ignore
            CCParamsCKKSRNS,
            GenCryptoContext,
            PKESchemeFeature,
            SecurityLevel,
        )

        self._CCParamsCKKSRNS = CCParamsCKKSRNS
        self._GenCryptoContext = GenCryptoContext
        self._PKESchemeFeature = PKESchemeFeature
        self._SecurityLevel = SecurityLevel
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.negative_slope = negative_slope
        # CKKS slot count must be >= out_channels (one ciphertext per node, slots = out_channels)
        self._batch_size = batch_size or max(out_channels, 8)
        self._mult_depth = mult_depth
        self._scale_mod_size = scale_mod_size

        self._cc: Any = None
        self._keys: Any = None
        self._W: np.ndarray = np.zeros((out_channels, in_channels), dtype=np.float64)
        self._a: np.ndarray = np.zeros(2 * out_channels, dtype=np.float64)
        self._use_cggi = use_cggi and cggi_available() and (CGGIContext is not None)
        self._cggi: Optional[Any] = cggi_context
        if self._use_cggi and self._cggi is None:
            self._cggi = CGGIContext(bootstrapping="GINX")
        self._build_context()

    def _build_context(self) -> None:
        parameters = self._CCParamsCKKSRNS()
        parameters.SetSecurityLevel(self._SecurityLevel.HEStd_NotSet)
        parameters.SetRingDim(1 << 12)
        parameters.SetMultiplicativeDepth(self._mult_depth)
        parameters.SetScalingModSize(self._scale_mod_size)
        parameters.SetBatchSize(self._batch_size)

        self._cc = self._GenCryptoContext(parameters)
        self._cc.Enable(self._PKESchemeFeature.PKE)
        self._cc.Enable(self._PKESchemeFeature.KEYSWITCH)
        self._cc.Enable(self._PKESchemeFeature.LEVELEDSHE)

        self._keys = self._cc.KeyGen()
        self._cc.EvalMultKeyGen(self._keys.secretKey)

        # Initialize W and a (Xavier-like)
        scale_w = math.sqrt(2.0 / (self.in_channels + self.out_channels))
        scale_a = 0.1
        self._W = np.random.randn(self.out_channels, self.in_channels).astype(np.float64) * scale_w
        self._a = np.random.randn(2 * self.out_channels).astype(np.float64) * scale_a

    @property
    def crypto_context(self) -> Any:
        return self._cc

    @property
    def keys(self) -> Any:
        return self._keys

    def set_weights(self, W: np.ndarray, a: np.ndarray) -> None:
        """Set linear weight W (out_channels, in_channels) and attention vector a (2*out_channels)."""
        W = np.asarray(W, dtype=np.float64)
        a = np.asarray(a, dtype=np.float64)
        if W.shape != (self.out_channels, self.in_channels):
            raise ValueError(f"W must have shape ({self.out_channels}, {self.in_channels})")
        if a.shape != (2 * self.out_channels,):
            raise ValueError(f"a must have shape ({2 * self.out_channels},)")
        self._W = W
        self._a = a

    def _linear_plain(self, x: np.ndarray) -> np.ndarray:
        """(N, F_in) -> (N, F_out)."""
        return x @ self._W.T

    def _attention_plain(
        self,
        edge_index: np.ndarray,
        h: np.ndarray,
        num_nodes: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute attention coefficients alpha over edges.
        row = edge_index[1] (target), col = edge_index[0] (source).
        Returns (e_scores, alpha) both shape (E,).
        """
        row = edge_index[1]
        col = edge_index[0]
        h_row = h[row]   # (E, F_out)
        h_col = h[col]   # (E, F_out)
        h_cat = np.concatenate([h_row, h_col], axis=1)  # (E, 2*F_out)
        e_raw = h_cat @ self._a
        # LeakyReLU: use CGGI for the if-else (sign) when available
        if self._use_cggi and self._cggi is not None and leaky_relu_with_cggi is not None:
            e = np.array(
                leaky_relu_with_cggi(self._cggi, e_raw.tolist(), self.negative_slope),
                dtype=np.float64,
            )
        else:
            e = np.where(e_raw >= 0, e_raw, self.negative_slope * e_raw)

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

    def encrypt_node_features(self, H: np.ndarray) -> List[Any]:
        """
        Encrypt node features H (N, F_out). Returns list of N ciphertexts.
        Each ciphertext encodes one row; batch size must be >= F_out.
        """
        N = H.shape[0]
        F = H.shape[1]
        if F > self._batch_size:
            raise ValueError(f"Feature dim {F} > batch_size {self._batch_size}")
        cts = []
        for i in range(N):
            row = np.zeros(self._batch_size, dtype=np.float64)
            row[:F] = H[i]
            pt = self._cc.MakeCKKSPackedPlaintext(row.tolist())
            ct = self._cc.Encrypt(self._keys.publicKey, pt)
            cts.append(ct)
        return cts

    def decrypt_node_features(self, cts: List[Any], length: int) -> np.ndarray:
        """Decrypt list of ciphertexts to (N, length) array."""
        out = []
        for ct in cts:
            pt = self._cc.Decrypt(ct, self._keys.secretKey)
            pt.SetLength(length)
            vals = pt.GetCKKSPackedValue()
            # CKKS may return complex values with tiny imaginary parts; keep real part.
            arr = np.asarray(vals[:length])
            if np.iscomplexobj(arr):
                arr = np.real(arr)
            out.append(arr.astype(np.float64, copy=False))
        return np.stack(out, axis=0)

    def aggregate_fhe(
        self,
        ct_list: List[Any],
        edge_index: np.ndarray,
        alpha: np.ndarray,
        num_nodes: int,
    ) -> List[Any]:
        """
        Homomorphic aggregation: for each node i, out_i = sum_{j in N(i)} alpha_ij * ct_j.
        ct_list[j] = ciphertext for node j (slots = out_channels).
        edge_index (2, E), alpha (E,) with alpha_ij at edge (col->row).
        """
        row = edge_index[1]
        col = edge_index[0]
        out_cts = []
        for i in range(num_nodes):
            mask = row == i
            cols_j = col[mask]
            alphas = alpha[mask]
            if len(cols_j) == 0:
                # Isolated node: zero ciphertext
                zero = np.zeros(self._batch_size, dtype=np.float64)
                pt0 = self._cc.MakeCKKSPackedPlaintext(zero.tolist())
                ct0 = self._cc.Encrypt(self._keys.publicKey, pt0)
                out_cts.append(ct0)
                continue
            # Slot-wise scalar: plaintext (alpha, alpha, ...) so EvalMult(ct, pt) = alpha * ct
            scale = np.full(self._batch_size, alphas[0], dtype=np.float64)
            pt_scale = self._cc.MakeCKKSPackedPlaintext(scale.tolist())
            acc = self._cc.EvalMult(ct_list[cols_j[0]], pt_scale)
            for k in range(1, len(cols_j)):
                scale = np.full(self._batch_size, alphas[k], dtype=np.float64)
                pt_scale = self._cc.MakeCKKSPackedPlaintext(scale.tolist())
                term = self._cc.EvalMult(ct_list[cols_j[k]], pt_scale)
                acc = self._cc.EvalAdd(acc, term)
            out_cts.append(acc)
        return out_cts

    def forward_plain(self, graph: FHEGraph) -> np.ndarray:
        """Full forward in plaintext (no encryption). For testing and comparison."""
        x = graph.to_plain()
        h = self._linear_plain(x)
        _, alpha = self._attention_plain(graph.edge_index, h, graph.num_nodes)
        # Aggregate in plaintext
        out = np.zeros((graph.num_nodes, self.out_channels), dtype=np.float64)
        row = graph.edge_index[1]
        col = graph.edge_index[0]
        for e in range(len(alpha)):
            i = row[e]
            j = col[e]
            out[i] += alpha[e] * h[j]
        return out

    def forward_fhe(self, graph: FHEGraph) -> np.ndarray:
        """
        Forward with FHE aggregation: linear and attention in plaintext,
        then encrypt h', aggregate in CKKS, decrypt.
        """
        x = graph.to_plain()
        h = self._linear_plain(x)
        _, alpha = self._attention_plain(graph.edge_index, h, graph.num_nodes)
        ct_list = self.encrypt_node_features(h)
        out_cts = self.aggregate_fhe(
            ct_list, graph.edge_index, alpha, graph.num_nodes
        )
        return self.decrypt_node_features(out_cts, self.out_channels)


def openfhe_available() -> bool:
    """Return True if OpenFHE Python is installed and usable."""
    ok, _, _ = _try_import_openfhe()
    return ok


def openfhe_import_error() -> Optional[str]:
    """If OpenFHE import fails, return the error string (else None)."""
    ok, err, _ = _try_import_openfhe()
    return None if ok else err
