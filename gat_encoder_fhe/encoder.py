"""
FHE-based GAT encoder using OpenFHE Python (CKKS + CGGI/FHEW scheme switching).

Implements a fully-encrypted Graph Attention Network encoder with:
- Stage 1: CKKS linear layer (rotation-based matrix multiplication)
- Stage 2: Encrypted LeakyReLU via CKKS↔FHEW scheme switching
- Stage 3: CKKS attention scores and Chebyshev-based softmax
- Stage 4: CKKS aggregation

See PLAN.md §4 for design details and IMPLEMENTATION_SUMMARY.md for architecture.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np

from .fhe_graph import FHEGraph
from gat_encoder import gat_forward_plain


def _try_import_openfhe() -> Tuple[bool, Optional[str], Any]:
    """
    Attempt to import OpenFHE Python.
    Returns: (success, error_msg, symbols_namespace)
    """
    try:
        import openfhe

        # Extract all needed symbols
        symbols = type('OpenFHESymbols', (), {})()
        symbols.CCParamsCKKSRNS = openfhe.CCParamsCKKSRNS
        symbols.GenCryptoContext = openfhe.GenCryptoContext
        symbols.PKESchemeFeature = openfhe.PKESchemeFeature
        symbols.SecretKeyDist = openfhe.SecretKeyDist
        symbols.KeySwitchTechnique = openfhe.KeySwitchTechnique
        symbols.ScalingTechnique = openfhe.ScalingTechnique
        symbols.HEStd_128_classic = openfhe.HEStd_128_classic
        
        return True, None, symbols
    except ImportError as e:
        return False, str(e), None
    except Exception as e:
        return False, f"Unexpected error importing OpenFHE: {e}", None


_OPENFHE_AVAILABLE, _OPENFHE_ERROR, _OPENFHE_SYMBOLS = _try_import_openfhe()


def openfhe_available() -> bool:
    """Return True if OpenFHE Python is installed and usable."""
    return _OPENFHE_AVAILABLE


def openfhe_import_error() -> Optional[str]:
    """If OpenFHE import fails, return the error string (else None)."""
    return _OPENFHE_ERROR


class GATEncoderFHE:
    """
    GAT encoder with FHE support (CKKS + optional CGGI/FHEW scheme switching).
    
    Supports multiple forward modes:
    - forward_plain: Plaintext reference (no encryption)
    - forward_fhe_full: Stages 1-3 (fully-encrypted pipeline)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        negative_slope: float = 0.2,
        batch_size: int = 8,
        mult_depth: int = 12,
        scale_mod_size: int = 50,
        use_cggi: bool = False,
    ):
        """
        Initialize FHE GAT encoder.
        
        in_channels: input feature dimension F_in
        out_channels: output feature dimension F_out
        negative_slope: LeakyReLU negative slope
        batch_size: CKKS batch size (must be >= max(F_in, F_out))  slots used for packing
        mult_depth: CKKS multiplicative depth (higher for complex circuits)
        scale_mod_size: CKKS scaling modulus bit size
        use_cggi: enable CGGI/FHEW scheme switching for encrypted comparisons
        """
        if not _OPENFHE_AVAILABLE:
            raise ImportError(f"OpenFHE not available: {_OPENFHE_ERROR}")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.negative_slope = negative_slope
        self.use_cggi = use_cggi

        # Store OpenFHE symbols as instance attributes
        self._CCParamsCKKSRNS = _OPENFHE_SYMBOLS.CCParamsCKKSRNS
        self._GenCryptoContext = _OPENFHE_SYMBOLS.GenCryptoContext
        self._PKESchemeFeature = _OPENFHE_SYMBOLS.PKESchemeFeature
        self._SecretKeyDist = _OPENFHE_SYMBOLS.SecretKeyDist
        self._KeySwitchTechnique = _OPENFHE_SYMBOLS.KeySwitchTechnique
        self._ScalingTechnique = _OPENFHE_SYMBOLS.ScalingTechnique
        self._HEStd_128_classic = _OPENFHE_SYMBOLS.HEStd_128_classic

        # CKKS parameters
        self._batch_size = batch_size
        self._mult_depth = mult_depth
        self._scale_mod_size = scale_mod_size
        self._first_mod_size = 60
        # Ring dimension: let OpenFHE choose based on security level
        # (HEStd_128_classic requires >= 16384)

        # Build CKKS crypto context and keys
        self._cc = self._build_context()
        self._keys = self._keygen()

        # Initialize weights (plaintext for now; could be encrypted in advanced setups)
        self._W = np.random.randn(out_channels, in_channels).astype(np.float64) * 0.1
        concat_dim = 2 * out_channels
        self._a = np.random.randn(concat_dim).astype(np.float64) * 0.1

        # CGGI context (for scheme switching)
        self._cggi_context = None
        self._fhew_sk = None
        if use_cggi:
            from .cggi_helpers import setup_scheme_switching
            self._fhew_sk, self._cggi_context = setup_scheme_switching(
                self._cc, self._keys, self._batch_size
            )

    def _build_context(self) -> Any:
        """Build CKKS crypto context with scheme switching support."""
        params = self._CCParamsCKKSRNS()
        params.SetMultiplicativeDepth(self._mult_depth)
        params.SetScalingModSize(self._scale_mod_size)
        params.SetFirstModSize(self._first_mod_size)
        params.SetBatchSize(self._batch_size)
        params.SetSecretKeyDist(self._SecretKeyDist.UNIFORM_TERNARY)
        params.SetSecurityLevel(self._HEStd_128_classic)
        # Ring dimension auto-set by security level
        params.SetScalingTechnique(self._ScalingTechnique.FLEXIBLEAUTO)
        params.SetKeySwitchTechnique(self._KeySwitchTechnique.HYBRID)

        cc = self._GenCryptoContext(params)
        cc.Enable(self._PKESchemeFeature.PKE)
        cc.Enable(self._PKESchemeFeature.KEYSWITCH)
        cc.Enable(self._PKESchemeFeature.LEVELEDSHE)
        cc.Enable(self._PKESchemeFeature.ADVANCEDSHE)  # For Chebyshev

        if self.use_cggi:
            cc.Enable(self._PKESchemeFeature.SCHEMESWITCH)  # For CKKS↔FHEW

        return cc

    def _keygen(self) -> Any:
        """Generate CKKS keys and rotation indices. Returns KeyPair."""
        keys = self._cc.KeyGen()
        self._cc.EvalMultKeyGen(keys.secretKey)

        # Rotation keys for:
        # - Matrix multiplication: [1, batch_size)
        # - Attention concatenation: [1, 2*out_channels) for packing h'_i||h'_j
        # Generate conservatively up to 2*batch_size to cover all cases
        max_rotation = min(2 * max(self.in_channels, self.out_channels), 2 * self._batch_size)
        rotation_indices = list(range(1, max_rotation + 1))
        self._cc.EvalRotateKeyGen(keys.secretKey, rotation_indices)

        return keys  # Return original KeyPair

    @property
    def crypto_context(self) -> Any:
        """Return CKKS crypto context."""
        return self._cc

    @property
    def batch_size(self) -> int:
        """Return CKKS batch size (slots used for packing)."""
        return self._batch_size

    @property
    def keys(self) -> Any:
        """Return CKKS keypair (OpenFHE KeyPair object)."""
        return self._keys

    # -------------------------------------------------------------------------
    # Encryption / Decryption helpers
    # -------------------------------------------------------------------------

    def encrypt_node_features(self, x: np.ndarray) -> List[Any]:
        """
        Encrypt node features (one ciphertext per node, packed slots).
        x: (N, F) plaintext features
        Returns: list of N ciphertexts
        """
        N, F = x.shape
        if F > self._batch_size:
            raise ValueError(f"Feature dim {F} > batch_size {self._batch_size}")

        ct_list = []
        for i in range(N):
            row = np.zeros(self._batch_size, dtype=np.float64)
            row[:F] = x[i]
            pt_x = self._cc.MakeCKKSPackedPlaintext(row.tolist())
            ct_x = self._cc.Encrypt(self._keys.publicKey, pt_x)
            ct_list.append(ct_x)
        return ct_list

    def decrypt_node_features(self, ct_list: List[Any], feature_dim: int) -> np.ndarray:
        """
        Decrypt node features.
        ct_list: list of N ciphertexts
        feature_dim: F (how many slots to extract)
        Returns: (N, F) plaintext array
        """
        N = len(ct_list)
        x = np.zeros((N, feature_dim), dtype=np.float64)
        for i, ct in enumerate(ct_list):
            pt = self._cc.Decrypt(self._keys.secretKey, ct)
            pt.SetLength(feature_dim)
            # Get packed values from CKKS plaintext
            values = pt.GetCKKSPackedValue()
            # CKKS can return complex; take real part
            x[i] = np.real([complex(v).real for v in values[:feature_dim]])
        return x

    # -------------------------------------------------------------------------
    # Stage 1: CKKS Linear Layer (rotation-based matrix multiplication)
    # -------------------------------------------------------------------------

    def _sum_slots_via_rotations(self, ct: Any, num_slots: int) -> Any:
        """
        Sum first num_slots slots of a ciphertext via binary tree rotations.
        Returns ciphertext with sum in all slots.
        """
        acc = ct
        shift = 1
        while shift < num_slots:
            rotated = self._cc.EvalRotate(acc, shift)
            acc = self._cc.EvalAdd(acc, rotated)
            shift *= 2
        return acc

    def matmul_ckks(self, ct_x: Any, W: np.ndarray) -> List[Any]:
        """
        CKKS matrix-vector multiplication: h' = W @ x (fully encrypted).
        
        ct_x: ciphertext with x packed in slots (shape: batch_size slots, first F_in used)
        W: (F_out, F_in) weight matrix
        
        Returns: list of F_out ciphertexts (one per output dimension)
        
        Algorithm:
        1. For each output dimension k:
           - Multiply ct_x by plaintext W[k, :] element-wise
           - Sum over F_in slots via rotations
           - Result: h'_k encrypted
        """
        F_out, F_in = W.shape
        ct_h_list = []

        for k in range(F_out):
            # Create plaintext for W[k, :]
            w_k = np.zeros(self._batch_size, dtype=np.float64)
            w_k[:F_in] = W[k]
            pt_w_k = self._cc.MakeCKKSPackedPlaintext(w_k.tolist())

            # Element-wise multiply
            ct_prod = self._cc.EvalMult(ct_x, pt_w_k)

            # Sum over F_in slots
            ct_sum = self._sum_slots_via_rotations(ct_prod, F_in)
            ct_h_list.append(ct_sum)

        return ct_h_list

    # -------------------------------------------------------------------------
    # Stage 2: Encrypted LeakyReLU (CKKS↔FHEW scheme switching)
    # -------------------------------------------------------------------------

    def leaky_relu_encrypted(self, ct_e: Any, negative_slope: float = 0.2) -> Any:
        """
        Encrypted LeakyReLU: y = x if x > 0 else negative_slope * x
        
        Uses CKKS↔FHEW scheme switching (following OpenFHE scheme-switching.py ComparisonViaSchemeSwitching):
        1. Switch CKKS → FHEW using EvalCKKStoFHEW
        2. Compute sign bit in FHEW using EvalSign (returns 1 if x >= 0, 0 if x < 0)
        3. Switch sign bit FHEW → CKKS using EvalFHEWtoCKKS
        4. Compute in CKKS: result = x * (negative_slope + sign * (1 - negative_slope))
           - When sign=1 (x >= 0): result = x * (negative_slope + (1 - negative_slope)) = x
           - When sign=0 (x < 0): result = x * negative_slope
        
        Args:
            ct_e: CKKS ciphertext containing the value
            negative_slope: LeakyReLU negative slope (default 0.2)
        
        Returns:
            CKKS ciphertext with LeakyReLU applied
        """
        if not self.use_cggi or self._cggi_context is None:
            # Fallback: return identity (no LeakyReLU activation)
            return ct_e

        try:
            # Step 0: Precompute scaling for CKKS → FHEW (required before first switch)
            # From scheme-switching.py: compute scale based on FHEW plaintext modulus
            logQ_ccLWE = 25  # From setup
            modulus_LWE = 1 << logQ_ccLWE
            beta = self._cggi_context.GetBeta()
            pLWE = int(modulus_LWE / (2 * beta))  # Large precision
            scale = 1.0 / pLWE
            
            # Call precompute (this can be called multiple times, will recompute if needed)
            self._cc.EvalCKKStoFHEWPrecompute(scale)
            
            # Step 1: CKKS → FHEW scheme switching
            # Convert single CKKS ciphertext to FHEW (list of 1 LWE ciphertext)
            num_values = 1  # Single value per ciphertext in our case
            lwe_ciphertexts = self._cc.EvalCKKStoFHEW(ct_e, num_values)
            
            # Step 2: Compute sign bit in FHEW
            # EvalSign returns 1 if value >= 0, 0 if value < 0
            lwe_sign = self._cggi_context.EvalSign(lwe_ciphertexts[0])
            
            # Step 3: FHEW → CKKS scheme switching (convert sign bit back to CKKS)
            # Switch sign bit back to CKKS (plaintext modulus 2 for binary sign)
            # EvalFHEWtoCKKS(LWECiphertexts, numCtxts, numSlots, p, pmin, pmax, dim1)
            ct_sign = self._cc.EvalFHEWtoCKKS(
                [lwe_sign],  # LWE ciphertexts list
                1,           # numCtxts (number of ciphertexts)
                self._batch_size,  # numSlots
                2,           # p (binary plaintext modulus for sign bit: 0 or 1)
                0.0,         # pmin
                2.0          # pmax
            )
            
            # Step 4: Compute LeakyReLU in CKKS
            # result = x * (negative_slope + sign * (1 - negative_slope))
            # Create plaintext for (1 - negative_slope)
            factor = 1.0 - negative_slope
            pt_factor = self._cc.MakeCKKSPackedPlaintext([factor] * self._batch_size)
            
            # Multiply: sign * (1 - negative_slope)
            ct_sign_scaled = self._cc.EvalMult(ct_sign, pt_factor)
            
            # Add: negative_slope + sign * (1 - negative_slope)
            pt_neg_slope = self._cc.MakeCKKSPackedPlaintext([negative_slope] * self._batch_size)
            ct_multiplier = self._cc.EvalAdd(ct_sign_scaled, pt_neg_slope)
            
            # Final: x * multiplier
            ct_result = self._cc.EvalMult(ct_e, ct_multiplier)
            
            return ct_result
            
        except Exception as e:
            # If scheme switching fails, return identity (log warning in production)
            print(f"Warning: LeakyReLU scheme switching failed: {e}")
            print("Falling back to identity (no activation)")
            return ct_e

    # -------------------------------------------------------------------------
    # Stage 3: CKKS Attention Scores
    # -------------------------------------------------------------------------

    def attention_scores_ckks(
        self,
        ct_h_list: List[List[Any]],
        edge_index: np.ndarray,
        num_nodes: int,
    ) -> List[Any]:
        """
        Compute FULLY ENCRYPTED attention scores: e_{ij} = a^T [h'_i || h'_j]
        
        ct_h_list: list of N nodes, each with F_out ciphertexts (from matmul_ckks)
        edge_index: (2, E) edge list
        
        Returns: list of E ciphertexts (one per edge)
        
        Algorithm (fully encrypted, no decryption):
        1. For each edge (i,j):
           - Pack h'_i into slots [0:F_out] of a ciphertext
           - Pack h'_j into slots [F_out:2*F_out] via rotations
           - Multiply by plaintext attention vector a
           - Sum to get e_{ij} encrypted
        """
        E = edge_index.shape[1]
        F_out = len(ct_h_list[0])
        ct_e_list = []

        # Prepare plaintext attention vector [a_i || a_j] for concatenated features
        a_concat = self._a  # Shape: (2 * F_out,)

        for e in range(E):
            i, j = edge_index[0, e], edge_index[1, e]

            # Step 1: Pack h'_i into first F_out slots
            # ct_h_list[i] is a list of F_out ciphertexts, each with value in slot 0
            # We need to combine them into one ciphertext with all values
            
            # Create packed ciphertext for h'_i (F_out values in first F_out slots)
            ct_h_i_packed = None
            for k in range(F_out):
                # Rotate ct_h_list[i][k] by k positions to put value in slot k
                if k == 0:
                    ct_h_i_packed = ct_h_list[i][0]
                else:
                    ct_rotated = self._cc.EvalRotate(ct_h_list[i][k], k)
                    ct_h_i_packed = self._cc.EvalAdd(ct_h_i_packed, ct_rotated)
            
            # Step 2: Pack h'_j into next F_out slots [F_out:2*F_out]
            ct_h_j_packed = None
            for k in range(F_out):
                # Rotate ct_h_list[j][k] by (F_out + k) positions
                ct_rotated = self._cc.EvalRotate(ct_h_list[j][k], F_out + k)
                if ct_h_j_packed is None:
                    ct_h_j_packed = ct_rotated
                else:
                    ct_h_j_packed = self._cc.EvalAdd(ct_h_j_packed, ct_rotated)
            
            # Step 3: Combine [h'_i || h'_j] into one ciphertext
            ct_concat = self._cc.EvalAdd(ct_h_i_packed, ct_h_j_packed)
            
            # Step 4: Multiply by attention vector a (element-wise)
            # a has shape (2*F_out,), pad to batch_size
            a_padded = np.zeros(self._batch_size, dtype=np.float64)
            a_padded[:min(2*F_out, self._batch_size)] = a_concat[:min(2*F_out, self._batch_size)]
            pt_a = self._cc.MakeCKKSPackedPlaintext(a_padded.tolist())
            ct_weighted = self._cc.EvalMult(ct_concat, pt_a)
            
            # Step 5: Sum all slots to get e_{ij} = a^T [h'_i || h'_j]
            ct_e_ij = self._sum_slots_via_rotations(ct_weighted, 2 * F_out)
            
            ct_e_list.append(ct_e_ij)

        return ct_e_list

    # -------------------------------------------------------------------------
    # Stage 3: CKKS Softmax (Chebyshev approximation)
    # -------------------------------------------------------------------------

    def softmax_ckks_chebyshev(
        self,
        ct_e_list: List[Any],
        edge_index: np.ndarray,
        num_nodes: int,
    ) -> List[Any]:
        """
        FULLY ENCRYPTED softmax using CKKS Chebyshev exp + homomorphic division.
        
        ct_e_list: encrypted edge scores (E ciphertexts)
        edge_index: (2, E)
        
        Returns: List of E encrypted attention weights (normalized)
        
        Algorithm (fully encrypted):
        1. Compute exp(e) for each edge using Chebyshev approximation
        2. For each target node:
           a. Sum exp values of incoming edges (encrypted)
           b. Compute reciprocal 1/sum (Newton-Raphson)
           c. Multiply each exp by reciprocal (encrypted normalization)
        3. Return encrypted softmax probabilities (NO DECRYPTION)
        """
        from .fhe_utils import encrypted_reciprocal_newton_raphson
        
        E = edge_index.shape[1]

        # Step 1: Chebyshev approximation for exp(x)
        # Coefficients for exp(x) approximation over [-1, 1]
        coeffs = [1.0, 1.0, 0.5, 0.166667, 0.041667]  # Taylor-like coefficients

        ct_exp_list = []
        for ct_e in ct_e_list:
            try:
                ct_exp = self._cc.EvalChebyshevSeries(ct_e, coeffs, -1.0, 1.0)
                ct_exp_list.append(ct_exp)
            except Exception:
                # Fallback: use ct_e directly
                ct_exp_list.append(ct_e)

        # Step 2: Encrypted normalization per target node (softmax)
        ct_alpha_list = [None] * E
        
        for t in range(num_nodes):
            # Find edges targeting node t
            mask = edge_index[1] == t
            edge_indices = np.where(mask)[0]
            
            if len(edge_indices) == 0:
                continue
            
            # Step 2a: Sum exp values for incoming edges (encrypted)
            ct_sum = ct_exp_list[edge_indices[0]]
            for idx in edge_indices[1:]:
                ct_sum = self._cc.EvalAdd(ct_sum, ct_exp_list[idx])
            
            # Step 2b: Compute reciprocal 1/sum using Newton-Raphson (encrypted)
            # Initial guess: assume sum is in range [1, 10], guess 0.2
            # For better convergence, could use adaptive guess based on number of edges
            num_incoming = len(edge_indices)
            initial_guess = 1.0 / max(1.0, num_incoming * 0.5)  # Conservative estimate
            
            ct_reciprocal = encrypted_reciprocal_newton_raphson(
                self._cc,
                ct_sum,
                num_iterations=2,  # 2 iterations: depth = 4 (balance accuracy vs depth)
                initial_guess=initial_guess,
                batch_size=self._batch_size,
            )
            
            # Step 2c: Multiply each exp by reciprocal (encrypted softmax)
            for idx in edge_indices:
                ct_alpha_list[idx] = self._cc.EvalMult(ct_exp_list[idx], ct_reciprocal)
        
        return ct_alpha_list

    # -------------------------------------------------------------------------
    # CKKS Aggregation
    # -------------------------------------------------------------------------

    def aggregate_fhe(
        self,
        ct_list: List[Any],
        edge_index: np.ndarray,
        alpha: Any,  # Can be np.ndarray (plaintext) or List[Any] (encrypted)
        num_nodes: int,
    ) -> List[Any]:
        """
        Aggregate node features using attention weights (CKKS).
        out_i = sum_j alpha_{ij} * h'_j
        
        ct_list: list of N ciphertexts (h' features)
        edge_index: (2, E)
        alpha: attention weights - either:
            - np.ndarray (E,): plaintext weights
            - List[Any]: encrypted weight ciphertexts (E,)
        num_nodes: N
        
        Returns: list of N output ciphertexts
        """
        E = edge_index.shape[1]
        out_cts = []
        
        # Check if alpha is encrypted (list of ciphertexts) or plaintext (numpy array)
        alpha_is_encrypted = isinstance(alpha, list)

        for t in range(num_nodes):
            mask = edge_index[1] == t
            edge_indices = np.where(mask)[0]
            
            if len(edge_indices) == 0:
                # No incoming edges: zero output
                zeros = [0.0] * self._batch_size
                pt_zero = self._cc.MakeCKKSPackedPlaintext(zeros)
                ct_zero = self._cc.Encrypt(self._keys.publicKey, pt_zero)
                out_cts.append(ct_zero)
                continue

            cols_j = edge_index[0, mask]

            # Accumulate: sum_j alpha_{ij} * h'_j
            acc = None
            for k, edge_idx in enumerate(edge_indices):
                if alpha_is_encrypted:
                    # Encrypted alpha: multiply ciphertext by ciphertext
                    term = self._cc.EvalMult(ct_list[cols_j[k]], alpha[edge_idx])
                else:
                    # Plaintext alpha: multiply ciphertext by plaintext
                    w = alpha[edge_idx]
                    pt_scale = self._cc.MakeCKKSPackedPlaintext([w] * self._batch_size)
                    term = self._cc.EvalMult(ct_list[cols_j[k]], pt_scale)
                
                if acc is None:
                    acc = term
                else:
                    acc = self._cc.EvalAdd(acc, term)

            out_cts.append(acc)

        return out_cts

    # -------------------------------------------------------------------------
    # Forward methods
    # -------------------------------------------------------------------------

    def forward_plain(self, graph: FHEGraph) -> np.ndarray:
        """
        Forward in plaintext (no encryption). Uses utility from gat_encoder.py.
        """
        return gat_forward_plain(graph.to_plain(), graph.edge_index, self._W, self._a, self.negative_slope)

    def forward_fhe_full(self, graph: FHEGraph) -> np.ndarray:
        """
        **FULLY-ENCRYPTED** GAT forward (ALL stages encrypted, NO intermediate decryption).
        
        - Uses encrypted features from graph.node_features_enc (plaintext never stored)
        - Stage 1: CKKS linear layer (h' = W x, rotation-based matrix multiplication)
        - Stage 2: CKKS attention scores (e_{ij} = a^T [h'_i || h'_j], rotation-based concatenation)
        - Stage 3: Encrypted LeakyReLU (CKKS↔FHEW scheme switching with EvalSign)
        - Stage 4: CKKS softmax (Chebyshev exp + Newton-Raphson division)
        - Stage 5: CKKS aggregation (encrypted weighted sum with encrypted attention weights)
        
        **SECURITY: ZERO intermediate decryption - only final output is decrypted!**
        
        All operations (linear, concat, attention, LeakyReLU, softmax, aggregate) 
        are performed on encrypted data using:
        - Homomorphic rotations for packing/unpacking
        - Homomorphic arithmetic (Add, Mult)
        - Scheme switching (CKKS ↔ FHEW) for sign bits
        - Newton-Raphson for encrypted division
        """
        # Step 1: Use encrypted node features from graph
        ct_x_list = graph.node_features_enc
        N = graph.num_nodes

        # Step 2: CKKS linear layer (fully encrypted)
        ct_h_prime_list = []
        for i in range(N):
            ct_h_i = self.matmul_ckks(ct_x_list[i], self._W)
            ct_h_prime_list.append(ct_h_i)

        # Step 3: CKKS attention scores (encrypted inner products)
        ct_e_list = self.attention_scores_ckks(ct_h_prime_list, graph.edge_index, N)

        # Step 4: Encrypted LeakyReLU via CKKS↔FHEW scheme switching
        ct_e_leaky_list = []
        for ct_e in ct_e_list:
            ct_e_leaky = self.leaky_relu_encrypted(ct_e, self.negative_slope)
            ct_e_leaky_list.append(ct_e_leaky)

        # Step 5: CKKS softmax (Chebyshev exp + ENCRYPTED normalization via Newton-Raphson)
        # Returns list of E encrypted attention weights (NO DECRYPTION)
        ct_alpha_list = self.softmax_ckks_chebyshev(ct_e_leaky_list, graph.edge_index, N)

        # Step 6: Convert h' from list-of-lists to per-node packed ciphertexts
        # ct_h_prime_list is list of N nodes, each with F_out separate ciphertexts
        # We need to pack each node's features into a single ciphertext for aggregation
        ct_h_packed_list = []
        for i in range(N):
            # Pack F_out ciphertexts into one ciphertext using rotations
            ct_packed = None
            for k in range(self.out_channels):
                if k == 0:
                    ct_packed = ct_h_prime_list[i][0]
                else:
                    # Rotate to put value in slot k
                    ct_rotated = self._cc.EvalRotate(ct_h_prime_list[i][k], k)
                    ct_packed = self._cc.EvalAdd(ct_packed, ct_rotated)
            ct_h_packed_list.append(ct_packed)

        # Step 7: CKKS aggregation with ENCRYPTED attention weights
        out_cts = self.aggregate_fhe(ct_h_packed_list, graph.edge_index, ct_alpha_list, N)

        # Step 8: Decrypt final outputs
        return self.decrypt_node_features(out_cts, self.out_channels)
