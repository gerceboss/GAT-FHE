"""
Pipeline runner for FHE GAT encoder with configurable per-step encryption.

Supports per-step enc/dec modes and detailed time+RSS metrics (see `metrics.py`).

Step-3 (LeakyReLU) supports TWO EvalSign modes:
1) Fully encrypted via scheme switching (CKKS -> FHEW -> EvalSign -> CKKS)
2) No scheme switching baseline:
   decrypt CKKS -> encrypt into standalone BinFHE (CGGI/GINX) -> EvalFunc(LUT) sign ->
   decrypt bit -> encrypt bit into CKKS (and re-encrypt edge scores into CKKS) to continue.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Any

import numpy as np

from .encoder import GATEncoderFHE
from .fhe_graph import FHEGraph
from .metrics import MetricsRecorder


Mode = Literal["enc", "dec"]
EvalSignMode = Literal[
    "schemeswitch",  # fully encrypted CKKS<->FHEW + EvalSign
    "decrypt_encrypt_fhew_evalfunc",  # no scheme switching baseline via BinFHE EvalFunc(LUT)
    "ckks_poly_leakyrelu",  # CKKS-only: polynomial approximation of LeakyReLU (no scheme switch, no decrypt)
]


@dataclass
class GATRunConfig:
    """
    Each main step can be:
    - "enc": encrypted (OpenFHE operations)
    - "dec": plaintext (Python/NumPy)
    """

    # Pipeline steps
    step1_linear: Mode = "enc"
    step2_attention: Mode = "enc"
    step3_leakyrelu: Mode = "enc"
    step4_softmax: Mode = "enc"
    step5_aggregation: Mode = "enc"

    # Step-3 (LeakyReLU) sign mode
    step3_evalsign_mode: EvalSignMode = "schemeswitch"

    # FHE parameters (informational; encoder owns actual CC/keys)
    batch_size: int = 8
    mult_depth: int = 15
    scale_mod_size: int = 40
    use_cggi: bool = False
    negative_slope: float = 0.2

    # Client-key mode: return encrypted ciphertexts instead of decrypted output
    return_encrypted: bool = False

    # FHE training: num_epochs and lr for homomorphic gradient descent
    num_epochs: int = 0  # 0 = inference only
    lr: float = 0.01

    # Reporting
    print_metrics: bool = True
    print_shapes: bool = False


def run_gat_pipeline(
    *,
    encoder: GATEncoderFHE,
    graph: FHEGraph,
    cfg: GATRunConfig,
) -> tuple[np.ndarray | list[Any], dict[str, Any]]:
    metrics = MetricsRecorder()

    num_nodes = graph.num_nodes
    edge_index = graph.edge_index
    ct_x_list = graph.node_features_enc

    def _decrypt_scalar_ct_list(ct_list: list) -> np.ndarray:
        vals = np.zeros(len(ct_list), dtype=np.float64)
        for i, ct in enumerate(ct_list):
            pt = encoder.crypto_context.Decrypt(encoder.keys.secretKey, ct)
            pt.SetLength(1)
            v = pt.GetCKKSPackedValue()[0]
            vals[i] = float(np.real(complex(v).real))
        return vals

    def _encrypt_scalar_list_to_ckks_ct_list(values: np.ndarray) -> list:
        ct_list = []
        for v in np.asarray(values, dtype=np.float64):
            packed = [float(v)] * encoder.batch_size
            pt = encoder.crypto_context.MakeCKKSPackedPlaintext(packed)
            ct_list.append(encoder.crypto_context.Encrypt(encoder.keys.publicKey, pt))
        return ct_list

    def _encrypt_matrix_to_ct_h_list(h_plain: np.ndarray) -> list[list]:
        # (N, F_out) -> N × F_out scalar ciphertexts (like matmul_ckks output)
        N, F = h_plain.shape
        ct_h_list: list[list] = []
        for i in range(N):
            node_cts = []
            for k in range(F):
                packed = [float(h_plain[i, k])] * encoder.batch_size
                pt = encoder.crypto_context.MakeCKKSPackedPlaintext(packed)
                node_cts.append(encoder.crypto_context.Encrypt(encoder.keys.publicKey, pt))
            ct_h_list.append(node_cts)
        return ct_h_list

    # ---------------------------------------------------------------------
    # Step 1: Linear layer
    # ---------------------------------------------------------------------
    with metrics.step("1_linear_layer", encrypted=(cfg.step1_linear == "enc")):
        if cfg.step1_linear == "enc":
            ct_h_list = []
            for ct_x in ct_x_list:
                ct_h_list.append(encoder._matmul_ckks_dispatch(ct_x))
            h_transformed: Any = ct_h_list
            if cfg.print_shapes:
                print(f"  Linear(enc): {num_nodes} nodes × {len(ct_h_list[0])} dims")
        else:
            x_plain = encoder.decrypt_node_features(ct_x_list, encoder.in_channels)
            h_plain = x_plain @ encoder._W.T
            h_transformed = h_plain
            if cfg.print_shapes:
                print(f"  Linear(dec): {h_plain.shape}")

    # ---------------------------------------------------------------------
    # Step 2: Attention scores
    # ---------------------------------------------------------------------
    with metrics.step("2_attention_scores", encrypted=(cfg.step2_attention == "enc")):
        if cfg.step2_attention == "enc":
            if not isinstance(h_transformed[0], list):
                h_transformed = _encrypt_matrix_to_ct_h_list(h_transformed)
            attention_scores = encoder.attention_scores_ckks(h_transformed, edge_index, num_nodes)
            if cfg.print_shapes:
                print(f"  Attention(enc): E={len(attention_scores)}")
        else:
            if isinstance(h_transformed[0], list):
                h_plain = np.zeros((num_nodes, encoder.out_channels), dtype=np.float64)
                for i, ct_h_node in enumerate(h_transformed):
                    for k, ct_h_k in enumerate(ct_h_node):
                        pt = encoder.crypto_context.Decrypt(encoder.keys.secretKey, ct_h_k)
                        pt.SetLength(1)
                        h_plain[i, k] = float(np.real(complex(pt.GetCKKSPackedValue()[0]).real))
            else:
                h_plain = h_transformed
            from gat_encoder import attention_plain

            e_plain, _alpha_plain = attention_plain(edge_index, h_plain, encoder._a, num_nodes, cfg.negative_slope)
            attention_scores = e_plain
            if cfg.print_shapes:
                print(f"  Attention(dec): {e_plain.shape}")

    # ---------------------------------------------------------------------
    # Step 3: LeakyReLU (two modes only)
    # ---------------------------------------------------------------------
    if cfg.step3_leakyrelu == "dec":
        with metrics.step("3_leakyrelu_plain", encrypted=False):
            if isinstance(attention_scores, list):
                e_plain = _decrypt_scalar_ct_list(attention_scores)
            else:
                e_plain = np.asarray(attention_scores, dtype=np.float64)
            e_after = np.where(e_plain >= 0.0, e_plain, cfg.negative_slope * e_plain)
    else:
        # step3_leakyrelu == "enc"
        if cfg.step3_evalsign_mode == "schemeswitch":
            if not cfg.use_cggi or encoder._cggi_context is None:
                raise ValueError("schemeswitch mode requires use_cggi=True")

            # ensure CKKS ciphertext list
            if not isinstance(attention_scores, list):
                attention_scores = _encrypt_scalar_list_to_ckks_ct_list(np.asarray(attention_scores, dtype=np.float64))
            elif len(attention_scores) > 0 and isinstance(attention_scores[0], (float, int, np.floating, np.integer)):
                attention_scores = _encrypt_scalar_list_to_ckks_ct_list(np.asarray(attention_scores, dtype=np.float64))

            # 3a: precompute/setup ONCE per encoder
            if not hasattr(encoder, "_schemeswitch_precomputed"):
                encoder._schemeswitch_precomputed = False  # type: ignore[attr-defined]
            if not encoder._schemeswitch_precomputed:  # type: ignore[attr-defined]
                with metrics.step("3a_schemeswitch_precompute_setup", encrypted=True):
                    logQ_ccLWE = 25
                    modulus_LWE = 1 << logQ_ccLWE
                    beta = encoder._cggi_context.GetBeta()
                    pLWE = int(modulus_LWE / (2 * beta))
                    scale = 1.0 / pLWE
                    encoder.crypto_context.EvalCKKStoFHEWPrecompute(scale)
                    encoder._schemeswitch_precomputed = True  # type: ignore[attr-defined]
            else:
                with metrics.step("3a_schemeswitch_precompute_cached", encrypted=True):
                    pass

            # 3b: CKKS -> FHEW
            with metrics.step("3b_ckks_to_fhew", encrypted=True):
                lwe_list = []
                for ct_e in attention_scores:
                    lwe_list.append(encoder.crypto_context.EvalCKKStoFHEW(ct_e, 1)[0])

            # 3c: EvalSign in CGGI/FHEW
            with metrics.step("3c_evalsign_cggi", encrypted=True):
                lwe_sign_list = [encoder._cggi_context.EvalSign(lwe) for lwe in lwe_list]

            # 3d: FHEW -> CKKS
            with metrics.step("3d_fhew_to_ckks", encrypted=True):
                ct_sign_list = []
                for lwe_sign in lwe_sign_list:
                    ct_sign_list.append(
                        encoder.crypto_context.EvalFHEWtoCKKS(
                            [lwe_sign], 1, encoder.batch_size, 2, 0.0, 2.0
                        )
                    )

            # 3e: combine
            with metrics.step("3e_leakyrelu_combine_ckks", encrypted=True):
                factor = 1.0 - cfg.negative_slope
                pt_factor = encoder.crypto_context.MakeCKKSPackedPlaintext([factor] * encoder.batch_size)
                pt_neg = encoder.crypto_context.MakeCKKSPackedPlaintext([cfg.negative_slope] * encoder.batch_size)
                e_after = []
                for ct_e, ct_sign in zip(attention_scores, ct_sign_list):
                    ct_sign_scaled = encoder.crypto_context.EvalMult(ct_sign, pt_factor)
                    ct_multiplier = encoder.crypto_context.EvalAdd(ct_sign_scaled, pt_neg)
                    e_after.append(encoder.crypto_context.EvalMult(ct_e, ct_multiplier))

        elif cfg.step3_evalsign_mode == "ckks_poly_leakyrelu":
            # CKKS-only: polynomial approximation of LeakyReLU, no scheme switch, no decrypt
            from .fhe_utils import get_leaky_relu_chebyshev_coefficients

            with metrics.step("3_ckks_poly_leakyrelu", encrypted=True):
                if not isinstance(attention_scores, list):
                    attention_scores = _encrypt_scalar_list_to_ckks_ct_list(
                        np.asarray(attention_scores, dtype=np.float64)
                    )
                elif len(attention_scores) > 0 and isinstance(
                    attention_scores[0], (float, int, np.floating, np.integer)
                ):
                    attention_scores = _encrypt_scalar_list_to_ckks_ct_list(
                        np.asarray(attention_scores, dtype=np.float64)
                    )

                coeffs = get_leaky_relu_chebyshev_coefficients(
                    negative_slope=cfg.negative_slope,
                    domain_low=-3.0,
                    domain_high=3.0,
                    degree=7,
                )
                e_after = []
                for ct_e in attention_scores:
                    try:
                        ct_leaky = encoder.crypto_context.EvalChebyshevSeries(
                            ct_e, coeffs, -3.0, 3.0
                        )
                        e_after.append(ct_leaky)
                    except Exception:
                        e_after.append(ct_e)

        elif cfg.step3_evalsign_mode == "decrypt_encrypt_fhew_evalfunc":
            # No scheme switching at all.
            # Decrypt CKKS -> encrypt into standalone BinFHE -> EvalFunc(LUT) -> decrypt bit -> encrypt into CKKS.
            from openfhe import STD128, GINX  # type: ignore
            from .evalsign_self_implementation import (
                create_evalsign_self_context,
                evalsign_via_evalfunc,
                quantize_float_to_centered_int,
                encode_centered_to_modp,
            )

            # 3a: decrypt CKKS edge scores (or use plaintext if already)
            with metrics.step("3a_decrypt_edge_scores_ckks", encrypted=False):
                if isinstance(attention_scores, list):
                    e_plain = _decrypt_scalar_ct_list(attention_scores)
                else:
                    e_plain = np.asarray(attention_scores, dtype=np.float64)

            # 3b: setup standalone BinFHE selfctx ONCE per encoder (expensive)
            if not hasattr(encoder, "_self_evalsign_ctx"):
                encoder._self_evalsign_ctx = None  # type: ignore[attr-defined]
            if encoder._self_evalsign_ctx is None:  # type: ignore[attr-defined]
                with metrics.step("3b_binfhe_selfctx_setup", encrypted=True):
                    encoder._self_evalsign_ctx = create_evalsign_self_context(  # type: ignore[attr-defined]
                        paramset=STD128, method=GINX, max_plaintext_bits=12
                    )
            else:
                with metrics.step("3b_binfhe_selfctx_cached", encrypted=True):
                    pass
            selfctx = encoder._self_evalsign_ctx  # type: ignore[attr-defined]

            # 3c: encrypt into BinFHE (CGGI)
            with metrics.step("3c_encrypt_fhew_inputs", encrypted=True):
                ct_lwe_list = []
                # quantization scale: map float -> centered int in Z_p
                scale = 512.0
                for v in e_plain:
                    q = quantize_float_to_centered_int(float(v), scale=scale, p=selfctx.p)
                    m = encode_centered_to_modp(q, selfctx.p)
                    ct_lwe_list.append(selfctx.cc_lwe.Encrypt(selfctx.sk_lwe, int(m)))

            # 3d: EvalFunc(LUT) sign bit
            with metrics.step("3d_evalfunc_sign_lut", encrypted=True):
                ct_sign_lwe_list = [
                    evalsign_via_evalfunc(cc_lwe=selfctx.cc_lwe, ct_lwe=ct, lut_sign=selfctx.lut_sign)
                    for ct in ct_lwe_list
                ]

            # 3e: decrypt sign bits (0/1)
            with metrics.step("3e_decrypt_signbits_fhew", encrypted=False):
                sign_bits = np.array(
                    [int(selfctx.cc_lwe.Decrypt(selfctx.sk_lwe, ct, 2)) for ct in ct_sign_lwe_list],
                    dtype=np.float64,
                )

            # 3f: encrypt sign bits to CKKS and also encrypt edge scores to CKKS
            with metrics.step("3f_encrypt_signbits_ckks", encrypted=True):
                ct_sign_list = _encrypt_scalar_list_to_ckks_ct_list(sign_bits)
            with metrics.step("3g_encrypt_edgescores_ckks", encrypted=True):
                attention_scores_ckks = _encrypt_scalar_list_to_ckks_ct_list(e_plain)

            # 3h: combine in CKKS (no scheme switching)
            with metrics.step("3h_leakyrelu_combine_ckks", encrypted=True):
                factor = 1.0 - cfg.negative_slope
                pt_factor = encoder.crypto_context.MakeCKKSPackedPlaintext([factor] * encoder.batch_size)
                pt_neg = encoder.crypto_context.MakeCKKSPackedPlaintext([cfg.negative_slope] * encoder.batch_size)
                e_after = []
                for ct_e, ct_sign in zip(attention_scores_ckks, ct_sign_list):
                    ct_sign_scaled = encoder.crypto_context.EvalMult(ct_sign, pt_factor)
                    ct_multiplier = encoder.crypto_context.EvalAdd(ct_sign_scaled, pt_neg)
                    e_after.append(encoder.crypto_context.EvalMult(ct_e, ct_multiplier))

        else:
            raise ValueError(f"Unknown step3_evalsign_mode: {cfg.step3_evalsign_mode}")

    # ---------------------------------------------------------------------
    # Step 4: Softmax
    # ---------------------------------------------------------------------
    with metrics.step("4_softmax", encrypted=(cfg.step4_softmax == "enc")):
        if cfg.step4_softmax == "enc":
            if not isinstance(e_after, list):
                e_after = _encrypt_scalar_list_to_ckks_ct_list(np.asarray(e_after, dtype=np.float64))
            ct_alpha_list = encoder.softmax_ckks_chebyshev(e_after, edge_index, num_nodes)
            alpha_final = ct_alpha_list
        else:
            if isinstance(e_after, list):
                e_plain = _decrypt_scalar_ct_list(e_after)
            else:
                e_plain = np.asarray(e_after, dtype=np.float64)
            alpha_plain = np.zeros(edge_index.shape[1], dtype=np.float64)
            for j in range(num_nodes):
                mask = edge_index[1] == j
                if mask.sum() > 0:
                    e_j = e_plain[mask]
                    e_j_exp = np.exp(e_j - e_j.max())
                    alpha_plain[mask] = e_j_exp / e_j_exp.sum()
            alpha_final = alpha_plain

    # ---------------------------------------------------------------------
    # Step 5: Aggregation
    # ---------------------------------------------------------------------
    with metrics.step("5_aggregation", encrypted=(cfg.step5_aggregation == "enc")):
        if cfg.step5_aggregation == "enc":
            # pack h' for aggregation
            if isinstance(h_transformed[0], list):
                ct_h_packed_list = []
                for i in range(num_nodes):
                    ct_packed = None
                    for k in range(encoder.out_channels):
                        if k == 0:
                            ct_packed = h_transformed[i][0]
                        else:
                            ct_rot = encoder.crypto_context.EvalRotate(h_transformed[i][k], k)
                            ct_packed = encoder.crypto_context.EvalAdd(ct_packed, ct_rot)
                    ct_h_packed_list.append(ct_packed)
                h_to_agg = ct_h_packed_list
            else:
                # encrypt plaintext node vectors into packed CKKS
                ct_h_packed_list = []
                for i in range(num_nodes):
                    packed = [float(h_transformed[i, k]) if k < encoder.out_channels else 0.0 for k in range(encoder.batch_size)]
                    pt = encoder.crypto_context.MakeCKKSPackedPlaintext(packed)
                    ct_h_packed_list.append(encoder.crypto_context.Encrypt(encoder.keys.publicKey, pt))
                h_to_agg = ct_h_packed_list

            out_cts = encoder.aggregate_fhe(h_to_agg, edge_index, alpha_final, num_nodes)
            if cfg.return_encrypted:
                output = out_cts  # type: ignore[assignment]  # List of ciphertexts
            else:
                output = encoder.decrypt_node_features(out_cts, encoder.out_channels)
        else:
            # plaintext aggregation
            if isinstance(h_transformed[0], list):
                h_plain = np.zeros((num_nodes, encoder.out_channels), dtype=np.float64)
                for i, ct_h_node in enumerate(h_transformed):
                    for k, ct_h_k in enumerate(ct_h_node):
                        pt = encoder.crypto_context.Decrypt(encoder.keys.secretKey, ct_h_k)
                        pt.SetLength(1)
                        h_plain[i, k] = float(np.real(complex(pt.GetCKKSPackedValue()[0]).real))
            else:
                h_plain = h_transformed

            if isinstance(alpha_final, list):
                alpha_plain = _decrypt_scalar_ct_list(alpha_final)
            else:
                alpha_plain = np.asarray(alpha_final, dtype=np.float64)

            output = np.zeros((num_nodes, encoder.out_channels), dtype=np.float64)
            for idx in range(edge_index.shape[1]):
                src, dst = edge_index[:, idx]
                output[dst] += alpha_plain[idx] * h_plain[src]

    if cfg.print_metrics:
        metrics.print_report()

    metrics_dict = metrics.to_dict()
    return output, metrics_dict


def run_gat_pipeline_client_keys(
    *,
    encoder: GATEncoderFHE,
    graph: FHEGraph,
    print_metrics: bool = True,
) -> tuple[list[Any], dict[str, Any]]:
    """
    Run GAT pipeline in client-key mode: all computation in CKKS, returns
    encrypted output. Client must decrypt with their secret key.

    encoder: Must be created via GATEncoderFHE.from_client_keys (server has
        only public key, cannot decrypt)
    graph: FHEGraph with client-encrypted node features
    """
    cfg = GATRunConfig(
        step1_linear="enc",
        step2_attention="enc",
        step3_leakyrelu="enc",
        step4_softmax="enc",
        step5_aggregation="enc",
        step3_evalsign_mode="ckks_poly_leakyrelu",
        use_cggi=False,
        return_encrypted=True,
        print_metrics=print_metrics,
    )
    out, metrics_dict = run_gat_pipeline(encoder=encoder, graph=graph, cfg=cfg)
    return out, metrics_dict  # type: ignore[return-value]


def run_gat_pipeline_fhe_training(
    *,
    encoder: GATEncoderFHE,
    graph: FHEGraph,
    ct_labels: list[Any],
    train_mask: np.ndarray,
    num_epochs: int = 3,
    lr: float = 0.01,
    print_metrics: bool = True,
) -> tuple[list[Any], dict[str, Any]]:
    """
    FHE training loop: forward pass, homomorphic gradient computation, weight update.

    Two modes:
    - Encrypted weights (no secret key): encoder from from_client_keys_with_encrypted_weights.
      Weights stay encrypted; update is ct_W_new = ct_W_old - lr * ct_grad_W (all in FHE).
    - Plaintext weights (has secret key): encoder from GATEncoderFHE(...).
      Gradient is decrypted, weights updated in plaintext (legacy, server has SK).

    encoder: GATEncoderFHE (with or without secret key; must have encrypted weights for no-SK mode)
    graph: FHEGraph with encrypted node features
    ct_labels: list of N ciphertexts (0 or 1 per node) - encrypted labels
    train_mask: (N,) bool array, True for train nodes (loss computed only on these)
    num_epochs: number of training epochs
    lr: learning rate for weight update

    Returns: (encrypted output ciphertexts, metrics dict)
    """
    from .fhe_utils import get_sigmoid_chebyshev_coefficients

    has_secret_key = True
    try:
        _ = encoder.keys.secretKey
    except (RuntimeError, AttributeError):
        has_secret_key = False

    has_encrypted_weights = (
        hasattr(encoder, "_ct_W_list") and encoder._ct_W_list is not None
    )

    if not has_encrypted_weights and not has_secret_key:
        raise ValueError(
            "run_gat_pipeline_fhe_training: encoder must either have secret key "
            "(plaintext weights) or encrypted weights (from_client_keys_with_encrypted_weights)."
        )

    metrics = MetricsRecorder()
    num_nodes = graph.num_nodes
    edge_index = graph.edge_index
    ct_x_list = graph.node_features_enc

    def _decrypt_scalar_ct_list(ct_list: list) -> np.ndarray:
        vals = np.zeros(len(ct_list), dtype=np.float64)
        for i, ct in enumerate(ct_list):
            pt = encoder.crypto_context.Decrypt(encoder.keys.secretKey, ct)
            pt.SetLength(1)
            v = pt.GetCKKSPackedValue()[0]
            vals[i] = float(np.real(complex(v).real))
        return vals

    def _encrypt_scalar_list_to_ckks_ct_list(values: np.ndarray) -> list:
        ct_list = []
        for v in np.asarray(values, dtype=np.float64):
            packed = [float(v)] * encoder.batch_size
            pt = encoder.crypto_context.MakeCKKSPackedPlaintext(packed)
            ct_list.append(encoder.crypto_context.Encrypt(encoder.keys.publicKey, pt))
        return ct_list

    cfg = GATRunConfig(
        step1_linear="enc",
        step2_attention="enc",
        step3_leakyrelu="enc",
        step4_softmax="enc",
        step5_aggregation="enc",
        step3_evalsign_mode="ckks_poly_leakyrelu",
        use_cggi=False,
        return_encrypted=True,
        print_metrics=False,
    )

    sigmoid_coeffs = get_sigmoid_chebyshev_coefficients(domain_low=-5.0, domain_high=5.0, degree=7)
    cc = encoder.crypto_context

    for epoch in range(num_epochs):
        with metrics.step(f"epoch_{epoch+1}_forward", encrypted=True):
            # Forward pass (reuse run_gat_pipeline logic)
            out_cts, ct_h_list, ct_alpha_list = _run_forward_with_intermediates(
                encoder, graph, cfg, metrics
            )

        with metrics.step(f"epoch_{epoch+1}_grad_out", encrypted=True):
            # grad_out = sigmoid(out) - y  (BCE gradient); zero for non-train nodes
            pt_zero = cc.MakeCKKSPackedPlaintext([0.0] * encoder.batch_size)
            ct_grad_out_list = []
            for i in range(num_nodes):
                if not train_mask[i]:
                    ct_grad_out_list.append(cc.Encrypt(encoder.keys.publicKey, pt_zero))
                else:
                    ct_sig = cc.EvalChebyshevSeries(out_cts[i], sigmoid_coeffs, -5.0, 5.0)
                    ct_grad = cc.EvalSub(ct_sig, ct_labels[i])
                    ct_grad_out_list.append(ct_grad)

        with metrics.step(f"epoch_{epoch+1}_aggregate_backward", encrypted=True):
            # grad_h[src] += alpha[edge] * grad_out[dst]  (aggregation backward)
            pt_zero = cc.MakeCKKSPackedPlaintext([0.0] * encoder.batch_size)
            ct_grad_h = [
                [cc.Encrypt(encoder.keys.publicKey, pt_zero) for _ in range(encoder.out_channels)]
                for _ in range(num_nodes)
            ]
            for e in range(edge_index.shape[1]):
                src, dst = edge_index[0, e], edge_index[1, e]
                term = cc.EvalMult(ct_alpha_list[e], ct_grad_out_list[dst])
                for k in range(encoder.out_channels):
                    ct_grad_h[src][k] = cc.EvalAdd(ct_grad_h[src][k], term)

        with metrics.step(f"epoch_{epoch+1}_linear_backward", encrypted=True):
            # grad_W[k] = sum_n (x_n * grad_h_n[k]) for train nodes only
            F_in = encoder.in_channels
            F_out = encoder.out_channels
            grad_W_enc = []
            for k in range(F_out):
                acc = None
                for n in range(num_nodes):
                    if not train_mask[n]:
                        continue
                    term = cc.EvalMult(ct_x_list[n], ct_grad_h[n][k])
                    if acc is None:
                        acc = term
                    else:
                        acc = cc.EvalAdd(acc, term)
                grad_W_enc.append(acc)

        with metrics.step(f"epoch_{epoch+1}_weight_update", encrypted=has_encrypted_weights):
            F_out = encoder.out_channels
            pt_lr = cc.MakeCKKSPackedPlaintext([lr] * encoder.batch_size)
            if has_encrypted_weights:
                # FHE update: ct_W_new = ct_W_old - lr * ct_grad_W (no secret key)
                for k in range(F_out):
                    if grad_W_enc[k] is None:
                        continue
                    ct_scaled_grad = cc.EvalMult(grad_W_enc[k], pt_lr)
                    encoder._ct_W_list[k] = cc.EvalSub(encoder._ct_W_list[k], ct_scaled_grad)
            else:
                # Plaintext update (legacy: server has secret key)
                F_in = encoder.in_channels
                grad_W_plain = np.zeros((F_out, F_in), dtype=np.float64)
                for k in range(F_out):
                    if grad_W_enc[k] is None:
                        continue
                    pt = cc.Decrypt(encoder.keys.secretKey, grad_W_enc[k])
                    pt.SetLength(F_in)
                    vals = pt.GetCKKSPackedValue()
                    grad_W_plain[k] = np.real([complex(v).real for v in vals[:F_in]])
                encoder._W = encoder._W.astype(np.float64) - lr * grad_W_plain

        if print_metrics and (epoch + 1) % 1 == 0:
            if has_encrypted_weights:
                print(f"  FHE epoch {epoch+1}/{num_epochs} (encrypted weights)")
            else:
                print(f"  FHE epoch {epoch+1}/{num_epochs}  |W|={np.linalg.norm(encoder._W):.4f}")

    cfg_final = GATRunConfig(
        step1_linear="enc", step2_attention="enc", step3_leakyrelu="enc",
        step4_softmax="enc", step5_aggregation="enc",
        step3_evalsign_mode="ckks_poly_leakyrelu", use_cggi=False,
        return_encrypted=True, print_metrics=print_metrics,
    )
    out_cts, _, _ = _run_forward_with_intermediates(encoder, graph, cfg_final, metrics)
    return out_cts, metrics.to_dict()


def _run_forward_with_intermediates(
    encoder: GATEncoderFHE,
    graph: FHEGraph,
    cfg: GATRunConfig,
    metrics: "MetricsRecorder",
) -> tuple[list[Any], list[list[Any]], list[Any]]:
    """Run forward pass and return (out_cts, ct_h_list, ct_alpha_list) for training."""
    num_nodes = graph.num_nodes
    edge_index = graph.edge_index
    ct_x_list = graph.node_features_enc

    with metrics.step("1_linear", encrypted=True):
        ct_h_list = [encoder._matmul_ckks_dispatch(ct_x) for ct_x in ct_x_list]

    with metrics.step("2_attention", encrypted=True):
        attention_scores = encoder.attention_scores_ckks(ct_h_list, edge_index, num_nodes)

    with metrics.step("3_leakyrelu", encrypted=True):
        from .fhe_utils import get_leaky_relu_chebyshev_coefficients
        coeffs = get_leaky_relu_chebyshev_coefficients(
            negative_slope=cfg.negative_slope, domain_low=-3.0, domain_high=3.0, degree=7
        )
        e_after = []
        for ct_e in attention_scores:
            ct_leaky = encoder.crypto_context.EvalChebyshevSeries(ct_e, coeffs, -3.0, 3.0)
            e_after.append(ct_leaky)

    with metrics.step("4_softmax", encrypted=True):
        ct_alpha_list = encoder.softmax_ckks_chebyshev(e_after, edge_index, num_nodes)

    with metrics.step("5_aggregation", encrypted=True):
        ct_h_packed = []
        for i in range(num_nodes):
            ct_packed = ct_h_list[i][0]
            for k in range(1, encoder.out_channels):
                ct_rot = encoder.crypto_context.EvalRotate(ct_h_list[i][k], k)
                ct_packed = encoder.crypto_context.EvalAdd(ct_packed, ct_rot)
            ct_h_packed.append(ct_packed)
        out_cts = encoder.aggregate_fhe(ct_h_packed, edge_index, ct_alpha_list, num_nodes)

    return out_cts, ct_h_list, ct_alpha_list

