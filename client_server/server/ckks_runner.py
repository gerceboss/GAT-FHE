"""
CKKS-only FHE training pipeline (streaming version).
Forward + streaming softmax + streaming backward + weight update.
Memory optimized.
"""

from __future__ import annotations
from typing import Any
import numpy as np
import gc
import os

from .encoder_ckks import GATEncoderCKKS
from .fhe_graph import FHEGraph
from .fhe_utils_ckks import (
    early_bootstrap_enabled,
    early_bootstrap_threshold_for_path,
    encrypted_reciprocal_newton_raphson,
    get_leaky_relu_chebyshev_coefficients,
    get_sigmoid_chebyshev_coefficients,
)
from .metrics import MetricsRecorder


def _ct_state(ct: Any) -> dict[str, int]:
    """Best-effort ciphertext state for debugging depth/noise."""
    out: dict[str, int] = {}
    try:
        out["ct_level"] = int(ct.GetLevel())
    except Exception:
        out["ct_level"] = -1
    try:
        out["ct_noise_scale_deg"] = int(ct.GetNoiseScaleDeg())
    except Exception:
        out["ct_noise_scale_deg"] = -1
    return out


def _debug_enabled() -> bool:
    return os.environ.get("GAT_FHE_DEBUG_CT", "").strip() not in ("", "0", "false", "False")


def _debug_print_ct(prefix: str, ct: Any) -> None:
    if not _debug_enabled():
        return
    s = _ct_state(ct)
    print(f"[ct] {prefix} level={s['ct_level']} noise_scale_deg={s['ct_noise_scale_deg']}")


def _bootstrap_after_pack_enabled() -> bool:
    """
    After ct_h_packed is built, EvalBootstrap each packed node embedding (num_nodes calls).
    Default: on. Disable with GAT_FHE_BOOTSTRAP_AFTER_PACK=0|false|no (faster; less refresh).
    """
    v = os.environ.get("GAT_FHE_BOOTSTRAP_AFTER_PACK", "").strip().lower()
    if v in ("0", "false", "no"):
        return False
    return True


def run_gat_forward_only(
    *,
    encoder: GATEncoderCKKS,
    graph: FHEGraph,
    print_metrics: bool = False,
) -> tuple[list[Any], dict[str, Any]]:

    metrics = MetricsRecorder()
    out_res, _ = _run_forward_with_intermediates(
        encoder, graph, metrics, training=False
    )
    # _run_forward_with_intermediates returns (out_cts, grad_W_enc_or_None)
    out_cts = out_res
    return out_cts, metrics.to_dict()


def run_gat_pipeline_fhe_training(
    *,
    encoder: GATEncoderCKKS,
    graph: FHEGraph,
    ct_labels: list[Any],
    train_mask: np.ndarray,
    num_epochs: int = 3,
    lr: float = 0.01,
    print_metrics: bool = True,
    bootstrap_weights: bool = True,
) -> tuple[list[Any], dict[str, Any], list[Any]]:

    metrics = MetricsRecorder()
    num_nodes = graph.num_nodes
    edge_index = graph.edge_index
    ct_x_list = graph.node_features_enc
    cc = encoder.crypto_context

    sigmoid_coeffs = get_sigmoid_chebyshev_coefficients(
        domain_low=-3.0, domain_high=3.0, degree=2
    )

    for epoch in range(num_epochs):

        with metrics.step(f"epoch_{epoch+1}_forward_backward_stream", encrypted=True):

            out_cts, grad_W_enc = _run_forward_with_intermediates(
                encoder,
                graph,
                metrics,
                training=True,
                ct_labels=ct_labels,
                train_mask=train_mask,
                sigmoid_coeffs=sigmoid_coeffs,
            )

        # ---- weight update ----
        with metrics.step(f"epoch_{epoch+1}_weight_update", encrypted=True):

            pt_lr = cc.MakeCKKSPackedPlaintext([lr] * encoder.slots)

            for k in range(encoder.out_channels):
                if grad_W_enc[k] is None:
                    continue
                ct_scaled_grad = cc.EvalMult(grad_W_enc[k], pt_lr)
                encoder._ct_W_list[k] = cc.EvalSub(
                    encoder._ct_W_list[k], ct_scaled_grad
                )
                del ct_scaled_grad
            del pt_lr, grad_W_enc
            gc.collect()

        gc.collect()

        # ---- bootstrap weights AFTER update so next epoch starts fresh ----
        if bootstrap_weights:
            with metrics.step(f"epoch_{epoch+1}_bootstrap_weights", encrypted=True):
                for k in range(encoder.out_channels):
                    _debug_print_ct(f"epoch{epoch+1} W{k} pre_bootstrap", encoder._ct_W_list[k])
                    encoder._ct_W_list[k] = cc.EvalBootstrap(encoder._ct_W_list[k])
                    _debug_print_ct(f"epoch{epoch+1} W{k} post_bootstrap", encoder._ct_W_list[k])
            gc.collect()

        # if bootstrap_weights:
        #     with metrics.step(f"epoch_{epoch+1}_bootstrap_weights", encrypted=True):
        #         for k in range(encoder.out_channels):
        #             encoder._ct_W_list[k] = cc.EvalBootstrap(
        #                 encoder._ct_W_list[k]
        #             )
        #     gc.collect()

        if print_metrics:
            print(f"  FHE epoch {epoch+1}/{num_epochs} (streaming)")

    out_cts, _ = _run_forward_with_intermediates(
        encoder, graph, metrics, training=False
    )

    ct_W_list_trained = list(encoder._ct_W_list)
    return out_cts, metrics.to_dict(), ct_W_list_trained


def _run_forward_with_intermediates(
    encoder: GATEncoderCKKS,
    graph: FHEGraph,
    metrics: MetricsRecorder,
    training: bool = False,
    ct_labels=None,
    train_mask=None,
    sigmoid_coeffs=None,
):

    num_nodes = graph.num_nodes
    edge_index = graph.edge_index
    ct_x_list = graph.node_features_enc
    cc = encoder.crypto_context
    _thr_early = early_bootstrap_threshold_for_path(training)

    # ---- LINEAR ----
    with metrics.step("1_linear", encrypted=True):
        ct_h_list = [encoder._matmul_ckks_dispatch(ct_x) for ct_x in ct_x_list]
    if _debug_enabled() and ct_h_list and ct_h_list[0]:
        _debug_print_ct("after linear h[0][0]", ct_h_list[0][0])

    # ---- ATTENTION ----
    with metrics.step("2_attention", encrypted=True):
        attention_scores = encoder.attention_scores_ckks(
            ct_h_list, edge_index, num_nodes, training=training
        )
    if _debug_enabled() and attention_scores:
        _debug_print_ct("after attention e[0]", attention_scores[0])

    # ---- LEAKY RELU ----
    # Attention scores are per-edge (global order). LeakyReLU adds depth on every edge
    # before softmax groups by destination node — refresh here when enabled so exp/NR
    # downstream does not stack on depleted ciphertexts.
    with metrics.step("3_leakyrelu", encrypted=True):
        coeffs = get_leaky_relu_chebyshev_coefficients(
            negative_slope=encoder.negative_slope,
            domain_low=-3.0,
            domain_high=3.0,
            degree=2,
        )
        e_after = []
        for ct_e in attention_scores:
            e1 = cc.EvalChebyshevSeries(ct_e, coeffs, -3.0, 3.0)
            if early_bootstrap_enabled():
                try:
                    if int(e1.GetLevel()) >= _thr_early:
                        e1 = cc.EvalBootstrap(e1)
                except Exception:
                    pass
            e_after.append(e1)
    del attention_scores
    gc.collect()
    if _debug_enabled() and e_after:
        _debug_print_ct("after leakyrelu e_after[0]", e_after[0])

    # ---- PACK NODE EMBEDDINGS ----
    ct_h_packed = []
    for i in range(num_nodes):
        ct_packed = ct_h_list[i][0]
        for k in range(1, encoder.out_channels):
            ct_rot = cc.EvalRotate(ct_h_list[i][k], k)
            ct_packed = cc.EvalAdd(ct_packed, ct_rot)
            del ct_rot
            gc.collect()
        ct_h_packed.append(ct_packed)
        del ct_packed
        gc.collect()

    # ---- REFRESH PACKED EMBEDDINGS (default on; targeted decode-margin vs softmax path) ----
    if _bootstrap_after_pack_enabled():
        with metrics.step("4_bootstrap_after_pack", encrypted=True):
            for i in range(num_nodes):
                if _debug_enabled():
                    _debug_print_ct(f"ct_h_packed[{i}] pre_bootstrap", ct_h_packed[i])
                ct_h_packed[i] = cc.EvalBootstrap(ct_h_packed[i])
                if _debug_enabled():
                    _debug_print_ct(f"ct_h_packed[{i}] post_bootstrap", ct_h_packed[i])
            gc.collect()

    del ct_h_list, coeffs
    gc.collect()

    # ---- STREAM SOFTMAX + (OPTIONAL) BACKWARD ----
    out_cts = []
    grad_W_enc = [None] * encoder.out_channels if training else None

    for t in range(num_nodes):

        mask = edge_index[1] == t
        edge_indices = np.where(mask)[0]

        if len(edge_indices) == 0:
            pt_zero = cc.MakeCKKSPackedPlaintext([0.0] * encoder.slots)
            ct_zero = cc.Encrypt(encoder.keys.publicKey, pt_zero)
            out_cts.append(ct_zero)
            continue

        # compute exp locally
        ct_exp_local = []
        for idx in edge_indices:
            ct_exp = cc.EvalChebyshevSeries(
                e_after[idx], [1.0, 1.0, 0.5], -1.0, 1.0
            )
            ct_exp_local.append(ct_exp)

        ct_sum = ct_exp_local[0]
        for ct in ct_exp_local[1:]:
            ct_sum = cc.EvalAdd(ct_sum, ct)
            if early_bootstrap_enabled():
                try:
                    if int(ct_sum.GetLevel()) >= _thr_early:
                        ct_sum = cc.EvalBootstrap(ct_sum)
                except Exception:
                    pass

        if early_bootstrap_enabled():
            try:
                if int(ct_sum.GetLevel()) >= _thr_early:
                    ct_sum = cc.EvalBootstrap(ct_sum)
            except Exception:
                pass

        initial_guess = 1.0 / max(1.0, len(ct_exp_local) * 0.5)

        ct_recip = encrypted_reciprocal_newton_raphson(
            cc,
            ct_sum,
            num_iterations=1,
            initial_guess=initial_guess,
            slots=encoder.slots,
        )
        del ct_sum

        acc = None

        for k, edge_idx in enumerate(edge_indices):

            alpha_ij = cc.EvalMult(ct_exp_local[k], ct_recip)
            src = edge_index[0, edge_idx]

            term = cc.EvalMult(ct_h_packed[src], alpha_ij)

            if acc is None:
                acc = term
            else:
                acc = cc.EvalAdd(acc, term)
                del term

            if early_bootstrap_enabled():
                try:
                    if int(acc.GetLevel()) >= _thr_early:
                        if _debug_enabled():
                            _debug_print_ct(f"t{t} edge{k} acc pre_early_bootstrap", acc)
                        acc = cc.EvalBootstrap(acc)
                        if _debug_enabled():
                            _debug_print_ct(f"t{t} edge{k} acc post_early_bootstrap", acc)
                except Exception:
                    pass

            # ---- STREAM BACKWARD ----
            if training and train_mask[t]:
                ct_sig = cc.EvalChebyshevSeries(
                    acc, sigmoid_coeffs, -5.0, 5.0
                )
                ct_grad_out = cc.EvalSub(ct_sig, ct_labels[t])
                del ct_sig

                for out_k in range(encoder.out_channels):
                    grad_term = cc.EvalMult(
                        ct_x_list[src], ct_grad_out
                    )
                    if grad_W_enc[out_k] is None:
                        grad_W_enc[out_k] = grad_term
                    else:
                        grad_W_enc[out_k] = cc.EvalAdd(
                            grad_W_enc[out_k], grad_term
                        )
                    del grad_term

        # Guard: if acc is still None despite non-empty edge_indices (e.g. all
        # src indices were out-of-range), fall back to an encrypted zero so we
        # never append None to out_cts and crash downstream decryption.
        if acc is None:
            pt_zero = cc.MakeCKKSPackedPlaintext([0.0] * encoder.slots)
            acc = cc.Encrypt(encoder.keys.publicKey, pt_zero)

        out_cts.append(acc)

        del ct_exp_local, ct_recip
        gc.collect()

    del e_after
    del ct_h_packed
    gc.collect()

    if training:
        return out_cts, grad_W_enc
    else:
        return out_cts, None
