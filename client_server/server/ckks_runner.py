"""
CKKS-only FHE training pipeline (streaming version).
Forward + streaming softmax + streaming backward + weight update.
Memory optimized.
"""

from __future__ import annotations
from typing import Any
import numpy as np
import gc

from .encoder_ckks import GATEncoderCKKS
from .fhe_graph import FHEGraph
from .fhe_utils_ckks import (
    encrypted_reciprocal_newton_raphson,
    get_leaky_relu_chebyshev_coefficients,
    get_sigmoid_chebyshev_coefficients,
)
from .metrics import MetricsRecorder


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
        domain_low=-5.0, domain_high=5.0, degree=3
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

        gc.collect()

        if bootstrap_weights:
            with metrics.step(f"epoch_{epoch+1}_bootstrap_weights", encrypted=True):
                for k in range(encoder.out_channels):
                    encoder._ct_W_list[k] = cc.EvalBootstrap(
                        encoder._ct_W_list[k]
                    )

        gc.collect()

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

    # ---- LINEAR ----
    with metrics.step("1_linear", encrypted=True):
        ct_h_list = [encoder._matmul_ckks_dispatch(ct_x) for ct_x in ct_x_list]

    # ---- ATTENTION ----
    with metrics.step("2_attention", encrypted=True):
        attention_scores = encoder.attention_scores_ckks(
            ct_h_list, edge_index, num_nodes
        )

    # ---- LEAKY RELU ----
    with metrics.step("3_leakyrelu", encrypted=True):
        coeffs = get_leaky_relu_chebyshev_coefficients(
            negative_slope=encoder.negative_slope,
            domain_low=-3.0,
            domain_high=3.0,
            degree=3,
        )
        e_after = [
            cc.EvalChebyshevSeries(ct_e, coeffs, -3.0, 3.0)
            for ct_e in attention_scores
        ]

    # ---- PACK NODE EMBEDDINGS ----
    ct_h_packed = []
    for i in range(num_nodes):
        ct_packed = ct_h_list[i][0]
        for k in range(1, encoder.out_channels):
            ct_rot = cc.EvalRotate(ct_h_list[i][k], k)
            ct_packed = cc.EvalAdd(ct_packed, ct_rot)
        ct_h_packed.append(ct_packed)

    del ct_h_list
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

        initial_guess = 1.0 / max(1.0, len(ct_exp_local) * 0.5)

        ct_recip = encrypted_reciprocal_newton_raphson(
            cc,
            ct_sum,
            num_iterations=1,
            initial_guess=initial_guess,
            slots=encoder.slots,
        )

        acc = None

        for k, edge_idx in enumerate(edge_indices):

            alpha_ij = cc.EvalMult(ct_exp_local[k], ct_recip)
            src = edge_index[0, edge_idx]

            term = cc.EvalMult(ct_h_packed[src], alpha_ij)

            if acc is None:
                acc = term
            else:
                acc = cc.EvalAdd(acc, term)

            # ---- STREAM BACKWARD ----
            if training and train_mask[t]:
                ct_sig = cc.EvalChebyshevSeries(
                    acc, sigmoid_coeffs, -5.0, 5.0
                )
                ct_grad_out = cc.EvalSub(ct_sig, ct_labels[t])

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

        # Guard: if acc is still None despite non-empty edge_indices (e.g. all
        # src indices were out-of-range), fall back to an encrypted zero so we
        # never append None to out_cts and crash downstream decryption.
        if acc is None:
            pt_zero = cc.MakeCKKSPackedPlaintext([0.0] * encoder.slots)
            acc = cc.Encrypt(encoder.keys.publicKey, pt_zero)

        out_cts.append(acc)

        del ct_exp_local
        gc.collect()

    del e_after
    del ct_h_packed
    gc.collect()

    if training:
        return out_cts, grad_W_enc
    else:
        return out_cts, None
