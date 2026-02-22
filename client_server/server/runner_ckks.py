"""
CKKS-only FHE training pipeline: forward + homomorphic gradient + weight update.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .encoder_ckks import GATEncoderCKKS
from .fhe_graph import FHEGraph
from .fhe_utils_ckks import get_leaky_relu_chebyshev_coefficients, get_sigmoid_chebyshev_coefficients
from .metrics import MetricsRecorder


def run_gat_forward_only(
    *,
    encoder: GATEncoderCKKS,
    graph: FHEGraph,
    print_metrics: bool = False,
) -> tuple[list[Any], dict[str, Any]]:
    """
    Single forward pass (inference only). No training, no gradient, no weight update.
    Returns (out_cts, metrics_dict).
    """
    metrics = MetricsRecorder()
    out_cts, _, _ = _run_forward_with_intermediates(encoder, graph, metrics)
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
    """
    FHE training: forward, homomorphic gradient, encrypted weight update.
    Encoder must be from_client_keys_with_encrypted_weights (no secret key on server).
    Returns (out_cts, metrics_dict, ct_W_list_trained) - trained weights for inference.
    """
    metrics = MetricsRecorder()
    num_nodes = graph.num_nodes
    edge_index = graph.edge_index
    ct_x_list = graph.node_features_enc
    cc = encoder.crypto_context
    # Use degree=3 to stay within multiplicative depth budget (degree 7 can exceed 50 levels over epochs)
    sigmoid_coeffs = get_sigmoid_chebyshev_coefficients(domain_low=-5.0, domain_high=5.0, degree=3)

    for epoch in range(num_epochs):
        with metrics.step(f"epoch_{epoch+1}_forward", encrypted=True):
            out_cts, ct_h_list, ct_alpha_list = _run_forward_with_intermediates(
                encoder, graph, metrics
            )

        with metrics.step(f"epoch_{epoch+1}_grad_out", encrypted=True):
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

        with metrics.step(f"epoch_{epoch+1}_weight_update", encrypted=True):
            F_out = encoder.out_channels
            pt_lr = cc.MakeCKKSPackedPlaintext([lr] * encoder.batch_size)
            for k in range(F_out):
                if grad_W_enc[k] is None:
                    continue
                ct_scaled_grad = cc.EvalMult(grad_W_enc[k], pt_lr)
                encoder._ct_W_list[k] = cc.EvalSub(encoder._ct_W_list[k], ct_scaled_grad)

        # Bootstrap only the encrypted weights after each epoch (refreshes levels; do NOT bootstrap activations/gradients)
        if bootstrap_weights:
            with metrics.step(f"epoch_{epoch+1}_bootstrap_weights", encrypted=True):
                for k in range(encoder.out_channels):
                    encoder._ct_W_list[k] = cc.EvalBootstrap(encoder._ct_W_list[k])
            if print_metrics:
                print(f"  Bootstrap weights (refresh levels) done for epoch {epoch+1}")

        if print_metrics and (epoch + 1) % 1 == 0:
            print(f"  FHE epoch {epoch+1}/{num_epochs} (encrypted weights)")

    out_cts, _, _ = _run_forward_with_intermediates(encoder, graph, metrics)
    # Return trained encrypted weights so client can use them for inference
    ct_W_list_trained = list(encoder._ct_W_list)
    return out_cts, metrics.to_dict(), ct_W_list_trained


def _run_forward_with_intermediates(
    encoder: GATEncoderCKKS,
    graph: FHEGraph,
    metrics: MetricsRecorder,
) -> tuple[list[Any], list[list[Any]], list[Any]]:
    """CKKS-only forward; returns (out_cts, ct_h_list, ct_alpha_list)."""
    num_nodes = graph.num_nodes
    edge_index = graph.edge_index
    ct_x_list = graph.node_features_enc

    with metrics.step("1_linear", encrypted=True):
        ct_h_list = [encoder._matmul_ckks_dispatch(ct_x) for ct_x in ct_x_list]

    with metrics.step("2_attention", encrypted=True):
        attention_scores = encoder.attention_scores_ckks(ct_h_list, edge_index, num_nodes)

    with metrics.step("3_leakyrelu", encrypted=True):
        # degree=3 to reduce multiplicative depth (degree 7 exceeds budget over 2 epochs)
        coeffs = get_leaky_relu_chebyshev_coefficients(
            negative_slope=encoder.negative_slope, domain_low=-3.0, domain_high=3.0, degree=3
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
