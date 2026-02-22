#!/usr/bin/env python3
"""
CKKS-only FHE GAT server: exposes compute API for FHE training.
No secret key on server; client sends encrypted weights, features, labels.

API (in-process):
  from client_server.server import compute_fhe_training
  out_cts, metrics = compute_fhe_training(crypto_context=..., public_key=..., ...)

HTTP API: POST /compute with JSON body describing payload format.
For cross-process, client can POST base64-encoded pickle of the same inputs;
server returns base64-encoded pickle of (out_cts, metrics_dict).
"""

from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from .encoder_ckks import GATEncoderCKKS
from .fhe_graph import FHEGraph
from .runner_ckks import run_gat_pipeline_fhe_training, run_gat_forward_only


def compute_fhe_training(
    *,
    crypto_context: Any,
    public_key: Any,
    in_channels: int,
    out_channels: int,
    batch_size: int,
    ct_W_list: list[Any],
    a: Any,
    negative_slope: float,
    num_nodes: int,
    edge_index: Any,
    node_features_enc: list[Any],
    ct_labels: list[Any],
    train_mask: Any,
    num_epochs: int = 3,
    lr: float = 0.01,
    print_metrics: bool = True,
    bootstrap_weights: bool = True,
) -> tuple[list[Any], dict[str, Any], list[Any]]:
    """
    Run FHE training on server using TRAIN DATA ONLY. All inputs from client;
    server never has the secret key.

    Returns:
        (out_cts_train, metrics_dict, ct_W_list_trained) - train outputs, metrics,
        and trained encrypted weights for use in compute_forward_only.
    """
    import numpy as np
    a_np = np.asarray(a, dtype=np.float64)
    train_mask_np = np.asarray(train_mask, dtype=bool)
    edge_index_np = np.asarray(edge_index, dtype=np.int64)
    if edge_index_np.shape[0] != 2:
        edge_index_np = edge_index_np.T

    graph = FHEGraph.from_encrypted(
        num_nodes=num_nodes,
        in_channels=in_channels,
        edge_index=edge_index_np,
        node_features_enc=node_features_enc,
    )
    encoder = GATEncoderCKKS.from_client_keys_with_encrypted_weights(
        crypto_context=crypto_context,
        public_key=public_key,
        in_channels=in_channels,
        out_channels=out_channels,
        batch_size=batch_size,
        ct_W_list=ct_W_list,
        a=a_np,
        negative_slope=negative_slope,
    )
    out_cts, metrics_dict, ct_W_list_trained = run_gat_pipeline_fhe_training(
        encoder=encoder,
        graph=graph,
        ct_labels=ct_labels,
        train_mask=train_mask_np,
        num_epochs=num_epochs,
        lr=lr,
        print_metrics=print_metrics,
        bootstrap_weights=bootstrap_weights,
    )
    return out_cts, metrics_dict, ct_W_list_trained


def compute_forward_only(
    *,
    crypto_context: Any,
    public_key: Any,
    in_channels: int,
    out_channels: int,
    batch_size: int,
    ct_W_list: list[Any],
    a: Any,
    negative_slope: float,
    num_nodes: int,
    edge_index: Any,
    node_features_enc: list[Any],
    print_metrics: bool = False,
) -> tuple[list[Any], dict[str, Any]]:
    """
    Run one forward pass (inference only). Use trained ct_W_list from compute_fhe_training.
    Graph can include test nodes; no labels or training. Returns (out_cts, metrics_dict).
    """
    import numpy as np
    a_np = np.asarray(a, dtype=np.float64)
    edge_index_np = np.asarray(edge_index, dtype=np.int64)
    if edge_index_np.shape[0] != 2:
        edge_index_np = edge_index_np.T

    graph = FHEGraph.from_encrypted(
        num_nodes=num_nodes,
        in_channels=in_channels,
        edge_index=edge_index_np,
        node_features_enc=node_features_enc,
    )
    encoder = GATEncoderCKKS.from_client_keys_with_encrypted_weights(
        crypto_context=crypto_context,
        public_key=public_key,
        in_channels=in_channels,
        out_channels=out_channels,
        batch_size=batch_size,
        ct_W_list=ct_W_list,
        a=a_np,
        negative_slope=negative_slope,
    )
    out_cts, metrics_dict = run_gat_forward_only(
        encoder=encoder,
        graph=graph,
        print_metrics=print_metrics,
    )
    return out_cts, metrics_dict


# ---------- HTTP server: accepts serialized JSON payload (openfhe_serializer) ----------

def _send_json(self: BaseHTTPRequestHandler, status: int, payload: Dict[str, Any]) -> None:
    body = json.dumps(payload).encode("utf-8")
    self.send_response(status)
    self.send_header("Content-Type", "application/json")
    self.send_header("Content-Length", str(len(body)))
    self.end_headers()
    self.wfile.write(body)


def _handle_post(self: BaseHTTPRequestHandler, path: str) -> bool:
    from client_server.openfhe_serializer import (
        deserialize_train_payload,
        deserialize_infer_payload,
        serialize_train_result,
        serialize_infer_result,
    )
    length = int(self.headers.get("Content-Length", "0"))
    raw = self.rfile.read(length)
    try:
        data = json.loads(raw.decode("utf-8") or "{}")
    except json.JSONDecodeError:
        _send_json(self, 400, {"error": "invalid JSON"})
        return True
    try:
        if path == "/train":
            train_data = data.get("train_payload")
            if not train_data:
                _send_json(self, 400, {"error": "missing 'train_payload' (serialized train kwargs)"})
                return True
            payload = deserialize_train_payload(train_data)
            out_cts, metrics_dict, ct_W_list = compute_fhe_training(**payload)
            result = serialize_train_result(out_cts, metrics_dict, ct_W_list)
            _send_json(self, 200, {"status": "ok", "result": result})
        else:  # /infer
            infer_data = data.get("infer_payload")
            if not infer_data:
                _send_json(self, 400, {"error": "missing 'infer_payload' (serialized infer kwargs)"})
                return True
            payload = deserialize_infer_payload(infer_data)
            out_cts, metrics_dict = compute_forward_only(**payload)
            result = serialize_infer_result(out_cts, metrics_dict)
            _send_json(self, 200, {"status": "ok", "result": result})
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        err_msg = f"{type(e).__name__}: {e}\n{tb}"
        try:
            _send_json(self, 500, {"error": f"{type(e).__name__}: {e}", "traceback": tb})
        except Exception:
            _send_json(self, 500, {"error": str(e)})
        import sys
        print(err_msg, file=sys.stderr)
    return True


class RequestHandler(BaseHTTPRequestHandler):
    def _send_json(self, status: int, payload: Dict[str, Any]) -> None:
        _send_json(self, status, payload)

    def do_POST(self) -> None:
        if self.path == "/train":
            _handle_post(self, "/train")
        elif self.path == "/infer":
            _handle_post(self, "/infer")
        else:
            self._send_json(404, {"error": "not found"})

    def log_message(self, format: str, *args: Any) -> None:
        pass


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="CKKS-only FHE GAT server")
    ap.add_argument("--host", default="127.0.0.1", help="Bind address (use 0.0.0.0 for LAN)")
    ap.add_argument("--port", type=int, default=8080, help="Port")
    a = ap.parse_args()
    host, port = a.host, a.port
    print(f"CKKS-only FHE GAT server on http://{host}:{port}")
    print("  POST /train  -> (out_cts_train, metrics, ct_W_trained)")
    print("  POST /infer  -> (out_cts, metrics) using ct_W_trained")
    server = HTTPServer((host, port), RequestHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
