"""
Tests for client_server.openfhe_serializer JSON payload transport.

Goal: ensure that the client can serialize a payload to JSON, and the server
can deserialize it back into usable OpenFHE objects (across a JSON boundary).
"""

from __future__ import annotations

import json
import threading
from http.server import HTTPServer

import numpy as np
import pytest

from gat_encoder_fhe import openfhe_available


pytestmark = pytest.mark.skipif(
    not openfhe_available(),
    reason="OpenFHE not installed (pip install openfhe)",
)


def _make_toy_train_payload():
    """Smallest train payload that still exercises CC/PK/eval keys/ct lists."""
    import openfhe
    from client_server.openfhe_serializer import serialize_train_payload

    in_channels = 2
    out_channels = 1
    batch_size = 8
    num_nodes = 2
    edge_index = np.array([[0, 1], [1, 0]], dtype=np.int64)
    train_mask = np.array([True, True], dtype=bool)

    # Build CKKS crypto context + keys (minimal features needed by serializer keys)
    params = openfhe.CCParamsCKKSRNS()
    params.SetMultiplicativeDepth(10)
    params.SetScalingModSize(50)
    params.SetFirstModSize(60)
    params.SetBatchSize(batch_size)
    cc = openfhe.GenCryptoContext(params)
    cc.Enable(openfhe.PKESchemeFeature.PKE)
    cc.Enable(openfhe.PKESchemeFeature.KEYSWITCH)
    cc.Enable(openfhe.PKESchemeFeature.LEVELEDSHE)
    cc.Enable(openfhe.PKESchemeFeature.ADVANCEDSHE)

    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey)
    cc.EvalRotateKeyGen(keys.secretKey, list(range(1, 2 * batch_size + 1)))

    # Encrypted weights (one row)
    W_init = np.array([[0.1, -0.2]], dtype=np.float64)
    ct_W_list = []
    for k in range(out_channels):
        row = np.zeros(batch_size, dtype=np.float64)
        row[:in_channels] = W_init[k]
        pt = cc.MakeCKKSPackedPlaintext(row.tolist())
        ct_W_list.append(cc.Encrypt(keys.publicKey, pt))

    # Encrypted node features
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    node_features_enc = []
    for i in range(num_nodes):
        row = np.zeros(batch_size, dtype=np.float64)
        row[:in_channels] = x[i]
        pt = cc.MakeCKKSPackedPlaintext(row.tolist())
        node_features_enc.append(cc.Encrypt(keys.publicKey, pt))

    # Encrypted labels (binary)
    y = np.array([1.0, 0.0], dtype=np.float64)
    ct_labels = []
    for i in range(num_nodes):
        pt = cc.MakeCKKSPackedPlaintext([float(y[i])] * batch_size)
        ct_labels.append(cc.Encrypt(keys.publicKey, pt))

    a = np.array([0.1, 0.2], dtype=np.float64)  # concat_dim = 2*out_channels = 2

    payload = {
        "crypto_context": cc,
        "public_key": keys.publicKey,
        "ct_W_list": ct_W_list,
        "node_features_enc": node_features_enc,
        "ct_labels": ct_labels,
        "in_channels": in_channels,
        "out_channels": out_channels,
        "batch_size": batch_size,
        "a": a,
        "negative_slope": 0.2,
        "num_nodes": num_nodes,
        "edge_index": edge_index,
        "train_mask": train_mask,
        "num_epochs": 1,
        "lr": 0.01,
        "print_metrics": False,
        "bootstrap_weights": False,
    }
    # ensure it's JSON serializable
    serialized = serialize_train_payload(payload)
    json.dumps({"train_payload": serialized})
    return payload, serialized


def test_serializer_roundtrip_train_payload_json():
    """Serialize->JSON->deserialize gives usable OpenFHE objects."""
    from client_server.openfhe_serializer import deserialize_train_payload

    _payload, serialized = _make_toy_train_payload()

    # Simulate HTTP boundary: JSON encode/decode
    wire = json.loads(json.dumps(serialized))
    decoded = deserialize_train_payload(wire)

    # Basic structural checks
    assert decoded["in_channels"] == 2
    assert decoded["out_channels"] == 1
    assert decoded["batch_size"] == 8
    assert decoded["num_nodes"] == 2
    assert len(decoded["ct_W_list"]) == 1
    assert len(decoded["node_features_enc"]) == 2
    assert len(decoded["ct_labels"]) == 2

    # Usability check: can perform a simple EvalAdd on decoded objects
    cc = decoded["crypto_context"]
    ct0 = decoded["node_features_enc"][0]
    ct1 = decoded["node_features_enc"][1]
    _ = cc.EvalAdd(ct0, ct1)


def test_http_server_accepts_train_payload(tmp_path):
    """
    End-to-end: start the HTTP server and POST /train.
    This validates deserialize_train_payload runs successfully in a separate handler.
    """
    import urllib.request

    from client_server.openfhe_serializer import serialize_train_payload
    from client_server.server.server import RequestHandler

    payload, _serialized = _make_toy_train_payload()
    body = json.dumps({"train_payload": serialize_train_payload(payload)}).encode("utf-8")

    httpd = HTTPServer(("127.0.0.1", 0), RequestHandler)
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()

    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/train",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            out = json.loads(resp.read().decode("utf-8"))
        assert out.get("status") == "ok"
        assert "result" in out
        assert "out_cts_b64" in out["result"]
    finally:
        httpd.shutdown()
