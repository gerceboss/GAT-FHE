"""
Serialize/deserialize OpenFHE objects (CryptoContext, PublicKey, Ciphertext) for HTTP.
Uses OpenFHE's native Serialize(..., BINARY) and Deserialize* from file.
Payloads are converted to/from a JSON-serializable dict of base64-encoded bytes.
"""

from __future__ import annotations

import base64
import json
import os
import tempfile
from typing import Any, Dict, List

import numpy as np

# OpenFHE enum instance for binary serialization (openfhe.BINARY, not openfhe.SERBINARY)
_OPENFHE = None


def _openfhe():
    global _OPENFHE
    if _OPENFHE is None:
        import openfhe
        # Use enum instance: openfhe.BINARY / openfhe.JSON (not SERBINARY/SERJSON class)
        if not hasattr(openfhe, "BINARY"):
            raise ImportError("OpenFHE Python must provide BINARY enum for serialization")
        _OPENFHE = openfhe
    return _OPENFHE


def _bytes_to_file(data: bytes, suffix: str = ".bin") -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    try:
        os.write(fd, data)
    finally:
        os.close(fd)
    return path


def _serialize_obj(obj: Any) -> bytes:
    of = _openfhe()
    return of.Serialize(obj, of.BINARY)


def _deserialize_cc(data: bytes):
    of = _openfhe()
    path = _bytes_to_file(data, ".cc")
    try:
        cc, ok = of.DeserializeCryptoContext(path, of.BINARY)
        if not ok:
            raise RuntimeError("DeserializeCryptoContext failed")
        return cc
    finally:
        os.unlink(path)


def _deserialize_pk(data: bytes):
    of = _openfhe()
    path = _bytes_to_file(data, ".pk")
    try:
        pk, ok = of.DeserializePublicKey(path, of.BINARY)
        if not ok:
            raise RuntimeError("DeserializePublicKey failed")
        return pk
    finally:
        os.unlink(path)


def _deserialize_ct(data: bytes):
    of = _openfhe()
    path = _bytes_to_file(data, ".ct")
    try:
        ct, ok = of.DeserializeCiphertext(path, of.BINARY)
        if not ok:
            raise RuntimeError("DeserializeCiphertext failed")
        return ct
    finally:
        os.unlink(path)


def _serialize_eval_mult_key(crypto_context: Any) -> bytes:
    """Serialize EvalMultKey from the given context to bytes (keyTag default)."""
    of = _openfhe()
    fd, path = tempfile.mkstemp(suffix=".emk")
    os.close(fd)
    try:
        # Static method on context's class; keyTag "" uses default key
        ok = type(crypto_context).SerializeEvalMultKey(path, of.BINARY, "")
        if not ok:
            raise RuntimeError("SerializeEvalMultKey failed")
        with open(path, "rb") as f:
            return f.read()
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def _serialize_eval_automorphism_key(crypto_context: Any) -> bytes:
    """Serialize EvalAutomorphismKey (rotation keys) from the given context to bytes."""
    of = _openfhe()
    fd, path = tempfile.mkstemp(suffix=".erk")
    os.close(fd)
    try:
        ok = type(crypto_context).SerializeEvalAutomorphismKey(path, of.BINARY, "")
        if not ok:
            raise RuntimeError("SerializeEvalAutomorphismKey failed")
        with open(path, "rb") as f:
            return f.read()
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def _deserialize_eval_mult_key_into_context(data: bytes, crypto_context: Any) -> None:
    """Deserialize EvalMultKey from bytes and load into the given context (in-place)."""
    of = _openfhe()
    path = _bytes_to_file(data, ".emk")
    try:
        ok = type(crypto_context).DeserializeEvalMultKey(path, of.BINARY)
        if not ok:
            raise RuntimeError("DeserializeEvalMultKey failed")
    finally:
        os.unlink(path)


def _deserialize_eval_automorphism_key_into_context(data: bytes, crypto_context: Any) -> None:
    """Deserialize EvalAutomorphismKey from bytes and load into the given context (in-place)."""
    of = _openfhe()
    path = _bytes_to_file(data, ".erk")
    try:
        ok = type(crypto_context).DeserializeEvalAutomorphismKey(path, of.BINARY)
        if not ok:
            raise RuntimeError("DeserializeEvalAutomorphismKey failed")
    finally:
        os.unlink(path)


def _serialize_ct_list(ct_list: List[Any]) -> List[str]:
    return [base64.b64encode(_serialize_obj(ct)).decode("ascii") for ct in ct_list]


def _deserialize_ct_list(b64_list: List[str]) -> List[Any]:
    return [_deserialize_ct(base64.b64decode(s)) for s in b64_list]


def _ndarray_to_b64(arr: np.ndarray) -> str:
    data = arr.tobytes()
    meta = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
    return base64.b64encode(json.dumps({"meta": meta, "data": base64.b64encode(data).decode("ascii")}).encode()).decode("ascii")


def _ndarray_from_b64(s: str) -> np.ndarray:
    raw = json.loads(base64.b64decode(s).decode())
    meta = raw["meta"]
    data = base64.b64decode(raw["data"])
    return np.frombuffer(data, dtype=meta["dtype"]).reshape(meta["shape"])


# ---------------------------------------------------------------------------
# Public API: payload dict <-> JSON-serializable dict for HTTP
# ---------------------------------------------------------------------------

def serialize_train_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Convert train payload (with OpenFHE objects) to JSON-serializable dict."""
    cc = payload["crypto_context"]
    return {
        "crypto_context_b64": base64.b64encode(_serialize_obj(cc)).decode("ascii"),
        "public_key_b64": base64.b64encode(_serialize_obj(payload["public_key"])).decode("ascii"),
        "eval_mult_key_b64": base64.b64encode(_serialize_eval_mult_key(cc)).decode("ascii"),
        "eval_automorphism_key_b64": base64.b64encode(_serialize_eval_automorphism_key(cc)).decode("ascii"),
        "ct_W_list_b64": _serialize_ct_list(payload["ct_W_list"]),
        "node_features_enc_b64": _serialize_ct_list(payload["node_features_enc"]),
        "ct_labels_b64": _serialize_ct_list(payload["ct_labels"]),
        "in_channels": int(payload["in_channels"]),
        "out_channels": int(payload["out_channels"]),
        "batch_size": int(payload["batch_size"]),
        "a_b64": _ndarray_to_b64(np.asarray(payload["a"], dtype=np.float64)),
        "negative_slope": float(payload["negative_slope"]),
        "num_nodes": int(payload["num_nodes"]),
        "edge_index_b64": _ndarray_to_b64(np.asarray(payload["edge_index"], dtype=np.int64)),
        "train_mask_b64": _ndarray_to_b64(np.asarray(payload["train_mask"], dtype=bool)),
        "num_epochs": int(payload["num_epochs"]),
        "lr": float(payload["lr"]),
        "print_metrics": bool(payload.get("print_metrics", True)),
        "bootstrap_weights": bool(payload.get("bootstrap_weights", True)),
    }


def deserialize_train_payload(data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert JSON-serializable dict back to train payload with OpenFHE objects."""
    cc = _deserialize_cc(base64.b64decode(data["crypto_context_b64"]))
    pk = _deserialize_pk(base64.b64decode(data["public_key_b64"]))
    _deserialize_eval_mult_key_into_context(base64.b64decode(data["eval_mult_key_b64"]), cc)
    _deserialize_eval_automorphism_key_into_context(base64.b64decode(data["eval_automorphism_key_b64"]), cc)
    return {
        "crypto_context": cc,
        "public_key": pk,
        "ct_W_list": _deserialize_ct_list(data["ct_W_list_b64"]),
        "node_features_enc": _deserialize_ct_list(data["node_features_enc_b64"]),
        "ct_labels": _deserialize_ct_list(data["ct_labels_b64"]),
        "in_channels": data["in_channels"],
        "out_channels": data["out_channels"],
        "batch_size": data["batch_size"],
        "a": _ndarray_from_b64(data["a_b64"]),
        "negative_slope": data["negative_slope"],
        "num_nodes": data["num_nodes"],
        "edge_index": _ndarray_from_b64(data["edge_index_b64"]),
        "train_mask": _ndarray_from_b64(data["train_mask_b64"]),
        "num_epochs": data["num_epochs"],
        "lr": data["lr"],
        "print_metrics": data["print_metrics"],
        "bootstrap_weights": data.get("bootstrap_weights", True),
    }


def serialize_infer_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Convert infer payload to JSON-serializable dict."""
    cc = payload["crypto_context"]
    return {
        "crypto_context_b64": base64.b64encode(_serialize_obj(cc)).decode("ascii"),
        "public_key_b64": base64.b64encode(_serialize_obj(payload["public_key"])).decode("ascii"),
        "eval_mult_key_b64": base64.b64encode(_serialize_eval_mult_key(cc)).decode("ascii"),
        "eval_automorphism_key_b64": base64.b64encode(_serialize_eval_automorphism_key(cc)).decode("ascii"),
        "ct_W_list_b64": _serialize_ct_list(payload["ct_W_list"]),
        "node_features_enc_b64": _serialize_ct_list(payload["node_features_enc"]),
        "in_channels": int(payload["in_channels"]),
        "out_channels": int(payload["out_channels"]),
        "batch_size": int(payload["batch_size"]),
        "a_b64": _ndarray_to_b64(np.asarray(payload["a"], dtype=np.float64)),
        "negative_slope": float(payload["negative_slope"]),
        "num_nodes": int(payload["num_nodes"]),
        "edge_index_b64": _ndarray_to_b64(np.asarray(payload["edge_index"], dtype=np.int64)),
        "print_metrics": bool(payload.get("print_metrics", False)),
    }


def deserialize_infer_payload(data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert JSON-serializable dict back to infer payload."""
    cc = _deserialize_cc(base64.b64decode(data["crypto_context_b64"]))
    pk = _deserialize_pk(base64.b64decode(data["public_key_b64"]))
    _deserialize_eval_mult_key_into_context(base64.b64decode(data["eval_mult_key_b64"]), cc)
    _deserialize_eval_automorphism_key_into_context(base64.b64decode(data["eval_automorphism_key_b64"]), cc)
    return {
        "crypto_context": cc,
        "public_key": pk,
        "ct_W_list": _deserialize_ct_list(data["ct_W_list_b64"]),
        "node_features_enc": _deserialize_ct_list(data["node_features_enc_b64"]),
        "in_channels": data["in_channels"],
        "out_channels": data["out_channels"],
        "batch_size": data["batch_size"],
        "a": _ndarray_from_b64(data["a_b64"]),
        "negative_slope": data["negative_slope"],
        "num_nodes": data["num_nodes"],
        "edge_index": _ndarray_from_b64(data["edge_index_b64"]),
        "print_metrics": data["print_metrics"],
    }


def serialize_train_result(out_cts: List[Any], metrics_dict: Dict, ct_W_trained: List[Any]) -> Dict[str, Any]:
    """Serialize (out_cts, metrics_dict, ct_W_trained) to JSON-serializable dict."""
    return {
        "out_cts_b64": _serialize_ct_list(out_cts),
        "metrics": metrics_dict,
        "ct_W_trained_b64": _serialize_ct_list(ct_W_trained),
    }


def deserialize_train_result(data: Dict[str, Any]) -> tuple:
    """Deserialize to (out_cts, metrics_dict, ct_W_trained)."""
    out_cts = _deserialize_ct_list(data["out_cts_b64"])
    ct_W_trained = _deserialize_ct_list(data["ct_W_trained_b64"])
    return out_cts, data["metrics"], ct_W_trained


def serialize_infer_result(out_cts: List[Any], metrics_dict: Dict) -> Dict[str, Any]:
    """Serialize (out_cts, metrics_dict) to JSON-serializable dict."""
    return {
        "out_cts_b64": _serialize_ct_list(out_cts),
        "metrics": metrics_dict,
    }


def deserialize_infer_result(data: Dict[str, Any]) -> tuple:
    """Deserialize to (out_cts, metrics_dict)."""
    return _deserialize_ct_list(data["out_cts_b64"]), data["metrics"]
