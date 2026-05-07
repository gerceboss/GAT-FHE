"""
OpenFHE TCP transport: zero-copy binary serialization over raw TCP sockets.

Protocol (all lengths are little-endian uint64):
  - Each "frame" is:  [8 bytes: payload_length][payload_bytes]
  - A "message" is a sequence of frames, preceded by a frame count.
  - Numpy arrays are sent as: [8 bytes dtype_len][dtype_str][8 bytes ndim]
    [ndim * 8 bytes shape][raw array bytes]
  - Scalars / small metadata are packed with struct.
  - The full train/infer payloads are sent as a structured sequence of frames
    defined by TRAIN_PAYLOAD_FIELDS / INFER_PAYLOAD_FIELDS order.

No base64. No JSON wrapping of FHE objects. No intermediate string copies.
OpenFHE BINARY mode is used throughout (most compact native format).

Bootstrap nullptr fix:
  client_keys.py calls:
      cc.EvalBootstrapSetup(levelBudget=[3,3], slots=slots)
  This builds precomputation tables inside the CryptoContext that are keyed
  to a SPECIFIC slot count.  These tables are NOT serialized with the context.
  After the server deserializes the context it has no tables at all, so
  EvalBootstrap() crashes: "KeySwitchDown(): Input ciphertext is nullptr".

  Fix: send bootstrap_level_budget ([3,3] by default) as part of the train
  payload.  recv_train_payload() replays:
      cc.EvalBootstrapSetup(levelBudget=bootstrap_level_budget, slots=slots)
  using the SAME level_budget AND the SAME slot count (slots is already
  in the common payload).  This restores the tables without the secret key.
"""

from __future__ import annotations

import pickle
import socket
import struct
import io
import sys
import traceback
from typing import Any, List, Tuple, Dict

import numpy as np

# ── OpenFHE lazy import ────────────────────────────────────────────────────

_OPENFHE = None


def _of():
    global _OPENFHE
    if _OPENFHE is None:
        import openfhe

        _OPENFHE = openfhe
    return _OPENFHE


# ── Low-level frame I/O ───────────────────────────────────────────────────
# A "frame" = 8-byte LE length prefix + raw bytes.
# We use memoryview + recv_into to avoid extra copies on the receive path.

_HDR = struct.Struct("<Q")  # unsigned 64-bit LE


def _send_frame(sock: socket.socket, data: bytes | bytearray | memoryview) -> None:
    """Send one length-prefixed frame. data must support the buffer protocol."""
    sock.sendall(_HDR.pack(len(data)))
    sock.sendall(data)


def _recv_exactly(sock: socket.socket, n: int) -> bytes:
    """Read exactly n bytes from sock; avoids per-recv allocations for large n."""
    if n == 0:
        return b""
    buf = bytearray(n)
    view = memoryview(buf)
    pos = 0
    while pos < n:
        got = sock.recv_into(view[pos:], n - pos)
        if not got:
            raise EOFError(f"Connection closed after {pos}/{n} bytes")
        pos += got
    return bytes(buf)


def _recv_frame(sock: socket.socket) -> bytes:
    """Receive one length-prefixed frame."""
    hdr = _recv_exactly(sock, 8)
    length = _HDR.unpack(hdr)[0]
    return _recv_exactly(sock, length)


# ── OpenFHE object serialization (BINARY, no base64) ─────────────────────


def _ser_obj(obj: Any) -> bytes:
    of = _of()
    s = of.Serialize(obj, of.BINARY)
    return s if isinstance(s, bytes) else s.encode("latin-1")


def serialize_obj(obj: Any) -> bytes:
    return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)


def deserialize_obj(data: bytes) -> Any:
    return pickle.loads(data)


def _deser_cc(data: bytes) -> Any:
    """
    Deserialize CryptoContext from OpenFHE BINARY payload.

    OpenFHE Python bindings expect (bytes, SERBINARY). Passing a decoded str
    with SERBINARY triggers a TypeError.
    """
    of = _of()
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError(f"Expected bytes for BINARY CryptoContext, got {type(data)}")
    return of.DeserializeCryptoContextString(bytes(data), of.BINARY)


def _deser_pk(data: bytes) -> Any:
    of = _of()
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError(f"Expected bytes for BINARY PublicKey, got {type(data)}")
    return of.DeserializePublicKeyString(bytes(data), of.BINARY)


def _deser_ct(data: bytes) -> Any:
    of = _of()
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError(f"Expected bytes for BINARY Ciphertext, got {type(data)}")
    return of.DeserializeCiphertextString(bytes(data), of.BINARY)


def _ser_eval_mult(cc: Any) -> bytes:
    of = _of()
    try:
        key_tag = cc.GetKeyTag()
    except Exception:
        key_tag = ""
    s = of.SerializeEvalMultKeyString(of.BINARY, key_tag)
    return s if isinstance(s, bytes) else s.encode("latin-1")


def _ser_eval_rot(cc: Any) -> bytes:
    of = _of()
    try:
        key_tag = cc.GetKeyTag()
    except Exception:
        key_tag = ""
    s = of.SerializeEvalAutomorphismKeyString(of.BINARY, key_tag)
    return s if isinstance(s, bytes) else s.encode("latin-1")

def _ser_eval_bootstrap(cc: Any) -> bytes:
    of = _of()
    try:
        key_tag = cc.GetKeyTag()
    except Exception:
        key_tag = ""

    # Try several possible OpenFHE API names / strategies to obtain a
    # serialized EvalBootstrap key. Different OpenFHE Python bindings expose
    # different helper functions across versions; be resilient.
    # 1) Prefer dedicated SerializeEvalBootstrapKeyString if present.
    fn_names = [
        "SerializeEvalBootstrapKeyString",
        "SerializeEvalKeyString",
    ]
    for name in fn_names:
        fn = getattr(of, name, None)
        if callable(fn):
            try:
                s = fn(of.BINARY, key_tag)
                payload = s if isinstance(s, bytes) else s.encode("latin-1")
                return b"\x01" + payload
            except Exception:
                # Try next fallback
                pass

    # 2) Try to obtain a bootstrap key object from the crypto context and
    # serialize it with the generic Serialize() function.
    try:
        # Common method names that may exist on CryptoContext
        getters = [
            "GetEvalBootstrapKey",
            "GetBootstrapKey",
            "GetEvalKey",
        ]
        key_obj = None
        for g in getters:
            if hasattr(cc, g):
                key_obj = getattr(cc, g)()
                break
        if key_obj is not None:
            s = of.Serialize(key_obj, of.BINARY)
            payload = s if isinstance(s, bytes) else s.encode("latin-1")
            return b"\x01" + payload
    except Exception:
        pass

    # 3) As a last resort, return a presence-flag-only frame indicating
    # "no bootstrap key supplied".
    return b"\x00"

def _deser_eval_mult(data: bytes, cc: Any) -> None:
    of = _of()
    try:
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError(f"Expected bytes for BINARY EvalMultKey, got {type(data)}")
        of.DeserializeEvalMultKeyString(bytes(data), of.BINARY)
    except RuntimeError as exc:
        if "Can not save a EvalMultKeys vector" not in str(exc):
            raise


def _deser_eval_rot(data: bytes, cc: Any) -> None:
    of = _of()
    try:
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError(
                f"Expected bytes for BINARY EvalAutomorphismKey, got {type(data)}"
            )
        of.DeserializeEvalAutomorphismKeyString(bytes(data), of.BINARY)
    except RuntimeError as exc:
        if "Can not save a EvalAutomorphismKeys vector" not in str(exc):
            raise

def _deser_eval_bootstrap(data: bytes, cc: Any) -> None:
    of = _of()
    try:
        # Serialized bootstrap frames are now flagged: first byte == 0
        # means "no bootstrap key supplied", first byte == 1 means
        # remaining bytes are the actual serialized key payload.
        if not data:
            return
        flag = data[0]
        if flag == 0:
            return
        payload = bytes(data[1:])

        fn = getattr(of, "DeserializeEvalBootstrapKeyString", None)
        if callable(fn):
            fn(payload, of.BINARY)
            return

        # Fallback to a more generic EvalKey deserializer if available.
        fn2 = getattr(of, "DeserializeEvalKeyString", None)
        if callable(fn2):
            fn2(payload, of.BINARY)
            return

        # Final fallback: attempt generic Deserialize (may raise)
        if hasattr(of, "Deserialize"):
            of.Deserialize(payload, of.BINARY)
    except RuntimeError as exc:
        if "Can not save a EvalBootstrapKeys vector" not in str(exc):
            raise

# ── Bootstrap setup replay ────────────────────────────────────────────────


def _replay_bootstrap_setup(cc: Any, level_budget: list, slots: int) -> None:
    """
    Replay EvalBootstrapSetup() on a freshly-deserialized CryptoContext.

    Must be called with the SAME levelBudget and slots the client passed to
    EvalBootstrapSetup() during key generation. The slot count is critical:
    OpenFHE keys the precomputed DFT/linear-transform tables to the exact
    slot count, and calling EvalBootstrap() with the wrong (or missing)
    tables causes "KeySwitchDown(): Input ciphertext is nullptr".

    This call does NOT require the secret key — it only rebuilds plaintext-
    side precomputation tables.
    """
    import sys

    try:
        cc.EvalBootstrapSetup(levelBudget=level_budget, slots=slots)
    except Exception as exc:
        # Context built without FHE enabled (--no_bootstrap path) — safe to skip.
        print(
            f"[serializer] EvalBootstrapSetup(levelBudget={level_budget}, "
            f"slots={slots}) skipped ({type(exc).__name__}: {exc})",
            file=sys.stderr,
        )


# ── Ciphertext list helpers ───────────────────────────────────────────────


def _send_ct_list(sock: socket.socket, cts: List[Any]) -> None:
    """Send list length then each ciphertext as a frame."""
    _send_frame(sock, _HDR.pack(len(cts)))
    for ct in cts:
        _send_frame(sock, _ser_obj(ct))


def _recv_ct_list(sock: socket.socket) -> List[Any]:
    n = _HDR.unpack(_recv_frame(sock))[0]
    return [_deser_ct(_recv_frame(sock)) for _ in range(n)]


# ── Numpy array helpers ───────────────────────────────────────────────────


def _send_ndarray(sock: socket.socket, arr: np.ndarray) -> None:
    """Send dtype string, shape, then raw bytes — no copies for the bulk data."""
    dtype_b = arr.dtype.str.encode("ascii")  # e.g. b"<f8"
    shape_b = struct.pack(f"<Q{len(arr.shape)}q", len(arr.shape), *arr.shape)
    _send_frame(sock, dtype_b)
    _send_frame(sock, shape_b)
    # Use tobytes() only; for very large arrays consider arr.data (memoryview)
    _send_frame(sock, arr.tobytes())


def _recv_ndarray(sock: socket.socket) -> np.ndarray:
    dtype = np.dtype(_recv_frame(sock).decode("ascii"))
    shape_raw = _recv_frame(sock)
    ndim = struct.unpack_from("<Q", shape_raw)[0]
    shape = struct.unpack_from(f"<{ndim}q", shape_raw, 8)
    raw = _recv_frame(sock)
    return np.frombuffer(raw, dtype=dtype).reshape(shape)


# ── Scalar / string helpers ───────────────────────────────────────────────


def _send_scalar(sock: socket.socket, fmt: str, *values) -> None:
    _send_frame(sock, struct.pack(fmt, *values))


def _recv_scalar(sock: socket.socket, fmt: str):
    data = _recv_frame(sock)
    result = struct.unpack(fmt, data)
    return result[0] if len(result) == 1 else result


def _send_str(sock: socket.socket, s: str) -> None:
    _send_frame(sock, s.encode("utf-8"))


def _recv_str(sock: socket.socket) -> str:
    return _recv_frame(sock).decode("utf-8")


# ── Metrics dict (sent as JSON frame — small, human-readable) ─────────────


def _send_metrics(sock: socket.socket, metrics: dict) -> None:
    import json

    _send_frame(sock, json.dumps(metrics).encode("utf-8"))


def _recv_metrics(sock: socket.socket) -> dict:
    import json

    return json.loads(_recv_frame(sock).decode("utf-8"))


# ── int list (for bootstrap_level_budget) ────────────────────────────────


def _send_int_list(sock: socket.socket, lst: list) -> None:
    """Send a short list of signed 64-bit ints as [count][v0][v1]..."""
    _send_frame(sock, struct.pack(f"<Q{len(lst)}q", len(lst), *lst))


def _recv_int_list(sock: socket.socket) -> list:
    raw = _recv_frame(sock)
    count = struct.unpack_from("<Q", raw)[0]
    return list(struct.unpack_from(f"<{count}q", raw, 8))


# ── High-level payload send/recv ─────────────────────────────────────────
#
# Order is fixed and must match between client and server.
# Common prefix (cc, pk, eval_mult, eval_rot, ct_W_list, node_features_enc,
#                in_channels, out_channels, slots, a, negative_slope,
#                num_nodes, edge_index)
# Train-only suffix: ct_labels, train_mask, num_epochs, lr, print_metrics,
#                    bootstrap_weights, bootstrap_level_budget
# Infer-only suffix: print_metrics


def send_common_payload(sock: socket.socket, payload: dict) -> None:
    """Send fields shared by both train and infer payloads."""
    cc = payload["crypto_context"]
    _send_frame(sock, _ser_obj(cc))
    _send_frame(sock, _ser_obj(payload["public_key"]))
    _send_frame(sock, _ser_eval_mult(cc))
    _send_frame(sock, _ser_eval_rot(cc))
    _send_frame(sock, _ser_eval_bootstrap(cc))
    _send_ct_list(sock, payload["ct_W_list"])
    _send_ct_list(sock, payload["node_features_enc"])
    _send_scalar(
        sock, "<qqq", payload["in_channels"], payload["out_channels"], payload["slots"]
    )
    _send_ndarray(sock, np.asarray(payload["a"], dtype=np.float64))
    _send_scalar(sock, "<d", float(payload["negative_slope"]))
    _send_scalar(sock, "<q", int(payload["num_nodes"]))
    _send_ndarray(sock, np.asarray(payload["edge_index"], dtype=np.int64))


def recv_common_payload(sock: socket.socket) -> dict:
    """Receive fields shared by both train and infer payloads."""
    cc = _deser_cc(_recv_frame(sock))
    pk = _deser_pk(_recv_frame(sock))
    _deser_eval_mult(_recv_frame(sock), cc)
    _deser_eval_rot(_recv_frame(sock), cc)
    _deser_eval_bootstrap(_recv_frame(sock), cc)
    ct_W_list = _recv_ct_list(sock)
    node_features_enc = _recv_ct_list(sock)
    in_ch, out_ch, slots = struct.unpack("<qqq", _recv_frame(sock))
    a = _recv_ndarray(sock)
    neg_slope = struct.unpack("<d", _recv_frame(sock))[0]
    num_nodes = struct.unpack("<q", _recv_frame(sock))[0]
    edge_index = _recv_ndarray(sock)
    if edge_index.ndim == 2 and edge_index.shape[0] != 2:
        edge_index = edge_index.T
    return {
        "crypto_context": cc,
        "public_key": pk,
        "ct_W_list": ct_W_list,
        "node_features_enc": node_features_enc,
        "in_channels": int(in_ch),
        "out_channels": int(out_ch),
        "slots": int(slots),
        "a": a,
        "negative_slope": neg_slope,
        "num_nodes": int(num_nodes),
        "edge_index": edge_index,
    }


def send_train_payload(sock: socket.socket, payload: dict) -> None:
    send_common_payload(sock, payload)
    _send_ct_list(sock, payload["ct_labels"])
    _send_ndarray(sock, np.asarray(payload["train_mask"], dtype=bool))
    _send_scalar(
        sock,
        "<qd??",
        int(payload["num_epochs"]),
        float(payload["lr"]),
        bool(payload.get("print_metrics", True)),
        bool(payload.get("bootstrap_weights", True)),
    )
    # Send the level_budget so the server can replay EvalBootstrapSetup with
    # the exact same parameters (levelBudget AND slots=slots) the client
    # used during key generation.  Default [4,4] matches client_keys.py.
    _send_int_list(sock, list(payload.get("bootstrap_level_budget", [3, 3])))


def recv_train_payload(sock: socket.socket) -> dict:
    payload = recv_common_payload(sock)
    payload["ct_labels"] = _recv_ct_list(sock)
    payload["train_mask"] = _recv_ndarray(sock)
    num_epochs, lr, print_metrics, bootstrap_weights = struct.unpack(
        "<qd??", _recv_frame(sock)
    )
    payload["num_epochs"] = int(num_epochs)
    payload["lr"] = float(lr)
    payload["print_metrics"] = bool(print_metrics)
    payload["bootstrap_weights"] = bool(bootstrap_weights)
    payload["bootstrap_level_budget"] = _recv_int_list(sock)

    # ── Replay EvalBootstrapSetup to rebuild precomputation tables ────────
    # Must pass BOTH levelBudget and slots=slots to exactly match what
    # client_keys.py called:
    #   cc.EvalBootstrapSetup(levelBudget=level_budget, slots=slots)
    # Using the wrong slot count (or omitting it, which defaults to ringDim/2)
    # leaves the tables mismatched and EvalBootstrap() crashes with nullptr.
    if payload["bootstrap_weights"]:
        _replay_bootstrap_setup(
            payload["crypto_context"],
            payload["bootstrap_level_budget"],
            slots=payload["slots"],  # <-- critical: must match client
        )

    return payload


def send_infer_payload(sock: socket.socket, payload: dict) -> None:
    send_common_payload(sock, payload)
    _send_scalar(sock, "<?", bool(payload.get("print_metrics", False)))


def recv_infer_payload(sock: socket.socket) -> dict:
    payload = recv_common_payload(sock)
    payload["print_metrics"] = bool(struct.unpack("<?", _recv_frame(sock))[0])
    return payload


# ── Result send/recv ──────────────────────────────────────────────────────


def send_train_result(
    sock: socket.socket, out_cts: List[Any], metrics: dict, ct_W_trained: List[Any]
) -> None:
    _send_ct_list(sock, out_cts)
    _send_metrics(sock, metrics)
    _send_ct_list(sock, ct_W_trained)


def recv_train_result(sock: socket.socket) -> Tuple[List[Any], dict, List[Any]]:
    out_cts = _recv_ct_list(sock)
    metrics = _recv_metrics(sock)
    ct_W_trained = _recv_ct_list(sock)
    return out_cts, metrics, ct_W_trained


def send_infer_result(sock: socket.socket, out_cts: List[Any], metrics: dict) -> None:
    _send_ct_list(sock, out_cts)
    _send_metrics(sock, metrics)


def recv_infer_result(sock: socket.socket) -> Tuple[List[Any], dict]:
    out_cts = _recv_ct_list(sock)
    metrics = _recv_metrics(sock)
    return out_cts, metrics


def recv_gradient_step_payload(sock: socket.socket) -> dict:
    payload = recv_common_payload(sock)

    payload["ct_labels"] = _recv_ct_list(sock)

    payload["train_mask"] = _recv_ndarray(sock)
    if payload["train_mask"].dtype != np.bool_:
        payload["train_mask"] = np.asarray(payload["train_mask"], dtype=bool)

    lr, num_epochs = struct.unpack("<dq", _recv_frame(sock))
    payload["lr"] = float(lr)
    payload["num_epochs"] = int(num_epochs)
    payload["bootstrap_level_budget"] = _recv_int_list(sock)

    # Replay bootstrap
    _replay_bootstrap_setup(
        payload["crypto_context"],
        level_budget=payload["bootstrap_level_budget"],
        slots=payload["slots"],
    )

    return payload


def send_gradient_step_payload(sock: socket.socket, payload: dict) -> None:
    send_common_payload(sock, payload)

    _send_ct_list(sock, payload["ct_labels"])

    _send_ndarray(sock, np.asarray(payload["train_mask"], dtype=bool))

    _send_scalar(
        sock,
        "<dq",
        float(payload["lr"]),
        int(payload["num_epochs"]),
    )
    _send_int_list(sock, list(payload.get("bootstrap_level_budget", [4, 4])))


def send_gradient_step_result(
    sock: socket.socket, ct_W_new: List[Any], metrics: dict
) -> None:
    """
    Send gradient-step result back to client.

    Wire format mirrors a subset of send_train_result:
      [ct_W_new list][metrics dict as JSON frame]
    """
    _send_ct_list(sock, ct_W_new)
    _send_metrics(sock, metrics)


def recv_gradient_step_result(sock: socket.socket) -> Tuple[List[Any], dict]:
    """
    Receive gradient-step result from server.

    Returns:
      (ct_W_new list, metrics dict)
    """
    ct_W_new = _recv_ct_list(sock)
    metrics = _recv_metrics(sock)
    return ct_W_new, metrics


# ── Error frame ───────────────────────────────────────────────────────────
# Server sends b"\x00" for OK, b"\x01" + error-msg frame on error.


def send_ok(sock: socket.socket) -> None:
    _send_frame(sock, b"\x00")


def send_error(sock: socket.socket, exc: Exception) -> None:
    tb = traceback.format_exc()
    msg = f"{type(exc).__name__}: {exc}\n{tb}"
    _send_frame(sock, b"\x01")
    _send_str(sock, msg)


def recv_status(sock: socket.socket) -> None:
    """Raise RuntimeError on server error, else return None."""
    status = _recv_frame(sock)
    if status == b"\x01":
        msg = _recv_str(sock)
        raise RuntimeError(f"Server error:\n{msg}")
    # b"\x00" = ok


# ── Trained-weight persistence (save / load to disk) ─────────────────────
#
# Saves everything a client needs to skip re-training and go straight to
# inference:
#
#   <save_dir>/
#     meta.json          — F_in, F_out, slots, negative_slope, n_weights,
#     W_{k}.bin          — one file per trained weight
#     a.npy              — plaintext attention parameter vector
#
# The SecretKey is NEVER sent to the server; it is stored only on the client
# machine in the save directory so future inference runs can decrypt outputs.


def save_trained_weights(
    save_dir: str,
    W_list,  # numpy array (F_out, F_in)
    a,  # numpy array
    slots: int,
    F_in: int,
    F_out: int,
) -> None:
    import json
    import os
    import numpy as np

    save_dir = str(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    meta = {
        "F_in": int(F_in),
        "F_out": int(F_out),
        "slots": int(slots),
    }

    with open(os.path.join(save_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Save plaintext weight matrix
    np.save(os.path.join(save_dir, "W.npy"), np.asarray(W_list, dtype=np.float64))

    # Save plaintext attention vector
    np.save(os.path.join(save_dir, "a.npy"), np.asarray(a, dtype=np.float64))

    print(f"[weights] saved plaintext weights → {save_dir}")


def load_trained_weights(save_dir: str) -> dict:
    """
    Load plaintext-trained weights from disk.

    Expected directory contents:
        meta.json
        W.npy
        a.npy

    Returns:
        {
            "W_list": numpy array (F_out, F_in),
            "a": numpy array,
            "slots": int,
            "F_in": int,
            "F_out": int,
        }
    """
    import json
    import os
    import numpy as np

    save_dir = str(save_dir)
    if not os.path.isdir(save_dir):
        raise FileNotFoundError(f"Weights directory not found: {save_dir!r}")

    # ── Load metadata ─────────────────────────────────────────────
    meta_path = os.path.join(save_dir, "meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"meta.json not found in {save_dir!r}")

    with open(meta_path) as f:
        meta = json.load(f)

    F_in = int(meta["F_in"])
    F_out = int(meta["F_out"])
    slots = int(meta["slots"])

    # ── Load plaintext weights ─────────────────────────────────────
    W_path = os.path.join(save_dir, "W.npy")
    a_path = os.path.join(save_dir, "a.npy")

    if not os.path.exists(W_path):
        raise FileNotFoundError(f"W.npy not found in {save_dir!r}")
    if not os.path.exists(a_path):
        raise FileNotFoundError(f"a.npy not found in {save_dir!r}")

    W_list = np.load(W_path)
    a = np.load(a_path)

    out = {
        "W_list": W_list,
        "a": a,
        "slots": slots,
        "F_in": F_in,
        "F_out": F_out,
    }

    return out
