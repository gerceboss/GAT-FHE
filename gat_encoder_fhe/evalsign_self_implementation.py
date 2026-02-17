"""
Self-implemented sign-bit evaluation helpers for BinFHE (FHEW/CGGI).

This module is based on the BinFHE examples:
- Boolean gates + basic workflow: `examples/binfhe/boolean.py` in openfhe-python
  https://raw.githubusercontent.com/openfheorg/openfhe-python/main/examples/binfhe/boolean.py
- Small-precision arbitrary function evaluation via LUT: `eval-function.cpp` in openfhe-development
  https://raw.githubusercontent.com/openfheorg/openfhe-development/main/src/binfhe/examples/eval-function.cpp

Why this exists:
- OpenFHE provides `BinFHEContext.EvalSign()` for *large-precision* LWE ciphertexts (e.g., from CKKS→FHEW scheme switching).
- For *small-precision* LWE ciphertexts (fresh BinFHE encryptions), `EvalSign()` can error.
- BinFHE supports programmable bootstrapping via `EvalFunc(ct, lut)` which we can use to build a sign-bit ourselves.

Important:
- This helper operates on **BinFHE ciphertexts**. If your source values are in CKKS, you either:
  - use scheme switching to obtain large-precision LWE, then prefer built-in EvalSign, OR
  - decrypt to plaintext and re-encrypt into a standalone BinFHE context (NOT fully encrypted end-to-end).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class EvalSignSelfContext:
    """Holds a standalone BinFHE context + keys + LUT for sign evaluation."""

    cc_lwe: Any
    sk_lwe: Any
    p: int
    lut_sign: Any


def _nonneg_bit_lut_function(m: int, p: int) -> int:
    """
    Return 1 if m encodes a non-negative value under centered mod-p encoding, else 0.

    We treat messages as signed in the centered interval [-p/2, p/2):
    - encoded non-negative values are in [0, p/2)
    - encoded negative values are in [p/2, p)
    """
    half = p // 2
    return 1 if m < half else 0


def create_evalsign_self_context(
    *,
    paramset: Any,
    method: Any,
    max_plaintext_bits: int = 4,
) -> EvalSignSelfContext:
    """
    Create a standalone BinFHE context for *small precision* function evaluation.

    In openfhe-python, `GenerateBinFHEContext` has multiple overloads. The overload
    `(paramset, use_function, bits)` increases the plaintext space.

    Example (works in our environment):
        cc.GenerateBinFHEContext(STD128, True, 12)  -> GetMaxPlaintextSpace() == 8

    Args:
        paramset: e.g. STD128
        method: e.g. GINX
        max_plaintext_bits: controls the plaintext space indirectly; we use the overload
            `GenerateBinFHEContext(paramset, True, max_plaintext_bits)`.

    Returns:
        EvalSignSelfContext with LUT for sign bit.
    """
    from openfhe import BinFHEContext  # type: ignore

    cc = BinFHEContext()
    # Overload: (paramset, useFunction: bool, bits: int)
    cc.GenerateBinFHEContext(paramset, True, int(max_plaintext_bits))

    sk = cc.KeyGen()
    cc.BTKeyGen(sk)

    p = int(cc.GetMaxPlaintextSpace())
    lut = cc.GenerateLUTviaFunction(_nonneg_bit_lut_function, p)

    return EvalSignSelfContext(cc_lwe=cc, sk_lwe=sk, p=p, lut_sign=lut)


def evalsign_via_evalfunc(
    *,
    cc_lwe: Any,
    ct_lwe: Any,
    lut_sign: Any,
) -> Any:
    """
    Evaluate sign-bit using programmable bootstrapping (EvalFunc + LUT).

    Returns an LWE ciphertext encrypting {0,1} in the **same plaintext space** used by the context.
    """
    return cc_lwe.EvalFunc(ct_lwe, lut_sign)


def encode_centered_to_modp(value: int, p: int) -> int:
    """
    Encode a signed integer in centered interval [-p/2, p/2) into Z_p.
    """
    return int(value) % int(p)


def quantize_float_to_centered_int(x: float, *, scale: float, p: int) -> int:
    """
    Quantize float x into centered integer range [-p/2, p/2).
    """
    half = p // 2
    v = int(round(float(x) * float(scale)))
    if v < -half:
        v = -half
    if v > half - 1:
        v = half - 1
    return v

