"""
CGGI (BinFHE) helpers for boolean operations (if-else, sign, select).
Uses OpenFHE BinFHE with GINX bootstrapping (CGGI scheme).
Ref: https://github.com/openfheorg/openfhe-python/tree/main/examples/binfhe
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

def _try_import_openfhe_binfhe() -> Tuple[bool, Optional[str], Any]:
    """Attempt to import OpenFHE and BinFHE/CGGI symbols."""
    try:
        from openfhe import (  # type: ignore
            BinFHEContext,
            AND,
            OR,
            NAND,
            NOR,
            XOR,
            XNOR,
            STD128,
            GINX,
            AP,
        )
        return True, None, {
            "BinFHEContext": BinFHEContext,
            "AND": AND,
            "OR": OR,
            "NAND": NAND,
            "NOR": NOR,
            "XOR": XOR,
            "XNOR": XNOR,
            "STD128": STD128,
            "GINX": GINX,
            "AP": AP,
        }
    except Exception as e:
        return False, f"{type(e).__name__}: {e}", None


def cggi_available() -> bool:
    """Return True if OpenFHE BinFHE (CGGI) is available."""
    ok, _, _ = _try_import_openfhe_binfhe()
    return ok


class CGGIContext:
    """
    CGGI (BinFHE/GINX) context for encrypted boolean operations.
    Use for if-else, sign bits, and select (mux) on encrypted bits.
    """

    def __init__(self, bootstrapping: str = "GINX"):
        """
        bootstrapping: "GINX" (CGGI, default) or "AP".
        """
        ok, err, sym = _try_import_openfhe_binfhe()
        if not ok:
            raise RuntimeError(
                "OpenFHE BinFHE (CGGI) is not available.\n"
                f"Import error: {err}"
            )
        self._sym = sym
        self._cc = sym["BinFHEContext"]()
        method = sym["GINX"] if bootstrapping.upper() == "GINX" else sym["AP"]
        self._cc.GenerateBinFHEContext(sym["STD128"], method)
        self._sk = self._cc.KeyGen()
        self._cc.BTKeyGen(self._sk)
        self._AND = sym["AND"]
        self._OR = sym["OR"]

    @property
    def cc(self) -> Any:
        return self._cc

    @property
    def secret_key(self) -> Any:
        return self._sk

    def encrypt_bit(self, bit: int) -> Any:
        """Encrypt a single bit (0 or 1). Returns ciphertext."""
        if bit not in (0, 1):
            raise ValueError("bit must be 0 or 1")
        return self._cc.Encrypt(self._sk, bit)

    def decrypt_bit(self, ct: Any) -> int:
        """Decrypt a single-bit ciphertext to 0 or 1."""
        return int(self._cc.Decrypt(self._sk, ct))

    def eval_not(self, ct: Any) -> Any:
        """Homomorphic NOT. Returns ciphertext."""
        return self._cc.EvalNOT(ct)

    def eval_and(self, ct_a: Any, ct_b: Any) -> Any:
        """Homomorphic AND. Returns ciphertext."""
        return self._cc.EvalBinGate(self._AND, ct_a, ct_b)

    def eval_or(self, ct_a: Any, ct_b: Any) -> Any:
        """Homomorphic OR. Returns ciphertext."""
        return self._cc.EvalBinGate(self._OR, ct_a, ct_b)

    def eval_select_bit(self, cond_ct: Any, ct_then: Any, ct_else: Any) -> Any:
        """
        Homomorphic mux on bits: out = (cond AND ct_then) OR ((NOT cond) AND ct_else).
        All inputs and output are single-bit ciphertexts.
        """
        not_cond = self._cc.EvalNOT(cond_ct)
        a = self._cc.EvalBinGate(self._AND, cond_ct, ct_then)
        b = self._cc.EvalBinGate(self._AND, not_cond, ct_else)
        return self._cc.EvalBinGate(self._OR, a, b)

    def sign_bits_encrypted(self, values: List[float]) -> List[Any]:
        """
        For each value, encrypt sign bit: 1 if value >= 0, else 0.
        Returns list of ciphertexts (one per value).
        """
        return [self.encrypt_bit(1 if v >= 0 else 0) for v in values]

    def sign_bits_decrypt(self, ct_list: List[Any]) -> List[int]:
        """Decrypt list of single-bit ciphertexts to 0/1 list."""
        return [self.decrypt_bit(ct) for ct in ct_list]


def leaky_relu_with_cggi(
    cggi: CGGIContext,
    raw_scores: List[float],
    negative_slope: float = 0.2,
) -> List[float]:
    """
    Apply LeakyReLU using CGGI for the branch (sign).
    For each score: sign = (score >= 0) is computed under FHE; then
    out = sign * score + (1 - sign) * negative_slope * score.
    """
    # Encrypt sign bits under CGGI
    sign_cts = cggi.sign_bits_encrypted(raw_scores)
    # Decrypt to get sign (in full pipeline you might use scheme switching to keep encrypted)
    signs = cggi.sign_bits_decrypt(sign_cts)
    out = []
    for s, score in zip(signs, raw_scores):
        if s == 1:
            out.append(score)
        else:
            out.append(negative_slope * score)
    return out
