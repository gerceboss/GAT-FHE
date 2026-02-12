"""
FHEW/CGGI + CKKS↔FHEW scheme-switching helpers for encrypted boolean operations.
Supports:
- Standalone BinFHE/CGGI context for boolean gates
- CKKS↔FHEW scheme switching for sign, comparison, encrypted branching
Pattern from: https://github.com/openfheorg/openfhe-python/blob/main/examples/pke/scheme-switching.py
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


def setup_scheme_switching(
    cc_ckks: Any,
    keys_ckks: Any,
    slots: int,
    logQ_ccLWE: int = 25,
    security_level_ckks: Any = None,
    security_level_fhew: Any = None,
) -> Tuple[Any, Any]:
    """
    Setup CKKS↔FHEW scheme switching following scheme-switching.py ComparisonViaSchemeSwitching().
    
    cc_ckks: CKKS CryptoContext (must have SCHEMESWITCH enabled)
    keys_ckks: CKKS KeyPair (CKKSKeys dataclass with publicKey, secretKey) or PrivateKey
    slots: number of CKKS slots
    logQ_ccLWE: log2 of FHEW ciphertext modulus (default 25)
    security_level_ckks, security_level_fhew: SecurityLevel enums (default HEStd_NotSet, TOY)
    
    Returns: (privateKeyFHEW, ccLWE)
    - privateKeyFHEW: FHEW secret key
    - ccLWE: BinFHEContext for FHEW operations
    
    After calling this, use:
    - cc_ckks.EvalCKKStoFHEWPrecompute(scale) to set scaling before CKKS→FHEW
    - cc_ckks.EvalCKKStoFHEW(ct_ckks, num_values) to switch
    - ccLWE.EvalSign(ct_fhew) for encrypted sign
    - cc_ckks.EvalFHEWtoCKKS([ct_fhew], ...) to switch back
    """
    ok, err, sym = _try_import_openfhe_binfhe()
    if not ok:
        raise RuntimeError(f"Scheme switching requires OpenFHE BinFHE. Import error: {err}")
    
    # Import scheme-switching parameter class
    try:
        from openfhe import SchSwchParams, KeyPair  # type: ignore
    except ImportError:
        raise RuntimeError("SchSwchParams not available; check OpenFHE version for scheme switching support.")
    
    # Default security levels
    if security_level_ckks is None:
        from openfhe import HEStd_NotSet  # type: ignore
        security_level_ckks = HEStd_NotSet
    if security_level_fhew is None:
        security_level_fhew = sym["STD128"]  # TOY for faster demo; use STD128 for real use
    
    # Setup scheme switching
    params = SchSwchParams()
    params.SetSecurityLevelCKKS(security_level_ckks)
    params.SetSecurityLevelFHEW(security_level_fhew)
    params.SetCtxtModSizeFHEWLargePrec(logQ_ccLWE)
    params.SetNumSlotsCKKS(slots)
    params.SetNumValues(slots)
    
    privateKeyFHEW = cc_ckks.EvalSchemeSwitchingSetup(params)
    ccLWE = cc_ckks.GetBinCCForSchemeSwitch()
    
    # EvalSchemeSwitchingKeyGen requires an openfhe.KeyPair (cannot be constructed manually)
    # Verify we have a valid KeyPair
    if not (hasattr(keys_ckks, 'publicKey') and hasattr(keys_ckks, 'secretKey')):
        raise TypeError(f"keys_ckks must be a KeyPair with publicKey and secretKey, got {type(keys_ckks)}")
    
    cc_ckks.EvalSchemeSwitchingKeyGen(keys_ckks, privateKeyFHEW)
    
    return privateKeyFHEW, ccLWE
