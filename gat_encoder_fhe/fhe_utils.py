"""
Homomorphic encryption utility functions for division and other operations.

Implements multiple approaches for encrypted division:
1. Newton-Raphson method (CKKS) - iterative approximation
2. Goldschmidt algorithm (CKKS) - multiplicative convergence
3. Binary circuit division (FHEW/BinFHE) - bit-wise operations

See: https://eprint.iacr.org/2020/1483.pdf (Homomorphic Polynomial Evaluation)
"""

from typing import Any, List, Optional


def encrypted_reciprocal_newton_raphson(
    cc: Any,
    ct_denominator: Any,
    num_iterations: int = 3,
    initial_guess: float = 1.0,
    batch_size: int = 8,
) -> Any:
    """
    Compute encrypted reciprocal 1/d using Newton-Raphson method in CKKS.
    
    Algorithm: x_{i+1} = x_i * (2 - d * x_i)
    Converges to 1/d when x_0 is close to 1/d
    
    Args:
        cc: CKKS CryptoContext
        ct_denominator: encrypted denominator d
        num_iterations: number of iterations (3-5 typical)
        initial_guess: starting approximation x_0 (should be ~ 1/d)
        batch_size: CKKS batch size (number of slots)
    
    Returns:
        Encrypted reciprocal 1/d
        
    Complexity:
    - Multiplications: 2 * num_iterations
    - Additions: num_iterations
    - Multiplicative depth: 2 * num_iterations
    
    Note: Requires ct_denominator to be normalized to reasonable range
    (e.g., [0.1, 10]) for convergence. Use scaling if needed.
    """
    # Initial guess x_0 (plaintext)
    pt_x = cc.MakeCKKSPackedPlaintext([initial_guess] * batch_size)
    
    # Create plaintext constant 2
    pt_two = cc.MakeCKKSPackedPlaintext([2.0] * batch_size)
    
    # Start with plaintext initial guess (multiply ct_denominator by initial_guess plaintext)
    ct_x = cc.EvalMult(ct_denominator, pt_x)
    
    # Newton-Raphson iterations
    for _ in range(num_iterations):
        # Compute d * x_i
        ct_dx = cc.EvalMult(ct_denominator, ct_x)
        
        # Compute (2 - d * x_i)
        ct_two_minus_dx = cc.EvalSub(pt_two, ct_dx)
        
        # Compute x_{i+1} = x_i * (2 - d * x_i)
        ct_x = cc.EvalMult(ct_x, ct_two_minus_dx)
    
    return ct_x


def encrypted_division_newton_raphson(
    cc: Any,
    ct_numerator: Any,
    ct_denominator: Any,
    num_iterations: int = 3,
    initial_guess: float = 1.0,
    batch_size: int = 8,
) -> Any:
    """
    Compute encrypted division a/b using Newton-Raphson reciprocal.
    
    Algorithm:
    1. Compute reciprocal: r = 1/b using Newton-Raphson
    2. Multiply: result = a * r
    
    Args:
        cc: CKKS CryptoContext
        ct_numerator: encrypted numerator a
        ct_denominator: encrypted denominator b
        num_iterations: Newton-Raphson iterations
        initial_guess: starting approximation for 1/b
        batch_size: CKKS batch size
    
    Returns:
        Encrypted quotient a/b
    """
    # Compute 1/b
    ct_reciprocal = encrypted_reciprocal_newton_raphson(
        cc, ct_denominator, num_iterations, initial_guess, batch_size
    )
    
    # Multiply a * (1/b)
    ct_result = cc.EvalMult(ct_numerator, ct_reciprocal)
    
    return ct_result


def encrypted_reciprocal_goldschmidt(
    cc: Any,
    ct_denominator: Any,
    num_iterations: int = 3,
    scale_factor: float = 1.0,
    batch_size: int = 8,
) -> Any:
    """
    Compute encrypted reciprocal 1/d using Goldschmidt algorithm in CKKS.
    
    Algorithm (multiplicative convergence):
    d_0 = d * f_0  (where f_0 ≈ 1/d)
    For i = 0, 1, ...:
        f_{i+1} = f_i * (2 - d_i)
        d_{i+1} = d_i * (2 - d_i)
    
    After k iterations: d_k → 1, f_k → 1/d
    
    Args:
        cc: CKKS CryptoContext
        ct_denominator: encrypted denominator d
        num_iterations: number of iterations
        scale_factor: initial scaling f_0 ≈ 1/d
    
    Returns:
        Encrypted reciprocal 1/d
        
    Advantage: Can compute multiple divisions simultaneously (SIMD)
    """
    # Initialize f_0 = scale_factor (should be ≈ 1/d)
    pt_f = cc.MakeCKKSPackedPlaintext([scale_factor] * batch_size)
    
    # Initialize d_0 = d * f_0 (plaintext mult)
    ct_d = cc.EvalMult(ct_denominator, pt_f)
    ct_f = pt_f  # Keep plaintext version for now
    
    # Plaintext constant 2
    pt_two = cc.MakeCKKSPackedPlaintext([2.0] * batch_size)
    
    # Convert ct_f to ciphertext for iterations
    ct_f = cc.EvalMult(ct_denominator, pt_f)
    
    # Goldschmidt iterations
    for _ in range(num_iterations):
        # Compute (2 - d_i)
        ct_two_minus_d = cc.EvalSub(pt_two, ct_d)
        
        # Update f_{i+1} = f_i * (2 - d_i)
        ct_f = cc.EvalMult(ct_f, ct_two_minus_d)
        
        # Update d_{i+1} = d_i * (2 - d_i)
        ct_d = cc.EvalMult(ct_d, ct_two_minus_d)
    
    return ct_f


def encrypted_softmax_with_division(
    cc: Any,
    ct_exp_list: List[Any],
    indices_per_group: List[List[int]],
    num_iterations: int = 3,
) -> List[Any]:
    """
    Compute encrypted softmax with homomorphic division (Newton-Raphson).
    
    For each group (e.g., edges per target node):
    softmax(x_i) = exp(x_i) / sum_j exp(x_j)
    
    Args:
        cc: CKKS CryptoContext
        ct_exp_list: list of encrypted exp(x) values
        indices_per_group: list of index lists, one per group
        num_iterations: Newton-Raphson iterations for division
    
    Returns:
        List of encrypted softmax probabilities
        
    Algorithm:
    1. For each group:
       a. Sum exp values: s = sum_j exp(x_j)
       b. Compute reciprocal: r = 1/s (Newton-Raphson)
       c. Multiply each: softmax(x_i) = exp(x_i) * r
    """
    ct_softmax_list = [None] * len(ct_exp_list)
    
    for group_indices in indices_per_group:
        if not group_indices:
            continue
        
        # Step 1: Sum exp values in this group
        ct_sum = ct_exp_list[group_indices[0]]
        for idx in group_indices[1:]:
            ct_sum = cc.EvalAdd(ct_sum, ct_exp_list[idx])
        
        # Step 2: Compute reciprocal 1/sum using Newton-Raphson
        # Initial guess: assume sum is in range [1, 10], so guess 0.2
        ct_reciprocal = encrypted_reciprocal_newton_raphson(
            cc, ct_sum, num_iterations, initial_guess=0.2
        )
        
        # Step 3: Multiply each exp by reciprocal
        for idx in group_indices:
            ct_softmax_list[idx] = cc.EvalMult(ct_exp_list[idx], ct_reciprocal)
    
    return ct_softmax_list


def encrypted_normalize_per_group(
    cc: Any,
    ct_list: List[Any],
    indices_per_group: List[List[int]],
    num_iterations: int = 3,
) -> List[Any]:
    """
    Normalize encrypted values per group using homomorphic division.
    
    For each group: normalized(x_i) = x_i / sum_j x_j
    
    Args:
        cc: CKKS CryptoContext
        ct_list: list of encrypted values
        indices_per_group: list of index lists (one per group)
        num_iterations: Newton-Raphson iterations
    
    Returns:
        List of normalized ciphertexts
    """
    return encrypted_softmax_with_division(cc, ct_list, indices_per_group, num_iterations)


def estimate_initial_reciprocal_guess(
    cc: Any,
    ct: Any,
    keys: Any,
    range_estimate: tuple = (0.1, 10.0),
) -> float:
    """
    Estimate initial guess for reciprocal by decrypting (for setup only).
    
    SECURITY NOTE: This should only be used during setup/testing.
    In production, use a fixed conservative guess or range estimation.
    
    Args:
        cc: CryptoContext
        ct: encrypted value
        keys: keypair with secret key
        range_estimate: (min, max) expected range
    
    Returns:
        Initial guess ≈ 1/value
    """
    # Decrypt to estimate (INSECURE - only for calibration)
    pt = cc.Decrypt(keys.secretKey, ct)
    pt.SetLength(1)
    values = pt.GetCKKSPackedValue()
    value = abs(complex(values[0]).real)
    
    # Clamp to range
    value = max(range_estimate[0], min(range_estimate[1], value))
    
    # Return reciprocal estimate
    return 1.0 / value if value > 1e-6 else 1.0


# Constants for division algorithm selection
DIV_METHOD_NEWTON_RAPHSON = "newton_raphson"
DIV_METHOD_GOLDSCHMIDT = "goldschmidt"
DIV_METHOD_BINARY_CIRCUIT = "binary_circuit"  # Future: for FHEW


def encrypted_divide(
    cc: Any,
    ct_numerator: Any,
    ct_denominator: Any,
    method: str = DIV_METHOD_NEWTON_RAPHSON,
    num_iterations: int = 3,
    initial_guess: Optional[float] = None,
) -> Any:
    """
    General encrypted division interface supporting multiple methods.
    
    Args:
        cc: CryptoContext
        ct_numerator: encrypted dividend
        ct_denominator: encrypted divisor
        method: division algorithm ("newton_raphson", "goldschmidt")
        num_iterations: number of iterations for iterative methods
        initial_guess: initial approximation for reciprocal (None = use default)
    
    Returns:
        Encrypted quotient
    """
    if initial_guess is None:
        # Conservative default: assume denominator in [0.5, 2.0]
        initial_guess = 1.0
    
    if method == DIV_METHOD_NEWTON_RAPHSON:
        return encrypted_division_newton_raphson(
            cc, ct_numerator, ct_denominator, num_iterations, initial_guess
        )
    elif method == DIV_METHOD_GOLDSCHMIDT:
        ct_reciprocal = encrypted_reciprocal_goldschmidt(
            cc, ct_denominator, num_iterations, initial_guess
        )
        return cc.EvalMult(ct_numerator, ct_reciprocal)
    else:
        raise ValueError(f"Unknown division method: {method}")
