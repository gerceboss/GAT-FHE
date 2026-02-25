"""
CKKS-only FHE utils: Newton-Raphson reciprocal, Chebyshev coefficients.
"""

from typing import Any


def encrypted_reciprocal_newton_raphson(
    cc: Any,
    ct_denominator: Any,
    num_iterations: int = 3,
    initial_guess: float = 1.0,
    slots: int = 8,
) -> Any:
    """Compute encrypted 1/d via Newton-Raphson in CKKS."""
    pt_x = cc.MakeCKKSPackedPlaintext([initial_guess] * slots)
    pt_two = cc.MakeCKKSPackedPlaintext([2.0] * slots)
    ct_x = cc.EvalMult(ct_denominator, pt_x)
    for _ in range(num_iterations):
        ct_dx = cc.EvalMult(ct_denominator, ct_x)
        ct_two_minus_dx = cc.EvalSub(pt_two, ct_dx)
        ct_x = cc.EvalMult(ct_x, ct_two_minus_dx)
    return ct_x


def get_leaky_relu_chebyshev_coefficients(
    negative_slope: float = 0.2,
    domain_low: float = -3.0,
    domain_high: float = 3.0,
    degree: int = 7,
) -> list:
    """Chebyshev coefficients for LeakyReLU in CKKS."""
    import numpy as np
    from numpy.polynomial import chebyshev as Ch
    x = np.linspace(domain_low, domain_high, 300)
    y = np.where(x >= 0.0, x, negative_slope * x)
    coeffs = Ch.chebfit(x, y, degree)
    return [float(c) for c in coeffs]


def get_sigmoid_chebyshev_coefficients(
    domain_low: float = -5.0,
    domain_high: float = 5.0,
    degree: int = 7,
) -> list:
    """Chebyshev coefficients for sigmoid in CKKS."""
    import numpy as np
    from numpy.polynomial import chebyshev as Ch
    x = np.linspace(domain_low, domain_high, 300)
    y = 1.0 / (1.0 + np.exp(-x))
    coeffs = Ch.chebfit(x, y, degree)
    return [float(c) for c in coeffs]
