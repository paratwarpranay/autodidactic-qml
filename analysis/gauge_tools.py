from __future__ import annotations
import numpy as np
from typing import Optional

def random_orthogonal(n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Generate a random orthogonal matrix U using QR decomposition.

    Uses a standard normal matrix A, QR factorization A = QR, then
    fixes the sign ambiguity so Q is uniformly distributed over O(n)
    (up to numerical details).

    Returns:
        U: (n, n) orthogonal matrix with U.T @ U ≈ I.
    """
    if rng is None:
        rng = np.random.default_rng()
    A = rng.standard_normal((n, n))
    Q, R = np.linalg.qr(A)
    # Fix signs so diag(R) positive
    d = np.sign(np.diag(R))
    d[d == 0] = 1.0
    Q = Q * d
    return Q

def gauge_transform(M: np.ndarray, U: np.ndarray) -> np.ndarray:
    """Apply conjugation M' = U M U^T (orthogonal gauge)."""
    return U @ M @ U.T
