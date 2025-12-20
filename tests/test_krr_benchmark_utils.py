import numpy as np

def _krr_fit(K, y, lam):
    n = K.shape[0]
    return np.linalg.solve(K + lam*np.eye(n), y)

def test_krr_solve_identity():
    K = np.eye(5)
    y = np.array([1, -1, 1, -1, 1], dtype=float)
    a = _krr_fit(K, y, lam=1e-3)
    # should be close to y (since K approx identity)
    assert np.allclose(a, y/(1+1e-3), atol=1e-6)
