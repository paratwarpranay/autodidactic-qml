from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np

def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    # numerically stable sigmoid
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out

@dataclass(frozen=True)
class LogisticModel:
    w: np.ndarray   # (d,)
    b: float
    mu: np.ndarray  # feature standardization mean
    sd: np.ndarray  # feature standardization std

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        Xz = (X - self.mu) / (self.sd + 1e-12)
        return sigmoid(Xz @ self.w + float(self.b))

def standardize(X: np.ndarray, eps: float = 1e-12) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=float)
    mu = np.mean(X, axis=0)
    sd = np.std(X, axis=0)
    sd = np.where(sd < eps, 1.0, sd)
    return (X - mu) / sd, mu, sd

def fit_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    l2: float = 1e-2,
    max_iter: int = 50,
    tol: float = 1e-6,
    seed: int = 0,
) -> LogisticModel:
    """Fit logistic regression with L2 regularization via Newton steps.

    Uses standardized features internally.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    assert X.shape[0] == y.size
    Xz, mu, sd = standardize(X)

    n, d = Xz.shape
    w = np.zeros(d, dtype=float)
    b = 0.0

    # Newton-Raphson with damping
    for _ in range(int(max_iter)):
        z = Xz @ w + b
        p = sigmoid(z)
        r = p * (1.0 - p)  # (n,)
        # gradient
        g_w = Xz.T @ (p - y) + l2 * w
        g_b = float(np.sum(p - y))
        # Hessian
        # H_w = X^T R X + l2 I
        XR = Xz * r[:, None]
        H = XR.T @ Xz + l2 * np.eye(d)
        # cross terms with b
        h_wb = Xz.T @ r
        h_bb = float(np.sum(r)) + 1e-12

        # Solve block system:
        # [H  h_wb] [dw] = -[g_w]
        # [h_wb^T h_bb] [db]   -[g_b]
        # Schur complement
        try:
            H_inv_h = np.linalg.solve(H, h_wb)
            schur = h_bb - float(h_wb.T @ H_inv_h)
            schur = float(max(schur, 1e-12))
            rhs_w = -g_w
            rhs_b = -g_b
            db = (rhs_b - float(h_wb.T @ np.linalg.solve(H, rhs_w))) / schur
            dw = np.linalg.solve(H, rhs_w - h_wb * db)
        except np.linalg.LinAlgError:
            # fallback: small gradient step if ill-conditioned
            dw = -0.1 * g_w
            db = -0.1 * g_b

        step = float(np.linalg.norm(dw) + abs(db))
        # simple damping
        alpha = 1.0
        # backtracking to ensure improvement (optional)
        for _bt in range(10):
            w_new = w + alpha * dw
            b_new = b + alpha * db
            # objective (neg log-likelihood + l2)
            z_new = Xz @ w_new + b_new
            p_new = sigmoid(z_new)
            eps = 1e-12
            nll_new = -float(np.sum(y*np.log(p_new+eps) + (1-y)*np.log(1-p_new+eps))) + 0.5*l2*float(w_new@w_new)
            z_old = Xz @ w + b
            p_old = sigmoid(z_old)
            nll_old = -float(np.sum(y*np.log(p_old+eps) + (1-y)*np.log(1-p_old+eps))) + 0.5*l2*float(w@w)
            if nll_new <= nll_old + 1e-10:
                w, b = w_new, float(b_new)
                break
            alpha *= 0.5

        if step < tol:
            break

    return LogisticModel(w=w, b=float(b), mu=mu, sd=sd)

def boundary_gstar_from_scores(
    g_values: np.ndarray,
    axis2_values: np.ndarray,
    score_grid: np.ndarray,
    threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract boundary g*(axis2) by locating threshold crossings along g for each axis2 slice.

    Uses linear interpolation between nearest points bracketing threshold.
    Returns (axis2, g_star).
    """
    g = np.asarray(g_values, dtype=float).reshape(-1)
    a2 = np.asarray(axis2_values, dtype=float).reshape(-1)
    S = np.asarray(score_grid, dtype=float)
    assert S.shape == (a2.size, g.size)

    g_star = np.full(a2.size, np.nan, dtype=float)
    for i in range(a2.size):
        s = S[i, :]
        if not np.isfinite(s).any():
            continue
        # We assume a single crossing; pick first crossing from low g to high g
        for j in range(1, g.size):
            s0, s1 = float(s[j-1]), float(s[j])
            if not (np.isfinite(s0) and np.isfinite(s1)):
                continue
            if (s0 - threshold) * (s1 - threshold) <= 0 and s0 != s1:
                # interpolate
                t = (threshold - s0) / (s1 - s0)
                g_star[i] = float(g[j-1] + t * (g[j] - g[j-1]))
                break
    return a2, g_star

def bootstrap_logistic_boundary(
    X: np.ndarray,
    y: np.ndarray,
    g_values: np.ndarray,
    axis2_values: np.ndarray,
    feature_index_map: np.ndarray,
    l2: float = 1e-2,
    n_boot: int = 200,
    seed: int = 0,
    threshold: float = 0.5,
) -> Dict[str, object]:
    """Bootstrap uncertainty on boundary line.

    Inputs:
      X: (n_points,d) feature matrix (raw, unstandardized)
      y: (n_points,) labels in {0,1}
      g_values, axis2_values define the grid
      feature_index_map: (n_points,2) giving (i_axis2,i_g) for each row of X
    Returns dict with median/p05/p95 for g*(axis2).
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int).reshape(-1)
    idx_map = np.asarray(feature_index_map, dtype=int)
    a2 = np.asarray(axis2_values, dtype=float).reshape(-1)
    g = np.asarray(g_values, dtype=float).reshape(-1)

    n = X.shape[0]
    curves = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        model = fit_logistic_regression(X[idx], y[idx], l2=l2, max_iter=60, tol=1e-6, seed=seed)
        p = model.predict_proba(X)  # predict on full grid points
        # reconstruct score grid
        S = np.full((a2.size, g.size), np.nan, dtype=float)
        for pp, (ia2, ig) in zip(p, idx_map):
            if 0 <= ia2 < a2.size and 0 <= ig < g.size:
                S[int(ia2), int(ig)] = float(pp)
        _, g_star = boundary_gstar_from_scores(g, a2, S, threshold=threshold)
        curves.append(g_star)

    C = np.asarray(curves, dtype=float)  # (n_boot, axis2)
    med = np.nanmedian(C, axis=0)
    p05 = np.nanquantile(C, 0.05, axis=0)
    p95 = np.nanquantile(C, 0.95, axis=0)
    return {
        "n_boot": int(n_boot),
        "threshold": float(threshold),
        "g_star_median": med.tolist(),
        "g_star_p05": p05.tolist(),
        "g_star_p95": p95.tolist(),
    }
