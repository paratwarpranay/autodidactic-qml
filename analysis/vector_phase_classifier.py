from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np

def zscore(X: np.ndarray, eps: float = 1e-12) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=float)
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd = np.where(sd < eps, 1.0, sd)
    return (X - mu) / sd, mu, sd

def impute_nan_with_col_mean(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float).copy()
    mu = np.nanmean(X, axis=0)
    inds = np.where(~np.isfinite(X))
    if inds[0].size > 0:
        X[inds] = np.take(mu, inds[1])
    # still might have NaNs if an entire column was NaN
    X = np.where(np.isfinite(X), X, 0.0)
    return X

@dataclass(frozen=True)
class KMeansResult:
    centroids: np.ndarray      # (k,d) in standardized space
    labels: np.ndarray         # (n,)
    inertia: float

def kmeans(
    X: np.ndarray,
    k: int = 2,
    n_init: int = 10,
    max_iter: int = 100,
    seed: int = 0,
) -> KMeansResult:
    """Simple k-means (Lloyd) with multiple restarts. No external deps."""
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    assert n >= k, "Need at least k points"

    best: Optional[KMeansResult] = None
    for t in range(int(n_init)):
        # init centroids from random unique points
        idx = rng.choice(n, size=k, replace=False)
        C = X[idx].copy()

        labels = np.zeros(n, dtype=int)
        for _ in range(int(max_iter)):
            # assign
            dist2 = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)  # (n,k)
            new_labels = np.argmin(dist2, axis=1)
            if np.array_equal(new_labels, labels):
                labels = new_labels
                break
            labels = new_labels
            # update
            for j in range(k):
                pts = X[labels == j]
                if pts.shape[0] > 0:
                    C[j] = pts.mean(axis=0)
                else:
                    # re-seed empty cluster
                    C[j] = X[rng.integers(0, n)]
        # inertia
        dist2 = ((X - C[labels]) ** 2).sum(axis=1)
        inertia = float(dist2.sum())
        cur = KMeansResult(centroids=C, labels=labels, inertia=inertia)
        if best is None or cur.inertia < best.inertia:
            best = cur
    assert best is not None
    return best

def soft_assignment_scores(X: np.ndarray, centroids: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Return soft scores p(cluster=1) for k=2 based on distance softmax."""
    X = np.asarray(X, dtype=float)
    C = np.asarray(centroids, dtype=float)
    assert C.shape[0] == 2, "soft scores implemented for k=2"
    dist2 = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)  # (n,2)
    # softmax over -dist2/T
    z = -dist2 / max(float(temperature), 1e-9)
    z = z - z.max(axis=1, keepdims=True)
    ez = np.exp(z)
    p = ez / (ez.sum(axis=1, keepdims=True) + 1e-12)
    return p[:, 1]  # probability of cluster 1

def build_feature_matrix_from_grids(
    feature_grids: Dict[str, np.ndarray],
    axis2_values: np.ndarray,
    g_values: np.ndarray,
    feature_keys: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """Flatten grids into X (n_points, d) and keep mapping indices -> (i_axis2, i_g)."""
    a2 = np.asarray(axis2_values, dtype=float).reshape(-1)
    g = np.asarray(g_values, dtype=float).reshape(-1)
    idx_map = []
    rows = []
    for i in range(a2.size):
        for j in range(g.size):
            feat = []
            ok = True
            for k in feature_keys:
                Z = np.asarray(feature_grids[k], dtype=float)
                val = float(Z[i, j])
                feat.append(val)
            rows.append(feat)
            idx_map.append((i, j))
    X = np.asarray(rows, dtype=float)
    idx_map = np.asarray(idx_map, dtype=int)
    return X, idx_map

def classify_phases_from_feature_grids(
    feature_grids: Dict[str, np.ndarray],
    axis2_values: np.ndarray,
    g_values: np.ndarray,
    feature_keys: List[str],
    k: int = 2,
    seed: int = 0,
    n_init: int = 10,
    max_iter: int = 100,
    soft_temp: float = 1.0,
) -> Dict[str, object]:
    """Cluster phases in feature space and return label grid + soft score grid."""
    X, idx_map = build_feature_matrix_from_grids(feature_grids, axis2_values, g_values, feature_keys)
    X = impute_nan_with_col_mean(X)
    Xz, mu, sd = zscore(X)

    km = kmeans(Xz, k=k, n_init=n_init, max_iter=max_iter, seed=seed)
    labels = km.labels
    label_grid = np.full((len(axis2_values), len(g_values)), -1, dtype=int)
    for (lab, (i, j)) in zip(labels, idx_map):
        label_grid[i, j] = int(lab)

    result: Dict[str, object] = {
        "k": int(k),
        "feature_keys": list(feature_keys),
        "mu": mu.tolist(),
        "sd": sd.tolist(),
        "centroids_z": km.centroids.tolist(),
        "inertia": float(km.inertia),
        "label_grid": label_grid.tolist(),
        "labels_flat": labels.tolist(),

        "X_raw": X.tolist(),
        "idx_map": idx_map.tolist(),

    }

    if k == 2:
        # soft score grid for cluster 1
        score = soft_assignment_scores(Xz, km.centroids, temperature=soft_temp)
        score_grid = np.full((len(axis2_values), len(g_values)), np.nan, dtype=float)
        for (s, (i, j)) in zip(score, idx_map):
            score_grid[i, j] = float(s)
        result["score_grid"] = score_grid.tolist()
        result["soft_temp"] = float(soft_temp)

    return result

def choose_cluster_semantics(
    feature_grids: Dict[str, np.ndarray],
    axis2_values: np.ndarray,
    g_values: np.ndarray,
    label_grid: np.ndarray,
    feature_key: str = "lap_gap_Mhat",
) -> Dict[str, int]:
    """Map raw cluster labels to semantic phase labels based on a feature mean.

    Returns mapping dict: {raw_label -> semantic_label}, where semantic_label 0=low,1=high.
    """
    Z = np.asarray(feature_grids[feature_key], dtype=float)
    L = np.asarray(label_grid, dtype=int)
    means = {}
    for lab in np.unique(L[L >= 0]):
        vals = Z[L == lab]
        means[int(lab)] = float(np.nanmean(vals)) if vals.size else float("nan")
    # define "high" as the cluster with larger mean(feature_key)
    labs = sorted(means.keys())
    if len(labs) < 2:
        return {int(labs[0]): 0} if labs else {}
    hi = max(labs, key=lambda k: means[k])
    lo = min(labs, key=lambda k: means[k])
    return {int(lo): 0, int(hi): 1}
