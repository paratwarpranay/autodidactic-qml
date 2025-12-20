"""Adaptive thresholding for entanglement graph construction.

This module addresses the graph degeneracy problem identified in the audit:
in threshold mode, the entanglement graph can collapse to empty or fully
connected depending on a single hyperparameter.

Solution: adaptive threshold tied to spectral density percentile rather
than absolute value.

Key features:
- Percentile-based thresholding (invariant to matrix scale)
- Spectral density-aware thresholds
- Connectivity-preserving fallbacks
- Smooth interpolation for gradient-friendly dynamics

Author: Christopher Altman
Date: 2025
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Tuple, Dict, Literal


def compute_adaptive_threshold(
    M: np.ndarray,
    target_density: float = 0.3,
    method: Literal['percentile', 'spectral', 'otsu', 'connectivity'] = 'percentile',
    min_edges: Optional[int] = None,
    max_edges: Optional[int] = None,
) -> float:
    """Compute an adaptive threshold for graph construction.
    
    Unlike fixed thresholds that can cause degenerate graphs (empty or
    fully connected), adaptive thresholds scale with the matrix's
    actual value distribution.
    
    Args:
        M: Symmetric matrix (coupling strengths)
        target_density: Target edge density (0-1), used by percentile method
        method: Thresholding method
            - 'percentile': Threshold at (1 - target_density) percentile
            - 'spectral': Based on spectral gap of candidate graphs
            - 'otsu': Otsu's method for bimodal distributions
            - 'connectivity': Minimum threshold for connected graph
        min_edges: Minimum edges to preserve (optional)
        max_edges: Maximum edges to allow (optional)
        
    Returns:
        Adaptive threshold value
    """
    # Ensure symmetry and extract upper triangle (excluding diagonal)
    M_sym = (M + M.T) / 2.0
    np.fill_diagonal(M_sym, 0.0)
    M_abs = np.abs(M_sym)
    
    n = M.shape[0]
    offdiag = M_abs[np.triu_indices(n, k=1)]
    
    if len(offdiag) == 0:
        return 0.0
    
    if method == 'percentile':
        # Threshold such that target_density fraction of edges survive
        # Higher percentile → fewer edges → lower density
        percentile = 100.0 * (1.0 - target_density)
        tau = float(np.percentile(offdiag, percentile))
        
    elif method == 'spectral':
        # Find threshold that maximizes spectral gap
        tau = _spectral_gap_threshold(M_abs, n_candidates=20)
        
    elif method == 'otsu':
        # Otsu's method: maximize inter-class variance
        tau = _otsu_threshold(offdiag)
        
    elif method == 'connectivity':
        # Minimum threshold that keeps graph connected
        tau = _connectivity_threshold(M_abs)
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Apply edge count constraints if specified
    if min_edges is not None or max_edges is not None:
        tau = _adjust_threshold_for_edge_count(
            M_abs, tau, min_edges, max_edges
        )
    
    return tau


def _spectral_gap_threshold(M_abs: np.ndarray, n_candidates: int = 20) -> float:
    """Find threshold maximizing spectral gap (algebraic connectivity).
    
    The spectral gap λ_2 - λ_1 measures graph connectivity robustness.
    We want a threshold that produces a well-connected but not trivial graph.
    """
    n = M_abs.shape[0]
    offdiag = M_abs[np.triu_indices(n, k=1)]
    
    # Generate candidate thresholds
    candidates = np.linspace(
        np.min(offdiag),
        np.max(offdiag),
        n_candidates
    )
    
    best_tau = 0.0
    best_gap = -np.inf
    
    for tau in candidates:
        # Construct adjacency
        A = np.where(M_abs > tau, M_abs, 0.0)
        
        # Compute Laplacian spectrum
        D = np.diag(np.sum(A, axis=1))
        L = D - A
        eigvals = np.linalg.eigvalsh(L)
        eigvals = np.sort(np.maximum(eigvals, 0.0))
        
        # Skip if graph is disconnected or trivial
        if eigvals[1] < 1e-10 or np.sum(A > 0) < n:
            continue
        
        # Spectral gap weighted by edge sparsity (prefer sparser graphs)
        n_edges = np.sum(A > 0) / 2
        max_edges = n * (n - 1) / 2
        sparsity_bonus = 1.0 - (n_edges / max_edges)
        
        gap = eigvals[1] * (1.0 + 0.5 * sparsity_bonus)
        
        if gap > best_gap:
            best_gap = gap
            best_tau = tau
    
    return best_tau


def _otsu_threshold(values: np.ndarray) -> float:
    """Otsu's method for automatic threshold selection.
    
    Maximizes inter-class variance, assuming bimodal distribution
    (significant edges vs. noise).
    """
    # Compute histogram
    n_bins = min(256, len(values) // 10 + 1)
    hist, bin_edges = np.histogram(values, bins=n_bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Total mean and variance
    total_mean = np.sum(bin_centers * hist) / np.sum(hist)
    
    best_tau = bin_centers[0]
    best_variance = 0.0
    
    cumsum = 0.0
    cumsum_weighted = 0.0
    
    for i, (center, count) in enumerate(zip(bin_centers, hist)):
        cumsum += count
        cumsum_weighted += center * count
        
        if cumsum == 0 or cumsum == np.sum(hist):
            continue
        
        # Class probabilities
        w0 = cumsum / np.sum(hist)
        w1 = 1.0 - w0
        
        # Class means
        mu0 = cumsum_weighted / cumsum
        mu1 = (total_mean * np.sum(hist) - cumsum_weighted) / (np.sum(hist) - cumsum)
        
        # Inter-class variance
        variance = w0 * w1 * (mu0 - mu1) ** 2
        
        if variance > best_variance:
            best_variance = variance
            best_tau = center
    
    return best_tau


def _connectivity_threshold(M_abs: np.ndarray) -> float:
    """Find minimum threshold that maintains graph connectivity.
    
    Uses binary search to find the largest threshold where the graph
    remains connected (λ_2 > 0).
    """
    n = M_abs.shape[0]
    offdiag = M_abs[np.triu_indices(n, k=1)]
    
    low = np.min(offdiag)
    high = np.max(offdiag)
    
    # Binary search
    for _ in range(20):
        mid = (low + high) / 2
        
        A = np.where(M_abs > mid, M_abs, 0.0)
        D = np.diag(np.sum(A, axis=1))
        L = D - A
        eigvals = np.linalg.eigvalsh(L)
        eigvals = np.sort(np.maximum(eigvals, 0.0))
        
        if eigvals[1] > 1e-10:
            # Still connected, try higher threshold
            low = mid
        else:
            # Disconnected, lower threshold
            high = mid
    
    # Return slightly below the critical threshold
    return low * 0.95


def _adjust_threshold_for_edge_count(
    M_abs: np.ndarray,
    tau: float,
    min_edges: Optional[int],
    max_edges: Optional[int],
) -> float:
    """Adjust threshold to satisfy edge count constraints."""
    n = M_abs.shape[0]
    offdiag = M_abs[np.triu_indices(n, k=1)]
    
    # Current edge count
    n_edges = np.sum(offdiag > tau)
    
    if min_edges is not None and n_edges < min_edges:
        # Need more edges → lower threshold
        sorted_vals = np.sort(offdiag)[::-1]
        if min_edges < len(sorted_vals):
            tau = sorted_vals[min_edges - 1] * 0.99
    
    if max_edges is not None and n_edges > max_edges:
        # Need fewer edges → higher threshold
        sorted_vals = np.sort(offdiag)[::-1]
        if max_edges < len(sorted_vals):
            tau = sorted_vals[max_edges - 1] * 1.01
    
    return tau


def adaptive_adjacency(
    M: np.ndarray,
    target_density: float = 0.3,
    method: str = 'percentile',
    weighted: bool = True,
    alpha: float = 1.0,
    **kwargs,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Construct adjacency matrix with adaptive threshold.
    
    This is the recommended replacement for fixed-threshold graph
    construction to avoid degeneracy issues.
    
    Args:
        M: Symmetric matrix (coupling strengths)
        target_density: Target edge density for percentile method
        method: Thresholding method (see compute_adaptive_threshold)
        weighted: If True, use |M_ij|^alpha as weights; else binary
        alpha: Power for weight transformation
        **kwargs: Additional arguments for compute_adaptive_threshold
        
    Returns:
        Tuple of (adjacency_matrix, info_dict)
    """
    # Compute adaptive threshold
    tau = compute_adaptive_threshold(
        M, 
        target_density=target_density,
        method=method,
        **kwargs
    )
    
    # Construct adjacency
    M_sym = (M + M.T) / 2.0
    np.fill_diagonal(M_sym, 0.0)
    M_abs = np.abs(M_sym)
    
    if weighted:
        A = np.where(M_abs > tau, np.power(M_abs, alpha), 0.0)
    else:
        A = (M_abs > tau).astype(float)
    
    # Compute info
    n = A.shape[0]
    n_edges = int(np.sum(A > 0) / 2)
    max_edges = n * (n - 1) // 2
    actual_density = n_edges / max_edges if max_edges > 0 else 0.0
    
    info = {
        'threshold': tau,
        'method': method,
        'target_density': target_density,
        'actual_density': actual_density,
        'n_edges': n_edges,
        'max_edges': max_edges,
    }
    
    return A, info


def smooth_threshold_adjacency(
    M: np.ndarray,
    tau: float,
    sharpness: float = 10.0,
    weighted: bool = True,
) -> np.ndarray:
    """Construct adjacency with smooth (differentiable) threshold.
    
    Instead of hard threshold (step function), uses sigmoid for
    gradient-friendly topology adaptation.
    
    Args:
        M: Symmetric matrix
        tau: Center of sigmoid threshold
        sharpness: Steepness of transition (higher = sharper)
        weighted: If True, multiply by original values
        
    Returns:
        Smooth adjacency matrix
    """
    M_sym = (M + M.T) / 2.0
    np.fill_diagonal(M_sym, 0.0)
    M_abs = np.abs(M_sym)
    
    # Sigmoid mask
    mask = 1.0 / (1.0 + np.exp(-sharpness * (M_abs - tau)))
    
    if weighted:
        A = M_abs * mask
    else:
        A = mask
    
    return A
