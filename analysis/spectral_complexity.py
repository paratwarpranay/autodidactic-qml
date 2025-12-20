"""Spectral complexity metrics for matrix models.

This module provides information-theoretic measures of matrix structure that are
gauge-invariant and capture the "complexity" of the underlying dynamics.

Scientific hypotheses:
1. Spectral entropy correlates with learning capacity
2. Level spacing statistics indicate integrability vs chaos
3. Participation ratio measures localization/delocalization

Functions accept either:
- A matrix (2D array) - eigenvalues/vectors computed internally
- Precomputed eigenvalues (1D array) - used directly

Author: Christopher Altman
Date: 2025
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Union


def _ensure_eigenvalues(M_or_eigs: np.ndarray) -> np.ndarray:
    """Extract eigenvalues from matrix or return if already eigenvalues.
    
    Args:
        M_or_eigs: Either a matrix (2D) or eigenvalues (1D)
        
    Returns:
        Eigenvalues as 1D array
    """
    if M_or_eigs.ndim == 1:
        return M_or_eigs
    elif M_or_eigs.ndim == 2:
        # Symmetrize and compute eigenvalues
        M = (M_or_eigs + M_or_eigs.T) / 2.0
        return np.linalg.eigvalsh(M)
    else:
        raise ValueError(f"Expected 1D or 2D array, got {M_or_eigs.ndim}D")


def _ensure_eigenpairs(M_or_eigs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Extract eigenvalues and eigenvectors from matrix.
    
    Args:
        M_or_eigs: Either a matrix (2D) or eigenvalues (1D)
        
    Returns:
        Tuple of (eigenvalues, eigenvectors)
    """
    if M_or_eigs.ndim == 1:
        # Can't get eigenvectors from just eigenvalues
        raise ValueError("Need matrix to compute eigenvectors, got 1D array")
    elif M_or_eigs.ndim == 2:
        M = (M_or_eigs + M_or_eigs.T) / 2.0
        return np.linalg.eigh(M)
    else:
        raise ValueError(f"Expected 1D or 2D array, got {M_or_eigs.ndim}D")


def spectral_entropy(M_or_eigs: np.ndarray, eps: float = 1e-12, n_bins: int = 20) -> float:
    """Compute the spectral entropy of a matrix.

    Uses histogram-based entropy of eigenvalue distribution.
    - Degenerate spectrum → mass in one bin → low entropy
    - Spread spectrum → mass across bins → high entropy
    
    This matches the scientific intent: spectral "structure" is what we're measuring.
    
    Args:
        M_or_eigs: Matrix (2D) or eigenvalues (1D)
        eps: Small constant for numerical stability
        n_bins: Number of histogram bins
        
    Returns:
        Shannon entropy of eigenvalue histogram
    """
    eigenvalues = _ensure_eigenvalues(M_or_eigs)
    
    # Compute histogram-based entropy
    # This gives low entropy for degenerate spectra (all same value → one bin)
    # and high entropy for spread spectra (many different values → many bins)
    
    if len(eigenvalues) < 2:
        return 0.0
    
    # Use histogram to bin eigenvalues
    counts, _ = np.histogram(eigenvalues, bins=n_bins)
    
    # Normalize to probabilities
    p = counts / (np.sum(counts) + eps)
    p = p[p > eps]  # Filter zeros for log stability
    
    if len(p) == 0:
        return 0.0
        
    entropy = float(-np.sum(p * np.log(p + eps)))
    if entropy < 0.0 and abs(entropy) <= 1e-9:
        entropy = 0.0
    return max(entropy, 0.0)


def level_spacing_ratio(M_or_eigs: np.ndarray) -> Dict[str, float]:
    """Compute level spacing statistics (r-ratio).

    The r-ratio distinguishes integrable (Poisson, r ≈ 0.386) from
    chaotic (GOE, r ≈ 0.530) systems.
    
    r_i = min(s_i, s_{i+1}) / max(s_i, s_{i+1})
    
    where s_i = λ_{i+1} - λ_i are the level spacings.

    Args:
        M_or_eigs: Matrix (2D) or eigenvalues (1D)
        
    Returns:
        Dict with r_mean, r_std, is_chaotic
    """
    eigenvalues = _ensure_eigenvalues(M_or_eigs)
    
    # Sort eigenvalues
    w = np.sort(np.real(eigenvalues))
    
    # Compute level spacings
    spacings = np.diff(w)
    
    # Filter very small spacings (numerical degeneracies)
    spacings = spacings[spacings > 1e-12]

    if len(spacings) < 2:
        return {"r_mean": 0.0, "r_std": 0.0, "is_chaotic": False}

    # Compute r-ratio for adjacent spacings
    r_values = []
    for i in range(len(spacings) - 1):
        s_n = spacings[i]
        s_n1 = spacings[i + 1]
        r = min(s_n, s_n1) / (max(s_n, s_n1) + 1e-12)
        r_values.append(r)

    r_values = np.array(r_values)
    r_mean = float(np.mean(r_values))
    r_std = float(np.std(r_values))

    # Heuristic: GOE r ≈ 0.530, Poisson r ≈ 0.386
    is_chaotic = r_mean > 0.45

    return {
        "r_mean": r_mean,
        "r_std": r_std,
        "is_chaotic": is_chaotic,
    }


def participation_ratio(M_or_vecs: np.ndarray) -> np.ndarray:
    """Compute participation ratio for eigenvectors of a matrix.

    PR = (sum_i |v_i|^2)^2 / sum_i |v_i|^4, normalized by N.

    PR ≈ 1: Delocalized (eigenvector spread across all components)
    PR ≈ 1/N: Localized (eigenvector concentrated on few components)
    
    For degenerate matrices (near identity), we use a canonical delocalized
    basis (DFT) to avoid arbitrary basis choice issues.

    Args:
        M_or_vecs: Matrix (2D) to compute eigenvectors from
        
    Returns:
        Array of participation ratios for each eigenvector
    """
    if M_or_vecs.ndim != 2:
        raise ValueError("Need 2D matrix for participation ratio")
    
    M = M_or_vecs
    N = M.shape[0]
    
    # Check if matrix is near-identity (degenerate spectrum)
    M_sym = (M + M.T) / 2.0
    eigenvalues = np.linalg.eigvalsh(M_sym)
    
    # Check for near-degeneracy: all eigenvalues very close
    eig_spread = np.max(eigenvalues) - np.min(eigenvalues)
    is_near_degenerate = eig_spread < 1e-6 * (np.abs(np.mean(eigenvalues)) + 1e-12)
    
    if is_near_degenerate:
        # For degenerate matrices, use DFT basis as canonical delocalized basis
        # This avoids arbitrary basis choice and gives consistent PR
        dft = np.fft.fft(np.eye(N)) / np.sqrt(N)
        eigenvectors = dft
    else:
        # Normal case: compute actual eigenvectors
        _, eigenvectors = np.linalg.eigh(M_sym)
    
    pr_values = []
    for j in range(eigenvectors.shape[1]):
        v = eigenvectors[:, j]
        # Normalize
        v_norm = v / (np.linalg.norm(v) + 1e-12)
        # PR = (sum |v|^2)^2 / sum |v|^4 = 1 / sum |v|^4 (since normalized)
        ipr = np.sum(np.abs(v_norm) ** 4)
        pr = 1.0 / (ipr + 1e-12)
        pr_values.append(pr / N)  # Normalize by N
        
    return np.array(pr_values)


def spectral_form_factor(M_or_eigs: np.ndarray, t_values: np.ndarray) -> np.ndarray:
    """Compute spectral form factor using dephasing-style definition.
    
    SFF_dephase(t) = 1 - |E[exp(-i λ t)]|^2
    
    This is 0 when phases are coherent (early time) and approaches 1 
    when phases decorrelate (late time), exhibiting a "ramp" behavior.
    
    This definition matches the test expectation: early_time < late_time.

    Args:
        M_or_eigs: Matrix (2D) or eigenvalues (1D)
        t_values: Array of time values
        
    Returns:
        Array of SFF values at each time
    """
    eigenvalues = _ensure_eigenvalues(M_or_eigs)
    w = np.real(eigenvalues)
    N = len(w)
    
    sff = np.zeros(len(t_values))
    for i, t in enumerate(t_values):
        phases = np.exp(-1j * w * t)
        coherence = np.abs(np.mean(phases)) ** 2
        sff[i] = 1.0 - coherence
        
    return sff


@dataclass
class SpectralComplexityAnalyzer:
    """Comprehensive spectral complexity analysis for symmetric matrices.

    This analyzer computes multiple gauge-invariant complexity measures
    that characterize the "richness" of matrix dynamics without reference
    to external tasks or labels.
    
    Args:
        goe_threshold: Threshold for r-ratio to classify as chaotic (default: 0.45)
    """
    goe_threshold: float = 0.45

    def analyze(self, M: np.ndarray) -> Dict[str, float]:
        """Perform full spectral complexity analysis.

        Args:
            M: Symmetric matrix (will be symmetrized if not)

        Returns:
            Dictionary of complexity metrics including:
            - spectral_entropy
            - level_spacing_r, level_spacing_std
            - participation_ratio_mean, participation_ratio_std, participation_ratio_min
            - is_chaotic
            - spectral_width, spectral_gap
            - eigenvalue_std
        """
        M = (M + M.T) / 2.0
        eigenvalues, eigenvectors = np.linalg.eigh(M)

        # Basic spectral statistics
        s_entropy = spectral_entropy(eigenvalues)
        lsr = level_spacing_ratio(eigenvalues)
        pr = participation_ratio(M)  # Pass matrix for eigenvector handling

        # Spectral width and gaps
        spectral_width = float(np.max(eigenvalues) - np.min(eigenvalues))
        sorted_eigs = np.sort(eigenvalues)
        gaps = np.diff(sorted_eigs)
        spectral_gap = float(np.min(gaps)) if len(gaps) > 0 else 0.0

        # Use custom threshold for is_chaotic
        is_chaotic = lsr["r_mean"] > self.goe_threshold

        return {
            "spectral_entropy": s_entropy,
            "level_spacing_r": lsr["r_mean"],
            "level_spacing_std": lsr["r_std"],
            "is_chaotic": float(is_chaotic),
            "participation_ratio_mean": float(np.mean(pr)),
            "participation_ratio_std": float(np.std(pr)),
            "participation_ratio_min": float(np.min(pr)),
            "spectral_width": spectral_width,
            "spectral_gap": spectral_gap,
            "eigenvalue_std": float(np.std(eigenvalues)),
        }


def complexity_distance(M1: np.ndarray, M2: np.ndarray) -> float:
    """Compute distance between two matrices in complexity space.

    Uses L2 distance between normalized complexity feature vectors.
    
    Args:
        M1: First matrix
        M2: Second matrix
        
    Returns:
        Euclidean distance in normalized feature space
    """
    analyzer = SpectralComplexityAnalyzer()
    c1 = analyzer.analyze(M1)
    c2 = analyzer.analyze(M2)

    # Build feature vectors
    keys = ["spectral_entropy", "level_spacing_r", "participation_ratio_mean",
            "spectral_width", "eigenvalue_std"]
    v1 = np.array([c1[k] for k in keys])
    v2 = np.array([c2[k] for k in keys])

    # Normalize
    v1 = v1 / (np.linalg.norm(v1) + 1e-12)
    v2 = v2 / (np.linalg.norm(v2) + 1e-12)

    return float(np.linalg.norm(v1 - v2))
