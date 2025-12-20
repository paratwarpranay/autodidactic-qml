"""Spectral diagnostics for matrix analysis.

Provides eigenvalue-based metrics for symmetric/Hermitian matrices.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class SpectralDiagnostics:
    """Compute spectral diagnostics for symmetric matrices.
    
    Caches eigenvalue computation to avoid redundant decompositions
    when computing multiple derived metrics.
    """

    def summarize(
        self, 
        M: np.ndarray, 
        prefix: str = "",
        eigenvalues: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Compute spectral summary statistics.
        
        Args:
            M: Symmetric/Hermitian matrix
            prefix: Optional prefix for metric keys (e.g., "M_" -> "M_eig_mean")
            eigenvalues: Pre-computed eigenvalues (avoids recomputation)
            
        Returns:
            Dictionary of spectral metrics
        """
        if eigenvalues is not None:
            w = eigenvalues
        else:
            w = np.linalg.eigvalsh(M)
        
        abs_w = np.abs(w)
        
        return {
            f"{prefix}eig_mean": float(np.mean(w)),
            f"{prefix}eig_std": float(np.std(w)),
            f"{prefix}eig_min": float(np.min(w)),
            f"{prefix}eig_max": float(np.max(w)),
            f"{prefix}spectral_radius": float(np.max(abs_w)),
            f"{prefix}condition": float((np.max(abs_w) + 1e-12) / (np.min(abs_w) + 1e-12)),
        }
    
    def full_diagnostics(
        self,
        M: np.ndarray,
        prefix: str = "",
    ) -> Dict[str, float]:
        """Extended spectral diagnostics including entropy and gap.
        
        Args:
            M: Symmetric/Hermitian matrix
            prefix: Optional prefix for metric keys
            
        Returns:
            Extended dictionary of spectral metrics
        """
        w = np.linalg.eigvalsh(M)
        abs_w = np.abs(w)
        
        # Basic stats
        result = self.summarize(M, prefix=prefix, eigenvalues=w)
        
        # Spectral gap (difference between two largest eigenvalues)
        sorted_w = np.sort(w)[::-1]
        if len(sorted_w) >= 2:
            result[f"{prefix}spectral_gap"] = float(sorted_w[0] - sorted_w[1])
        else:
            result[f"{prefix}spectral_gap"] = 0.0
        
        # Spectral entropy (normalized eigenvalue distribution)
        pos_w = abs_w[abs_w > 1e-12]
        if len(pos_w) > 0:
            p = pos_w / np.sum(pos_w)
            entropy = -np.sum(p * np.log(p + 1e-12))
            max_entropy = np.log(len(pos_w)) if len(pos_w) > 1 else 1.0
            result[f"{prefix}spectral_entropy"] = float(entropy)
            result[f"{prefix}spectral_entropy_norm"] = float(entropy / max_entropy)
        else:
            result[f"{prefix}spectral_entropy"] = 0.0
            result[f"{prefix}spectral_entropy_norm"] = 0.0
        
        # Effective rank (participation ratio)
        if np.sum(abs_w) > 1e-12:
            p = abs_w / np.sum(abs_w)
            eff_rank = 1.0 / (np.sum(p**2) + 1e-12)
            result[f"{prefix}effective_rank"] = float(eff_rank)
        else:
            result[f"{prefix}effective_rank"] = 0.0
        
        return result
