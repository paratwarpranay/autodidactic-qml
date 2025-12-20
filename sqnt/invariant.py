"""Soft spectral mass invariant for constrained autodidactic dynamics.

This module implements the minimal invariant that stabilizes interpretation
without biasing the outcome. The invariant enforces a finite "budget" constraint
on the system, allowing structure to emerge through negotiation rather than
runaway amplification or trivial collapse.

Mathematical foundation:
    - Invariant: I(M) = ||M||_F^2 = Σ_{ij} |M_ij|^2
    - Penalty: Φ(M) = (I(M) - I_0)^2
    - Gradient: ∇_M Φ = 4(I(M) - I_0) M
    - Update correction: -ε · ∇_M Φ

Properties (Fletcher-clean):
    ✓ State-intrinsic: depends only on M
    ✓ Weakly enforced: violations incur pressure, not hard projection
    ✓ Interpretation-free: no encoded preference for structure
    ✓ Symmetry-respecting: gauge/basis invariant

What this does NOT constrain:
    - Spectral shape
    - Topology structure
    - Entropy direction
    - Module/cluster formation

What this DOES prevent:
    - Runaway spectral blow-up
    - Trivial zero collapse
    - Numerical domination artifacts

Author: Christopher Altman
Date: 2025
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple


@dataclass
class SpectralMassInvariant:
    """Soft spectral mass budget constraint.
    
    The spectral mass invariant tracks ||M||_F^2 and applies a restoring
    force when the system drifts from its initial budget. This creates
    a "thermostat" rather than a hard constraint.
    
    Usage:
        invariant = SpectralMassInvariant()
        invariant.initialize(M0)  # Record initial budget
        
        for t in range(T):
            delta_M = compute_update(M)
            correction = invariant.correction_term(M)
            M = M + delta_M + correction
            invariant.log_step(M)
    
    Attributes:
        epsilon: Strength of restoring force (default 1e-4)
        I_0: Initial spectral mass (set by initialize())
        history: List of (I_t, delta_I) tuples for diagnostics
    """
    epsilon: float = 1e-4
    I_0: Optional[float] = field(default=None, repr=False)
    _history: List[Tuple[float, float]] = field(default_factory=list, repr=False)
    
    def spectral_mass(self, M: np.ndarray) -> float:
        """Compute spectral mass I(M) = ||M||_F^2.
        
        Args:
            M: Matrix (any shape)
            
        Returns:
            Frobenius norm squared
        """
        return float(np.linalg.norm(M, ord='fro') ** 2)
    
    def initialize(self, M0: np.ndarray) -> float:
        """Record initial spectral mass budget.
        
        This value is stored and never recomputed. All future corrections
        reference this initial budget.
        
        Args:
            M0: Initial matrix state
            
        Returns:
            Initial spectral mass I_0
        """
        self.I_0 = self.spectral_mass(M0)
        self._history = []
        return self.I_0
    
    def violation(self, M: np.ndarray) -> float:
        """Compute invariant violation δI = I(M) - I_0.
        
        Args:
            M: Current matrix state
            
        Returns:
            Signed violation (positive = overshoot, negative = undershoot)
            
        Raises:
            RuntimeError: If initialize() not called
        """
        if self.I_0 is None:
            raise RuntimeError("SpectralMassInvariant not initialized. Call initialize(M0) first.")
        I_t = self.spectral_mass(M)
        return I_t - self.I_0
    
    def correction_term(self, M: np.ndarray) -> np.ndarray:
        """Compute the soft invariant correction term.
        
        This is the gradient descent step on the penalty functional,
        with normalization and saturation for numerical stability:
        
            relative_violation = (I(M) - I_0) / I_0
            saturated_violation = tanh(relative_violation)  # bounded [-1, 1]
            correction = -ε · saturated_violation · M
        
        The correction:
            - Vanishes when invariant is satisfied
            - Grows smoothly with violation magnitude (up to saturation)
            - Is bounded to prevent numerical explosion
            - Points in direction to restore budget
            - Respects matrix symmetry
        
        Args:
            M: Current matrix state
            
        Returns:
            Correction term (same shape as M)
        """
        delta_I = self.violation(M)
        
        # Normalize by I_0 to make dimensionless
        relative_violation = delta_I / (self.I_0 + 1e-12)
        
        # Saturate to prevent numerical explosion (tanh bounds to [-1, 1])
        saturated_violation = np.tanh(relative_violation)
        
        # Apply correction: negative feedback toward I_0
        return -self.epsilon * saturated_violation * M
    
    def corrected_update(
        self, 
        M: np.ndarray, 
        delta_M: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Apply update with invariant correction.
        
        Computes: M_new = M + delta_M + correction_term(M)
        
        Uses the stable, normalized correction term with saturation.
        
        Args:
            M: Current matrix state
            delta_M: Proposed update (from dynamics)
            
        Returns:
            Tuple of:
                - M_new: Updated matrix with correction
                - diagnostics: Dict with violation, correction_norm, etc.
        """
        delta_I = self.violation(M)
        
        # Normalized and saturated correction (matches correction_term)
        relative_violation = delta_I / (self.I_0 + 1e-12)
        saturated_violation = np.tanh(relative_violation)
        correction = -self.epsilon * saturated_violation * M
        
        M_new = M + delta_M + correction
        
        # Preserve symmetry if input was symmetric
        if np.allclose(M, M.T):
            M_new = (M_new + M_new.T) / 2.0
        
        diagnostics = {
            'I_current': self.spectral_mass(M),
            'I_0': self.I_0,
            'delta_I': delta_I,
            'relative_violation': abs(relative_violation),
            'saturated_violation': float(saturated_violation),
            'correction_norm': float(np.linalg.norm(correction, 'fro')),
            'update_norm': float(np.linalg.norm(delta_M, 'fro')),
            'correction_ratio': float(np.linalg.norm(correction, 'fro')) / (float(np.linalg.norm(delta_M, 'fro')) + 1e-12),
        }
        
        return M_new, diagnostics
    
    def log_step(self, M: np.ndarray) -> Dict[str, float]:
        """Log current state for diagnostics.
        
        Args:
            M: Current matrix state
            
        Returns:
            Dict with I_t, delta_I, relative_violation
        """
        I_t = self.spectral_mass(M)
        delta_I = I_t - self.I_0 if self.I_0 is not None else 0.0
        self._history.append((I_t, delta_I))
        
        return {
            'I_t': I_t,
            'I_0': self.I_0,
            'delta_I': delta_I,
            'relative_violation': abs(delta_I) / (self.I_0 + 1e-12) if self.I_0 else 0.0,
        }
    
    @property
    def history(self) -> np.ndarray:
        """Get history as (T, 2) array of [I_t, delta_I]."""
        if not self._history:
            return np.empty((0, 2))
        return np.array(self._history)
    
    def is_stable(self, threshold: float = 0.1) -> bool:
        """Check if invariant is approximately satisfied.
        
        Args:
            threshold: Maximum allowed relative violation
            
        Returns:
            True if |δI/I_0| < threshold
        """
        if not self._history or self.I_0 is None:
            return True
        _, delta_I = self._history[-1]
        return abs(delta_I) / (self.I_0 + 1e-12) < threshold
    
    def summary(self) -> Dict[str, float]:
        """Get summary statistics of invariant behavior.
        
        Returns:
            Dict with mean/std/max violations and stability flag
        """
        if not self._history:
            return {
                'I_0': self.I_0,
                'n_steps': 0,
            }
        
        history = np.array(self._history)
        I_values = history[:, 0]
        delta_values = history[:, 1]
        rel_violations = np.abs(delta_values) / (self.I_0 + 1e-12)
        
        return {
            'I_0': self.I_0,
            'n_steps': len(self._history),
            'I_mean': float(np.mean(I_values)),
            'I_std': float(np.std(I_values)),
            'delta_I_mean': float(np.mean(delta_values)),
            'delta_I_std': float(np.std(delta_values)),
            'max_relative_violation': float(np.max(rel_violations)),
            'mean_relative_violation': float(np.mean(rel_violations)),
            'is_stable': bool(rel_violations[-1] < 0.1) if len(rel_violations) > 0 else True,
        }


def apply_invariant_correction(
    M: np.ndarray,
    delta_M: np.ndarray,
    I_0: float,
    epsilon: float = 1e-4,
) -> np.ndarray:
    """Functional interface for invariant-corrected update.
    
    Computes: M_new = M + delta_M - ε · tanh((||M||²_F - I_0) / I_0) · M
    
    Uses normalized, saturated correction for numerical stability.
    This is the stateless version for integration into existing loops.
    
    Args:
        M: Current matrix state
        delta_M: Proposed update
        I_0: Reference spectral mass
        epsilon: Correction strength
        
    Returns:
        Updated matrix with invariant correction
    """
    I_t = np.linalg.norm(M, ord='fro') ** 2
    delta_I = I_t - I_0
    
    # Normalized, saturated correction
    relative_violation = delta_I / (I_0 + 1e-12)
    saturated_violation = np.tanh(relative_violation)
    correction = -epsilon * saturated_violation * M
    
    M_new = M + delta_M + correction
    
    # Preserve symmetry
    if np.allclose(M, M.T):
        M_new = (M_new + M_new.T) / 2.0
    
    return M_new
