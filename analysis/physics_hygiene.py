"""Physics hygiene layer for autodidactic dynamics.

This module implements invariant assertions and state validation
to prevent silent shape drift, numerical instabilities, and
hidden Hilbert sector jumps.

The hygiene layer should be inserted after EVERY evolution step
to catch problems early rather than letting them propagate.

Key invariants enforced:
1. Shape consistency (N×N)
2. Hermiticity (M = M†)
3. Finiteness (no NaN, Inf)
4. Spectral bounds (eigenvalues in expected range)
5. Trace conservation (optional)

Author: Christopher Altman
Date: 2025
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple, Callable
import warnings


class PhysicsViolation(Exception):
    """Raised when physics invariants are violated."""
    pass


@dataclass
class HygieneConfig:
    """Configuration for physics hygiene checks.
    
    Attributes:
        check_shape: Verify matrix shape is (N, N)
        check_hermiticity: Verify M ≈ M†
        check_finite: Verify no NaN/Inf values
        check_spectral_bounds: Verify eigenvalues in expected range
        check_trace: Verify trace conservation
        hermiticity_atol: Absolute tolerance for Hermiticity check
        spectral_max: Maximum allowed eigenvalue magnitude
        trace_rtol: Relative tolerance for trace conservation
        strict: If True, raise exception on violation; else warn
        auto_correct: If True, attempt to fix minor violations
    """
    check_shape: bool = True
    check_hermiticity: bool = True
    check_finite: bool = True
    check_spectral_bounds: bool = True
    check_trace: bool = False
    hermiticity_atol: float = 1e-10
    spectral_max: float = 1e6
    trace_rtol: float = 0.1
    strict: bool = False
    auto_correct: bool = True


@dataclass
class HygieneReport:
    """Report from a hygiene check.
    
    Attributes:
        passed: Whether all checks passed
        violations: List of violated invariants
        corrections: List of auto-corrections applied
        metrics: Computed validation metrics
    """
    passed: bool
    violations: List[str] = field(default_factory=list)
    corrections: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)


class PhysicsHygiene:
    """Physics hygiene checker for matrix dynamics.
    
    This class validates matrix states against physical invariants
    and optionally auto-corrects minor violations.
    
    Usage:
        hygiene = PhysicsHygiene(config)
        
        # Set reference state
        hygiene.set_reference(M0)
        
        # After each evolution step
        for t in range(T):
            M = evolve(M)
            report = hygiene.check(M)
            if not report.passed and hygiene.config.strict:
                raise PhysicsViolation(report.violations)
    
    The hygiene checker tracks:
    - Reference trace (for conservation checks)
    - Expected dimension
    - Violation history
    """
    
    def __init__(self, config: Optional[HygieneConfig] = None):
        """Initialize hygiene checker.
        
        Args:
            config: Configuration (uses defaults if None)
        """
        self.config = config or HygieneConfig()
        self._reference_trace: Optional[float] = None
        self._expected_dim: Optional[int] = None
        self._history: List[HygieneReport] = []
    
    def set_reference(self, M: np.ndarray) -> None:
        """Set reference state for conservation checks.
        
        Args:
            M: Reference matrix (typically initial state)
        """
        self._expected_dim = M.shape[0]
        self._reference_trace = float(np.trace(M))
    
    def check(
        self,
        M: np.ndarray,
        context: str = "",
    ) -> HygieneReport:
        """Run all configured hygiene checks.
        
        Args:
            M: Matrix to check
            context: Optional context string for error messages
            
        Returns:
            HygieneReport with results
            
        Raises:
            PhysicsViolation: If strict mode and violations found
        """
        violations = []
        corrections = []
        metrics = {}
        
        # Shape check
        if self.config.check_shape:
            if M.ndim != 2:
                violations.append(f"Matrix is not 2D: ndim={M.ndim}")
            elif M.shape[0] != M.shape[1]:
                violations.append(f"Matrix is not square: shape={M.shape}")
            elif self._expected_dim is not None and M.shape[0] != self._expected_dim:
                violations.append(
                    f"Dimension changed: expected {self._expected_dim}, got {M.shape[0]}"
                )
            metrics['dim'] = float(M.shape[0]) if M.ndim == 2 else 0.0
        
        # Finiteness check
        if self.config.check_finite:
            n_nan = np.sum(np.isnan(M))
            n_inf = np.sum(np.isinf(M))
            if n_nan > 0:
                violations.append(f"Matrix contains {n_nan} NaN values")
            if n_inf > 0:
                violations.append(f"Matrix contains {n_inf} Inf values")
            metrics['n_nan'] = float(n_nan)
            metrics['n_inf'] = float(n_inf)
        
        # Hermiticity check
        if self.config.check_hermiticity and M.ndim == 2 and M.shape[0] == M.shape[1]:
            hermiticity_error = np.max(np.abs(M - M.conj().T))
            metrics['hermiticity_error'] = float(hermiticity_error)
            
            if hermiticity_error > self.config.hermiticity_atol:
                if self.config.auto_correct:
                    # Auto-correct by symmetrizing
                    M_corrected = (M + M.conj().T) / 2.0
                    new_error = np.max(np.abs(M_corrected - M_corrected.conj().T))
                    if new_error <= self.config.hermiticity_atol:
                        corrections.append(
                            f"Symmetrized matrix (error: {hermiticity_error:.2e} → {new_error:.2e})"
                        )
                        np.copyto(M, M_corrected)
                    else:
                        violations.append(
                            f"Hermiticity violation: max|M - M†| = {hermiticity_error:.2e}"
                        )
                else:
                    violations.append(
                        f"Hermiticity violation: max|M - M†| = {hermiticity_error:.2e}"
                    )
        
        # Spectral bounds check
        if self.config.check_spectral_bounds and M.ndim == 2 and M.shape[0] == M.shape[1]:
            try:
                # Use Hermitian eigensolver if matrix is approximately Hermitian
                if np.max(np.abs(M - M.conj().T)) < 1e-8:
                    eigvals = np.linalg.eigvalsh(M)
                else:
                    eigvals = np.linalg.eigvals(M)
                
                max_eig = np.max(np.abs(eigvals))
                min_eig = np.min(np.real(eigvals))
                metrics['max_eigenvalue'] = float(max_eig)
                metrics['min_eigenvalue'] = float(min_eig)
                
                if max_eig > self.config.spectral_max:
                    violations.append(
                        f"Spectral bound exceeded: max|λ| = {max_eig:.2e} > {self.config.spectral_max:.2e}"
                    )
            except np.linalg.LinAlgError:
                violations.append("Eigenvalue computation failed")
        
        # Trace conservation check
        if self.config.check_trace and self._reference_trace is not None:
            current_trace = float(np.trace(M))
            trace_change = abs(current_trace - self._reference_trace)
            relative_change = trace_change / (abs(self._reference_trace) + 1e-12)
            metrics['trace'] = current_trace
            metrics['trace_change'] = float(trace_change)
            metrics['trace_relative_change'] = float(relative_change)
            
            if relative_change > self.config.trace_rtol:
                violations.append(
                    f"Trace drift: {self._reference_trace:.4f} → {current_trace:.4f} "
                    f"(relative change: {relative_change:.2e})"
                )
        
        # Create report
        passed = len(violations) == 0
        report = HygieneReport(
            passed=passed,
            violations=violations,
            corrections=corrections,
            metrics=metrics,
        )
        
        self._history.append(report)
        
        # Handle violations
        if not passed:
            msg = f"Physics hygiene violations{' at ' + context if context else ''}:\n"
            msg += "\n".join(f"  - {v}" for v in violations)
            
            if self.config.strict:
                raise PhysicsViolation(msg)
            else:
                warnings.warn(msg, RuntimeWarning)
        
        return report
    
    def check_and_correct(
        self,
        M: np.ndarray,
        context: str = "",
    ) -> Tuple[np.ndarray, HygieneReport]:
        """Check and return corrected matrix.
        
        Args:
            M: Matrix to check
            context: Optional context string
            
        Returns:
            Tuple of (corrected_matrix, report)
        """
        M_copy = M.copy()
        report = self.check(M_copy, context)
        return M_copy, report
    
    @property
    def history(self) -> List[HygieneReport]:
        """Get history of hygiene checks."""
        return self._history
    
    def summary(self) -> Dict[str, any]:
        """Get summary statistics."""
        if not self._history:
            return {'n_checks': 0}
        
        n_passed = sum(1 for r in self._history if r.passed)
        n_failed = len(self._history) - n_passed
        n_corrected = sum(len(r.corrections) for r in self._history)
        
        return {
            'n_checks': len(self._history),
            'n_passed': n_passed,
            'n_failed': n_failed,
            'n_corrected': n_corrected,
            'pass_rate': n_passed / len(self._history),
        }
    
    def reset(self) -> None:
        """Reset history and reference."""
        self._history = []
        self._reference_trace = None
        self._expected_dim = None


def assert_physics_invariants(
    M: np.ndarray,
    N: Optional[int] = None,
    atol: float = 1e-10,
    context: str = "",
) -> None:
    """Assert critical physics invariants (one-liner version).
    
    Use this for quick inline checks:
        M = evolve(M)
        assert_physics_invariants(M, N=dim)
    
    Args:
        M: Matrix to check
        N: Expected dimension (optional)
        atol: Tolerance for Hermiticity
        context: Context for error messages
        
    Raises:
        AssertionError: If any invariant violated
    """
    prefix = f"[{context}] " if context else ""
    
    # Shape check
    assert M.ndim == 2, f"{prefix}Matrix is not 2D: ndim={M.ndim}"
    assert M.shape[0] == M.shape[1], f"{prefix}Matrix is not square: shape={M.shape}"
    if N is not None:
        assert M.shape[0] == N, f"{prefix}Dimension changed: expected {N}, got {M.shape[0]}"
    
    # Finiteness check FIRST (before Hermiticity, since NaN propagates)
    assert np.isfinite(M).all(), f"{prefix}Matrix contains NaN or Inf values"
    
    # Hermiticity check (safe now that we know no NaN/Inf)
    herm_error = np.max(np.abs(M - M.conj().T))
    assert herm_error <= atol, f"{prefix}Hermiticity violation: max|M - M†| = {herm_error:.2e}"


def create_atomic_updater(
    hygiene: Optional[PhysicsHygiene] = None,
) -> Callable:
    """Create an atomic update wrapper.
    
    This ensures updates follow the pattern:
        snapshot → compute → update → normalize → commit → validate
    
    Args:
        hygiene: Optional hygiene checker
        
    Returns:
        Decorator for update functions
    """
    def decorator(update_fn: Callable) -> Callable:
        def atomic_update(M: np.ndarray, *args, **kwargs) -> np.ndarray:
            # 1. Snapshot
            M_snapshot = M.copy()
            N = M.shape[0]
            
            # 2. Compute update
            delta_M = update_fn(M_snapshot, *args, **kwargs)
            
            # 3. Apply update
            M_new = M_snapshot + delta_M
            
            # 4. Normalize (preserve Hermiticity)
            M_new = (M_new + M_new.conj().T) / 2.0
            
            # 5. Validate
            if hygiene is not None:
                hygiene.check(M_new, context="atomic_update")
            else:
                assert_physics_invariants(M_new, N=N)
            
            # 6. Commit
            return M_new
        
        return atomic_update
    
    return decorator


@dataclass
class AtomicUpdateResult:
    """Result of an atomic update operation.
    
    Attributes:
        M_new: Updated matrix
        delta_M: Applied update
        hygiene_report: Optional hygiene report
    """
    M_new: np.ndarray
    delta_M: np.ndarray
    hygiene_report: Optional[HygieneReport] = None


def atomic_step(
    M: np.ndarray,
    compute_update: Callable[[np.ndarray], np.ndarray],
    hygiene: Optional[PhysicsHygiene] = None,
    context: str = "",
) -> AtomicUpdateResult:
    """Execute a single atomic update step.
    
    Enforces the pattern:
        snapshot → compute → update → normalize → validate → commit
    
    Args:
        M: Current matrix state
        compute_update: Function M -> delta_M
        hygiene: Optional hygiene checker
        context: Context for error messages
        
    Returns:
        AtomicUpdateResult with new state and diagnostics
    """
    # 1. Snapshot (immutable reference)
    M_snapshot = M.copy()
    N = M.shape[0]
    
    # 2. Compute update from snapshot (not live state)
    delta_M = compute_update(M_snapshot)
    
    # 3. Apply update
    M_new = M_snapshot + delta_M
    
    # 4. Normalize (enforce Hermiticity)
    M_new = (M_new + M_new.conj().T) / 2.0
    
    # 5. Validate
    report = None
    if hygiene is not None:
        report = hygiene.check(M_new, context=context)
    else:
        assert_physics_invariants(M_new, N=N, context=context)
    
    # 6. Return result (caller decides whether to commit)
    return AtomicUpdateResult(
        M_new=M_new,
        delta_M=delta_M,
        hygiene_report=report,
    )
