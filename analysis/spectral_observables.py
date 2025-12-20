"""Spectral observables module for tracking matrix-model observables over time.

This module provides a unified interface for computing and tracking gauge-invariant
spectral observables that can be monitored during learning or time evolution.

Scientific context:
- Observables are gauge-invariant functions of the matrix that characterize its state
- Tracking observables over time reveals phase transitions, learning dynamics, and stability
- The observable-based view connects to statistical mechanics and thermodynamics

Performance optimizations:
- Eigenvalue decomposition cached and shared across observables
- Vectorized rolling statistics using cumulative sums (O(n) vs O(n*window))
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Callable, Optional, Any, Tuple
from collections import deque

from .spectral_diagnostics import SpectralDiagnostics
from .spectral_complexity import (
    spectral_entropy,
    level_spacing_ratio,
    participation_ratio,
)


@dataclass
class Observable:
    """A single observable function.

    Attributes:
        name: Identifier for the observable
        compute: Function that takes (M, cache) and returns a scalar
        is_extensive: Whether the observable scales with system size
        category: Classification (e.g., "spectral", "topological", "complexity")
        needs_eigenvectors: Whether full eigendecomposition needed (vs just eigenvalues)
    """
    name: str
    compute: Callable[[np.ndarray, Dict[str, Any]], float]
    is_extensive: bool = False
    category: str = "spectral"
    needs_eigenvectors: bool = False


def _vectorized_rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    """O(n) rolling mean using cumulative sums."""
    if len(values) < window:
        return np.full(len(values), np.nan)
    
    cumsum = np.cumsum(np.insert(values, 0, 0))
    rolling = (cumsum[window:] - cumsum[:-window]) / window
    
    # Pad beginning with NaN
    return np.concatenate([np.full(window - 1, np.nan), rolling])


def _vectorized_rolling_std(values: np.ndarray, window: int) -> np.ndarray:
    """O(n) rolling standard deviation using cumulative sums."""
    if len(values) < window:
        return np.full(len(values), np.nan)
    
    n = len(values)
    # Cumsum for mean
    cumsum = np.cumsum(np.insert(values, 0, 0))
    cumsum_sq = np.cumsum(np.insert(values ** 2, 0, 0))
    
    # Rolling sums
    sum_x = cumsum[window:] - cumsum[:-window]
    sum_x2 = cumsum_sq[window:] - cumsum_sq[:-window]
    
    # Variance = E[X²] - E[X]²
    mean_x = sum_x / window
    mean_x2 = sum_x2 / window
    variance = mean_x2 - mean_x ** 2
    variance = np.maximum(variance, 0)  # Numerical stability
    
    std = np.sqrt(variance)
    return np.concatenate([np.full(window - 1, np.nan), std])


@dataclass
class ObservableHistory:
    """Track the history of an observable over time.

    Provides running statistics and change detection with O(n) vectorized operations.
    """
    name: str
    maxlen: int = 1000
    values: deque = field(default_factory=lambda: deque(maxlen=1000))
    
    # Cached statistics (invalidated on append)
    _cache_valid: bool = field(default=False, repr=False)
    _cached_array: Optional[np.ndarray] = field(default=None, repr=False)

    def __post_init__(self):
        self.values = deque(maxlen=self.maxlen)
        self._cache_valid = False
        self._cached_array = None

    def append(self, value: float) -> None:
        """Add a new observation (invalidates cache)."""
        self.values.append(value)
        self._cache_valid = False

    def _get_array(self) -> np.ndarray:
        """Get values as numpy array (cached)."""
        if not self._cache_valid or self._cached_array is None:
            self._cached_array = np.array(list(self.values), dtype=float)
            self._cache_valid = True
        return self._cached_array

    def mean(self) -> float:
        """Running mean."""
        arr = self._get_array()
        if len(arr) == 0:
            return 0.0
        return float(np.mean(arr))

    def std(self) -> float:
        """Running standard deviation."""
        arr = self._get_array()
        if len(arr) < 2:
            return 0.0
        return float(np.std(arr))

    def trend(self, window: int = 10) -> float:
        """Linear trend in recent window (slope)."""
        arr = self._get_array()
        if len(arr) < window:
            return 0.0
        recent = arr[-window:]
        x = np.arange(len(recent))
        slope = np.polyfit(x, recent, 1)[0]
        return float(slope)

    def rolling_mean(self, window: int = 10) -> np.ndarray:
        """Vectorized rolling mean (O(n))."""
        return _vectorized_rolling_mean(self._get_array(), window)

    def rolling_std(self, window: int = 10) -> np.ndarray:
        """Vectorized rolling std (O(n))."""
        return _vectorized_rolling_std(self._get_array(), window)

    def jump_detected(self, threshold: float = 3.0) -> bool:
        """Detect if latest value is a significant jump."""
        arr = self._get_array()
        if len(arr) < 10:
            return False
        baseline = arr[:-1]
        mu = np.mean(baseline)
        sigma = np.std(baseline) + 1e-12
        return abs(arr[-1] - mu) > threshold * sigma

    def to_array(self) -> np.ndarray:
        """Convert history to numpy array."""
        return self._get_array().copy()


class EigenCache:
    """Cache for eigenvalue decomposition to avoid redundant computation."""
    
    def __init__(self):
        self._matrix_hash: Optional[int] = None
        self._eigenvalues: Optional[np.ndarray] = None
        self._eigenvectors: Optional[np.ndarray] = None
    
    def _hash_matrix(self, M: np.ndarray) -> int:
        """Compute fast hash for matrix identity check."""
        return hash((M.shape, M.dtype.name, M.tobytes()[:1000]))
    
    def get(self, M: np.ndarray, need_vectors: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Get eigenvalues/vectors, computing only if necessary.
        
        Args:
            M: Matrix to decompose (will be symmetrized)
            need_vectors: Whether eigenvectors are needed
            
        Returns:
            (eigenvalues, eigenvectors or None)
        """
        M_sym = (M + M.T) / 2
        current_hash = self._hash_matrix(M_sym)
        
        # Check if we need to recompute
        if self._matrix_hash != current_hash:
            self._matrix_hash = current_hash
            self._eigenvectors = None
            
            if need_vectors:
                self._eigenvalues, self._eigenvectors = np.linalg.eigh(M_sym)
            else:
                self._eigenvalues = np.linalg.eigvalsh(M_sym)
        
        # Upgrade to full decomposition if vectors now needed
        if need_vectors and self._eigenvectors is None:
            self._eigenvalues, self._eigenvectors = np.linalg.eigh(M_sym)
        
        return self._eigenvalues, self._eigenvectors
    
    def clear(self):
        """Clear the cache."""
        self._matrix_hash = None
        self._eigenvalues = None
        self._eigenvectors = None


class SpectralObservables:
    """Unified interface for computing spectral observables.

    This class provides a standard set of observables that characterize
    the spectral properties of matrices in a gauge-invariant way.
    
    Optimization: Uses EigenCache to compute eigendecomposition once per matrix,
    then share across all observables.

    Example usage:
        obs = SpectralObservables()
        M = np.random.randn(10, 10)
        M = (M + M.T) / 2

        # Compute all observables (eigendecomposition done once)
        values = obs.compute_all(M)

        # Track over time
        for t in range(100):
            M_t = evolve(M)
            obs.update(M_t)

        # Get statistics
        stats = obs.get_statistics()
    """

    def __init__(self, track_history: bool = True, maxlen: int = 1000):
        """Initialize spectral observables tracker.

        Args:
            track_history: Whether to maintain history for all observables
            maxlen: Maximum history length per observable
        """
        self.track_history = track_history
        self.maxlen = maxlen
        self._observables: Dict[str, Observable] = {}
        self._history: Dict[str, ObservableHistory] = {}
        self._eigen_cache = EigenCache()

        # Register default observables
        self._register_defaults()

    def _register_defaults(self) -> None:
        """Register the standard set of spectral observables."""

        # Basic trace powers (don't need eigenvalues)
        self.register(Observable(
            name="trace",
            compute=lambda M, c: float(np.trace(M)),
            is_extensive=True,
            category="trace_powers",
        ))

        self.register(Observable(
            name="trace_squared",
            compute=lambda M, c: float(np.trace(M @ M)),
            is_extensive=True,
            category="trace_powers",
        ))

        self.register(Observable(
            name="trace_fourth",
            compute=lambda M, c: float(np.trace(np.linalg.matrix_power(M, 4))),
            is_extensive=True,
            category="trace_powers",
        ))

        # Eigenvalue statistics (use cache)
        self.register(Observable(
            name="spectral_radius",
            compute=lambda M, c: float(np.max(np.abs(c["eigenvalues"]))),
            category="spectral",
        ))

        self.register(Observable(
            name="eigenvalue_mean",
            compute=lambda M, c: float(np.mean(c["eigenvalues"])),
            category="spectral",
        ))

        self.register(Observable(
            name="eigenvalue_std",
            compute=lambda M, c: float(np.std(c["eigenvalues"])),
            category="spectral",
        ))

        self.register(Observable(
            name="spectral_gap",
            compute=self._compute_spectral_gap,
            category="spectral",
        ))

        # Complexity measures (use cache)
        self.register(Observable(
            name="spectral_entropy",
            compute=lambda M, c: spectral_entropy(c["eigenvalues"]),
            category="complexity",
        ))

        self.register(Observable(
            name="level_spacing_r",
            compute=lambda M, c: level_spacing_ratio(c["eigenvalues"])["r_mean"],
            category="complexity",
        ))

        self.register(Observable(
            name="participation_ratio_mean",
            compute=self._compute_mean_pr,
            category="complexity",
            needs_eigenvectors=True,
        ))

        # Norms (don't need eigenvalues)
        self.register(Observable(
            name="frobenius_norm",
            compute=lambda M, c: float(np.linalg.norm(M, "fro")),
            is_extensive=True,
            category="norms",
        ))

        self.register(Observable(
            name="operator_norm",
            compute=lambda M, c: float(np.max(np.abs(c["eigenvalues"]))),
            category="norms",
        ))

        # Condition (use cache)
        self.register(Observable(
            name="condition_number",
            compute=self._compute_condition,
            category="stability",
        ))

    def _compute_spectral_gap(self, M: np.ndarray, cache: Dict[str, Any]) -> float:
        """Compute gap between two smallest eigenvalues."""
        eigs = np.sort(cache["eigenvalues"])
        if len(eigs) < 2:
            return 0.0
        return float(eigs[1] - eigs[0])

    def _compute_mean_pr(self, M: np.ndarray, cache: Dict[str, Any]) -> float:
        """Compute mean participation ratio."""
        vecs = cache.get("eigenvectors")
        if vecs is None:
            # Fallback: compute eigenvectors
            M_sym = (M + M.T) / 2
            _, vecs = np.linalg.eigh(M_sym)
        pr = participation_ratio(vecs)
        return float(np.mean(pr))

    def _compute_condition(self, M: np.ndarray, cache: Dict[str, Any]) -> float:
        """Compute condition number robustly."""
        eigs = np.abs(cache["eigenvalues"])
        eigs_pos = eigs[eigs > 1e-12]
        if len(eigs_pos) < 2:
            return 1.0
        return float(np.max(eigs_pos) / np.min(eigs_pos))

    def register(self, obs: Observable) -> None:
        """Register a new observable.

        Args:
            obs: Observable object to register
        """
        self._observables[obs.name] = obs
        if self.track_history:
            self._history[obs.name] = ObservableHistory(
                name=obs.name, maxlen=self.maxlen
            )

    def _build_cache(self, M: np.ndarray) -> Dict[str, Any]:
        """Build computation cache for observables.
        
        Computes eigendecomposition once and shares across all observables.
        """
        # Check if any observable needs eigenvectors
        need_vectors = any(
            obs.needs_eigenvectors for obs in self._observables.values()
        )
        
        eigenvalues, eigenvectors = self._eigen_cache.get(M, need_vectors)
        
        return {
            "eigenvalues": eigenvalues,
            "eigenvectors": eigenvectors,
        }

    def compute(self, name: str, M: np.ndarray) -> float:
        """Compute a single observable.

        Args:
            name: Observable name
            M: Matrix to analyze

        Returns:
            Observable value
        """
        if name not in self._observables:
            raise KeyError(f"Observable '{name}' not registered")

        cache = self._build_cache(M)
        value = self._observables[name].compute(M, cache)
        if isinstance(value, dict):
            # Handle observables that return dicts
            value = list(value.values())[0]
        return float(value)

    def compute_all(self, M: np.ndarray) -> Dict[str, float]:
        """Compute all registered observables.

        Optimized: eigendecomposition computed once and shared.

        Args:
            M: Matrix to analyze

        Returns:
            Dictionary of observable name -> value
        """
        cache = self._build_cache(M)
        
        results = {}
        for name, obs in self._observables.items():
            try:
                results[name] = float(obs.compute(M, cache))
            except Exception:
                results[name] = np.nan
        return results

    def update(self, M: np.ndarray) -> Dict[str, float]:
        """Compute all observables and update history.

        Args:
            M: Matrix to analyze

        Returns:
            Dictionary of current observable values
        """
        values = self.compute_all(M)

        if self.track_history:
            for name, value in values.items():
                if np.isfinite(value):
                    self._history[name].append(value)

        return values

    def get_history(self, name: str) -> np.ndarray:
        """Get history array for an observable.

        Args:
            name: Observable name

        Returns:
            Array of historical values
        """
        if name not in self._history:
            raise KeyError(f"No history for '{name}'")
        return self._history[name].to_array()

    def get_rolling_statistics(
        self, 
        name: str, 
        window: int = 10,
    ) -> Dict[str, np.ndarray]:
        """Get vectorized rolling statistics for an observable.
        
        Args:
            name: Observable name
            window: Window size for rolling computation
            
        Returns:
            Dictionary with rolling_mean and rolling_std arrays
        """
        if name not in self._history:
            raise KeyError(f"No history for '{name}'")
        
        hist = self._history[name]
        return {
            "rolling_mean": hist.rolling_mean(window),
            "rolling_std": hist.rolling_std(window),
        }

    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all tracked observables.

        Returns:
            Nested dict: observable_name -> {mean, std, trend, n_samples}
        """
        stats = {}
        for name, hist in self._history.items():
            stats[name] = {
                "mean": hist.mean(),
                "std": hist.std(),
                "trend": hist.trend(),
                "n_samples": len(hist.values),
                "jump_detected": hist.jump_detected(),
            }
        return stats

    def detect_phase_transitions(
        self,
        threshold: float = 3.0,
    ) -> Dict[str, bool]:
        """Detect phase transitions based on observable jumps.

        Args:
            threshold: Number of standard deviations for jump detection

        Returns:
            Dictionary of observable name -> transition_detected
        """
        transitions = {}
        for name, hist in self._history.items():
            transitions[name] = hist.jump_detected(threshold=threshold)
        return transitions

    def reset_history(self) -> None:
        """Clear all history."""
        for hist in self._history.values():
            hist.values.clear()
            hist._cache_valid = False
        self._eigen_cache.clear()

    def list_observables(self) -> List[str]:
        """List all registered observable names."""
        return list(self._observables.keys())

    def by_category(self, category: str) -> List[str]:
        """Get observable names by category.

        Args:
            category: Category to filter by

        Returns:
            List of observable names in that category
        """
        return [
            name for name, obs in self._observables.items()
            if obs.category == category
        ]


def compute_thermodynamic_observables(M: np.ndarray, beta: float = 1.0) -> Dict[str, float]:
    """Compute observables with thermodynamic interpretation.

    Uses the matrix eigenvalues as an "energy spectrum" and computes
    thermodynamic quantities at inverse temperature beta.

    Args:
        M: Symmetric matrix (Hamiltonian-like)
        beta: Inverse temperature

    Returns:
        Dictionary of thermodynamic observables
    """
    M_sym = (M + M.T) / 2
    eigs = np.linalg.eigvalsh(M_sym)

    # Partition function
    Z = np.sum(np.exp(-beta * eigs))

    # Boltzmann probabilities
    p = np.exp(-beta * eigs) / Z

    # Energy (internal energy)
    E = np.sum(eigs * p)

    # Free energy
    F = -np.log(Z) / beta if beta > 0 else E

    # Entropy
    S = -np.sum(p * np.log(p + 1e-12))

    # Heat capacity (from energy fluctuations)
    E2 = np.sum(eigs**2 * p)
    C = beta**2 * (E2 - E**2)

    # Susceptibility (from order parameter fluctuations)
    # Using largest eigenvalue as "magnetization"
    m = np.max(np.abs(eigs))
    chi = beta * np.var(np.abs(eigs) * p * len(eigs))

    return {
        "partition_function": float(Z),
        "internal_energy": float(E),
        "free_energy": float(F),
        "entropy": float(S),
        "heat_capacity": float(C),
        "susceptibility": float(chi),
        "order_parameter": float(m),
    }
