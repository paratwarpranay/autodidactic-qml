"""Autodidactic phase detection module.

This module implements methods to detect phase transitions in autodidactic
learning systems without external supervision or labeled data.

Scientific hypotheses:
1. Learning systems undergo phase transitions analogous to physical systems
2. These transitions can be detected via:
   - Sharp changes in spectral properties
   - Discontinuities in information-theoretic measures
   - Singularities in the learning dynamics
3. Phase transitions mark boundaries between qualitatively different regimes:
   - Exploration vs exploitation
   - Law emergence vs law collapse
   - Integrable vs chaotic dynamics
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Union
from collections import deque


def finite_difference_derivative(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Compute numerical derivative dy/dx using central differences."""
    dy = np.gradient(y, x)
    return dy


def cusp_detector(values: np.ndarray, window: int = 5) -> np.ndarray:
    """Detect cusps (sharp changes) in a sequence.

    Returns cusp score at each point - higher means sharper change.
    """
    n = len(values)
    if n < 2 * window + 1:
        return np.zeros(n)

    scores = np.zeros(n)
    for i in range(window, n - window):
        left = np.mean(values[i-window:i])
        right = np.mean(values[i+1:i+window+1])
        center = values[i]

        # Cusp score: deviation from linear interpolation
        expected = (left + right) / 2
        scores[i] = abs(center - expected)

    return scores


def susceptibility(values: np.ndarray, parameter: np.ndarray) -> np.ndarray:
    """Compute susceptibility χ = |dO/dp| where O is observable, p is parameter.

    In physics, diverging susceptibility indicates a phase transition.
    """
    return np.abs(finite_difference_derivative(values, parameter))


def variance_peak_detector(
    values: np.ndarray,
    parameter: np.ndarray,
    window: int = 3,
) -> Dict[str, float]:
    """Detect peaks in local variance (fluctuations).

    At phase transitions, fluctuations often diverge.
    """
    n = len(values)
    local_var = np.zeros(n)

    for i in range(window, n - window):
        local_var[i] = np.var(values[i-window:i+window+1])

    # Find peaks
    peaks = []
    for i in range(1, n - 1):
        if local_var[i] > local_var[i-1] and local_var[i] > local_var[i+1]:
            peaks.append((parameter[i], local_var[i]))

    if not peaks:
        return {"transition_point": np.nan, "peak_variance": 0.0, "n_peaks": 0}

    # Return strongest peak
    best = max(peaks, key=lambda x: x[1])
    return {
        "transition_point": float(best[0]),
        "peak_variance": float(best[1]),
        "n_peaks": len(peaks),
    }


def binder_cumulant(values: np.ndarray) -> float:
    """Compute Binder cumulant U for distribution analysis.

    The Binder cumulant measures distribution shape/peakedness.
    Standard definition: U = <m²>² / <m⁴>
    
    Values:
        U = 1 for constant (delta function)  
        U ≈ 1/3 for Gaussian
        U ≈ 5/9 ≈ 0.56 for uniform
        
    At critical points in physical systems, U takes a universal value
    that depends only on symmetry class and dimensionality.
    """
    m2 = np.mean(values ** 2)
    m4 = np.mean(values ** 4)

    if m4 < 1e-12:
        return 1.0  # Effectively constant
    
    # Standard Binder cumulant: U = <m²>² / <m⁴>
    return float((m2 * m2) / m4)


class OnlinePhaseDetector:
    """Online detector for phase transitions during training.

    Monitors observables in real-time and flags potential phase transitions
    without needing the full trajectory in advance.
    
    Args:
        window_size: Size of the moving window for baseline statistics
        threshold: Number of standard deviations for anomaly detection
        sensitivity: Alias for threshold (backwards compatibility)
    """
    
    def __init__(
        self, 
        window_size: int = 50, 
        threshold: float = None,
        sensitivity: float = None,
    ):
        """Initialize detector with threshold or sensitivity parameter.
        
        Args:
            window_size: Size of moving window
            threshold: Standard deviations for anomaly (preferred name)
            sensitivity: Alias for threshold (for backwards compatibility)
        """
        self.window_size = window_size
        
        # Accept both 'threshold' and 'sensitivity' as parameter names
        if threshold is not None:
            self._threshold = threshold
        elif sensitivity is not None:
            self._threshold = sensitivity
        else:
            self._threshold = 2.0
        
        self.history: Dict[str, deque] = {}
        self.baselines: Dict[str, Tuple[float, float]] = {}
    
    @property
    def threshold(self) -> float:
        """Anomaly detection threshold in standard deviations."""
        return self._threshold
    
    @threshold.setter
    def threshold(self, value: float):
        self._threshold = value
    
    @property 
    def sensitivity(self) -> float:
        """Alias for threshold (backwards compatibility)."""
        return self._threshold
    
    @sensitivity.setter
    def sensitivity(self, value: float):
        self._threshold = value

    def update(self, observables: Dict[str, float]) -> Dict[str, bool]:
        """Update with new observations and check for transitions.

        Args:
            observables: Dictionary of metric_name -> value

        Returns:
            Dictionary of metric_name -> is_transition_detected
        """
        transitions = {}

        for name, value in observables.items():
            if name not in self.history:
                self.history[name] = deque(maxlen=self.window_size)

            self.history[name].append(value)

            if len(self.history[name]) < self.window_size:
                transitions[name] = False
                continue

            # Update baseline from history excluding current value
            values = np.array(self.history[name])
            mean = np.mean(values[:-1])
            std = np.std(values[:-1]) + 1e-12

            # Check for anomaly via z-score
            z_score = abs(value - mean) / std
            transitions[name] = z_score > self._threshold

            self.baselines[name] = (mean, std)

        return transitions

    def get_diagnostics(self) -> Dict[str, Dict[str, float]]:
        """Get diagnostic information about current state."""
        diagnostics = {}
        for name, values in self.history.items():
            if len(values) > 1:
                arr = np.array(values)
                diagnostics[name] = {
                    "current": float(arr[-1]),
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr)),
                    "trend": float(np.polyfit(np.arange(len(arr)), arr, 1)[0]),
                }
        return diagnostics


@dataclass
class PhaseTransitionAnalyzer:
    """Comprehensive analysis of phase transitions in parameter sweeps.

    Given a sweep of some control parameter (e.g., coupling strength g,
    temperature T, or learning rate), this analyzer identifies potential
    phase transition points using multiple complementary methods.
    """
    # Peak detection parameters
    peak_prominence: float = 0.5
    min_peak_distance: int = 3

    def analyze_sweep(
        self,
        parameter: np.ndarray,
        observables: Dict[str, np.ndarray],
    ) -> Dict[str, Dict]:
        """Analyze a parameter sweep for phase transitions.

        Args:
            parameter: Array of control parameter values
            observables: Dict of observable_name -> values_array

        Returns:
            Comprehensive analysis results with keys including:
            - susceptibility: array of susceptibility values
            - phase_boundaries: list of detected boundary locations
        """
        results = {}

        for name, values in observables.items():
            values = np.asarray(values)

            # Susceptibility (derivative magnitude)
            chi = susceptibility(values, parameter)

            # Cusp detection
            cusps = cusp_detector(values)

            # Variance peaks
            var_peaks = variance_peak_detector(values, parameter)

            # Find transition candidates from multiple methods
            candidates = []
            phase_boundaries = []

            # From susceptibility peaks
            chi_mean = np.mean(chi)
            chi_std = np.std(chi)
            chi_threshold = chi_mean + 2 * chi_std if chi_std > 1e-12 else chi_mean + 1e-6
            
            for i in range(1, len(chi) - 1):
                if chi[i] > chi_threshold and chi[i] > chi[i-1] and chi[i] > chi[i+1]:
                    candidates.append({
                        "parameter": float(parameter[i]),
                        "method": "susceptibility",
                        "strength": float(chi[i]),
                    })
                    phase_boundaries.append(float(parameter[i]))

            # From cusp peaks
            cusp_mean = np.mean(cusps)
            cusp_std = np.std(cusps)
            cusp_threshold = cusp_mean + 2 * cusp_std if cusp_std > 1e-12 else cusp_mean + 1e-6
            
            for i in range(1, len(cusps) - 1):
                if cusps[i] > cusp_threshold and cusps[i] > cusps[i-1] and cusps[i] > cusps[i+1]:
                    candidates.append({
                        "parameter": float(parameter[i]),
                        "method": "cusp",
                        "strength": float(cusps[i]),
                    })
                    if float(parameter[i]) not in phase_boundaries:
                        phase_boundaries.append(float(parameter[i]))

            # Consensus transition point
            if candidates:
                params = [c["parameter"] for c in candidates]
                consensus = float(np.median(params))
            else:
                consensus = np.nan

            results[name] = {
                "susceptibility": chi,
                "susceptibility_max": float(np.max(chi)),
                "susceptibility_max_at": float(parameter[np.argmax(chi)]),
                "cusp_max": float(np.max(cusps)),
                "variance_peak": var_peaks,
                "transition_candidates": candidates,
                "consensus_transition": consensus,
                "phase_boundaries": phase_boundaries,
                "binder_cumulant": binder_cumulant(values - np.mean(values)),
            }

        return results
    
    def find_critical_points(
        self,
        parameter: np.ndarray,
        order_param: np.ndarray,
        method: str = "susceptibility",
    ) -> List[float]:
        """Find critical points (phase transition locations) in a sweep.
        
        Args:
            parameter: Array of control parameter values
            order_param: Order parameter values (e.g., magnetization)
            method: Detection method ('susceptibility', 'cusp', 'variance', 'gradient')
            
        Returns:
            List of critical point locations (parameter values)
        """
        parameter = np.asarray(parameter)
        order_param = np.asarray(order_param)
        
        critical_points = []
        
        if method == "susceptibility":
            chi = susceptibility(order_param, parameter)
            chi_mean = np.mean(chi)
            chi_std = np.std(chi)
            threshold = chi_mean + 2 * chi_std if chi_std > 1e-12 else chi_mean * 2
            
            for i in range(1, len(chi) - 1):
                if chi[i] > threshold and chi[i] > chi[i-1] and chi[i] > chi[i+1]:
                    critical_points.append(float(parameter[i]))
                        
        elif method == "cusp":
            cusps = cusp_detector(order_param)
            cusp_mean = np.mean(cusps)
            cusp_std = np.std(cusps)
            threshold = cusp_mean + 2 * cusp_std if cusp_std > 1e-12 else cusp_mean * 2
            
            for i in range(1, len(cusps) - 1):
                if cusps[i] > threshold and cusps[i] > cusps[i-1] and cusps[i] > cusps[i+1]:
                    critical_points.append(float(parameter[i]))
                        
        elif method == "variance":
            result = variance_peak_detector(order_param, parameter)
            if not np.isnan(result["transition_point"]):
                critical_points.append(result["transition_point"])
        
        # Fallback: gradient-based detection for sharp transitions
        if len(critical_points) == 0:
            grad = np.abs(np.gradient(order_param, parameter))
            grad_mean = np.mean(grad)
            grad_std = np.std(grad)
            grad_threshold = grad_mean + 1.5 * grad_std if grad_std > 1e-12 else grad_mean * 2
            
            for i in range(1, len(grad) - 1):
                if grad[i] > grad_threshold and grad[i] > grad[i-1] and grad[i] > grad[i+1]:
                    critical_points.append(float(parameter[i]))
        
        return critical_points


def _compute_rolling_stats_vectorized(
    loss: np.ndarray, 
    window: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute rolling mean, std, and trend using vectorized operations.
    
    O(n) complexity via cumulative sums instead of O(n·window) loop.
    """
    n = len(loss)
    
    # Cumulative sums for efficient windowed mean/variance
    cumsum = np.concatenate([[0], np.cumsum(loss)])
    cumsum_sq = np.concatenate([[0], np.cumsum(loss ** 2)])
    
    # Rolling mean
    local_mean = np.zeros(n)
    for i in range(n):
        start = max(0, i - window + 1)
        end = i + 1
        count = end - start
        local_mean[i] = (cumsum[end] - cumsum[start]) / count
    
    # Rolling variance -> std
    local_std = np.zeros(n)
    for i in range(n):
        start = max(0, i - window + 1)
        end = i + 1
        count = end - start
        mean_sq = (cumsum_sq[end] - cumsum_sq[start]) / count
        sq_mean = local_mean[i] ** 2
        var = max(0, mean_sq - sq_mean)  # Numerical stability
        local_std[i] = np.sqrt(var) + 1e-12
    
    # Rolling trend (linear regression slope)
    local_trend = np.zeros(n)
    for i in range(window, n):
        segment = loss[i-window+1:i+1]
        x = np.arange(len(segment))
        local_trend[i] = np.polyfit(x, segment, 1)[0]
    
    return local_mean, local_std, local_trend


def detect_learning_phases(
    loss_history: Union[List[float], np.ndarray],
    window: int = 20,
    min_phase_duration: int = 15,
    transition_threshold: float = 0.1,
) -> Dict[str, any]:
    """Detect learning phases from loss trajectory with hysteresis.

    Identifies:
    - initial: Early training period
    - learning: Decreasing loss (active learning)
    - converged: Stable low loss
    - plateau: Stable but not decreasing
    - unstable: High variance or erratic behavior
    
    Uses hysteresis to prevent rapid flip-flopping between phases.

    Args:
        loss_history: Loss values over training
        window: Window size for local statistics
        min_phase_duration: Minimum steps before allowing phase change
        transition_threshold: Relative change threshold for transitions
        
    Returns:
        Dict with keys:
        - phases: List of phase labels per step
        - transitions: List of transition events
        - final_phase: Last phase
        - n_transitions: Number of transitions
        - n_phases: Number of distinct phase segments
    """
    loss = np.array(loss_history, dtype=float)
    n = len(loss)

    if n < 3 * window:
        return {
            "phases": ["insufficient_data"] * n if n > 0 else [],
            "transitions": [],
            "final_phase": "insufficient_data" if n > 0 else "",
            "n_transitions": 0,
            "n_phases": 1,
        }

    # Compute rolling statistics
    local_mean, local_std, local_trend = _compute_rolling_stats_vectorized(loss, window)

    # Classify each point into a raw phase
    raw_phases = []
    for i in range(n):
        if i < window:
            phase = "initial"
        else:
            relative_std = local_std[i] / (abs(local_mean[i]) + 1e-12)
            relative_trend = local_trend[i] / (local_std[i] + 1e-12)
            
            if relative_trend < -0.3:  # Strong decreasing trend
                phase = "learning"
            elif relative_std < 0.08 and abs(relative_trend) < 0.15:
                phase = "converged"
            elif abs(relative_trend) > 0.8:  # Very unstable
                phase = "unstable"
            else:
                phase = "plateau"
        
        raw_phases.append(phase)

    # Apply hysteresis: require sustained phase before switching
    phases = []
    current_phase = raw_phases[0]
    phase_start = 0
    pending_phase = None
    pending_start = 0
    
    for i in range(n):
        if raw_phases[i] != current_phase:
            if pending_phase is None or raw_phases[i] != pending_phase:
                # Start tracking a new potential phase
                pending_phase = raw_phases[i]
                pending_start = i
            
            # Check if pending phase has persisted long enough
            if pending_phase == raw_phases[i]:
                duration = i - pending_start + 1
                if duration >= min_phase_duration:
                    # Commit the phase change
                    current_phase = pending_phase
                    phase_start = pending_start
                    pending_phase = None
        else:
            # Current phase continues, reset pending
            pending_phase = None
        
        phases.append(current_phase)

    # Detect transitions (only where phase actually changes after hysteresis)
    transitions = []
    for i in range(1, n):
        if phases[i] != phases[i-1]:
            transitions.append({
                "step": i,
                "from": phases[i-1],
                "to": phases[i],
                "loss": float(loss[i]),
            })

    # Count distinct phase segments
    n_phases = 1
    for i in range(1, n):
        if phases[i] != phases[i-1]:
            n_phases += 1

    return {
        "phases": phases,
        "transitions": transitions,
        "final_phase": phases[-1] if phases else "",
        "n_transitions": len(transitions),
        "n_phases": n_phases,
    }
