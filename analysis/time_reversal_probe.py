"""Time-reversal probe for autodidactic dynamics.

This module implements bidirectional evolution tests to probe the
irreversibility structure of the dynamics. This is a critical diagnostic
for distinguishing genuine learning/adaptation from trivial dynamics.

Physical interpretation:
- Reversible dynamics: No genuine information processing
- Irreversible dynamics: Real structural memory formation
- Partial reversibility: Indicates transition between regimes

Test protocol:
1. Run forward T steps
2. Reverse update rule (sign-flip learning rate)
3. Measure recovery fidelity

If recovery is complete → dynamics is reversible (no learning)
If recovery fails in specific ways → irreversibility is structural (real memory)

Author: Christopher Altman
Date: 2025
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Callable, Optional
from enum import Enum


class ReversibilityClass(Enum):
    """Classification of reversibility behavior."""
    FULLY_REVERSIBLE = "fully_reversible"
    PARTIALLY_REVERSIBLE = "partially_reversible"
    IRREVERSIBLE = "irreversible"
    CHAOTIC = "chaotic"  # No recovery even with perfect reversal
    UNSTABLE = "unstable"  # Diverges under reversal


@dataclass
class TimeReversalResult:
    """Result of a time-reversal probe.
    
    Attributes:
        forward_steps: Number of forward evolution steps
        reverse_steps: Number of reverse evolution steps
        initial_state: Initial matrix state
        forward_final: State after forward evolution
        reverse_final: State after reversal
        recovery_fidelity: How well the initial state was recovered (0-1)
        spectral_recovery: Recovery of eigenvalue spectrum
        topology_recovery: Recovery of graph topology
        reversibility_class: Classification of behavior
        trajectory_divergence: How trajectories differ in reverse
    """
    forward_steps: int
    reverse_steps: int
    initial_state: np.ndarray
    forward_final: np.ndarray
    reverse_final: np.ndarray
    recovery_fidelity: float
    spectral_recovery: float
    topology_recovery: float
    reversibility_class: ReversibilityClass
    trajectory_divergence: List[float] = field(default_factory=list)
    diagnostics: Dict[str, float] = field(default_factory=dict)


def matrix_fidelity(M1: np.ndarray, M2: np.ndarray) -> float:
    """Compute fidelity between two matrices.
    
    Uses cosine-style Frobenius overlap mapped to [0, 1]:
        F = 0.5 * (1 + Re(Tr(M1† M2)) / (||M1||_F ||M2||_F))
    
    This maps the normalized inner product from [-1, 1] to [0, 1], where:
        - 1.0 = identical (perfect positive overlap)
        - 0.5 = orthogonal (zero overlap)
        - 0.0 = anti-correlated (perfect negative overlap)
    
    Returns:
        Fidelity in [0, 1]
    """
    norm1 = np.linalg.norm(M1, 'fro')
    norm2 = np.linalg.norm(M2, 'fro')
    
    if norm1 < 1e-12 or norm2 < 1e-12:
        return 1.0 if (norm1 < 1e-12 and norm2 < 1e-12) else 0.0
    
    # Real part of normalized trace inner product
    inner = np.real(np.trace(M1.conj().T @ M2))
    cos_sim = inner / (norm1 * norm2)
    
    # Map [-1, 1] -> [0, 1] and clip for numerical safety
    fidelity = 0.5 * (1.0 + cos_sim)
    return float(np.clip(fidelity, 0.0, 1.0))


def spectral_fidelity(M1: np.ndarray, M2: np.ndarray) -> float:
    """Compute spectral recovery fidelity.
    
    Compares sorted eigenvalue spectra.
    """
    # Ensure Hermitian for real eigenvalues
    M1_h = (M1 + M1.conj().T) / 2
    M2_h = (M2 + M2.conj().T) / 2
    
    eig1 = np.sort(np.linalg.eigvalsh(M1_h))
    eig2 = np.sort(np.linalg.eigvalsh(M2_h))
    
    norm1 = np.linalg.norm(eig1)
    norm2 = np.linalg.norm(eig2)
    
    if norm1 < 1e-12 or norm2 < 1e-12:
        return 1.0 if (norm1 < 1e-12 and norm2 < 1e-12) else 0.0
    
    diff = np.linalg.norm(eig1 - eig2)
    return float(1.0 - diff / (norm1 + norm2))


def topology_fidelity(A1: np.ndarray, A2: np.ndarray) -> float:
    """Compute topology recovery fidelity.
    
    Uses correlation of flattened adjacency matrices.
    """
    a1 = A1.flatten()
    a2 = A2.flatten()
    
    if np.std(a1) < 1e-12 or np.std(a2) < 1e-12:
        return 1.0 if np.allclose(a1, a2) else 0.0
    
    corr = np.corrcoef(a1, a2)[0, 1]
    return float((corr + 1.0) / 2.0)  # Map [-1, 1] to [0, 1]


def classify_reversibility(
    recovery_fidelity: float,
    trajectory_variance: float,
    thresholds: Dict[str, float] = None,
) -> ReversibilityClass:
    """Classify reversibility behavior.
    
    Args:
        recovery_fidelity: How well initial state was recovered (0-1)
        trajectory_variance: Variance of trajectory divergence
        thresholds: Classification thresholds
        
    Returns:
        ReversibilityClass enum value
    """
    thresholds = thresholds or {
        'fully_reversible': 0.95,
        'partially_reversible': 0.5,
        'chaotic_variance': 0.1,
    }
    
    if recovery_fidelity > thresholds['fully_reversible']:
        return ReversibilityClass.FULLY_REVERSIBLE
    
    if trajectory_variance > thresholds['chaotic_variance']:
        return ReversibilityClass.CHAOTIC
    
    if recovery_fidelity > thresholds['partially_reversible']:
        return ReversibilityClass.PARTIALLY_REVERSIBLE
    
    if recovery_fidelity < 0.1:
        return ReversibilityClass.UNSTABLE
    
    return ReversibilityClass.IRREVERSIBLE


@dataclass
class TimeReversalProbe:
    """Time-reversal probe for testing dynamics reversibility.
    
    This probe runs the following protocol:
    1. Record initial state M_0
    2. Evolve forward: M_0 → M_1 → ... → M_T
    3. Reverse dynamics and evolve: M_T → M'_{T-1} → ... → M'_0
    4. Compare M'_0 with M_0
    
    The comparison reveals whether the dynamics is:
    - Reversible (Hamiltonian-like)
    - Irreversible (dissipative/learning)
    - Chaotic (sensitive to reversal direction)
    
    Usage:
        probe = TimeReversalProbe()
        
        def dynamics(M, A, sign=1.0):
            return sign * (-0.01 * M)  # sign flips for reversal
        
        result = probe.run(
            M0, A0,
            forward_dynamics=lambda M, A: dynamics(M, A, sign=1.0),
            reverse_dynamics=lambda M, A: dynamics(M, A, sign=-1.0),
            n_steps=50,
        )
        
        print(f"Reversibility: {result.reversibility_class.value}")
        print(f"Recovery fidelity: {result.recovery_fidelity:.4f}")
    """
    
    # Logging
    verbose: bool = False
    
    def run(
        self,
        M0: np.ndarray,
        A0: np.ndarray,
        forward_dynamics: Callable[[np.ndarray, np.ndarray], np.ndarray],
        reverse_dynamics: Callable[[np.ndarray, np.ndarray], np.ndarray],
        n_steps: int = 50,
        topology_update: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
        record_trajectory: bool = True,
    ) -> TimeReversalResult:
        """Run the time-reversal probe.
        
        Args:
            M0: Initial matrix state
            A0: Initial topology (adjacency)
            forward_dynamics: Forward update (M, A) -> delta_M
            reverse_dynamics: Reverse update (M, A) -> delta_M
            n_steps: Steps in each direction
            topology_update: Optional topology evolution (A, M) -> A_new
            record_trajectory: Whether to record full trajectories
            
        Returns:
            TimeReversalResult with recovery analysis
        """
        # Store initial state
        M_initial = M0.copy()
        A_initial = A0.copy()
        
        # Forward evolution
        M = M0.copy()
        A = A0.copy()
        forward_trajectory = [M.copy()] if record_trajectory else []
        
        if self.verbose:
            print(f"Forward evolution: {n_steps} steps")
        
        for t in range(n_steps):
            delta_M = forward_dynamics(M, A)
            M = M + delta_M
            M = (M + M.conj().T) / 2.0  # Enforce Hermiticity
            
            if topology_update is not None:
                A = topology_update(A, M)
            
            if record_trajectory:
                forward_trajectory.append(M.copy())
        
        M_forward_final = M.copy()
        A_forward_final = A.copy()
        
        # Reverse evolution
        reverse_trajectory = [M.copy()] if record_trajectory else []
        trajectory_divergence = []
        
        if self.verbose:
            print(f"Reverse evolution: {n_steps} steps")
        
        for t in range(n_steps):
            delta_M = reverse_dynamics(M, A)
            M = M + delta_M
            M = (M + M.conj().T) / 2.0  # Enforce Hermiticity
            
            if topology_update is not None:
                A = topology_update(A, M)
            
            if record_trajectory:
                reverse_trajectory.append(M.copy())
                
                # Compare with forward trajectory at corresponding point
                forward_idx = n_steps - t - 1
                if forward_idx >= 0 and forward_idx < len(forward_trajectory):
                    div = np.linalg.norm(M - forward_trajectory[forward_idx], 'fro')
                    trajectory_divergence.append(float(div))
        
        M_reverse_final = M.copy()
        A_reverse_final = A.copy()
        
        # Compute recovery metrics
        recovery_fidelity = matrix_fidelity(M_initial, M_reverse_final)
        spectral_recovery = spectral_fidelity(M_initial, M_reverse_final)
        topology_recovery = topology_fidelity(A_initial, A_reverse_final)
        
        # Classify reversibility
        traj_variance = float(np.var(trajectory_divergence)) if trajectory_divergence else 0.0
        reversibility_class = classify_reversibility(recovery_fidelity, traj_variance)
        
        # Diagnostics
        diagnostics = {
            'forward_spectral_change': float(
                np.linalg.norm(
                    np.sort(np.linalg.eigvalsh((M_initial + M_initial.T)/2)) -
                    np.sort(np.linalg.eigvalsh((M_forward_final + M_forward_final.T)/2))
                )
            ),
            'reverse_spectral_change': float(
                np.linalg.norm(
                    np.sort(np.linalg.eigvalsh((M_forward_final + M_forward_final.T)/2)) -
                    np.sort(np.linalg.eigvalsh((M_reverse_final + M_reverse_final.T)/2))
                )
            ),
            'trajectory_variance': traj_variance,
            'max_divergence': float(max(trajectory_divergence)) if trajectory_divergence else 0.0,
            'final_distance': float(np.linalg.norm(M_initial - M_reverse_final, 'fro')),
        }
        
        if self.verbose:
            print(f"\nResults:")
            print(f"  Recovery fidelity: {recovery_fidelity:.4f}")
            print(f"  Spectral recovery: {spectral_recovery:.4f}")
            print(f"  Topology recovery: {topology_recovery:.4f}")
            print(f"  Classification: {reversibility_class.value}")
        
        return TimeReversalResult(
            forward_steps=n_steps,
            reverse_steps=n_steps,
            initial_state=M_initial,
            forward_final=M_forward_final,
            reverse_final=M_reverse_final,
            recovery_fidelity=recovery_fidelity,
            spectral_recovery=spectral_recovery,
            topology_recovery=topology_recovery,
            reversibility_class=reversibility_class,
            trajectory_divergence=trajectory_divergence,
            diagnostics=diagnostics,
        )
    
    def run_with_perturbation(
        self,
        M0: np.ndarray,
        A0: np.ndarray,
        forward_dynamics: Callable,
        reverse_dynamics: Callable,
        n_steps: int = 50,
        perturbation_scale: float = 0.01,
        n_trials: int = 5,
        **kwargs,
    ) -> Dict[str, any]:
        """Run time-reversal with initial perturbation sensitivity analysis.
        
        Tests whether small perturbations to the final state before reversal
        lead to large deviations in recovery. This distinguishes:
        - Stable reversibility (perturbations don't matter)
        - Sensitive reversibility (chaos-like)
        - Robust irreversibility (perturbations don't change the failure mode)
        
        Args:
            M0: Initial matrix
            A0: Initial topology
            forward_dynamics: Forward dynamics
            reverse_dynamics: Reverse dynamics
            n_steps: Steps per direction
            perturbation_scale: Scale of perturbations
            n_trials: Number of perturbed trials
            **kwargs: Additional arguments for run()
            
        Returns:
            Dictionary with sensitivity analysis results
        """
        # Baseline run
        baseline = self.run(M0, A0, forward_dynamics, reverse_dynamics, n_steps, **kwargs)
        
        # Perturbed runs
        perturbed_fidelities = []
        
        for trial in range(n_trials):
            # Run forward to get final state
            M = M0.copy()
            A = A0.copy()
            
            for _ in range(n_steps):
                delta_M = forward_dynamics(M, A)
                M = M + delta_M
                M = (M + M.conj().T) / 2.0
            
            # Add perturbation before reversal
            perturbation = perturbation_scale * np.random.randn(*M.shape)
            perturbation = (perturbation + perturbation.T) / 2.0
            M_perturbed = M + perturbation
            
            # Reverse from perturbed state
            for _ in range(n_steps):
                delta_M = reverse_dynamics(M_perturbed, A)
                M_perturbed = M_perturbed + delta_M
                M_perturbed = (M_perturbed + M_perturbed.conj().T) / 2.0
            
            fidelity = matrix_fidelity(M0, M_perturbed)
            perturbed_fidelities.append(fidelity)
        
        # Analysis
        fidelity_mean = float(np.mean(perturbed_fidelities))
        fidelity_std = float(np.std(perturbed_fidelities))
        fidelity_drop = baseline.recovery_fidelity - fidelity_mean
        
        # Classify sensitivity
        if fidelity_std > 0.1:
            sensitivity = "high"  # Chaotic
        elif fidelity_drop > 0.1:
            sensitivity = "moderate"  # Sensitive but consistent
        else:
            sensitivity = "low"  # Stable
        
        return {
            'baseline_fidelity': baseline.recovery_fidelity,
            'baseline_class': baseline.reversibility_class.value,
            'perturbed_fidelity_mean': fidelity_mean,
            'perturbed_fidelity_std': fidelity_std,
            'fidelity_drop': fidelity_drop,
            'sensitivity': sensitivity,
            'n_trials': n_trials,
            'perturbation_scale': perturbation_scale,
        }


def create_reversible_dynamics_pair(
    base_update: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> Tuple[Callable, Callable]:
    """Create forward/reverse dynamics pair from base update.
    
    The reverse dynamics is simply the negation of the forward update.
    
    Args:
        base_update: Base dynamics function (M, A) -> delta_M
        
    Returns:
        Tuple of (forward_dynamics, reverse_dynamics)
    """
    def forward(M: np.ndarray, A: np.ndarray) -> np.ndarray:
        return base_update(M, A)
    
    def reverse(M: np.ndarray, A: np.ndarray) -> np.ndarray:
        return -base_update(M, A)
    
    return forward, reverse


def quick_reversibility_check(
    M0: np.ndarray,
    A0: np.ndarray,
    dynamics: Callable[[np.ndarray, np.ndarray], np.ndarray],
    n_steps: int = 20,
) -> Tuple[float, ReversibilityClass]:
    """Quick check for dynamics reversibility.
    
    Args:
        M0: Initial matrix
        A0: Initial topology
        dynamics: Dynamics function
        n_steps: Number of steps
        
    Returns:
        Tuple of (recovery_fidelity, reversibility_class)
    """
    forward_dynamics, reverse_dynamics = create_reversible_dynamics_pair(dynamics)
    
    probe = TimeReversalProbe(verbose=False)
    result = probe.run(
        M0, A0,
        forward_dynamics,
        reverse_dynamics,
        n_steps=n_steps,
        record_trajectory=False,
    )
    
    return result.recovery_fidelity, result.reversibility_class
