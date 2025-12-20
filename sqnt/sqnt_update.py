"""SQNT-inspired integrated update system.

This module provides a classical instantiation of SQNT (Superpositional Quantum
Network Topologies) principles, implementing topology-adaptive dynamics under
spectral constraints.

**Theoretical Background:**

The original SQNT framework (Altman, Pykacz & Zapatrin, 2004) introduced:
    - Quantum superposition of network topologies via Rota algebras
    - Training on superpositions where topology becomes trainable
    - Collapse to classical network via measurement

This implementation captures the *spirit* of SQNT in a classical setting:
    1. Matrix dynamics (endogenous updates)
    2. Soft spectral mass invariant (budget constraint)
    3. Topology plasticity (participation-based edge adaptation)
    4. Topology-matrix coupling (bidirectional feedback)

The key insight from SQNT is that topology should be adaptive and trainable,
not fixed a priori. This module tests that hypothesis classically.

**Scientific Claim (Falsifiable):**

    "Adaptive topology under a conserved spectral budget yields perturbation
    responses that cannot be replicated by fixed or random topologies."

References:
    [1] C. Altman, J. Pykacz & R. Zapatrin (2004). "Superpositional Quantum
        Network Topologies." Int. J. Theor. Phys. 43, 2029–2041.
    [2] C. Altman & R. Zapatrin (2010). "Backpropagation in Adaptive Quantum
        Networks." Int. J. Theor. Phys. 49, 2991–2997.

Author: Christopher Altman
Date: 2025
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple, Literal, Callable
from enum import Enum

from .invariant import SpectralMassInvariant, apply_invariant_correction
from .topology_plasticity import (
    TopologyPlasticity, 
    TopologyFeedback, 
    random_rewire,
    apply_perturbation,
)


class SystemMode(Enum):
    """System operating modes for controlled experiments.
    
    SQNT: Full adaptive topology with invariant
    FROZEN: Fixed topology (control 1)
    RANDOM: Randomly rewired topology each step (control 2)
    """
    SQNT = "sqnt"
    FROZEN = "frozen"
    RANDOM = "random"


@dataclass
class SQNTConfig:
    """Configuration for SQNT system.
    
    Attributes:
        mode: System operating mode (sqnt, frozen, random)
        invariant_epsilon: Strength of spectral mass restoring force
        topology_eta: Learning rate for edge plasticity
        n_dominant_modes: Number of eigenmodes for participation
        feedback_mode: How topology modulates matrix updates
        clip_weights: Whether to bound edge weights
    """
    mode: SystemMode = SystemMode.SQNT
    invariant_epsilon: float = 1e-4
    topology_eta: float = 1e-3
    n_dominant_modes: int = 5
    feedback_mode: Literal['hadamard', 'laplacian', 'degree', 'none'] = 'hadamard'
    clip_weights: bool = True
    seed: Optional[int] = None


@dataclass
class SQNTState:
    """State of the SQNT system at a time step.
    
    Immutable snapshot for logging and analysis.
    """
    t: int
    M: np.ndarray
    A: np.ndarray  # Adjacency/topology
    spectral_mass: float
    invariant_violation: float
    topology_summary: Dict[str, float]
    diagnostics: Dict[str, float]


class SQNTSystem:
    """Integrated SQNT dynamical system.
    
    This class manages the full SQNT evolution loop:
        1. Matrix evolves via user-provided dynamics
        2. Invariant correction maintains spectral budget
        3. Topology adapts based on spectral participation
        4. Topology feeds back into matrix dynamics
    
    The system can operate in three modes:
        - SQNT: Full adaptive system (experimental)
        - FROZEN: Fixed topology (control 1)
        - RANDOM: Random topology (control 2)
    
    Example usage:
        config = SQNTConfig(mode=SystemMode.SQNT)
        system = SQNTSystem(config)
        
        # Initialize
        M0 = ... # initial matrix
        A0 = ... # initial topology (from matrix_to_graph or similar)
        system.initialize(M0, A0)
        
        # Define dynamics
        def dynamics(M, A):
            # Your update rule here
            return delta_M
        
        # Evolution loop
        for t in range(T):
            state = system.step(dynamics)
            print(f"t={t}, violation={state.invariant_violation:.4f}")
    """
    
    def __init__(self, config: SQNTConfig):
        """Initialize SQNT system.
        
        Args:
            config: System configuration
        """
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        
        # Components
        self.invariant = SpectralMassInvariant(epsilon=config.invariant_epsilon)
        self.plasticity = TopologyPlasticity(
            eta=config.topology_eta,
            n_dominant=config.n_dominant_modes,
            clip_method='tanh' if config.clip_weights else 'none',
        )
        self.feedback = TopologyFeedback()
        
        # State
        self.M: Optional[np.ndarray] = None
        self.A: Optional[np.ndarray] = None
        self.A_initial: Optional[np.ndarray] = None  # For frozen mode
        self.t: int = 0
        self._history: List[SQNTState] = []
        
    def initialize(self, M0: np.ndarray, A0: np.ndarray) -> SQNTState:
        """Initialize system with initial matrix and topology.
        
        Args:
            M0: Initial matrix
            A0: Initial adjacency matrix (topology)
            
        Returns:
            Initial state snapshot
        """
        self.M = M0.copy()
        self.A = A0.copy()
        self.A_initial = A0.copy()  # Store for frozen mode
        self.t = 0
        self._history = []
        
        # Initialize invariant
        self.invariant.initialize(M0)
        
        # Log initial state
        state = self._create_state({})
        self._history.append(state)
        
        return state
    
    def _create_state(self, step_diagnostics: Dict[str, float]) -> SQNTState:
        """Create state snapshot."""
        # Topology summary
        n = self.A.shape[0]
        n_edges = np.sum(self.A > 0.01)
        mean_weight = np.mean(self.A[~np.eye(n, dtype=bool)])
        
        D = np.diag(np.sum(self.A, axis=1))
        L = D - self.A
        eigvals = np.linalg.eigvalsh(L)
        
        topology_summary = {
            'n_edges': float(n_edges),
            'mean_weight': float(mean_weight),
            'max_weight': float(np.max(self.A)),
            'laplacian_gap': float(eigvals[1] - eigvals[0]) if len(eigvals) > 1 else 0.0,
            'algebraic_connectivity': float(eigvals[1]) if len(eigvals) > 1 else 0.0,
        }
        
        return SQNTState(
            t=self.t,
            M=self.M.copy(),
            A=self.A.copy(),
            spectral_mass=self.invariant.spectral_mass(self.M),
            invariant_violation=self.invariant.violation(self.M) if self.invariant.I_0 else 0.0,
            topology_summary=topology_summary,
            diagnostics=step_diagnostics,
        )
    
    def step(
        self,
        dynamics: Callable[[np.ndarray, np.ndarray], np.ndarray],
    ) -> SQNTState:
        """Perform one SQNT update step.
        
        The update sequence is:
            1. Compute raw dynamics: δM = dynamics(M, A)
            2. Apply topology feedback: δM = feedback(δM, A)
            3. Apply invariant correction: M_new = M + δM + correction
            4. Update topology (mode-dependent)
        
        Args:
            dynamics: Function (M, A) -> δM that computes raw matrix update
            
        Returns:
            New state snapshot
        """
        step_diagnostics = {}
        
        # 1. Compute raw dynamics
        delta_M = dynamics(self.M, self.A)
        step_diagnostics['raw_update_norm'] = float(np.linalg.norm(delta_M, 'fro'))
        
        # 2. Apply topology feedback (if not 'none')
        if self.config.feedback_mode != 'none':
            delta_M = self.feedback.modulate_update(
                delta_M, 
                self.A, 
                mode=self.config.feedback_mode,
            )
            step_diagnostics['modulated_update_norm'] = float(np.linalg.norm(delta_M, 'fro'))
        
        # 3. Apply invariant correction
        self.M, inv_diag = self.invariant.corrected_update(self.M, delta_M)
        step_diagnostics.update({f'inv_{k}': v for k, v in inv_diag.items()})
        
        # 4. Update topology (mode-dependent)
        if self.config.mode == SystemMode.SQNT:
            # Adaptive topology
            D = np.diag(np.sum(self.A, axis=1))
            L = D - self.A
            self.A, topo_diag = self.plasticity.update(self.A, L)
            step_diagnostics.update({f'topo_{k}': v for k, v in topo_diag.items()})
            
        elif self.config.mode == SystemMode.FROZEN:
            # Fixed topology - no update
            pass
            
        elif self.config.mode == SystemMode.RANDOM:
            # Random rewiring each step
            self.A = random_rewire(self.A, preserve_degree=True, seed=self.rng.integers(1e9))
            step_diagnostics['random_rewire'] = True
        
        # Increment time
        self.t += 1
        
        # Log invariant
        self.invariant.log_step(self.M)
        
        # Create and store state
        state = self._create_state(step_diagnostics)
        self._history.append(state)
        
        return state
    
    def apply_perturbation(
        self,
        fraction: float = 0.25,
        mode: Literal['contiguous', 'random', 'targeted'] = 'contiguous',
    ) -> Dict[str, any]:
        """Apply structural perturbation to topology.
        
        Args:
            fraction: Fraction of edges to remove
            mode: Perturbation type
            
        Returns:
            Perturbation info dict
        """
        self.A, info = apply_perturbation(
            self.A, 
            fraction=fraction, 
            mode=mode,
            seed=self.rng.integers(1e9),
        )
        info['applied_at_t'] = self.t
        return info
    
    def get_baseline_snapshot(self) -> Dict[str, np.ndarray]:
        """Get current state as baseline for recovery analysis.
        
        Returns:
            Dict with M, A, eigenvalues for comparison
        """
        D = np.diag(np.sum(self.A, axis=1))
        L = D - self.A
        eigvals_L = np.linalg.eigvalsh(L)
        eigvals_M = np.linalg.eigvalsh((self.M + self.M.T) / 2)
        
        return {
            'M': self.M.copy(),
            'A': self.A.copy(),
            'eigvals_L': eigvals_L,
            'eigvals_M': eigvals_M,
            't': self.t,
        }
    
    def compute_recovery_metrics(
        self,
        baseline: Dict[str, np.ndarray],
    ) -> Dict[str, float]:
        """Compute recovery metrics relative to baseline.
        
        Args:
            baseline: Baseline snapshot from get_baseline_snapshot()
            
        Returns:
            Dict with spectral_distance, topology_similarity, etc.
        """
        # Current eigenvalues
        D = np.diag(np.sum(self.A, axis=1))
        L = D - self.A
        eigvals_L_current = np.linalg.eigvalsh(L)
        eigvals_M_current = np.linalg.eigvalsh((self.M + self.M.T) / 2)
        
        # Spectral distances
        spectral_distance_L = np.linalg.norm(
            eigvals_L_current - baseline['eigvals_L']
        )
        spectral_distance_M = np.linalg.norm(
            eigvals_M_current - baseline['eigvals_M']
        )
        
        # Topology similarity (correlation of adjacency matrices)
        A_flat = self.A.flatten()
        A_base_flat = baseline['A'].flatten()
        if np.std(A_flat) > 1e-12 and np.std(A_base_flat) > 1e-12:
            topology_similarity = float(np.corrcoef(A_flat, A_base_flat)[0, 1])
        else:
            topology_similarity = 1.0 if np.allclose(A_flat, A_base_flat) else 0.0
        
        # Matrix similarity
        M_flat = self.M.flatten()
        M_base_flat = baseline['M'].flatten()
        if np.std(M_flat) > 1e-12 and np.std(M_base_flat) > 1e-12:
            matrix_similarity = float(np.corrcoef(M_flat, M_base_flat)[0, 1])
        else:
            matrix_similarity = 1.0 if np.allclose(M_flat, M_base_flat) else 0.0
        
        return {
            'spectral_distance_L': float(spectral_distance_L),
            'spectral_distance_M': float(spectral_distance_M),
            'topology_similarity': topology_similarity,
            'matrix_similarity': matrix_similarity,
            'steps_since_baseline': self.t - baseline['t'],
        }
    
    @property
    def history(self) -> List[SQNTState]:
        """Get evolution history."""
        return self._history
    
    def get_trajectory(self, observable: str) -> np.ndarray:
        """Get time series of an observable.
        
        Args:
            observable: One of 'spectral_mass', 'invariant_violation',
                       or a key from topology_summary or diagnostics
            
        Returns:
            Array of values over time
        """
        values = []
        for state in self._history:
            if observable == 'spectral_mass':
                values.append(state.spectral_mass)
            elif observable == 'invariant_violation':
                values.append(state.invariant_violation)
            elif observable in state.topology_summary:
                values.append(state.topology_summary[observable])
            elif observable in state.diagnostics:
                values.append(state.diagnostics[observable])
            else:
                values.append(np.nan)
        return np.array(values)
    
    def summary(self) -> Dict[str, any]:
        """Get system summary statistics."""
        return {
            'mode': self.config.mode.value,
            'n_steps': self.t,
            'invariant_summary': self.invariant.summary(),
            'plasticity_summary': self.plasticity.summary() if self.config.mode == SystemMode.SQNT else {},
            'final_spectral_mass': self._history[-1].spectral_mass if self._history else None,
            'final_violation': self._history[-1].invariant_violation if self._history else None,
        }


def run_comparison_experiment(
    M0: np.ndarray,
    A0: np.ndarray,
    dynamics: Callable[[np.ndarray, np.ndarray], np.ndarray],
    n_steps: int = 100,
    perturb_at: Optional[int] = None,
    perturb_fraction: float = 0.25,
    seed: int = 0,
    **config_kwargs,
) -> Dict[str, Dict]:
    """Run comparison experiment across all three system modes.
    
    This is the primary interface for KT-1 falsification experiments.
    Runs identical experiments with SQNT, Frozen, and Random topologies.
    
    Args:
        M0: Initial matrix
        A0: Initial topology
        dynamics: Dynamics function
        n_steps: Number of evolution steps
        perturb_at: Time step to apply perturbation (None = no perturbation)
        perturb_fraction: Fraction of edges to remove
        seed: Base random seed
        **config_kwargs: Additional config parameters
        
    Returns:
        Dict with results for each mode: 'sqnt', 'frozen', 'random'
    """
    results = {}
    
    for mode in [SystemMode.SQNT, SystemMode.FROZEN, SystemMode.RANDOM]:
        config = SQNTConfig(
            mode=mode,
            seed=seed,
            **config_kwargs,
        )
        system = SQNTSystem(config)
        system.initialize(M0.copy(), A0.copy())
        
        # Get baseline before perturbation
        baseline = None
        perturbation_info = None
        
        for t in range(n_steps):
            # Apply perturbation at specified time
            if t == perturb_at:
                baseline = system.get_baseline_snapshot()
                perturbation_info = system.apply_perturbation(
                    fraction=perturb_fraction,
                    mode='contiguous',
                )
            
            state = system.step(dynamics)
        
        # Compute final recovery metrics
        recovery_metrics = None
        if baseline is not None:
            recovery_metrics = system.compute_recovery_metrics(baseline)
        
        results[mode.value] = {
            'summary': system.summary(),
            'trajectory_spectral_mass': system.get_trajectory('spectral_mass'),
            'trajectory_violation': system.get_trajectory('invariant_violation'),
            'trajectory_n_edges': system.get_trajectory('n_edges'),
            'perturbation_info': perturbation_info,
            'recovery_metrics': recovery_metrics,
            'final_state': {
                'M': system.M.copy(),
                'A': system.A.copy(),
            },
        }
    
    return results
