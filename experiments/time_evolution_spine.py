"""Time-evolution experiment spine: dynamic topology-aware experiments.

This module wraps the static experiment spine in a time-evolution loop,
allowing observation of how topology, kernels, and phase indicators
evolve over simulated "time" or learning steps.

Scientific motivation:
- Static snapshots miss the dynamics of learning and phase transitions
- Time evolution reveals emergent structure and critical slowing down
- Observables tracked over time connect to thermalization and equilibration

Usage:
    python -m experiments.time_evolution_spine --n-steps 50 --dt 0.1

This implements the time-evolution wrapper for the experiment spine.
"""

from __future__ import annotations
import argparse
import json
import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict, field
import time

# Matrix model imports
from matrix_models import (
    HermitianEnsemble,
    CubicAction,
    QuarticAction,
    LangevinSampler,
    invariants,
)

# Topology imports
from topology import Graph, matrix_to_graph, GraphFamily, sample_graph_family

# Quantum kernel imports
from quantum_kernels import (
    GraphConditionedKernel,
    random_fourier_features_kernel,
)

# Analysis imports
from analysis import (
    SpectralDiagnostics,
    SpectralComplexityAnalyzer,
    OnlinePhaseDetector,
)
from analysis.spectral_observables import (
    SpectralObservables,
    compute_thermodynamic_observables,
)


@dataclass
class TimeEvolutionConfig:
    """Configuration for time-evolution experiment."""
    # Matrix model
    dim: int = 12
    g_initial: float = 0.0
    g_final: float = 0.3
    g_schedule: str = "linear"  # "linear", "step", "cosine"
    langevin_steps_per_update: int = 50
    temperature: float = 0.01

    # Time evolution
    n_steps: int = 100
    dt: float = 0.1

    # Graph parameters
    graph_mode: str = "weighted"
    graph_tau_q: float = 0.5
    graph_weighted: bool = True

    # Quantum kernel
    n_qubits: int = 4
    n_layers: int = 2
    n_samples: int = 30

    # Observation
    observe_kernel_every: int = 10
    observe_full_every: int = 20

    # Output
    seed: int = 0
    verbose: bool = False


@dataclass
class TimeStep:
    """Data from a single time step."""
    t: int
    time: float
    g: float
    observables: Dict[str, float]
    graph_summary: Optional[Dict[str, float]] = None
    kernel_metrics: Optional[Dict[str, float]] = None
    phase_detected: bool = False


class TimeEvolutionSpine:
    """Time-evolution wrapper for the experiment spine.

    This class manages the temporal evolution of:
    1. The coupling constant g (via a schedule)
    2. The matrix M (via Langevin dynamics)
    3. The derived topology G
    4. Spectral observables
    5. Kernel performance metrics

    Example:
        config = TimeEvolutionConfig(n_steps=100, g_initial=0.0, g_final=0.3)
        spine = TimeEvolutionSpine(config)
        history = spine.run()

        # Plot observable evolution
        times = [step.time for step in history]
        entropy = [step.observables["spectral_entropy"] for step in history]
        plt.plot(times, entropy)
    """

    def __init__(self, config: TimeEvolutionConfig):
        """Initialize time-evolution spine.

        Args:
            config: Configuration object
        """
        self.config = config
        self.rng = np.random.default_rng(config.seed)

        # Initialize matrix model
        self.ensemble = HermitianEnsemble(
            dim=config.dim,
            scale=1.0,
            seed=config.seed,
        )
        self.M = self.ensemble.sample(rng=self.rng)

        # Initialize analysis tools
        self.spec_diag = SpectralDiagnostics()
        self.spec_complex = SpectralComplexityAnalyzer()
        self.observables = SpectralObservables(track_history=True)
        self.phase_detector = OnlinePhaseDetector(window_size=20, sensitivity=2.0)

        # State tracking
        self.t = 0
        self.history: List[TimeStep] = []
        self.current_g = config.g_initial

    def g_schedule(self, t: int) -> float:
        """Compute coupling constant at time t.

        Args:
            t: Time step index

        Returns:
            Coupling constant g
        """
        progress = t / max(self.config.n_steps - 1, 1)

        if self.config.g_schedule == "linear":
            return (
                self.config.g_initial
                + (self.config.g_final - self.config.g_initial) * progress
            )

        elif self.config.g_schedule == "step":
            # Sudden quench at midpoint
            if progress < 0.5:
                return self.config.g_initial
            else:
                return self.config.g_final

        elif self.config.g_schedule == "cosine":
            # Smooth interpolation
            cos_progress = 0.5 * (1 - np.cos(np.pi * progress))
            return (
                self.config.g_initial
                + (self.config.g_final - self.config.g_initial) * cos_progress
            )

        return self.config.g_initial

    def evolve_matrix(self) -> np.ndarray:
        """Evolve matrix M by one time step using Langevin dynamics.

        Returns:
            Updated matrix M
        """
        action = CubicAction(g=self.current_g)
        sampler = LangevinSampler(
            action=action,
            dt=self.config.dt,
            temperature=self.config.temperature,
            seed=self.config.seed + self.t,
        )
        self.M = sampler.run(
            self.M,
            steps=self.config.langevin_steps_per_update,
            rng=self.rng,
        )
        return self.M

    def compute_observables(self) -> Dict[str, float]:
        """Compute spectral observables for current matrix state.

        Returns:
            Dictionary of observable values
        """
        return self.observables.update(self.M)

    def compute_graph(self) -> Graph:
        """Compute graph topology from current matrix.

        Returns:
            Graph object
        """
        params = {
            "mode": self.config.graph_mode,
            "tau_quantile": self.config.graph_tau_q,
            "weighted": self.config.graph_weighted,
        }
        return matrix_to_graph(self.M, mode=self.config.graph_mode, params=params)

    def compute_kernel_metrics(self, G: Graph) -> Dict[str, float]:
        """Compute kernel performance metrics.

        Args:
            G: Graph topology

        Returns:
            Dictionary of kernel metrics
        """
        # Generate test data
        X = self.rng.normal(size=(self.config.n_samples, self.config.dim))
        y = np.sin(X[:, 0]) + 0.1 * self.rng.normal(size=self.config.n_samples)

        n_train = int(0.7 * self.config.n_samples)
        X_train, X_test = X[:n_train], X[n_train:]
        y_train, y_test = y[:n_train], y[n_train:]

        # Graph-conditioned kernel
        gc_kernel = GraphConditionedKernel(
            n_qubits=self.config.n_qubits,
            n_layers=self.config.n_layers,
            seed=self.config.seed,
        )
        gc_kernel.set_graph(G)
        K = gc_kernel.gram(X)

        # Simple kernel ridge regression
        K_train = K[:n_train, :n_train]
        K_test = K[n_train:, :n_train]
        ridge = 1e-3

        alpha = np.linalg.solve(K_train + ridge * np.eye(n_train), y_train)
        y_pred = K_test @ alpha

        mse = float(np.mean((y_pred - y_test) ** 2))
        r2 = 1.0 - mse / (np.var(y_test) + 1e-12)

        # Classical baseline
        K_rbf = random_fourier_features_kernel(X, gamma=1.0, n_features=128)
        K_train_c = K_rbf[:n_train, :n_train]
        K_test_c = K_rbf[n_train:, :n_train]

        alpha_c = np.linalg.solve(K_train_c + ridge * np.eye(n_train), y_train)
        y_pred_c = K_test_c @ alpha_c
        mse_c = float(np.mean((y_pred_c - y_test) ** 2))
        r2_c = 1.0 - mse_c / (np.var(y_test) + 1e-12)

        return {
            "r2_quantum": r2,
            "r2_classical": r2_c,
            "advantage": r2 - r2_c,
            "mse_quantum": mse,
            "mse_classical": mse_c,
        }

    def step(self) -> TimeStep:
        """Perform one time step of evolution.

        Returns:
            TimeStep object with all computed quantities
        """
        # Update coupling constant
        self.current_g = self.g_schedule(self.t)

        # Evolve matrix
        self.evolve_matrix()

        # Compute observables
        obs = self.compute_observables()
        obs["g"] = self.current_g

        # Check for phase transition
        phase_result = self.phase_detector.update({"loss": obs.get("spectral_entropy", 0)})
        phase_detected = phase_result.get("loss", False)

        # Create time step record
        step = TimeStep(
            t=self.t,
            time=self.t * self.config.dt,
            g=self.current_g,
            observables=obs,
            phase_detected=phase_detected,
        )

        # Compute graph (periodically)
        if self.t % self.config.observe_kernel_every == 0:
            G = self.compute_graph()
            step.graph_summary = G.summary()

        # Compute kernel metrics (less frequently)
        if self.t % self.config.observe_full_every == 0:
            G = self.compute_graph() if step.graph_summary is None else None
            if G is None:
                G = self.compute_graph()
            step.kernel_metrics = self.compute_kernel_metrics(G)

        # Log if verbose
        if self.config.verbose and self.t % 10 == 0:
            entropy = obs.get("spectral_entropy", 0)
            print(f"t={self.t:4d} | g={self.current_g:.3f} | entropy={entropy:.3f}")
            if phase_detected:
                print("  *** PHASE TRANSITION DETECTED ***")

        self.history.append(step)
        self.t += 1

        return step

    def run(self) -> List[TimeStep]:
        """Run full time evolution.

        Returns:
            List of TimeStep objects
        """
        if self.config.verbose:
            print("=" * 60)
            print("TIME EVOLUTION EXPERIMENT")
            print("=" * 60)
            print(f"Steps: {self.config.n_steps}")
            print(f"g: {self.config.g_initial} -> {self.config.g_final} ({self.config.g_schedule})")
            print(f"Matrix dim: {self.config.dim}")
            print("=" * 60)

        for _ in range(self.config.n_steps):
            self.step()

        return self.history

    def get_observable_trajectory(self, name: str) -> np.ndarray:
        """Get time series of a specific observable.

        Args:
            name: Observable name

        Returns:
            Array of observable values over time
        """
        return np.array([step.observables.get(name, np.nan) for step in self.history])

    def get_times(self) -> np.ndarray:
        """Get array of time points."""
        return np.array([step.time for step in self.history])

    def get_g_values(self) -> np.ndarray:
        """Get array of coupling constants."""
        return np.array([step.g for step in self.history])

    def find_phase_transitions(self) -> List[int]:
        """Find indices where phase transitions were detected.

        Returns:
            List of time step indices
        """
        return [step.t for step in self.history if step.phase_detected]

    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary for JSON serialization.

        Returns:
            Dictionary representation
        """
        return {
            "config": asdict(self.config),
            "n_steps": len(self.history),
            "times": self.get_times().tolist(),
            "g_values": self.get_g_values().tolist(),
            "phase_transitions": self.find_phase_transitions(),
            "final_observables": self.history[-1].observables if self.history else {},
            "observable_statistics": self.observables.get_statistics(),
        }


def parse_args():
    """Parse command line arguments."""
    p = argparse.ArgumentParser(
        description="Time-evolution experiment spine"
    )

    # Matrix model
    p.add_argument("--dim", type=int, default=12, help="Matrix dimension")
    p.add_argument("--g-initial", type=float, default=0.0, help="Initial coupling")
    p.add_argument("--g-final", type=float, default=0.3, help="Final coupling")
    p.add_argument("--g-schedule", choices=["linear", "step", "cosine"],
                   default="linear", help="Coupling schedule")
    p.add_argument("--temperature", type=float, default=0.01, help="Langevin temperature")

    # Time evolution
    p.add_argument("--n-steps", type=int, default=100, help="Number of time steps")
    p.add_argument("--dt", type=float, default=0.1, help="Time step size")
    p.add_argument("--langevin-steps", type=int, default=50,
                   help="Langevin steps per time step")

    # Graph
    p.add_argument("--graph-mode", choices=["threshold", "weighted", "laplacian"],
                   default="weighted", help="Graph extraction mode")

    # Quantum kernel
    p.add_argument("--n-qubits", type=int, default=4, help="Number of qubits")
    p.add_argument("--n-layers", type=int, default=2, help="Ansatz layers")
    p.add_argument("--n-samples", type=int, default=30, help="Samples for kernel eval")

    # Observation frequency
    p.add_argument("--observe-kernel-every", type=int, default=10,
                   help="Compute graph every N steps")
    p.add_argument("--observe-full-every", type=int, default=20,
                   help="Compute full metrics every N steps")

    # Output
    p.add_argument("--output", type=str, default=None, help="Output JSON file")
    p.add_argument("--seed", type=int, default=0, help="Random seed")
    p.add_argument("--verbose", action="store_true", help="Verbose output")

    return p.parse_args()


def main():
    args = parse_args()

    config = TimeEvolutionConfig(
        dim=args.dim,
        g_initial=args.g_initial,
        g_final=args.g_final,
        g_schedule=args.g_schedule,
        temperature=args.temperature,
        n_steps=args.n_steps,
        dt=args.dt,
        langevin_steps_per_update=args.langevin_steps,
        graph_mode=args.graph_mode,
        n_qubits=args.n_qubits,
        n_layers=args.n_layers,
        n_samples=args.n_samples,
        observe_kernel_every=args.observe_kernel_every,
        observe_full_every=args.observe_full_every,
        seed=args.seed,
        verbose=args.verbose,
    )

    spine = TimeEvolutionSpine(config)
    history = spine.run()

    # Output
    if args.output:
        with open(args.output, "w") as f:
            json.dump(spine.to_dict(), f, indent=2, default=str)
        print(f"Results saved to: {args.output}")
    else:
        print("\n" + "=" * 60)
        print("TIME EVOLUTION SUMMARY")
        print("=" * 60)

        print(f"\nTotal steps: {len(history)}")
        print(f"Phase transitions detected at: {spine.find_phase_transitions()}")

        print("\nFinal observables:")
        for k, v in history[-1].observables.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")

        print("\nObservable statistics:")
        stats = spine.observables.get_statistics()
        for name in ["spectral_entropy", "level_spacing_r", "eigenvalue_std"]:
            if name in stats:
                s = stats[name]
                print(f"  {name}: mean={s['mean']:.4f}, std={s['std']:.4f}, trend={s['trend']:.4f}")

        if history[-1].kernel_metrics:
            print("\nFinal kernel metrics:")
            for k, v in history[-1].kernel_metrics.items():
                print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
