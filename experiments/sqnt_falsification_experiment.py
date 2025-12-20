"""SQNT Falsification Experiment: Run KT-1 protocol.

This script runs the pre-registered KT-1 falsification test to determine
whether SQNT demonstrates genuine adaptive topology behavior.

Usage:
    python -m experiments.sqnt_falsification_experiment --verbose
    python -m experiments.sqnt_falsification_experiment --output results/kt1_results.json

Author: Christopher Altman
Date: 2025
"""

from __future__ import annotations
import argparse
import numpy as np
import json
from typing import Callable

# Matrix model imports
from matrix_models import HermitianEnsemble, CubicAction, LangevinSampler

# Topology imports
from topology import matrix_to_graph

# SQNT imports
from sqnt import (
    SQNTSystem,
    SQNTConfig,
    SystemMode,
    KT1Protocol,
    run_comparison_experiment,
)


def create_langevin_dynamics(
    g: float = 0.1,
    dt: float = 0.01,
    temperature: float = 0.01,
    langevin_steps: int = 10,
    seed: int = 0,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Create Langevin dynamics function for matrix evolution.
    
    This wraps the matrix model's Langevin sampler to provide dynamics
    that can be used with the SQNT system.
    
    Args:
        g: Cubic coupling constant
        dt: Langevin time step
        temperature: Noise temperature
        langevin_steps: Number of Langevin steps per SQNT step
        seed: Random seed
        
    Returns:
        Dynamics function (M, A) -> delta_M
    """
    action = CubicAction(g=g)
    sampler = LangevinSampler(
        action=action,
        dt=dt,
        temperature=temperature,
        seed=seed,
    )
    rng = np.random.default_rng(seed)
    
    def dynamics(M: np.ndarray, A: np.ndarray) -> np.ndarray:
        """Compute delta_M using Langevin dynamics."""
        M_new = sampler.run(M, steps=langevin_steps, rng=rng)
        delta_M = M_new - M
        return delta_M
    
    return dynamics


def create_simple_dynamics(
    decay_rate: float = 0.01,
    noise_scale: float = 0.001,
    seed: int = 0,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Create simple decay + noise dynamics.
    
    This is a minimal dynamics for testing that doesn't require
    the full matrix model infrastructure.
    
    Args:
        decay_rate: Rate of exponential decay
        noise_scale: Scale of random perturbations
        seed: Random seed
        
    Returns:
        Dynamics function (M, A) -> delta_M
    """
    rng = np.random.default_rng(seed)
    
    def dynamics(M: np.ndarray, A: np.ndarray) -> np.ndarray:
        """Simple decay + noise dynamics."""
        # Decay toward zero
        decay = -decay_rate * M
        
        # Add small noise
        noise = noise_scale * rng.normal(size=M.shape)
        noise = (noise + noise.T) / 2  # Symmetrize
        
        return decay + noise
    
    return dynamics


def run_demo_experiment(verbose: bool = True) -> dict:
    """Run a demonstration SQNT experiment.
    
    This shows the basic SQNT workflow without the full falsification protocol.
    
    Args:
        verbose: Print progress
        
    Returns:
        Results dictionary
    """
    if verbose:
        print("=" * 60)
        print("SQNT DEMONSTRATION EXPERIMENT")
        print("=" * 60)
    
    # Initialize matrix model
    dim = 12
    seed = 42
    
    ensemble = HermitianEnsemble(dim=dim, scale=1.0, seed=seed)
    rng = np.random.default_rng(seed)
    M0 = ensemble.sample(rng=rng)
    
    # Extract initial topology
    G = matrix_to_graph(M0, mode='weighted')
    A0 = G.adjacency
    
    if verbose:
        print(f"\nInitial matrix dimension: {dim}")
        print(f"Initial spectral mass: {np.linalg.norm(M0, 'fro')**2:.4f}")
        print(f"Initial topology edges: {np.sum(A0 > 0.01)}")
    
    # Create dynamics
    dynamics = create_simple_dynamics(decay_rate=0.01, noise_scale=0.001, seed=seed)
    
    # Run SQNT system
    config = SQNTConfig(
        mode=SystemMode.SQNT,
        invariant_epsilon=1e-4,
        topology_eta=1e-3,
        n_dominant_modes=5,
        seed=seed,
    )
    system = SQNTSystem(config)
    system.initialize(M0, A0)
    
    # Evolution
    n_steps = 50
    if verbose:
        print(f"\nRunning {n_steps} steps...")
    
    for t in range(n_steps):
        state = system.step(dynamics)
        
        if verbose and t % 10 == 0:
            print(f"  t={t:3d}: I={state.spectral_mass:.4f}, "
                  f"δI={state.invariant_violation:.4f}, "
                  f"edges={state.topology_summary['n_edges']:.0f}")
    
    # Apply perturbation
    if verbose:
        print("\nApplying perturbation...")
    
    baseline = system.get_baseline_snapshot()
    info = system.apply_perturbation(fraction=0.25)
    
    if verbose:
        print(f"  Removed {info['n_edges_removed']} edges")
    
    # Post-perturbation evolution
    n_recovery = 50
    if verbose:
        print(f"\nRunning {n_recovery} recovery steps...")
    
    for t in range(n_recovery):
        state = system.step(dynamics)
        
        if verbose and t % 10 == 0:
            metrics = system.compute_recovery_metrics(baseline)
            print(f"  t={t:3d}: spec_dist={metrics['spectral_distance_L']:.4f}, "
                  f"topo_sim={metrics['topology_similarity']:.4f}")
    
    # Final summary
    summary = system.summary()
    final_metrics = system.compute_recovery_metrics(baseline)
    
    if verbose:
        print("\n" + "=" * 60)
        print("FINAL SUMMARY")
        print("=" * 60)
        print(f"Mode: {summary['mode']}")
        print(f"Total steps: {summary['n_steps']}")
        print(f"Final spectral mass: {summary['final_spectral_mass']:.4f}")
        print(f"Final violation: {summary['final_violation']:.4f}")
        print(f"Topology similarity: {final_metrics['topology_similarity']:.4f}")
    
    return {
        'summary': summary,
        'final_metrics': final_metrics,
        'perturbation_info': info,
    }


def run_falsification_experiment(
    output_path: str = None,
    verbose: bool = True,
) -> dict:
    """Run the full KT-1 falsification protocol.
    
    Args:
        output_path: Path to save results JSON
        verbose: Print progress
        
    Returns:
        Falsification result
    """
    # Initialize
    dim = 12
    seed = 42
    
    ensemble = HermitianEnsemble(dim=dim, scale=1.0, seed=seed)
    rng = np.random.default_rng(seed)
    M0 = ensemble.sample(rng=rng)
    
    G = matrix_to_graph(M0, mode='weighted')
    A0 = G.adjacency
    
    # Create dynamics
    dynamics = create_simple_dynamics(decay_rate=0.01, noise_scale=0.001, seed=seed)
    
    # Run protocol
    protocol = KT1Protocol(
        n_warmup=50,
        n_post_perturbation=100,
        perturbation_fraction=0.25,
        n_seeds=5,
        base_seed=seed,
    )
    
    result = protocol.run(M0, A0, dynamics, verbose=verbose)
    
    # Save if requested
    if output_path:
        protocol.to_json(result, output_path)
        if verbose:
            print(f"\nResults saved to: {output_path}")
    
    return result


def parse_args():
    """Parse command line arguments."""
    p = argparse.ArgumentParser(
        description="SQNT Falsification Experiment"
    )
    
    p.add_argument("--mode", choices=["demo", "falsification"],
                   default="demo",
                   help="Experiment mode")
    p.add_argument("--output", type=str, default=None,
                   help="Output JSON file for results")
    p.add_argument("--verbose", action="store_true",
                   help="Verbose output")
    p.add_argument("--dim", type=int, default=12,
                   help="Matrix dimension")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed")
    
    return p.parse_args()


def main():
    args = parse_args()
    
    if args.mode == "demo":
        run_demo_experiment(verbose=args.verbose or True)
    else:
        run_falsification_experiment(
            output_path=args.output,
            verbose=args.verbose or True,
        )


if __name__ == "__main__":
    main()
