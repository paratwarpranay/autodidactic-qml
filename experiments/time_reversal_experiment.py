"""Time-reversal experiment for SQNT dynamics.

This experiment probes the reversibility structure of SQNT dynamics
to distinguish genuine learning/adaptation from trivial dynamics.

Scientific question:
    Does adaptive topology create genuine irreversibility (real memory),
    or is the system fundamentally reversible?

Protocol:
    1. Run SQNT system forward T steps
    2. Reverse the dynamics (sign-flip learning rates)
    3. Measure recovery fidelity
    4. Compare SQNT vs Frozen vs Random modes

Key insight:
    - If SQNT is irreversible but Frozen is reversible → topology adaptation
      creates genuine structural memory
    - If both are equally (ir)reversible → irreversibility is not from topology

Author: Christopher Altman
Date: 2025
"""

from __future__ import annotations
import argparse
import numpy as np
import json
from typing import Dict, Any
from dataclasses import dataclass, asdict

# Matrix model imports
from matrix_models import HermitianEnsemble

# Topology imports
from topology import matrix_to_graph

# SQNT imports
from sqnt import SQNTSystem, SQNTConfig, SystemMode

# Analysis imports
from analysis import (
    TimeReversalProbe,
    ReversibilityClass,
    matrix_fidelity,
    spectral_fidelity,
    topology_fidelity,
    quick_reversibility_check,
)


@dataclass
class TimeReversalExperimentConfig:
    """Configuration for time-reversal experiment."""
    # Matrix model
    dim: int = 12
    
    # Evolution
    n_forward_steps: int = 50
    n_reverse_steps: int = 50
    
    # Dynamics
    decay_rate: float = 0.01
    noise_scale: float = 0.001
    
    # Perturbation sensitivity
    run_sensitivity_analysis: bool = True
    perturbation_scale: float = 0.01
    n_sensitivity_trials: int = 5
    
    # Output
    seed: int = 42
    verbose: bool = True


def create_sqnt_forward_dynamics(
    system: SQNTSystem,
    decay_rate: float = 0.01,
    noise_scale: float = 0.001,
    seed: int = 0,
) -> callable:
    """Create forward dynamics for SQNT system."""
    rng = np.random.default_rng(seed)
    
    def forward_dynamics(M: np.ndarray, A: np.ndarray) -> np.ndarray:
        # Decay
        delta_M = -decay_rate * M
        
        # Small noise
        noise = noise_scale * rng.normal(size=M.shape)
        noise = (noise + noise.T) / 2
        delta_M = delta_M + noise
        
        return delta_M
    
    return forward_dynamics


def create_sqnt_reverse_dynamics(
    system: SQNTSystem,
    decay_rate: float = 0.01,
    noise_scale: float = 0.001,
    seed: int = 0,
) -> callable:
    """Create reverse dynamics for SQNT system."""
    rng = np.random.default_rng(seed + 1000)  # Different seed
    
    def reverse_dynamics(M: np.ndarray, A: np.ndarray) -> np.ndarray:
        # Reverse decay (growth)
        delta_M = decay_rate * M
        
        # Noise (cannot be reversed exactly)
        noise = noise_scale * rng.normal(size=M.shape)
        noise = (noise + noise.T) / 2
        delta_M = delta_M + noise
        
        return delta_M
    
    return reverse_dynamics


def run_single_mode_reversal(
    mode: SystemMode,
    M0: np.ndarray,
    A0: np.ndarray,
    config: TimeReversalExperimentConfig,
) -> Dict[str, Any]:
    """Run time-reversal test for a single system mode."""
    
    # Create system
    sqnt_config = SQNTConfig(
        mode=mode,
        invariant_epsilon=1e-4,
        topology_eta=1e-3,
        seed=config.seed,
    )
    system = SQNTSystem(sqnt_config)
    system.initialize(M0.copy(), A0.copy())
    
    # Create dynamics
    forward_dynamics = create_sqnt_forward_dynamics(
        system,
        decay_rate=config.decay_rate,
        noise_scale=config.noise_scale,
        seed=config.seed,
    )
    reverse_dynamics = create_sqnt_reverse_dynamics(
        system,
        decay_rate=config.decay_rate,
        noise_scale=config.noise_scale,
        seed=config.seed,
    )
    
    # Run time-reversal probe
    probe = TimeReversalProbe(verbose=False)
    
    result = probe.run(
        M0.copy(),
        A0.copy(),
        forward_dynamics,
        reverse_dynamics,
        n_steps=config.n_forward_steps,
        record_trajectory=True,
    )
    
    # Package results
    return {
        'mode': mode.value,
        'recovery_fidelity': result.recovery_fidelity,
        'spectral_recovery': result.spectral_recovery,
        'topology_recovery': result.topology_recovery,
        'reversibility_class': result.reversibility_class.value,
        'trajectory_divergence_mean': float(np.mean(result.trajectory_divergence)) if result.trajectory_divergence else 0.0,
        'trajectory_divergence_max': float(max(result.trajectory_divergence)) if result.trajectory_divergence else 0.0,
        'diagnostics': result.diagnostics,
    }


def run_time_reversal_experiment(
    config: TimeReversalExperimentConfig,
) -> Dict[str, Any]:
    """Run full time-reversal experiment across all modes."""
    
    if config.verbose:
        print("=" * 60)
        print("TIME-REVERSAL EXPERIMENT")
        print("=" * 60)
        print(f"Dimension: {config.dim}")
        print(f"Forward steps: {config.n_forward_steps}")
        print(f"Reverse steps: {config.n_reverse_steps}")
        print(f"Decay rate: {config.decay_rate}")
        print(f"Noise scale: {config.noise_scale}")
        print("=" * 60)
    
    # Initialize matrix model
    ensemble = HermitianEnsemble(dim=config.dim, scale=1.0, seed=config.seed)
    rng = np.random.default_rng(config.seed)
    M0 = ensemble.sample(rng=rng)
    
    # Extract topology
    G = matrix_to_graph(M0, mode='weighted')
    A0 = G.adjacency
    
    # Run for each mode
    results = {}
    
    for mode in [SystemMode.SQNT, SystemMode.FROZEN, SystemMode.RANDOM]:
        if config.verbose:
            print(f"\n--- Mode: {mode.value.upper()} ---")
        
        mode_result = run_single_mode_reversal(mode, M0, A0, config)
        results[mode.value] = mode_result
        
        if config.verbose:
            print(f"  Recovery fidelity: {mode_result['recovery_fidelity']:.4f}")
            print(f"  Spectral recovery: {mode_result['spectral_recovery']:.4f}")
            print(f"  Reversibility class: {mode_result['reversibility_class']}")
    
    # Analysis
    analysis = analyze_results(results)
    
    if config.verbose:
        print("\n" + "=" * 60)
        print("ANALYSIS")
        print("=" * 60)
        print(f"SQNT vs Frozen fidelity diff: {analysis['sqnt_vs_frozen_fidelity_diff']:.4f}")
        print(f"SQNT vs Random fidelity diff: {analysis['sqnt_vs_random_fidelity_diff']:.4f}")
        print(f"Topology contribution to irreversibility: {analysis['topology_contribution']:.4f}")
        print(f"Conclusion: {analysis['conclusion']}")
    
    # Run sensitivity analysis if requested
    sensitivity = None
    if config.run_sensitivity_analysis:
        if config.verbose:
            print("\n--- Perturbation Sensitivity Analysis ---")
        
        sensitivity = run_sensitivity_analysis(M0, A0, config)
        
        if config.verbose:
            for mode, sens in sensitivity.items():
                print(f"  {mode}: sensitivity={sens['sensitivity']}, "
                      f"fidelity_drop={sens['fidelity_drop']:.4f}")
    
    return {
        'config': asdict(config),
        'results': results,
        'analysis': analysis,
        'sensitivity': sensitivity,
    }


def analyze_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze time-reversal results across modes."""
    
    sqnt = results['sqnt']
    frozen = results['frozen']
    random = results['random']
    
    # Fidelity differences
    sqnt_vs_frozen = sqnt['recovery_fidelity'] - frozen['recovery_fidelity']
    sqnt_vs_random = sqnt['recovery_fidelity'] - random['recovery_fidelity']
    
    # Topology contribution: how much does topology affect reversibility?
    # If SQNT is more irreversible than Frozen, topology creates genuine memory
    topology_contribution = frozen['recovery_fidelity'] - sqnt['recovery_fidelity']
    
    # Determine conclusion
    if topology_contribution > 0.1:
        conclusion = "Topology plasticity creates genuine irreversibility (structural memory)"
    elif topology_contribution < -0.1:
        conclusion = "Topology plasticity aids reversibility (unexpected)"
    else:
        if sqnt['recovery_fidelity'] > 0.9:
            conclusion = "Dynamics is fundamentally reversible (no learning)"
        elif sqnt['recovery_fidelity'] < 0.3:
            conclusion = "Dynamics is fundamentally irreversible (source not topology)"
        else:
            conclusion = "Partial reversibility; topology contribution unclear"
    
    return {
        'sqnt_vs_frozen_fidelity_diff': sqnt_vs_frozen,
        'sqnt_vs_random_fidelity_diff': sqnt_vs_random,
        'topology_contribution': topology_contribution,
        'conclusion': conclusion,
        'sqnt_class': sqnt['reversibility_class'],
        'frozen_class': frozen['reversibility_class'],
        'random_class': random['reversibility_class'],
    }


def run_sensitivity_analysis(
    M0: np.ndarray,
    A0: np.ndarray,
    config: TimeReversalExperimentConfig,
) -> Dict[str, Any]:
    """Run perturbation sensitivity analysis for each mode."""
    
    probe = TimeReversalProbe(verbose=False)
    
    results = {}
    
    for mode in [SystemMode.SQNT, SystemMode.FROZEN, SystemMode.RANDOM]:
        sqnt_config = SQNTConfig(mode=mode, seed=config.seed)
        system = SQNTSystem(sqnt_config)
        system.initialize(M0.copy(), A0.copy())
        
        forward = create_sqnt_forward_dynamics(
            system, config.decay_rate, config.noise_scale, config.seed
        )
        reverse = create_sqnt_reverse_dynamics(
            system, config.decay_rate, config.noise_scale, config.seed
        )
        
        sens = probe.run_with_perturbation(
            M0.copy(),
            A0.copy(),
            forward,
            reverse,
            n_steps=config.n_forward_steps,
            perturbation_scale=config.perturbation_scale,
            n_trials=config.n_sensitivity_trials,
        )
        
        results[mode.value] = sens
    
    return results


def parse_args():
    """Parse command line arguments."""
    p = argparse.ArgumentParser(
        description="Time-reversal experiment for SQNT dynamics"
    )
    
    p.add_argument("--dim", type=int, default=12, help="Matrix dimension")
    p.add_argument("--n-steps", type=int, default=50, help="Steps in each direction")
    p.add_argument("--decay-rate", type=float, default=0.01, help="Decay rate")
    p.add_argument("--noise-scale", type=float, default=0.001, help="Noise scale")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--output", type=str, default=None, help="Output JSON file")
    p.add_argument("--no-sensitivity", action="store_true", help="Skip sensitivity analysis")
    p.add_argument("--quiet", action="store_true", help="Quiet mode")
    
    return p.parse_args()


def main():
    args = parse_args()
    
    config = TimeReversalExperimentConfig(
        dim=args.dim,
        n_forward_steps=args.n_steps,
        n_reverse_steps=args.n_steps,
        decay_rate=args.decay_rate,
        noise_scale=args.noise_scale,
        run_sensitivity_analysis=not args.no_sensitivity,
        seed=args.seed,
        verbose=not args.quiet,
    )
    
    results = run_time_reversal_experiment(config)
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
