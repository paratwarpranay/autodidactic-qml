"""Sweep quantum kernel advantage across graph families.

This experiment implements Steps 7 and 8 of the patch plan:
- Test kernel advantage across different graph topologies
- Generate phase diagram over graph parameters

Usage:
    python -m experiments.graph_family_sweep --n-nodes 12 --output results.json
"""

from __future__ import annotations
import argparse
import json
import numpy as np
from typing import Dict, Any, List
from dataclasses import dataclass, asdict
import time

# Topology imports
from topology import Graph, matrix_to_graph, GraphFamily, sample_graph_family

# Matrix model imports
from matrix_models import HermitianEnsemble, CubicAction, LangevinSampler

# Quantum kernel imports
from quantum_kernels import (
    QuantumKernel,
    VariationalLawEncoder,
    GraphConditionedKernel,
    random_fourier_features_kernel,
    kernel_alignment,
)

# Analysis imports
from analysis import SpectralComplexityAnalyzer


def parse_args():
    p = argparse.ArgumentParser(description="Graph family kernel advantage sweep")
    p.add_argument("--n-nodes", type=int, default=12, help="Number of nodes")
    p.add_argument("--n-qubits", type=int, default=4, help="Number of qubits")
    p.add_argument("--n-layers", type=int, default=2, help="Ansatz layers")
    p.add_argument("--n-samples", type=int, default=40, help="Data samples")
    p.add_argument("--n-trials", type=int, default=3, help="Trials per family")
    p.add_argument("--seed", type=int, default=0, help="Random seed")
    p.add_argument("--output", type=str, default=None, help="Output JSON file")
    p.add_argument("--verbose", action="store_true", help="Verbose output")
    return p.parse_args()


def generate_data(n_samples: int, dim: int, seed: int):
    """Generate regression data."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, dim))
    y = np.sin(X[:, 0]) + 0.5 * np.cos(X[:, 1] * X[:, 2]) + 0.1 * rng.normal(size=n_samples)
    n_train = int(0.7 * n_samples)
    return X[:n_train], X[n_train:], y[:n_train], y[n_train:]


def krr_predict(K_train, y_train, K_test, ridge=1e-3):
    """Kernel ridge regression."""
    n = K_train.shape[0]
    alpha = np.linalg.solve(K_train + ridge * np.eye(n), y_train)
    return K_test @ alpha


def evaluate_kernel(
    K_train: np.ndarray,
    K_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, float]:
    """Evaluate kernel performance."""
    y_pred = krr_predict(K_train, y_train, K_test)
    mse = float(np.mean((y_pred - y_test) ** 2))
    r2 = 1.0 - mse / (np.var(y_test) + 1e-12)

    # Alignment with binary labels
    y_binary = (y_train > np.median(y_train)).astype(float) * 2 - 1
    align = kernel_alignment(K_train, y_binary)

    return {"mse": mse, "r2": r2, "alignment": align}


def run_family_experiment(
    family: GraphFamily,
    n_nodes: int,
    n_qubits: int,
    n_layers: int,
    n_samples: int,
    seed: int,
) -> Dict[str, Any]:
    """Run experiment for a single graph family."""

    # Sample graph
    _, params_used, G = sample_graph_family(family, n_nodes, seed=seed)

    # Generate data
    X_train, X_test, y_train, y_test = generate_data(n_samples, n_nodes, seed)
    X_all = np.vstack([X_train, X_test])
    n_train = len(y_train)

    # Graph-conditioned quantum kernel
    gc_kernel = GraphConditionedKernel(
        n_qubits=n_qubits,
        n_layers=n_layers,
        seed=seed,
    )
    gc_kernel.set_graph(G)
    K_gc = gc_kernel.gram(X_all)
    gc_metrics = evaluate_kernel(
        K_gc[:n_train, :n_train],
        K_gc[n_train:, :n_train],
        y_train, y_test
    )

    # Standard quantum kernel (all-to-all)
    std_encoder = VariationalLawEncoder(n_qubits=n_qubits, n_layers=n_layers, seed=seed)
    std_kernel = QuantumKernel(encode=std_encoder.encode)
    K_std = std_kernel.gram(X_all)
    std_metrics = evaluate_kernel(
        K_std[:n_train, :n_train],
        K_std[n_train:, :n_train],
        y_train, y_test
    )

    # Classical RBF
    K_rbf = random_fourier_features_kernel(X_all, gamma=1.0, n_features=256)
    rbf_metrics = evaluate_kernel(
        K_rbf[:n_train, :n_train],
        K_rbf[n_train:, :n_train],
        y_train, y_test
    )

    return {
        "family": family.value,
        "family_params": params_used,
        "graph_summary": G.summary(),
        "graph_conditioned": gc_metrics,
        "standard_quantum": std_metrics,
        "classical_rbf": rbf_metrics,
        "advantage_gc_vs_classical": gc_metrics["r2"] - rbf_metrics["r2"],
        "advantage_std_vs_classical": std_metrics["r2"] - rbf_metrics["r2"],
        "advantage_gc_vs_std": gc_metrics["r2"] - std_metrics["r2"],
    }


def run_matrix_baseline(
    n_nodes: int,
    n_qubits: int,
    n_layers: int,
    n_samples: int,
    seed: int,
    g: float = 0.1,
) -> Dict[str, Any]:
    """Run baseline with matrix-derived graph."""

    # Sample matrix
    rng = np.random.default_rng(seed)
    ens = HermitianEnsemble(dim=n_nodes, seed=seed)
    M0 = ens.sample(rng=rng)
    action = CubicAction(g=g)
    sampler = LangevinSampler(action=action, dt=1e-3, temperature=0.01, seed=seed)
    M = sampler.run(M0, steps=300, rng=rng)

    # Matrix to graph
    G = matrix_to_graph(M, mode="weighted")

    # Generate data
    X_train, X_test, y_train, y_test = generate_data(n_samples, n_nodes, seed)
    X_all = np.vstack([X_train, X_test])
    n_train = len(y_train)

    # Graph-conditioned quantum kernel
    gc_kernel = GraphConditionedKernel(n_qubits=n_qubits, n_layers=n_layers, seed=seed)
    gc_kernel.set_graph(G)
    K_gc = gc_kernel.gram(X_all)
    gc_metrics = evaluate_kernel(
        K_gc[:n_train, :n_train],
        K_gc[n_train:, :n_train],
        y_train, y_test
    )

    # Classical RBF
    K_rbf = random_fourier_features_kernel(X_all, gamma=1.0, n_features=256)
    rbf_metrics = evaluate_kernel(
        K_rbf[:n_train, :n_train],
        K_rbf[n_train:, :n_train],
        y_train, y_test
    )

    # Spectral complexity
    complexity = SpectralComplexityAnalyzer().analyze(M)

    return {
        "source": "matrix_to_graph",
        "g": g,
        "graph_summary": G.summary(),
        "spectral_complexity": complexity,
        "graph_conditioned": gc_metrics,
        "classical_rbf": rbf_metrics,
        "advantage": gc_metrics["r2"] - rbf_metrics["r2"],
    }


def main():
    args = parse_args()

    results = {
        "config": vars(args),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "families": {},
        "matrix_baseline": [],
    }

    if args.verbose:
        print("=" * 60)
        print("GRAPH FAMILY KERNEL ADVANTAGE SWEEP")
        print("=" * 60)

    # Run for each graph family
    for family in GraphFamily:
        if args.verbose:
            print(f"\n{family.value}...")

        family_results = []
        for trial in range(args.n_trials):
            trial_seed = args.seed + trial * 1000
            res = run_family_experiment(
                family=family,
                n_nodes=args.n_nodes,
                n_qubits=args.n_qubits,
                n_layers=args.n_layers,
                n_samples=args.n_samples,
                seed=trial_seed,
            )
            family_results.append(res)

            if args.verbose:
                print(f"  Trial {trial}: GC R²={res['graph_conditioned']['r2']:.3f}, "
                      f"Advantage={res['advantage_gc_vs_classical']:.3f}")

        # Aggregate
        advantages = [r["advantage_gc_vs_classical"] for r in family_results]
        results["families"][family.value] = {
            "trials": family_results,
            "mean_advantage": float(np.mean(advantages)),
            "std_advantage": float(np.std(advantages)),
            "mean_graph_density": float(np.mean([r["graph_summary"]["density"] for r in family_results])),
            "mean_connectivity": float(np.mean([r["graph_summary"]["algebraic_connectivity"] for r in family_results])),
        }

    # Run matrix baseline
    if args.verbose:
        print("\nMatrix-derived baseline...")

    for trial in range(args.n_trials):
        trial_seed = args.seed + trial * 1000
        res = run_matrix_baseline(
            n_nodes=args.n_nodes,
            n_qubits=args.n_qubits,
            n_layers=args.n_layers,
            n_samples=args.n_samples,
            seed=trial_seed,
        )
        results["matrix_baseline"].append(res)

        if args.verbose:
            print(f"  Trial {trial}: GC R²={res['graph_conditioned']['r2']:.3f}, "
                  f"Advantage={res['advantage']:.3f}")

    # Phase diagram data (Step 8)
    if args.verbose:
        print("\nGenerating phase diagram data...")

    phase_data = []
    for family_name, family_data in results["families"].items():
        for trial in family_data["trials"]:
            phase_data.append({
                "family": family_name,
                "density": trial["graph_summary"]["density"],
                "connectivity": trial["graph_summary"]["algebraic_connectivity"],
                "spectral_gap": trial["graph_summary"]["spectral_gap"],
                "clustering": trial["graph_summary"]["clustering_coefficient"],
                "advantage": trial["advantage_gc_vs_classical"],
                "r2_quantum": trial["graph_conditioned"]["r2"],
                "r2_classical": trial["classical_rbf"]["r2"],
            })

    results["phase_diagram_data"] = phase_data

    # Summary
    if args.verbose:
        print("\n" + "=" * 60)
        print("SUMMARY: Mean Advantage by Family")
        print("=" * 60)
        for family_name, family_data in results["families"].items():
            print(f"  {family_name:20s}: {family_data['mean_advantage']:+.4f} "
                  f"(±{family_data['std_advantage']:.4f})")

    # Output
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to: {args.output}")
    else:
        print("\nUse --output to save results to JSON file")


if __name__ == "__main__":
    main()
