"""Canonical experiment spine: topology-aware kernel advantage benchmark.

This is the main experimental runner that:
1. Samples matrix M from ensemble with configurable action
2. Lifts M to graph topology G via configurable mode
3. Runs quantum kernel ridge regression with graph-conditioned encoder
4. Compares against classical RBF baseline
5. Computes phase diagram over graph parameters
6. Outputs JSON report with SQNT/Loop/UCIP bridge metrics

Usage:
    python -m experiments.topology_kernel_benchmark --dim 12 --graph-mode weighted

This implements the 9-step patch plan for making topology a first-class object.
"""

from __future__ import annotations
import argparse
import json
import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
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
    QuantumKernel,
    VariationalLawEncoder,
    GraphConditionedKernel,
    random_fourier_features_kernel,
    kernel_alignment,
)

# Analysis imports
from analysis import (
    SpectralDiagnostics,
    SpectralComplexityAnalyzer,
    PhaseTransitionAnalyzer,
    LawStabilityTracker,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Topology-aware quantum kernel advantage benchmark"
    )

    # Matrix model parameters
    p.add_argument("--dim", type=int, default=12, help="Matrix dimension N")
    p.add_argument("--g", type=float, default=0.1, help="Cubic coupling constant")
    p.add_argument("--langevin-steps", type=int, default=500, help="Langevin sampling steps")
    p.add_argument("--temperature", type=float, default=0.01, help="Langevin temperature")
    p.add_argument("--seed", type=int, default=0, help="Random seed")

    # Graph parameters (Step 2)
    p.add_argument("--graph-mode", choices=["threshold", "weighted", "laplacian"],
                   default="weighted", help="Graph extraction mode")
    p.add_argument("--graph-tau-q", type=float, default=0.5,
                   help="Quantile threshold for edges")
    p.add_argument("--graph-weighted", type=bool, default=True,
                   help="Use weighted edges in threshold mode")
    p.add_argument("--graph-alpha", type=float, default=1.0,
                   help="Power for weight transformation")
    p.add_argument("--graph-sparsify", choices=["none", "topk", "quantile"],
                   default="none", help="Sparsification method")
    p.add_argument("--graph-topk", type=int, default=5,
                   help="Top-k edges per node")
    p.add_argument("--graph-q", type=float, default=0.1,
                   help="Quantile for sparsification")
    p.add_argument("--graph-metric", choices=["abs", "fro"], default="abs",
                   help="Distance metric for laplacian mode")
    p.add_argument("--graph-beta", type=float, default=1.0,
                   help="Kernel bandwidth for laplacian mode")
    p.add_argument("--graph-sigma", type=float, default=0.0,
                   help="Noise regularization")
    p.add_argument("--graph-normalize", type=bool, default=True,
                   help="Normalize adjacency")

    # Graph family parameters
    p.add_argument("--use-graph-family", action="store_true",
                   help="Use graph family instead of matrix-derived graph")
    p.add_argument("--graph-family", choices=[f.value for f in GraphFamily],
                   default="erdos_renyi", help="Graph family to sample")
    p.add_argument("--family-p", type=float, default=0.3,
                   help="Edge probability for ER graphs")
    p.add_argument("--family-m", type=int, default=2,
                   help="Edges per new node for BA graphs")
    p.add_argument("--family-k", type=int, default=4,
                   help="Neighbors for WS/ring graphs")

    # Quantum kernel parameters
    p.add_argument("--n-qubits", type=int, default=4, help="Number of qubits")
    p.add_argument("--n-layers", type=int, default=2, help="Ansatz layers")
    p.add_argument("--n-samples", type=int, default=50, help="Data samples for kernel")
    p.add_argument("--edge-eps", type=float, default=0.01,
                   help="Minimum edge weight for entanglers")

    # Output
    p.add_argument("--output", type=str, default=None,
                   help="Output JSON file path")
    p.add_argument("--verbose", action="store_true", help="Verbose output")

    return p.parse_args()


def generate_regression_data(n_samples: int, dim: int, seed: int) -> tuple:
    """Generate synthetic regression data for kernel comparison."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, dim))

    # Nonlinear target function
    y = np.sin(X[:, 0]) + 0.5 * np.cos(X[:, 1] * X[:, 2]) + 0.1 * rng.normal(size=n_samples)

    # Split train/test
    n_train = int(0.7 * n_samples)
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    return X_train, X_test, y_train, y_test


def kernel_ridge_regression(K_train: np.ndarray, y_train: np.ndarray,
                            K_test: np.ndarray, ridge: float = 1e-3) -> np.ndarray:
    """Kernel ridge regression prediction."""
    n = K_train.shape[0]
    alpha = np.linalg.solve(K_train + ridge * np.eye(n), y_train)
    return K_test @ alpha


def compute_kernel_advantage(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    graph: Graph,
    n_qubits: int = 4,
    n_layers: int = 2,
    seed: int = 0,
) -> Dict[str, float]:
    """Compute quantum vs classical kernel advantage metrics.

    Uses the GraphConditionedKernel which encodes the graph topology
    into the quantum circuit's entanglement structure.
    """
    X_all = np.vstack([X_train, X_test])
    n_train = len(y_train)

    # Graph-conditioned quantum kernel
    gc_kernel = GraphConditionedKernel(
        n_qubits=n_qubits,
        n_layers=n_layers,
        seed=seed,
    )
    gc_kernel.set_graph(graph)
    K_gc = gc_kernel.gram(X_all)

    K_train_gc = K_gc[:n_train, :n_train]
    K_test_gc = K_gc[n_train:, :n_train]

    y_pred_gc = kernel_ridge_regression(K_train_gc, y_train, K_test_gc)
    mse_gc = float(np.mean((y_pred_gc - y_test) ** 2))
    r2_gc = 1.0 - mse_gc / (np.var(y_test) + 1e-12)

    # Standard quantum kernel (all-to-all entanglement for comparison)
    std_encoder = VariationalLawEncoder(n_qubits=n_qubits, n_layers=n_layers, seed=seed)
    qk_std = QuantumKernel(encode=std_encoder.encode)
    K_std = qk_std.gram(X_all)

    K_train_std = K_std[:n_train, :n_train]
    K_test_std = K_std[n_train:, :n_train]

    y_pred_std = kernel_ridge_regression(K_train_std, y_train, K_test_std)
    mse_std = float(np.mean((y_pred_std - y_test) ** 2))
    r2_std = 1.0 - mse_std / (np.var(y_test) + 1e-12)

    # Classical RBF kernel
    K_rbf = random_fourier_features_kernel(X_all, gamma=1.0, n_features=256)
    K_train_c = K_rbf[:n_train, :n_train]
    K_test_c = K_rbf[n_train:, :n_train]

    y_pred_c = kernel_ridge_regression(K_train_c, y_train, K_test_c)
    mse_c = float(np.mean((y_pred_c - y_test) ** 2))
    r2_c = 1.0 - mse_c / (np.var(y_test) + 1e-12)

    # Kernel alignment (with binary labels for simplicity)
    y_binary = (y_train > np.median(y_train)).astype(float) * 2 - 1
    align_gc = kernel_alignment(K_train_gc, y_binary)
    align_std = kernel_alignment(K_train_std, y_binary)
    align_c = kernel_alignment(K_train_c, y_binary)

    return {
        "mse_graph_conditioned": mse_gc,
        "mse_standard_quantum": mse_std,
        "mse_classical": mse_c,
        "r2_graph_conditioned": r2_gc,
        "r2_standard_quantum": r2_std,
        "r2_classical": r2_c,
        "alignment_graph_conditioned": align_gc,
        "alignment_standard_quantum": align_std,
        "alignment_classical": align_c,
        "advantage_gc_vs_classical": r2_gc - r2_c,
        "advantage_gc_vs_standard": r2_gc - r2_std,
        "advantage_std_vs_classical": r2_std - r2_c,
    }


def run_experiment(args) -> Dict[str, Any]:
    """Run the full topology-aware experiment."""
    results = {
        "config": vars(args),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    rng = np.random.default_rng(args.seed)

    # === STEP 1: Sample matrix M ===
    if args.verbose:
        print("Sampling matrix M...")

    ens = HermitianEnsemble(dim=args.dim, scale=1.0, seed=args.seed)
    M0 = ens.sample(rng=rng)

    action = CubicAction(g=args.g)
    sampler = LangevinSampler(
        action=action,
        dt=1e-3,
        temperature=args.temperature,
        seed=args.seed,
    )
    M = sampler.run(M0, steps=args.langevin_steps, rng=rng)

    # Matrix invariants
    inv = invariants(M, max_power=4)
    results["matrix_invariants"] = inv

    # Spectral analysis
    spec_diag = SpectralDiagnostics()
    spec_complex = SpectralComplexityAnalyzer()
    results["spectral_diagnostics"] = spec_diag.summarize(M)
    results["spectral_complexity"] = spec_complex.analyze(M)

    # === STEP 3: Lift M to graph G ===
    if args.verbose:
        print("Constructing graph topology...")

    if args.use_graph_family:
        # Use graph family directly
        family = GraphFamily(args.graph_family)
        params = {"p": args.family_p, "m": args.family_m, "k": args.family_k}
        _, params_used, G = sample_graph_family(family, args.dim, params, args.seed)
        results["graph_source"] = "family"
        results["graph_family"] = args.graph_family
        results["graph_family_params"] = params_used
    else:
        # Derive from matrix
        graph_params = {
            "mode": args.graph_mode,
            "tau_quantile": args.graph_tau_q,
            "weighted": args.graph_weighted,
            "alpha": args.graph_alpha,
            "sparsify": args.graph_sparsify,
            "topk": args.graph_topk,
            "q": args.graph_q,
            "metric": args.graph_metric,
            "beta": args.graph_beta,
            "sigma": args.graph_sigma,
            "normalize": args.graph_normalize,
        }
        G = matrix_to_graph(M, mode=args.graph_mode, params=graph_params)
        results["graph_source"] = "matrix_to_graph"

    # Graph summary (Step 9: graph_summary)
    results["graph_summary"] = G.summary()

    if args.verbose:
        print(f"  Graph: {G.n_nodes} nodes, {G.n_edges()} edges")
        print(f"  Algebraic connectivity: {G.algebraic_connectivity():.4f}")

    # === STEP 5 + 7: Quantum kernel with graph-conditioned encoder ===
    if args.verbose:
        print("Computing kernel advantage...")

    # Generate regression data
    X_train, X_test, y_train, y_test = generate_regression_data(
        args.n_samples, args.dim, args.seed
    )

    # Compute kernel advantage with graph-conditioned kernel
    advantage = compute_kernel_advantage(
        X_train, X_test, y_train, y_test,
        graph=G,
        n_qubits=args.n_qubits,
        n_layers=args.n_layers,
        seed=args.seed,
    )
    results["kernel_advantage"] = advantage

    # Step 9: kernel_advantage_vs_topology
    results["kernel_advantage_vs_topology"] = {
        "advantage_gc_vs_classical": advantage["advantage_gc_vs_classical"],
        "advantage_gc_vs_standard": advantage["advantage_gc_vs_standard"],
        "algebraic_connectivity": G.algebraic_connectivity(),
        "spectral_gap": G.spectral_gap(),
        "density": G.density(),
        "clustering": G.clustering_coefficient(),
    }

    if args.verbose:
        print(f"  Graph-conditioned R²: {advantage['r2_graph_conditioned']:.4f}")
        print(f"  Standard quantum R²: {advantage['r2_standard_quantum']:.4f}")
        print(f"  Classical R²: {advantage['r2_classical']:.4f}")
        print(f"  GC vs Classical: {advantage['advantage_gc_vs_classical']:.4f}")
        print(f"  GC vs Standard: {advantage['advantage_gc_vs_standard']:.4f}")

    # === STEP 8: Phase boundary detection ===
    # Run parameter sweep for phase diagram data
    if args.verbose:
        print("Computing phase indicators...")

    phase_data = {
        "g": args.g,
        "laplacian_gap": G.spectral_gap(),
        "effective_dimension": results["spectral_complexity"]["participation_ratio_mean"] * args.dim,
        "spectral_entropy": results["spectral_complexity"]["spectral_entropy"],
        "is_chaotic": results["spectral_complexity"]["is_chaotic"],
    }
    results["phase_indicators"] = phase_data

    # === STEP 9: SQNT/Loop/UCIP bridge metrics ===
    results["sqnt_bridge"] = {
        "topology_n_edges": G.n_edges(),
        "topology_mean_degree": G.mean_degree(),
        "kernel_geometry_alignment": advantage["alignment_graph_conditioned"],
    }

    results["loop_bridge"] = {
        "spectral_radius": results["spectral_diagnostics"]["spectral_radius"],
        "level_spacing_r": results["spectral_complexity"]["level_spacing_r"],
        "is_chaotic": results["spectral_complexity"]["is_chaotic"],
    }

    # UCIP placeholder (would need actual training loop)
    results["ucip_vs_topology"] = {
        "topology_density": G.density(),
        "topology_connectivity": G.algebraic_connectivity(),
        # CI metrics would be computed during actual training
        "ci_score_placeholder": None,
        "recovery_time_placeholder": None,
    }

    return results


def main():
    args = parse_args()

    if args.verbose:
        print("=" * 60)
        print("TOPOLOGY-AWARE KERNEL BENCHMARK")
        print("=" * 60)

    results = run_experiment(args)

    # Output
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results saved to: {args.output}")
    else:
        print("\n" + "=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)

        print("\nGraph Summary:")
        for k, v in results["graph_summary"].items():
            print(f"  {k}: {v:.4f}")

        print("\nKernel Advantage:")
        for k, v in results["kernel_advantage"].items():
            print(f"  {k}: {v:.4f}")

        print("\nPhase Indicators:")
        for k, v in results["phase_indicators"].items():
            print(f"  {k}: {v}")

        print("\nSQNT Bridge:")
        for k, v in results["sqnt_bridge"].items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    main()
