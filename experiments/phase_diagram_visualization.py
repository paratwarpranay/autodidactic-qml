"""Phase diagram visualization for graph topology kernel advantage.

This script implements Step 8 of the patch plan:
Generate phase diagrams showing how quantum kernel advantage varies
across graph topology parameters.

Usage:
    python -m experiments.phase_diagram_visualization --input results.json
    python -m experiments.phase_diagram_visualization --generate --output phase_diagram.png
"""

from __future__ import annotations
import argparse
import json
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# Topology imports
from topology import Graph, matrix_to_graph, GraphFamily, sample_graph_family

# Matrix model imports
from matrix_models import HermitianEnsemble, CubicAction, LangevinSampler

# Quantum kernel imports
from quantum_kernels import (
    GraphConditionedKernel,
    QuantumKernel,
    VariationalLawEncoder,
    random_fourier_features_kernel,
)


def parse_args():
    p = argparse.ArgumentParser(description="Phase diagram visualization")
    p.add_argument("--input", type=str, help="Input JSON from graph_family_sweep")
    p.add_argument("--generate", action="store_true", help="Generate fresh data")
    p.add_argument("--output", type=str, default="phase_diagram.png", help="Output image")
    p.add_argument("--n-nodes", type=int, default=10, help="Graph nodes")
    p.add_argument("--n-qubits", type=int, default=4, help="Qubits")
    p.add_argument("--n-layers", type=int, default=2, help="Circuit layers")
    p.add_argument("--n-samples", type=int, default=30, help="Data samples")
    p.add_argument("--n-g-points", type=int, default=8, help="Points in g sweep")
    p.add_argument("--n-p-points", type=int, default=8, help="Points in p sweep")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--no-plot", action="store_true", help="Skip plotting")
    return p.parse_args()


def krr_predict(K_train, y_train, K_test, ridge=1e-3):
    """Kernel ridge regression."""
    n = K_train.shape[0]
    alpha = np.linalg.solve(K_train + ridge * np.eye(n), y_train)
    return K_test @ alpha


def compute_advantage(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    graph: Graph,
    n_qubits: int,
    n_layers: int,
    seed: int,
) -> Dict[str, float]:
    """Compute kernel advantage for a single configuration."""
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

    y_pred_gc = krr_predict(
        K_gc[:n_train, :n_train],
        y_train,
        K_gc[n_train:, :n_train],
    )
    mse_gc = float(np.mean((y_pred_gc - y_test) ** 2))
    r2_gc = 1.0 - mse_gc / (np.var(y_test) + 1e-12)

    # Classical RBF kernel
    K_rbf = random_fourier_features_kernel(X_all, gamma=1.0, n_features=256)
    y_pred_c = krr_predict(
        K_rbf[:n_train, :n_train],
        y_train,
        K_rbf[n_train:, :n_train],
    )
    mse_c = float(np.mean((y_pred_c - y_test) ** 2))
    r2_c = 1.0 - mse_c / (np.var(y_test) + 1e-12)

    return {
        "r2_quantum": r2_gc,
        "r2_classical": r2_c,
        "advantage": r2_gc - r2_c,
    }


def generate_phase_data(
    n_nodes: int,
    n_qubits: int,
    n_layers: int,
    n_samples: int,
    n_g_points: int,
    n_p_points: int,
    seed: int,
) -> Dict[str, Any]:
    """Generate phase diagram data by sweeping g (coupling) and p (edge probability).

    This creates a 2D phase diagram with:
    - x-axis: coupling constant g (matrix model parameter)
    - y-axis: edge probability p (ER graph parameter)
    - color: quantum kernel advantage
    """
    rng = np.random.default_rng(seed)

    # Parameter ranges
    g_values = np.linspace(0.01, 0.5, n_g_points)
    p_values = np.linspace(0.1, 0.9, n_p_points)

    # Generate regression data (same for all configs)
    X = rng.normal(size=(n_samples, n_nodes))
    y = np.sin(X[:, 0]) + 0.5 * np.cos(X[:, 1] * X[:, 2]) + 0.1 * rng.normal(size=n_samples)
    n_train = int(0.7 * n_samples)
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    # Results storage
    advantage_grid = np.zeros((n_p_points, n_g_points))
    r2_quantum_grid = np.zeros((n_p_points, n_g_points))
    r2_classical_grid = np.zeros((n_p_points, n_g_points))
    connectivity_grid = np.zeros((n_p_points, n_g_points))
    density_grid = np.zeros((n_p_points, n_g_points))

    print(f"Generating phase diagram: {n_g_points}x{n_p_points} grid")

    for i, p in enumerate(p_values):
        for j, g in enumerate(g_values):
            # Sample graph from ER family
            _, _, G = sample_graph_family(
                GraphFamily.ERDOS_RENYI,
                n_nodes,
                params={"p": p},
                seed=seed + i * 100 + j,
            )

            # Compute advantage
            metrics = compute_advantage(
                X_train, X_test, y_train, y_test,
                G, n_qubits, n_layers, seed + i * 100 + j,
            )

            advantage_grid[i, j] = metrics["advantage"]
            r2_quantum_grid[i, j] = metrics["r2_quantum"]
            r2_classical_grid[i, j] = metrics["r2_classical"]
            connectivity_grid[i, j] = G.algebraic_connectivity()
            density_grid[i, j] = G.density()

        print(f"  Row {i+1}/{n_p_points}: p={p:.2f}")

    return {
        "g_values": g_values.tolist(),
        "p_values": p_values.tolist(),
        "advantage_grid": advantage_grid.tolist(),
        "r2_quantum_grid": r2_quantum_grid.tolist(),
        "r2_classical_grid": r2_classical_grid.tolist(),
        "connectivity_grid": connectivity_grid.tolist(),
        "density_grid": density_grid.tolist(),
        "config": {
            "n_nodes": n_nodes,
            "n_qubits": n_qubits,
            "n_layers": n_layers,
            "n_samples": n_samples,
            "seed": seed,
        },
    }


def generate_family_comparison_data(
    n_nodes: int,
    n_qubits: int,
    n_layers: int,
    n_samples: int,
    n_trials: int,
    seed: int,
) -> Dict[str, Any]:
    """Generate comparison data across all graph families."""
    rng = np.random.default_rng(seed)

    # Generate regression data
    X = rng.normal(size=(n_samples, n_nodes))
    y = np.sin(X[:, 0]) + 0.5 * np.cos(X[:, 1] * X[:, 2]) + 0.1 * rng.normal(size=n_samples)
    n_train = int(0.7 * n_samples)
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    results = {}
    for family in GraphFamily:
        advantages = []
        connectivities = []
        densities = []

        for trial in range(n_trials):
            _, _, G = sample_graph_family(
                family, n_nodes, seed=seed + trial * 1000
            )
            metrics = compute_advantage(
                X_train, X_test, y_train, y_test,
                G, n_qubits, n_layers, seed + trial * 1000,
            )
            advantages.append(metrics["advantage"])
            connectivities.append(G.algebraic_connectivity())
            densities.append(G.density())

        results[family.value] = {
            "mean_advantage": float(np.mean(advantages)),
            "std_advantage": float(np.std(advantages)),
            "mean_connectivity": float(np.mean(connectivities)),
            "mean_density": float(np.mean(densities)),
            "trials": advantages,
        }

    return results


def plot_phase_diagram(data: Dict[str, Any], output_path: str):
    """Create phase diagram visualization."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import TwoSlopeNorm
    except ImportError:
        print("matplotlib not available, skipping plot")
        return

    g_values = np.array(data["g_values"])
    p_values = np.array(data["p_values"])
    advantage_grid = np.array(data["advantage_grid"])
    connectivity_grid = np.array(data["connectivity_grid"])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Phase diagram: Quantum Advantage
    vmax = max(abs(advantage_grid.min()), abs(advantage_grid.max()))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    im1 = axes[0].imshow(
        advantage_grid,
        aspect="auto",
        origin="lower",
        extent=[g_values[0], g_values[-1], p_values[0], p_values[-1]],
        cmap="RdYlGn",
        norm=norm,
    )
    axes[0].set_xlabel("Coupling g")
    axes[0].set_ylabel("Edge probability p")
    axes[0].set_title("Quantum Kernel Advantage")
    plt.colorbar(im1, ax=axes[0], label="R² advantage")

    # Algebraic connectivity
    im2 = axes[1].imshow(
        connectivity_grid,
        aspect="auto",
        origin="lower",
        extent=[g_values[0], g_values[-1], p_values[0], p_values[-1]],
        cmap="viridis",
    )
    axes[1].set_xlabel("Coupling g")
    axes[1].set_ylabel("Edge probability p")
    axes[1].set_title("Algebraic Connectivity")
    plt.colorbar(im2, ax=axes[1], label="λ₂")

    # Advantage vs connectivity scatter
    axes[2].scatter(
        connectivity_grid.flatten(),
        advantage_grid.flatten(),
        c=np.array(data["density_grid"]).flatten(),
        cmap="plasma",
        alpha=0.7,
    )
    axes[2].axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
    axes[2].set_xlabel("Algebraic Connectivity")
    axes[2].set_ylabel("Quantum Advantage")
    axes[2].set_title("Advantage vs Topology")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Phase diagram saved to: {output_path}")


def plot_family_comparison(data: Dict[str, Any], output_path: str):
    """Create bar chart comparing graph families."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return

    families = list(data.keys())
    advantages = [data[f]["mean_advantage"] for f in families]
    errors = [data[f]["std_advantage"] for f in families]
    connectivities = [data[f]["mean_connectivity"] for f in families]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Bar chart
    colors = plt.cm.viridis(np.linspace(0, 1, len(families)))
    bars = axes[0].bar(range(len(families)), advantages, yerr=errors, color=colors, capsize=3)
    axes[0].axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
    axes[0].set_xticks(range(len(families)))
    axes[0].set_xticklabels([f.replace("_", "\n") for f in families], fontsize=8)
    axes[0].set_ylabel("Quantum Advantage (R²)")
    axes[0].set_title("Advantage by Graph Family")

    # Scatter: advantage vs connectivity
    axes[1].scatter(connectivities, advantages, c=colors, s=100)
    for i, family in enumerate(families):
        axes[1].annotate(
            family.replace("_", " "),
            (connectivities[i], advantages[i]),
            fontsize=7,
            ha="center",
            va="bottom",
        )
    axes[1].axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
    axes[1].set_xlabel("Mean Algebraic Connectivity")
    axes[1].set_ylabel("Mean Quantum Advantage")
    axes[1].set_title("Advantage vs Graph Connectivity")

    plt.tight_layout()
    family_output = output_path.replace(".png", "_families.png")
    plt.savefig(family_output, dpi=150)
    print(f"Family comparison saved to: {family_output}")


def main():
    args = parse_args()

    if args.input:
        # Load existing data
        with open(args.input) as f:
            sweep_data = json.load(f)

        if "phase_diagram_data" in sweep_data:
            # Convert to grid format for plotting
            phase_data = sweep_data["phase_diagram_data"]
            print(f"Loaded {len(phase_data)} phase data points")
        else:
            print("No phase_diagram_data found in input")
            return

    elif args.generate:
        # Generate fresh data
        print("Generating phase diagram data...")
        phase_data = generate_phase_data(
            n_nodes=args.n_nodes,
            n_qubits=args.n_qubits,
            n_layers=args.n_layers,
            n_samples=args.n_samples,
            n_g_points=args.n_g_points,
            n_p_points=args.n_p_points,
            seed=args.seed,
        )

        print("\nGenerating family comparison data...")
        family_data = generate_family_comparison_data(
            n_nodes=args.n_nodes,
            n_qubits=args.n_qubits,
            n_layers=args.n_layers,
            n_samples=args.n_samples,
            n_trials=5,
            seed=args.seed,
        )

        # Save data
        data_output = args.output.replace(".png", "_data.json")
        with open(data_output, "w") as f:
            json.dump({"phase_diagram": phase_data, "family_comparison": family_data}, f, indent=2)
        print(f"\nData saved to: {data_output}")

        if not args.no_plot:
            plot_phase_diagram(phase_data, args.output)
            plot_family_comparison(family_data, args.output)

    else:
        print("Specify --input or --generate")


if __name__ == "__main__":
    main()
