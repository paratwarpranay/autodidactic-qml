"""Entanglement graph construction from matrix models.

This module implements the core idea from "The Autodidactic Universe":
interpret matrix elements as interaction strengths and construct a
graph that represents the system's "architecture".

Scientific hypotheses:
1. Graph connectivity correlates with learning capacity
2. Spectral gap of graph Laplacian indicates phase transitions
3. Community structure emerges during autodidactic learning
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Optional


def adjacency_from_matrix(
    M: np.ndarray,
    threshold: Optional[float] = None,
    weighted: bool = True,
) -> np.ndarray:
    """Construct adjacency matrix from a symmetric matrix.

    Args:
        M: Symmetric matrix (coupling strengths)
        threshold: If provided, zero out entries below threshold
        weighted: If True, use |M_ij| as edge weights; else binary

    Returns:
        Adjacency matrix A
    """
    M = (M + M.T) / 2.0  # Ensure symmetry
    A = np.abs(M).copy()
    np.fill_diagonal(A, 0.0)  # No self-loops

    if threshold is not None:
        A[A < threshold] = 0.0

    if not weighted:
        A = (A > 0).astype(float)

    return A


def graph_laplacian(A: np.ndarray, normalized: bool = False) -> np.ndarray:
    """Compute graph Laplacian from adjacency matrix.

    Args:
        A: Adjacency matrix
        normalized: If True, compute normalized Laplacian L_n = I - D^{-1/2} A D^{-1/2}

    Returns:
        Graph Laplacian L
    """
    d = np.sum(A, axis=1)

    if normalized:
        d_inv_sqrt = 1.0 / np.sqrt(d + 1e-12)
        D_inv_sqrt = np.diag(d_inv_sqrt)
        L = np.eye(A.shape[0]) - D_inv_sqrt @ A @ D_inv_sqrt
    else:
        D = np.diag(d)
        L = D - A

    return L


def laplacian_spectrum(L: np.ndarray) -> np.ndarray:
    """Compute eigenvalues of graph Laplacian (sorted ascending)."""
    evals = np.linalg.eigvalsh(L)
    return np.sort(np.maximum(evals, 0.0))  # Numerical stability


def algebraic_connectivity(L: np.ndarray) -> float:
    """Compute algebraic connectivity (second smallest eigenvalue of L).

    This measures how "connected" the graph is:
    - λ_2 = 0: Graph is disconnected
    - λ_2 > 0: Graph is connected
    - Larger λ_2: More robust connectivity
    """
    evals = laplacian_spectrum(L)
    if len(evals) < 2:
        return 0.0
    return float(evals[1])


def spectral_gap(L: np.ndarray) -> float:
    """Compute spectral gap λ_2 - λ_1 (should equal λ_2 for connected graphs)."""
    evals = laplacian_spectrum(L)
    if len(evals) < 2:
        return 0.0
    return float(evals[1] - evals[0])


def effective_resistance(L: np.ndarray) -> float:
    """Compute total effective resistance of the graph.

    R_total = n * sum_{i>0} 1/λ_i

    This measures the overall "resistance" of the graph to information flow.
    """
    evals = laplacian_spectrum(L)
    n = len(evals)
    # Skip zero eigenvalue(s)
    nonzero = evals[evals > 1e-10]
    if len(nonzero) == 0:
        return float('inf')
    return float(n * np.sum(1.0 / nonzero))


def von_neumann_entropy(L: np.ndarray) -> float:
    """Compute von Neumann entropy of the normalized Laplacian.

    S = -Tr(ρ log ρ) where ρ = L / Tr(L)

    This measures the "quantum" complexity of the graph structure.
    """
    evals = laplacian_spectrum(L)
    trace_L = np.sum(evals)
    if trace_L < 1e-12:
        return 0.0

    # Normalize eigenvalues to form density matrix spectrum
    rho_evals = evals / trace_L
    rho_evals = rho_evals[rho_evals > 1e-12]

    return float(-np.sum(rho_evals * np.log(rho_evals)))


@dataclass(frozen=True)
class EntanglementGraphAnalyzer:
    """Analyze the entanglement graph structure of a matrix model.

    This class constructs and analyzes the graph interpretation of a matrix,
    following the autodidactic universe framework where matrix elements
    define interaction strengths.
    """
    threshold: Optional[float] = None
    weighted: bool = True
    normalized_laplacian: bool = False

    def build_graph(self, M: np.ndarray) -> Dict[str, np.ndarray]:
        """Build graph structures from matrix.

        Returns:
            Dict with adjacency, laplacian, and eigenvalues
        """
        A = adjacency_from_matrix(M, self.threshold, self.weighted)
        L = graph_laplacian(A, self.normalized_laplacian)
        evals = laplacian_spectrum(L)

        return {
            "adjacency": A,
            "laplacian": L,
            "laplacian_eigenvalues": evals,
        }

    def analyze(self, M: np.ndarray) -> Dict[str, float]:
        """Full graph-theoretic analysis of matrix.

        Returns:
            Dictionary of graph metrics
        """
        graph = self.build_graph(M)
        A = graph["adjacency"]
        L = graph["laplacian"]
        evals = graph["laplacian_eigenvalues"]

        n = A.shape[0]
        n_edges = int(np.sum(A > 0) / 2)  # Undirected
        density = 2.0 * n_edges / (n * (n - 1)) if n > 1 else 0.0

        return {
            "n_vertices": float(n),
            "n_edges": float(n_edges),
            "edge_density": density,
            "mean_degree": float(np.mean(np.sum(A > 0, axis=1))),
            "max_degree": float(np.max(np.sum(A > 0, axis=1))),
            "algebraic_connectivity": algebraic_connectivity(L),
            "spectral_gap": spectral_gap(L),
            "effective_resistance": effective_resistance(L),
            "von_neumann_entropy": von_neumann_entropy(L),
            "laplacian_trace": float(np.sum(evals)),
        }


def graph_distance(M1: np.ndarray, M2: np.ndarray, **kwargs) -> float:
    """Compute distance between two matrices in graph-metric space.

    Uses spectral distance between graph Laplacians.
    """
    analyzer = EntanglementGraphAnalyzer(**kwargs)
    g1 = analyzer.build_graph(M1)
    g2 = analyzer.build_graph(M2)

    e1 = g1["laplacian_eigenvalues"]
    e2 = g2["laplacian_eigenvalues"]

    # Pad to same length if needed
    max_len = max(len(e1), len(e2))
    e1 = np.pad(e1, (0, max_len - len(e1)))
    e2 = np.pad(e2, (0, max_len - len(e2)))

    return float(np.linalg.norm(e1 - e2) / (np.linalg.norm(e1) + 1e-12))
