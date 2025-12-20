"""Canonical graph structure for the autodidactic experiment spine.

This module defines the Graph dataclass and the matrix_to_graph lift
that converts a symmetric matrix M into an explicit graph representation.

The graph is the "architecture" that emerges from the matrix model,
following the core insight of "The Autodidactic Universe": matrix elements
define interaction strengths, which we interpret as graph edges.

Modes:
- threshold: Binary edges where |M_ij| > tau
- weighted: Edge weights = |M_ij|^alpha (optionally thresholded)
- laplacian: Graph derived from distance-based kernel on matrix structure
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Literal, Iterator


@dataclass(frozen=True)
class GraphParams:
    """Parameters for matrix-to-graph conversion.

    Attributes:
        mode: Conversion mode ('threshold', 'weighted', 'laplacian')
        tau_quantile: Quantile threshold for edge inclusion (0-1)
        weighted: Whether to use weights in threshold mode
        alpha: Power for weight transformation |M_ij|^alpha
        sparsify: Sparsification method ('none', 'topk', 'quantile')
        topk: Number of edges to keep per node (for topk sparsification)
        q: Quantile for quantile-based sparsification
        metric: Distance metric for laplacian mode ('abs', 'fro')
        beta: Kernel bandwidth for laplacian mode
        sigma: Optional noise regularization
        normalize: Whether to normalize the final adjacency
    """
    mode: Literal["threshold", "weighted", "laplacian"] = "weighted"
    tau_quantile: float = 0.5
    weighted: bool = True
    alpha: float = 1.0
    sparsify: Literal["none", "topk", "quantile"] = "none"
    topk: int = 5
    q: float = 0.1
    metric: Literal["abs", "fro"] = "abs"
    beta: float = 1.0
    sigma: float = 0.0
    normalize: bool = True


@dataclass(frozen=True)
class Graph:
    """Immutable weighted graph representation.

    This is the canonical graph object used throughout the experiment spine.
    It stores the adjacency matrix and provides methods for:
    - Degree computation
    - Laplacian construction
    - Edge iteration (for quantum circuit construction)
    - Summary statistics

    Attributes:
        adjacency: (n, n) weighted adjacency matrix (symmetric, zero diagonal)
        n_nodes: Number of nodes
        metadata: Optional metadata about graph origin
    """
    adjacency: np.ndarray
    n_nodes: int = field(init=False)
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, 'n_nodes', self.adjacency.shape[0])

    def degree(self) -> np.ndarray:
        """Compute degree vector (sum of edge weights per node)."""
        return np.sum(self.adjacency, axis=1)

    def degree_matrix(self) -> np.ndarray:
        """Compute diagonal degree matrix."""
        return np.diag(self.degree())

    def laplacian(self, normalized: bool = False) -> np.ndarray:
        """Compute graph Laplacian.

        Args:
            normalized: If True, compute normalized Laplacian L_n = I - D^{-1/2} A D^{-1/2}

        Returns:
            (n, n) Laplacian matrix
        """
        d = self.degree()
        if normalized:
            d_inv_sqrt = 1.0 / np.sqrt(d + 1e-12)
            D_inv_sqrt = np.diag(d_inv_sqrt)
            return np.eye(self.n_nodes) - D_inv_sqrt @ self.adjacency @ D_inv_sqrt
        else:
            return self.degree_matrix() - self.adjacency

    def laplacian_spectrum(self) -> np.ndarray:
        """Compute eigenvalues of the Laplacian (sorted ascending)."""
        L = self.laplacian()
        evals = np.linalg.eigvalsh(L)
        return np.sort(np.maximum(evals, 0.0))

    def edges(self, eps: float = 1e-12) -> Iterator[Tuple[int, int, float]]:
        """Iterate over edges as (i, j, weight) tuples.

        Only yields edges with i < j (upper triangle) to avoid duplicates.
        Only yields edges with weight > eps.

        Args:
            eps: Minimum weight threshold

        Yields:
            (i, j, weight) tuples for each edge
        """
        n = self.n_nodes
        for i in range(n):
            for j in range(i + 1, n):
                w = self.adjacency[i, j]
                if abs(w) > eps:
                    yield (i, j, float(w))

    def edge_list(self, eps: float = 1e-12) -> List[Tuple[int, int, float]]:
        """Get all edges as a list."""
        return list(self.edges(eps))

    def n_edges(self, eps: float = 1e-12) -> int:
        """Count number of edges above threshold."""
        return len(self.edge_list(eps))

    def mean_degree(self) -> float:
        """Compute mean degree."""
        return float(np.mean(self.degree()))

    def density(self) -> float:
        """Compute edge density (fraction of possible edges present)."""
        n = self.n_nodes
        max_edges = n * (n - 1) / 2
        return self.n_edges() / max_edges if max_edges > 0 else 0.0

    def algebraic_connectivity(self) -> float:
        """Second smallest Laplacian eigenvalue (Fiedler value)."""
        spec = self.laplacian_spectrum()
        return float(spec[1]) if len(spec) > 1 else 0.0

    def spectral_gap(self) -> float:
        """Gap between first and second Laplacian eigenvalue."""
        spec = self.laplacian_spectrum()
        return float(spec[1] - spec[0]) if len(spec) > 1 else 0.0

    def clustering_coefficient(self) -> float:
        """Compute global clustering coefficient.
        
        C = 3T / τ where T = number of triangles, τ = connected triples.
        """
        A = (self.adjacency > 0).astype(float)
        A2 = A @ A
        triangles = np.trace(A @ A @ A) / 6.0
        triplets = np.sum(A2 * (1 - np.eye(self.n_nodes))) / 2.0
        return float(3.0 * triangles / (triplets + 1e-12))

    def summary(self) -> Dict[str, float]:
        """Compute summary statistics for the graph."""
        spec = self.laplacian_spectrum()
        return {
            "n_nodes": float(self.n_nodes),
            "n_edges": float(self.n_edges()),
            "density": self.density(),
            "mean_degree": self.mean_degree(),
            "max_degree": float(np.max(self.degree())),
            "algebraic_connectivity": self.algebraic_connectivity(),
            "spectral_gap": self.spectral_gap(),
            "clustering_coefficient": self.clustering_coefficient(),
            "laplacian_trace": float(np.sum(spec)),
        }


def _threshold_adjacency(
    M: np.ndarray,
    tau_quantile: float,
    weighted: bool,
    alpha: float,
) -> np.ndarray:
    """Construct adjacency via thresholding."""
    M_abs = np.abs(M)
    np.fill_diagonal(M_abs, 0.0)

    # Compute threshold from quantile of off-diagonal elements
    offdiag = M_abs[np.triu_indices_from(M_abs, k=1)]
    tau = np.quantile(offdiag, tau_quantile) if len(offdiag) > 0 else 0.0

    if weighted:
        A = np.where(M_abs > tau, np.power(M_abs, alpha), 0.0)
    else:
        A = (M_abs > tau).astype(float)

    return A


def _weighted_adjacency(M: np.ndarray, alpha: float) -> np.ndarray:
    """Construct weighted adjacency directly from matrix."""
    M_abs = np.abs(M)
    np.fill_diagonal(M_abs, 0.0)
    return np.power(M_abs, alpha)


def _laplacian_adjacency(
    M: np.ndarray,
    metric: str,
    beta: float,
) -> np.ndarray:
    """Construct adjacency via kernel on matrix-derived distances."""
    n = M.shape[0]
    A = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            if metric == "abs":
                # Distance based on difference of rows
                d = np.sum(np.abs(M[i, :] - M[j, :]))
            else:  # "fro"
                d = np.linalg.norm(M[i, :] - M[j, :])

            w = np.exp(-d / (beta + 1e-12))
            A[i, j] = A[j, i] = w

    return A


def _sparsify(
    A: np.ndarray,
    method: str,
    topk: int,
    q: float,
) -> np.ndarray:
    """Apply sparsification to adjacency matrix."""
    if method == "none":
        return A

    n = A.shape[0]
    A_sparse = np.zeros_like(A)

    if method == "topk":
        # Keep top-k edges per node
        for i in range(n):
            row = A[i, :].copy()
            row[i] = -np.inf  # Exclude self
            top_indices = np.argsort(row)[-topk:]
            for j in top_indices:
                if A[i, j] > 0:
                    A_sparse[i, j] = A[i, j]
                    A_sparse[j, i] = A[j, i]

    elif method == "quantile":
        # Keep edges above q-quantile
        offdiag = A[np.triu_indices_from(A, k=1)]
        threshold = np.quantile(offdiag, 1 - q) if len(offdiag) > 0 else 0.0
        A_sparse = np.where(A > threshold, A, 0.0)

    return A_sparse


def matrix_to_graph(
    M: np.ndarray,
    mode: str = "weighted",
    params: Optional[Dict] = None,
) -> Graph:
    """Convert a symmetric matrix to a Graph.

    This is the canonical lift from matrix model to graph topology,
    inserted into the experiment spine after sampling and before
    correspondence/kernel computation.

    Args:
        M: Symmetric matrix (will be symmetrized if not)
        mode: Conversion mode ('threshold', 'weighted', 'laplacian')
        params: Dictionary of parameters (see GraphParams for keys)

    Returns:
        Graph object with adjacency matrix and metadata
    """
    # Ensure symmetry
    M = (M + M.T) / 2.0

    # Parse parameters
    p = GraphParams()
    if params:
        p = GraphParams(
            mode=params.get("mode", mode),
            tau_quantile=params.get("tau_quantile", p.tau_quantile),
            weighted=params.get("weighted", p.weighted),
            alpha=params.get("alpha", p.alpha),
            sparsify=params.get("sparsify", p.sparsify),
            topk=params.get("topk", p.topk),
            q=params.get("q", p.q),
            metric=params.get("metric", p.metric),
            beta=params.get("beta", p.beta),
            sigma=params.get("sigma", p.sigma),
            normalize=params.get("normalize", p.normalize),
        )
        mode = p.mode

    # Construct adjacency based on mode
    if mode == "threshold":
        A = _threshold_adjacency(M, p.tau_quantile, p.weighted, p.alpha)
    elif mode == "weighted":
        A = _weighted_adjacency(M, p.alpha)
    elif mode == "laplacian":
        A = _laplacian_adjacency(M, p.metric, p.beta)
    else:
        raise ValueError(f"Unknown graph mode: {mode}")

    # Apply sparsification
    A = _sparsify(A, p.sparsify, p.topk, p.q)

    # Add noise regularization if specified
    if p.sigma > 0:
        noise = np.random.randn(*A.shape) * p.sigma
        noise = (noise + noise.T) / 2.0
        np.fill_diagonal(noise, 0.0)
        A = np.maximum(A + noise, 0.0)

    # Normalize if requested
    if p.normalize and np.max(A) > 0:
        A = A / np.max(A)

    metadata = {
        "mode": mode,
        "params": {
            "tau_quantile": p.tau_quantile,
            "weighted": p.weighted,
            "alpha": p.alpha,
            "sparsify": p.sparsify,
            "topk": p.topk,
            "q": p.q,
            "metric": p.metric,
            "beta": p.beta,
            "sigma": p.sigma,
            "normalize": p.normalize,
        },
        "source": "matrix_to_graph",
    }

    return Graph(adjacency=A, metadata=metadata)
