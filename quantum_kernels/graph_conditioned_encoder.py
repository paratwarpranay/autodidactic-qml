"""Graph-conditioned quantum encoder.

This module implements quantum feature maps where the entangling structure
is determined by an explicit graph topology, rather than all-to-all.

The key idea: edges in the graph G determine which qubits get entangled,
and edge weights determine entangling angles. This makes the quantum
circuit's structure match the "architecture" inferred from the matrix model.

Scientific motivation: If the graph represents the "interaction structure"
of the underlying law, then quantum circuits respecting this structure
may better capture the relevant correlations for learning.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable, Iterator

from .statevector_sim import Circuit, fidelity, H, RY, RZ


def weight_to_angle(w: float, scale: float = np.pi) -> float:
    """Convert edge weight to rotation angle.

    Uses tanh to bound angles to [-scale, scale].
    """
    return scale * np.tanh(w)


@dataclass
class GraphConditionedAnsatz:
    """Variational ansatz with entanglement structure from graph topology.

    Instead of all-to-all entanglement, we only entangle qubits that
    are connected by edges in the input graph.

    Structure per layer:
    - RY on each qubit (from parameters)
    - RZ on each qubit (from parameters)
    - For each edge (i,j,w) in graph: apply CZ or CNOT with weight-derived angle
    """
    n_qubits: int
    n_layers: int = 2
    entangler_type: str = "crz"  # "cnot", "crz", "cz"

    def n_params_per_layer(self) -> int:
        """Parameters per layer (just single-qubit rotations)."""
        return 2 * self.n_qubits

    def n_params(self) -> int:
        return self.n_layers * self.n_params_per_layer()

    def ops(
        self,
        params: np.ndarray,
        edges: List[Tuple[int, int, float]],
    ) -> List[Tuple[str, Tuple]]:
        """Generate circuit operations conditioned on graph edges.

        Args:
            params: Variational parameters for single-qubit gates
            edges: List of (i, j, weight) from graph

        Returns:
            List of (op_name, args) tuples
        """
        params = np.asarray(params, dtype=float).reshape(self.n_layers, self.n_qubits, 2)
        ops: List[Tuple[str, Tuple]] = []

        for layer in range(self.n_layers):
            # Single-qubit rotations
            for q in range(self.n_qubits):
                ops.append(("RY", (q, float(params[layer, q, 0]))))
                ops.append(("RZ", (q, float(params[layer, q, 1]))))

            # Graph-conditioned entanglers
            for (i, j, w) in edges:
                # Map to qubit indices (mod n_qubits to handle larger graphs)
                qi = i % self.n_qubits
                qj = j % self.n_qubits

                if qi == qj:
                    continue

                theta = weight_to_angle(w)

                if self.entangler_type == "cnot":
                    ops.append(("CNOT", (qi, qj)))
                elif self.entangler_type == "crz":
                    # Controlled-RZ via CNOT sandwich
                    ops.append(("RZ", (qj, theta / 2)))
                    ops.append(("CNOT", (qi, qj)))
                    ops.append(("RZ", (qj, -theta / 2)))
                    ops.append(("CNOT", (qi, qj)))
                else:  # "cz" approximation
                    ops.append(("RZ", (qi, theta / 2)))
                    ops.append(("RZ", (qj, theta / 2)))
                    ops.append(("CNOT", (qi, qj)))
                    ops.append(("RZ", (qj, -theta / 2)))
                    ops.append(("CNOT", (qi, qj)))

        return ops

    def state(
        self,
        params: np.ndarray,
        edges: List[Tuple[int, int, float]],
    ) -> np.ndarray:
        """Execute circuit and return final state."""
        c = Circuit(self.n_qubits)
        return c.run(self.ops(params, edges))


@dataclass
class GraphConditionedEncoder:
    """Encode data vectors using graph-conditioned variational circuit.

    The encoding uses data reuploading: params(x) = W*x + b
    The entangling structure comes from the graph edges.

    This is the "lift the matrix to explicit entanglement graph" made operational.
    """
    n_qubits: int
    n_layers: int = 2
    entangler_type: str = "crz"
    edge_eps: float = 0.01
    seed: int = 0

    def __post_init__(self):
        self.ansatz = GraphConditionedAnsatz(
            self.n_qubits,
            self.n_layers,
            self.entangler_type,
        )
        rng = np.random.default_rng(self.seed)
        n_params = self.ansatz.n_params()
        self.W = rng.normal(scale=0.2, size=(n_params,))
        self.b = rng.normal(scale=0.1, size=(n_params,))
        self._edges: List[Tuple[int, int, float]] = []

    def set_graph(self, graph) -> None:
        """Set the graph topology for entanglement structure.

        Args:
            graph: Graph object with edges() method
        """
        if hasattr(graph, 'edges'):
            self._edges = list(graph.edges(eps=self.edge_eps))
        elif hasattr(graph, 'edge_list'):
            self._edges = graph.edge_list(eps=self.edge_eps)
        else:
            # Fallback: treat as adjacency matrix
            self._edges = self._edges_from_adjacency(graph)

    def _edges_from_adjacency(self, A: np.ndarray) -> List[Tuple[int, int, float]]:
        """Extract edges from adjacency matrix."""
        edges = []
        n = A.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                if abs(A[i, j]) > self.edge_eps:
                    edges.append((i, j, float(A[i, j])))
        return edges

    def encode(self, x: np.ndarray, graph=None) -> np.ndarray:
        """Encode a data vector into a quantum state.

        Args:
            x: Input vector
            graph: Optional graph (if not set via set_graph)

        Returns:
            Quantum state vector
        """
        if graph is not None:
            self.set_graph(graph)

        x = np.asarray(x, dtype=float).reshape(-1)
        n_params = self.ansatz.n_params()

        # Data reuploading: expand x to match param count
        v = np.zeros(n_params)
        for i in range(n_params):
            v[i] = x[i % x.size] if x.size > 0 else 0.0

        params = self.W * v + self.b
        return self.ansatz.state(params, self._edges)

    def kernel(self, x1: np.ndarray, x2: np.ndarray, graph=None) -> float:
        """Compute quantum kernel between two data points."""
        if graph is not None:
            self.set_graph(graph)
        return fidelity(self.encode(x1), self.encode(x2))


@dataclass
class GraphConditionedKernel:
    """Quantum kernel using graph-conditioned encoder.

    This is the main interface for computing kernel matrices with
    graph-defined entanglement structure.
    """
    n_qubits: int = 4
    n_layers: int = 2
    entangler_type: str = "crz"
    edge_eps: float = 0.01
    seed: int = 0

    def __post_init__(self):
        self.encoder = GraphConditionedEncoder(
            n_qubits=self.n_qubits,
            n_layers=self.n_layers,
            entangler_type=self.entangler_type,
            edge_eps=self.edge_eps,
            seed=self.seed,
        )

    def set_graph(self, graph) -> None:
        """Set graph topology for entanglement."""
        self.encoder.set_graph(graph)

    def gram(self, X: np.ndarray, graph=None) -> np.ndarray:
        """Compute Gram matrix for data X.

        Args:
            X: (n, d) data matrix
            graph: Graph object defining entanglement

        Returns:
            (n, n) kernel matrix
        """
        if graph is not None:
            self.set_graph(graph)

        X = np.asarray(X, dtype=float)
        n = X.shape[0]

        # Encode all data points
        states = [self.encoder.encode(X[i]) for i in range(n)]

        # Compute kernel matrix
        K = np.zeros((n, n))
        for i in range(n):
            K[i, i] = 1.0
            for j in range(i + 1, n):
                fij = fidelity(states[i], states[j])
                K[i, j] = K[j, i] = fij

        return K
