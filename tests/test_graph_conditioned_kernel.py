"""Tests for graph-conditioned quantum kernel.

Tests cover:
- GraphConditionedAnsatz circuit generation
- GraphConditionedEncoder state encoding
- GraphConditionedKernel Gram matrix computation
- Comparison with standard quantum kernel
"""

import numpy as np
import pytest

from quantum_kernels import (
    GraphConditionedAnsatz,
    GraphConditionedEncoder,
    GraphConditionedKernel,
    weight_to_angle,
    QuantumKernel,
    VariationalLawEncoder,
)
from topology import Graph, sample_graph_family, GraphFamily


class TestWeightToAngle:
    """Tests for weight_to_angle conversion."""

    def test_zero_weight(self):
        """Test that zero weight gives zero angle."""
        angle = weight_to_angle(0.0)
        assert np.isclose(angle, 0.0)

    def test_positive_weight(self):
        """Test positive weight gives positive angle."""
        angle = weight_to_angle(1.0)
        assert angle > 0
        assert angle <= np.pi

    def test_negative_weight(self):
        """Test negative weight gives negative angle."""
        angle = weight_to_angle(-1.0)
        assert angle < 0
        assert angle >= -np.pi

    def test_bounded(self):
        """Test angles are bounded by scale."""
        scale = np.pi
        for w in [-10, -1, 0, 1, 10]:
            angle = weight_to_angle(w, scale=scale)
            assert -scale <= angle <= scale

    def test_monotonic(self):
        """Test that angle is monotonic in weight."""
        weights = [-2, -1, 0, 1, 2]
        angles = [weight_to_angle(w) for w in weights]
        assert all(a1 < a2 for a1, a2 in zip(angles[:-1], angles[1:]))


class TestGraphConditionedAnsatz:
    """Tests for GraphConditionedAnsatz circuit."""

    def test_n_params(self):
        """Test parameter count."""
        ansatz = GraphConditionedAnsatz(n_qubits=4, n_layers=2)
        # 2 params per qubit per layer (RY, RZ)
        expected = 2 * 4 * 2
        assert ansatz.n_params() == expected

    def test_ops_generation(self):
        """Test circuit operations generation."""
        ansatz = GraphConditionedAnsatz(n_qubits=3, n_layers=1)
        params = np.zeros(ansatz.n_params())
        edges = [(0, 1, 0.5), (1, 2, 0.8)]

        ops = ansatz.ops(params, edges)

        # Should have RY, RZ for each qubit, plus entanglers for edges
        assert len(ops) > 0
        op_names = [op[0] for op in ops]
        assert "RY" in op_names
        assert "RZ" in op_names

    def test_entangler_types(self):
        """Test different entangler types."""
        for ent_type in ["cnot", "crz", "cz"]:
            ansatz = GraphConditionedAnsatz(
                n_qubits=3, n_layers=1, entangler_type=ent_type
            )
            params = np.zeros(ansatz.n_params())
            edges = [(0, 1, 0.5)]

            ops = ansatz.ops(params, edges)
            assert len(ops) > 0

    def test_state_output_shape(self):
        """Test state vector output shape."""
        ansatz = GraphConditionedAnsatz(n_qubits=4, n_layers=2)
        params = np.random.default_rng(42).normal(size=ansatz.n_params())
        edges = [(0, 1, 0.5), (1, 2, 0.3), (2, 3, 0.7)]

        state = ansatz.state(params, edges)

        assert state.shape == (2**4,)
        assert np.isclose(np.linalg.norm(state), 1.0)

    def test_no_edges_runs(self):
        """Test ansatz works with empty edge list."""
        ansatz = GraphConditionedAnsatz(n_qubits=3, n_layers=1)
        params = np.zeros(ansatz.n_params())
        edges = []

        state = ansatz.state(params, edges)
        assert state.shape == (2**3,)

    def test_self_loop_ignored(self):
        """Test that self-loops (i==j after mod) are ignored."""
        ansatz = GraphConditionedAnsatz(n_qubits=3, n_layers=1)
        params = np.zeros(ansatz.n_params())
        # Edge that maps to same qubit
        edges = [(0, 3, 0.5)]  # 3 % 3 == 0

        # Should not raise
        ops = ansatz.ops(params, edges)
        assert len(ops) > 0


class TestGraphConditionedEncoder:
    """Tests for GraphConditionedEncoder."""

    def test_encode_shape(self):
        """Test encoded state shape."""
        encoder = GraphConditionedEncoder(n_qubits=4, n_layers=2, seed=42)
        _, _, G = sample_graph_family(GraphFamily.ERDOS_RENYI, n=8, seed=42)
        encoder.set_graph(G)

        x = np.random.default_rng(42).normal(size=8)
        state = encoder.encode(x)

        assert state.shape == (2**4,)
        assert np.isclose(np.linalg.norm(state), 1.0)

    def test_set_graph_from_graph_object(self):
        """Test setting graph from Graph object."""
        encoder = GraphConditionedEncoder(n_qubits=4, n_layers=2, seed=42)
        _, _, G = sample_graph_family(GraphFamily.COMPLETE, n=6, seed=42)

        encoder.set_graph(G)
        assert len(encoder._edges) > 0

    def test_set_graph_from_adjacency(self):
        """Test setting graph from adjacency matrix."""
        encoder = GraphConditionedEncoder(n_qubits=4, n_layers=2, seed=42)
        A = np.array([
            [0, 0.5, 0],
            [0.5, 0, 0.8],
            [0, 0.8, 0],
        ])

        encoder.set_graph(A)
        assert len(encoder._edges) == 2

    def test_kernel_computation(self):
        """Test kernel value between two points."""
        encoder = GraphConditionedEncoder(n_qubits=4, n_layers=2, seed=42)
        _, _, G = sample_graph_family(GraphFamily.ERDOS_RENYI, n=8, seed=42)
        encoder.set_graph(G)

        x1 = np.random.default_rng(42).normal(size=8)
        x2 = np.random.default_rng(43).normal(size=8)

        k = encoder.kernel(x1, x2)

        assert 0 <= k <= 1
        # Same point should give kernel = 1
        k_same = encoder.kernel(x1, x1)
        assert np.isclose(k_same, 1.0)

    def test_different_graphs_different_encoding(self):
        """Test that different graphs produce different encodings."""
        encoder1 = GraphConditionedEncoder(n_qubits=4, n_layers=2, seed=42)
        encoder2 = GraphConditionedEncoder(n_qubits=4, n_layers=2, seed=42)

        _, _, G1 = sample_graph_family(GraphFamily.COMPLETE, n=6, seed=42)
        _, _, G2 = sample_graph_family(GraphFamily.RING_LATTICE, n=6, seed=42)

        x = np.random.default_rng(42).normal(size=6)

        encoder1.set_graph(G1)
        encoder2.set_graph(G2)

        state1 = encoder1.encode(x)
        state2 = encoder2.encode(x)

        # Different graphs should give different states
        assert not np.allclose(state1, state2)


class TestGraphConditionedKernel:
    """Tests for GraphConditionedKernel."""

    def test_gram_shape(self):
        """Test Gram matrix shape."""
        kernel = GraphConditionedKernel(n_qubits=4, n_layers=2, seed=42)
        _, _, G = sample_graph_family(GraphFamily.ERDOS_RENYI, n=8, seed=42)
        kernel.set_graph(G)

        X = np.random.default_rng(42).normal(size=(10, 8))
        K = kernel.gram(X)

        assert K.shape == (10, 10)

    def test_gram_symmetric(self):
        """Test Gram matrix is symmetric."""
        kernel = GraphConditionedKernel(n_qubits=4, n_layers=2, seed=42)
        _, _, G = sample_graph_family(GraphFamily.ERDOS_RENYI, n=8, seed=42)
        kernel.set_graph(G)

        X = np.random.default_rng(42).normal(size=(10, 8))
        K = kernel.gram(X)

        assert np.allclose(K, K.T)

    def test_gram_diagonal_ones(self):
        """Test Gram matrix has ones on diagonal."""
        kernel = GraphConditionedKernel(n_qubits=4, n_layers=2, seed=42)
        _, _, G = sample_graph_family(GraphFamily.ERDOS_RENYI, n=8, seed=42)
        kernel.set_graph(G)

        X = np.random.default_rng(42).normal(size=(10, 8))
        K = kernel.gram(X)

        assert np.allclose(np.diag(K), 1.0)

    def test_gram_positive_semidefinite(self):
        """Test Gram matrix is positive semi-definite."""
        kernel = GraphConditionedKernel(n_qubits=4, n_layers=2, seed=42)
        _, _, G = sample_graph_family(GraphFamily.ERDOS_RENYI, n=8, seed=42)
        kernel.set_graph(G)

        X = np.random.default_rng(42).normal(size=(10, 8))
        K = kernel.gram(X)

        # All eigenvalues should be non-negative
        evals = np.linalg.eigvalsh(K)
        assert np.all(evals >= -1e-10)

    def test_gram_bounded(self):
        """Test Gram matrix entries in [0, 1]."""
        kernel = GraphConditionedKernel(n_qubits=4, n_layers=2, seed=42)
        _, _, G = sample_graph_family(GraphFamily.ERDOS_RENYI, n=8, seed=42)
        kernel.set_graph(G)

        X = np.random.default_rng(42).normal(size=(10, 8))
        K = kernel.gram(X)

        assert np.all(K >= -1e-10)
        assert np.all(K <= 1 + 1e-10)

    def test_graph_inline(self):
        """Test passing graph inline to gram."""
        kernel = GraphConditionedKernel(n_qubits=4, n_layers=2, seed=42)
        _, _, G = sample_graph_family(GraphFamily.ERDOS_RENYI, n=8, seed=42)

        X = np.random.default_rng(42).normal(size=(10, 8))
        K = kernel.gram(X, graph=G)

        assert K.shape == (10, 10)

    def test_reproducibility(self):
        """Test same seed gives same kernel."""
        _, _, G = sample_graph_family(GraphFamily.ERDOS_RENYI, n=8, seed=42)
        X = np.random.default_rng(42).normal(size=(5, 8))

        kernel1 = GraphConditionedKernel(n_qubits=4, n_layers=2, seed=42)
        kernel1.set_graph(G)
        K1 = kernel1.gram(X)

        kernel2 = GraphConditionedKernel(n_qubits=4, n_layers=2, seed=42)
        kernel2.set_graph(G)
        K2 = kernel2.gram(X)

        assert np.allclose(K1, K2)


class TestGraphConditionedVsStandard:
    """Tests comparing graph-conditioned to standard quantum kernel."""

    def test_different_from_standard(self):
        """Test graph-conditioned kernel differs from standard."""
        n_qubits = 4
        n_layers = 2
        seed = 42

        _, _, G = sample_graph_family(GraphFamily.RING_LATTICE, n=8, seed=42)
        X = np.random.default_rng(42).normal(size=(5, 8))

        # Graph-conditioned kernel
        gc_kernel = GraphConditionedKernel(
            n_qubits=n_qubits, n_layers=n_layers, seed=seed
        )
        gc_kernel.set_graph(G)
        K_gc = gc_kernel.gram(X)

        # Standard kernel (all-to-all)
        std_encoder = VariationalLawEncoder(
            n_qubits=n_qubits, n_layers=n_layers, seed=seed
        )
        std_kernel = QuantumKernel(encode=std_encoder.encode)
        K_std = std_kernel.gram(X)

        # Should be different
        assert not np.allclose(K_gc, K_std)

    def test_sparse_graph_vs_dense(self):
        """Test kernel differs between sparse and dense graphs."""
        X = np.random.default_rng(42).normal(size=(5, 8))

        # Sparse graph (ring)
        _, _, G_sparse = sample_graph_family(
            GraphFamily.RING_LATTICE, n=8, params={"k": 1}, seed=42
        )
        kernel_sparse = GraphConditionedKernel(n_qubits=4, n_layers=2, seed=42)
        kernel_sparse.set_graph(G_sparse)
        K_sparse = kernel_sparse.gram(X)

        # Dense graph (complete)
        _, _, G_dense = sample_graph_family(GraphFamily.COMPLETE, n=8, seed=42)
        kernel_dense = GraphConditionedKernel(n_qubits=4, n_layers=2, seed=42)
        kernel_dense.set_graph(G_dense)
        K_dense = kernel_dense.gram(X)

        # Should be different
        assert not np.allclose(K_sparse, K_dense)

    def test_all_families_produce_valid_kernels(self):
        """Test all graph families produce valid kernels."""
        X = np.random.default_rng(42).normal(size=(5, 10))

        for family in GraphFamily:
            _, _, G = sample_graph_family(family, n=10, seed=42)
            kernel = GraphConditionedKernel(n_qubits=4, n_layers=2, seed=42)
            kernel.set_graph(G)
            K = kernel.gram(X)

            # Valid kernel matrix
            assert K.shape == (5, 5)
            assert np.allclose(K, K.T)
            assert np.allclose(np.diag(K), 1.0)
            assert np.all(np.linalg.eigvalsh(K) >= -1e-10)
