"""Tests for the topology module.

Tests cover:
- Graph dataclass (construction, methods, properties)
- matrix_to_graph function (all modes)
- Graph family samplers
"""

import numpy as np
import pytest

from topology import Graph, matrix_to_graph, GraphParams, GraphFamily, sample_graph_family


class TestGraph:
    """Tests for the Graph dataclass."""

    def test_graph_construction(self):
        """Test basic graph construction from adjacency matrix."""
        A = np.array([
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0],
        ], dtype=float)
        G = Graph(adjacency=A)

        assert G.n_nodes == 3
        assert G.adjacency.shape == (3, 3)

    def test_graph_frozen(self):
        """Test that Graph is immutable (frozen dataclass)."""
        A = np.eye(3)
        G = Graph(adjacency=A)

        with pytest.raises(Exception):  # FrozenInstanceError
            G.n_nodes = 5

    def test_degree(self):
        """Test degree computation."""
        # Path graph: 0 -- 1 -- 2
        A = np.array([
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0],
        ], dtype=float)
        G = Graph(adjacency=A)
        deg = G.degree()

        assert deg[0] == 1  # Node 0 has degree 1
        assert deg[1] == 2  # Node 1 has degree 2
        assert deg[2] == 1  # Node 2 has degree 1

    def test_degree_weighted(self):
        """Test degree computation with weighted edges."""
        A = np.array([
            [0, 0.5, 0],
            [0.5, 0, 1.5],
            [0, 1.5, 0],
        ], dtype=float)
        G = Graph(adjacency=A)
        deg = G.degree()

        assert np.isclose(deg[0], 0.5)
        assert np.isclose(deg[1], 2.0)
        assert np.isclose(deg[2], 1.5)

    def test_laplacian_unnormalized(self):
        """Test unnormalized Laplacian computation."""
        # Complete graph K3
        A = np.array([
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
        ], dtype=float)
        G = Graph(adjacency=A)
        L = G.laplacian(normalized=False)

        # Laplacian of K3: diagonal = 2, off-diagonal = -1
        expected = np.array([
            [2, -1, -1],
            [-1, 2, -1],
            [-1, -1, 2],
        ], dtype=float)
        assert np.allclose(L, expected)

    def test_laplacian_normalized(self):
        """Test normalized Laplacian computation."""
        A = np.array([
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
        ], dtype=float)
        G = Graph(adjacency=A)
        L_norm = G.laplacian(normalized=True)

        # Normalized Laplacian: eigenvalues in [0, 2]
        evals = np.linalg.eigvalsh(L_norm)
        assert np.all(evals >= -1e-10)
        assert np.all(evals <= 2 + 1e-10)

    def test_laplacian_spectrum(self):
        """Test Laplacian eigenvalue computation."""
        # Path graph
        A = np.array([
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0],
        ], dtype=float)
        G = Graph(adjacency=A)
        spec = G.laplacian_spectrum()

        # Smallest eigenvalue should be 0 (connected graph)
        assert len(spec) == 3
        assert np.isclose(spec[0], 0, atol=1e-10)
        # Eigenvalues should be sorted ascending
        assert np.all(np.diff(spec) >= -1e-10)

    def test_edges_iterator(self):
        """Test edge iteration."""
        A = np.array([
            [0, 0.5, 0],
            [0.5, 0, 0.8],
            [0, 0.8, 0],
        ], dtype=float)
        G = Graph(adjacency=A)
        edges = list(G.edges())

        assert len(edges) == 2
        # Should be (i, j, weight) with i < j
        assert (0, 1, 0.5) in edges
        assert (1, 2, 0.8) in edges

    def test_edges_threshold(self):
        """Test edge iteration with threshold."""
        A = np.array([
            [0, 0.01, 0.5],
            [0.01, 0, 0.8],
            [0.5, 0.8, 0],
        ], dtype=float)
        G = Graph(adjacency=A)

        # With default eps, all edges
        edges_all = list(G.edges(eps=1e-12))
        assert len(edges_all) == 3

        # With higher eps, filter out small edge
        edges_filtered = list(G.edges(eps=0.1))
        assert len(edges_filtered) == 2

    def test_n_edges(self):
        """Test edge count."""
        A = np.array([
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
        ], dtype=float)
        G = Graph(adjacency=A)
        assert G.n_edges() == 3  # K3 has 3 edges

    def test_density(self):
        """Test edge density computation."""
        # Complete graph K3: density = 1.0
        A_complete = np.array([
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
        ], dtype=float)
        G_complete = Graph(adjacency=A_complete)
        assert np.isclose(G_complete.density(), 1.0)

        # Path graph: density = 2/3
        A_path = np.array([
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0],
        ], dtype=float)
        G_path = Graph(adjacency=A_path)
        assert np.isclose(G_path.density(), 2/3)

    def test_algebraic_connectivity(self):
        """Test algebraic connectivity (Fiedler value)."""
        # Disconnected graph: algebraic connectivity = 0
        A_disconnected = np.array([
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
        ], dtype=float)
        G_disconnected = Graph(adjacency=A_disconnected)
        assert np.isclose(G_disconnected.algebraic_connectivity(), 0, atol=1e-10)

        # Complete graph: high connectivity
        A_complete = np.ones((4, 4)) - np.eye(4)
        G_complete = Graph(adjacency=A_complete)
        assert G_complete.algebraic_connectivity() > 0

    def test_spectral_gap(self):
        """Test spectral gap computation."""
        A = np.array([
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
        ], dtype=float)
        G = Graph(adjacency=A)

        # For connected graph, spectral gap = algebraic connectivity
        gap = G.spectral_gap()
        conn = G.algebraic_connectivity()
        assert np.isclose(gap, conn)

    def test_clustering_coefficient(self):
        """Test clustering coefficient."""
        # Complete graph K3: all triangles, clustering = 1
        A_complete = np.array([
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
        ], dtype=float)
        G_complete = Graph(adjacency=A_complete)
        cc = G_complete.clustering_coefficient()
        assert np.isclose(cc, 1.0, atol=0.1)  # Should be close to 1

        # Star graph: no triangles, clustering ~ 0
        A_star = np.array([
            [0, 1, 1, 1],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
        ], dtype=float)
        G_star = Graph(adjacency=A_star)
        cc_star = G_star.clustering_coefficient()
        assert cc_star < 0.1

    def test_summary(self):
        """Test summary statistics."""
        A = np.array([
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
        ], dtype=float)
        G = Graph(adjacency=A)
        summary = G.summary()

        required_keys = [
            "n_nodes", "n_edges", "density", "mean_degree",
            "max_degree", "algebraic_connectivity", "spectral_gap",
            "clustering_coefficient", "laplacian_trace"
        ]
        for key in required_keys:
            assert key in summary
            assert isinstance(summary[key], float)


class TestMatrixToGraph:
    """Tests for matrix_to_graph function."""

    def test_weighted_mode(self):
        """Test weighted adjacency mode."""
        M = np.array([
            [1, 0.5, -0.3],
            [0.5, 2, 0.8],
            [-0.3, 0.8, 1],
        ], dtype=float)
        G = matrix_to_graph(M, mode="weighted")

        # Adjacency should use |M_ij|
        assert G.adjacency[0, 1] > 0
        assert G.adjacency[0, 2] > 0
        # Diagonal should be zero
        assert np.allclose(np.diag(G.adjacency), 0)

    def test_threshold_mode(self):
        """Test threshold adjacency mode."""
        M = np.array([
            [1, 0.9, 0.1],
            [0.9, 2, 0.8],
            [0.1, 0.8, 1],
        ], dtype=float)
        params = {"tau_quantile": 0.5, "weighted": False}
        G = matrix_to_graph(M, mode="threshold", params=params)

        # With tau_quantile=0.5, about half of edges should be present
        # Binary mode: edges are 0 or 1
        unique_vals = np.unique(G.adjacency)
        assert set(unique_vals).issubset({0.0, 1.0})

    def test_laplacian_mode(self):
        """Test laplacian-based adjacency mode."""
        M = np.random.default_rng(42).normal(size=(5, 5))
        M = (M + M.T) / 2
        G = matrix_to_graph(M, mode="laplacian")

        # Should produce valid adjacency
        assert G.n_nodes == 5
        assert np.allclose(G.adjacency, G.adjacency.T)
        assert np.allclose(np.diag(G.adjacency), 0)

    def test_symmetrization(self):
        """Test that non-symmetric input is symmetrized."""
        M = np.array([
            [1, 0.5, 0.3],
            [0.2, 2, 0.8],
            [0.1, 0.6, 1],
        ], dtype=float)
        G = matrix_to_graph(M, mode="weighted")

        # Adjacency should be symmetric
        assert np.allclose(G.adjacency, G.adjacency.T)

    def test_sparsification_topk(self):
        """Test top-k sparsification."""
        M = np.random.default_rng(42).normal(size=(8, 8))
        M = (M + M.T) / 2
        params = {"sparsify": "topk", "topk": 2}
        G = matrix_to_graph(M, mode="weighted", params=params)

        # Each node should have at most ~2*topk edges (due to symmetry)
        degrees = np.sum(G.adjacency > 0, axis=1)
        assert np.all(degrees <= 2 * params["topk"] + 1)

    def test_normalization(self):
        """Test adjacency normalization."""
        M = np.random.default_rng(42).normal(size=(5, 5)) * 10
        M = (M + M.T) / 2
        params = {"normalize": True}
        G = matrix_to_graph(M, mode="weighted", params=params)

        # Max edge weight should be 1 (or 0 if no edges)
        if G.n_edges() > 0:
            assert np.isclose(np.max(G.adjacency), 1.0)

    def test_metadata(self):
        """Test that metadata is stored correctly."""
        M = np.random.default_rng(42).normal(size=(5, 5))
        M = (M + M.T) / 2
        G = matrix_to_graph(M, mode="weighted")

        assert "mode" in G.metadata
        assert G.metadata["mode"] == "weighted"
        assert "source" in G.metadata
        assert G.metadata["source"] == "matrix_to_graph"


class TestGraphFamilies:
    """Tests for graph family samplers."""

    @pytest.mark.parametrize("family", list(GraphFamily))
    def test_family_returns_valid_graph(self, family):
        """Test that all families return valid Graph objects."""
        _, _, G = sample_graph_family(family, n=10, seed=42)

        assert isinstance(G, Graph)
        assert G.n_nodes == 10
        assert np.allclose(G.adjacency, G.adjacency.T)
        assert np.allclose(np.diag(G.adjacency), 0)

    def test_erdos_renyi_density(self):
        """Test Erdos-Renyi edge probability."""
        n = 50
        p = 0.3
        _, _, G = sample_graph_family(
            GraphFamily.ERDOS_RENYI, n=n, params={"p": p}, seed=42
        )

        # Density should be roughly p (with some variance)
        expected_edges = p * n * (n - 1) / 2
        actual_edges = G.n_edges()
        assert abs(actual_edges - expected_edges) < 0.3 * expected_edges

    def test_complete_graph(self):
        """Test complete graph has all edges."""
        n = 10
        _, _, G = sample_graph_family(GraphFamily.COMPLETE, n=n, seed=42)

        expected_edges = n * (n - 1) // 2
        assert G.n_edges() == expected_edges
        assert np.isclose(G.density(), 1.0)

    def test_ring_lattice_regularity(self):
        """Test ring lattice has regular structure."""
        n = 20
        k = 3
        _, _, G = sample_graph_family(
            GraphFamily.RING_LATTICE, n=n, params={"k": k}, seed=42
        )

        # Each node should have 2*k neighbors (k on each side)
        degrees = np.sum(G.adjacency > 0, axis=1)
        assert np.all(degrees == 2 * k)

    def test_barabasi_albert_scale_free(self):
        """Test Barabasi-Albert creates scale-free graph."""
        n = 50
        m = 2
        _, _, G = sample_graph_family(
            GraphFamily.BARABASI_ALBERT, n=n, params={"m": m}, seed=42
        )

        # Graph should be connected
        assert G.algebraic_connectivity() > 0

        # Should have roughly n*m edges
        expected_edges = (n - m - 1) * m + m * (m + 1) // 2
        assert abs(G.n_edges() - expected_edges) < 0.3 * expected_edges

    def test_watts_strogatz_small_world(self):
        """Test Watts-Strogatz has small-world properties."""
        n = 30
        k = 4
        p = 0.3
        _, _, G = sample_graph_family(
            GraphFamily.WATTS_STROGATZ, n=n, params={"k": k, "p": p}, seed=42
        )

        # Should maintain roughly same number of edges as ring
        ring_edges = n * k // 2
        assert abs(G.n_edges() - ring_edges) < 0.3 * ring_edges

        # Should have non-zero clustering
        assert G.clustering_coefficient() > 0

    def test_block_model_structure(self):
        """Test block model has community structure."""
        n = 20
        n_blocks = 2
        p_intra = 0.8
        p_inter = 0.1
        _, _, G = sample_graph_family(
            GraphFamily.BLOCK_MODEL, n=n,
            params={"n_blocks": n_blocks, "p_intra": p_intra, "p_inter": p_inter},
            seed=42
        )

        # Should have more intra-block edges than inter-block
        block_size = n // n_blocks
        intra_edges = 0
        inter_edges = 0
        for (i, j, w) in G.edges():
            if i // block_size == j // block_size:
                intra_edges += 1
            else:
                inter_edges += 1

        # Ratio should reflect p_intra/p_inter roughly
        if inter_edges > 0:
            ratio = intra_edges / inter_edges
            assert ratio > 1  # More intra than inter

    def test_reproducibility(self):
        """Test that same seed gives same graph."""
        _, _, G1 = sample_graph_family(GraphFamily.ERDOS_RENYI, n=10, seed=42)
        _, _, G2 = sample_graph_family(GraphFamily.ERDOS_RENYI, n=10, seed=42)

        assert np.allclose(G1.adjacency, G2.adjacency)

    def test_different_seeds_different_graphs(self):
        """Test that different seeds give different graphs."""
        _, _, G1 = sample_graph_family(GraphFamily.ERDOS_RENYI, n=10, seed=42)
        _, _, G2 = sample_graph_family(GraphFamily.ERDOS_RENYI, n=10, seed=43)

        assert not np.allclose(G1.adjacency, G2.adjacency)

    def test_sample_returns_params(self):
        """Test that sample returns family and params used."""
        family, params, G = sample_graph_family(
            GraphFamily.ERDOS_RENYI, n=10, params={"p": 0.5}, seed=42
        )

        assert family == GraphFamily.ERDOS_RENYI
        assert "p" in params
        assert params["p"] == 0.5

    def test_metadata_stored(self):
        """Test that graph metadata includes family info."""
        _, _, G = sample_graph_family(GraphFamily.WATTS_STROGATZ, n=10, seed=42)

        assert "family" in G.metadata
        assert G.metadata["family"] == "watts_strogatz"
        assert "source" in G.metadata
        assert G.metadata["source"] == "sample_graph_family"
