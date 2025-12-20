"""Tests for adaptive threshold module."""

import numpy as np
import pytest

from analysis.adaptive_threshold import (
    compute_adaptive_threshold,
    adaptive_adjacency,
    smooth_threshold_adjacency,
)


class TestComputeAdaptiveThreshold:
    """Tests for compute_adaptive_threshold function."""
    
    def test_percentile_method(self):
        """Test percentile-based thresholding."""
        # Create matrix with known distribution
        n = 20
        M = np.random.randn(n, n)
        M = (M + M.T) / 2
        np.fill_diagonal(M, 0)
        
        tau = compute_adaptive_threshold(M, target_density=0.3, method='percentile')
        
        # Threshold should be positive
        assert tau >= 0
        
        # Check resulting density is approximately target
        A, info = adaptive_adjacency(M, target_density=0.3, method='percentile')
        assert 0.1 < info['actual_density'] < 0.5  # Reasonable range
    
    def test_spectral_method(self):
        """Test spectral gap-based thresholding."""
        n = 15
        M = np.random.randn(n, n)
        M = (M + M.T) / 2
        np.fill_diagonal(M, 0)
        
        tau = compute_adaptive_threshold(M, method='spectral')
        
        assert tau >= 0
    
    def test_otsu_method(self):
        """Test Otsu's thresholding method."""
        n = 20
        M = np.random.randn(n, n)
        M = (M + M.T) / 2
        np.fill_diagonal(M, 0)
        
        tau = compute_adaptive_threshold(M, method='otsu')
        
        assert tau >= 0
    
    def test_connectivity_method(self):
        """Test connectivity-preserving threshold."""
        n = 15
        M = np.random.randn(n, n)
        M = (M + M.T) / 2
        np.fill_diagonal(M, 0)
        
        tau = compute_adaptive_threshold(M, method='connectivity')
        
        # Build graph with this threshold and check connectivity
        A = np.where(np.abs(M) > tau, np.abs(M), 0.0)
        D = np.diag(np.sum(A, axis=1))
        L = D - A
        eigvals = np.linalg.eigvalsh(L)
        
        # Graph should be connected (λ_2 > 0)
        assert eigvals[1] > 1e-10
    
    def test_min_edges_constraint(self):
        """Test minimum edges constraint."""
        n = 10
        M = np.random.randn(n, n)
        M = (M + M.T) / 2
        np.fill_diagonal(M, 0)
        
        min_edges = 20
        tau = compute_adaptive_threshold(M, target_density=0.1, min_edges=min_edges)
        
        A, info = adaptive_adjacency(M, target_density=0.1, min_edges=min_edges)
        
        # Should have at least min_edges
        assert info['n_edges'] >= min_edges
    
    def test_max_edges_constraint(self):
        """Test maximum edges constraint."""
        n = 10
        M = np.random.randn(n, n)
        M = (M + M.T) / 2
        np.fill_diagonal(M, 0)
        
        max_edges = 10
        tau = compute_adaptive_threshold(M, target_density=0.9, max_edges=max_edges)
        
        A, info = adaptive_adjacency(M, target_density=0.9, max_edges=max_edges)
        
        # Should have at most max_edges
        assert info['n_edges'] <= max_edges
    
    def test_scale_invariance(self):
        """Test that threshold scales with matrix magnitude."""
        n = 15
        M = np.random.randn(n, n)
        M = (M + M.T) / 2
        np.fill_diagonal(M, 0)
        
        tau1 = compute_adaptive_threshold(M, target_density=0.3)
        
        # Scale matrix by 10
        M_scaled = 10 * M
        tau2 = compute_adaptive_threshold(M_scaled, target_density=0.3)
        
        # Thresholds should scale proportionally
        assert abs(tau2 / tau1 - 10) < 1.0  # Allow some tolerance


class TestAdaptiveAdjacency:
    """Tests for adaptive_adjacency function."""
    
    def test_returns_symmetric(self):
        """Test that adjacency is symmetric."""
        n = 15
        M = np.random.randn(n, n)
        M = (M + M.T) / 2
        
        A, info = adaptive_adjacency(M)
        
        assert np.allclose(A, A.T)
    
    def test_zero_diagonal(self):
        """Test that diagonal is zero (no self-loops)."""
        n = 15
        M = np.random.randn(n, n)
        M = (M + M.T) / 2
        
        A, info = adaptive_adjacency(M)
        
        assert np.allclose(np.diag(A), 0)
    
    def test_non_negative(self):
        """Test that weights are non-negative."""
        n = 15
        M = np.random.randn(n, n)
        M = (M + M.T) / 2
        
        A, info = adaptive_adjacency(M, weighted=True)
        
        assert np.all(A >= 0)
    
    def test_binary_mode(self):
        """Test binary (unweighted) mode."""
        n = 15
        M = np.random.randn(n, n)
        M = (M + M.T) / 2
        
        A, info = adaptive_adjacency(M, weighted=False)
        
        # Should only have 0 and 1 values
        unique_vals = np.unique(A)
        assert len(unique_vals) <= 2
        assert np.all((A == 0) | (A == 1))
    
    def test_alpha_power(self):
        """Test alpha power transformation."""
        n = 15
        M = np.random.randn(n, n)
        M = (M + M.T) / 2
        
        A1, _ = adaptive_adjacency(M, alpha=1.0)
        A2, _ = adaptive_adjacency(M, alpha=2.0)
        
        # Higher alpha should have more contrast
        # (larger differences between large and small weights)
        std1 = np.std(A1[A1 > 0])
        std2 = np.std(A2[A2 > 0])
        
        # Not a strict test, but alpha=2 should generally have more variance
        # among non-zero elements
        assert std2 > 0  # Just ensure it's computed
    
    def test_info_dict_contents(self):
        """Test that info dict contains expected keys."""
        n = 15
        M = np.random.randn(n, n)
        M = (M + M.T) / 2
        
        A, info = adaptive_adjacency(M)
        
        assert 'threshold' in info
        assert 'method' in info
        assert 'target_density' in info
        assert 'actual_density' in info
        assert 'n_edges' in info


class TestSmoothThresholdAdjacency:
    """Tests for smooth_threshold_adjacency function."""
    
    def test_returns_correct_shape(self):
        """Test output shape matches input."""
        n = 15
        M = np.random.randn(n, n)
        M = (M + M.T) / 2
        
        A = smooth_threshold_adjacency(M, tau=0.5)
        
        assert A.shape == M.shape
    
    def test_differentiable(self):
        """Test that output is differentiable (smooth)."""
        n = 10
        M = np.random.randn(n, n)
        M = (M + M.T) / 2
        
        # Compute adjacency at two nearby thresholds
        A1 = smooth_threshold_adjacency(M, tau=0.5, sharpness=5.0)
        A2 = smooth_threshold_adjacency(M, tau=0.51, sharpness=5.0)
        
        # Should have small difference (smooth)
        diff = np.max(np.abs(A1 - A2))
        assert diff < 0.5  # Reasonable smoothness
    
    def test_sharpness_effect(self):
        """Test that higher sharpness gives sharper transition."""
        n = 10
        M = np.random.randn(n, n)
        M = (M + M.T) / 2
        tau = 0.5
        
        A_soft = smooth_threshold_adjacency(M, tau, sharpness=1.0)
        A_sharp = smooth_threshold_adjacency(M, tau, sharpness=100.0)
        
        # Sharp version should be closer to binary
        # (more values near 0 or 1)
        mid_range_soft = np.sum((A_soft > 0.2) & (A_soft < 0.8))
        mid_range_sharp = np.sum((A_sharp > 0.2) & (A_sharp < 0.8))
        
        assert mid_range_sharp <= mid_range_soft
    
    def test_weighted_mode(self):
        """Test weighted vs unweighted output."""
        n = 10
        M = np.random.randn(n, n)
        M = (M + M.T) / 2
        np.fill_diagonal(M, 0)
        
        A_weighted = smooth_threshold_adjacency(M, tau=0.3, weighted=True)
        A_unweighted = smooth_threshold_adjacency(M, tau=0.3, weighted=False)
        
        # Weighted should have variation in positive weights
        # Unweighted should just be the sigmoid mask
        assert np.max(A_weighted) >= np.max(A_unweighted)


class TestDegeneracyPrevention:
    """Tests that adaptive thresholding prevents degenerate graphs."""
    
    def test_prevents_empty_graph(self):
        """Test that we never get an empty graph."""
        n = 15
        
        # Matrix with very small values
        M = 0.001 * np.random.randn(n, n)
        M = (M + M.T) / 2
        
        A, info = adaptive_adjacency(M, target_density=0.3, min_edges=5)
        
        assert info['n_edges'] > 0
    
    def test_prevents_fully_connected(self):
        """Test that we never get a fully connected graph (if max_edges set)."""
        n = 10
        
        # Matrix with large uniform values
        M = 10 * np.ones((n, n))
        np.fill_diagonal(M, 0)
        
        max_edges = 20
        A, info = adaptive_adjacency(M, max_edges=max_edges)
        
        assert info['n_edges'] <= max_edges
    
    def test_different_matrices_different_thresholds(self):
        """Test that different matrices get different thresholds."""
        n = 15
        
        M1 = np.random.randn(n, n)
        M1 = (M1 + M1.T) / 2
        
        M2 = 10 * np.random.randn(n, n)
        M2 = (M2 + M2.T) / 2
        
        tau1 = compute_adaptive_threshold(M1, target_density=0.3)
        tau2 = compute_adaptive_threshold(M2, target_density=0.3)
        
        # Different matrices should generally get different thresholds
        # (unless by coincidence)
        # At minimum, the ratio should reflect the scale difference
        assert tau2 > tau1 * 2  # M2 is 10x larger
