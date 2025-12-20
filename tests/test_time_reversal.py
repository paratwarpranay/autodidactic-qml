"""Tests for time-reversal probe module."""

import numpy as np
import pytest

from analysis.time_reversal_probe import (
    TimeReversalProbe,
    TimeReversalResult,
    ReversibilityClass,
    matrix_fidelity,
    spectral_fidelity,
    topology_fidelity,
    quick_reversibility_check,
    create_reversible_dynamics_pair,
    classify_reversibility,
)


class TestMatrixFidelity:
    """Tests for matrix_fidelity function."""
    
    def test_identical_matrices(self):
        """Test fidelity of identical matrices."""
        M = np.random.randn(10, 10)
        M = (M + M.T) / 2
        
        fidelity = matrix_fidelity(M, M)
        
        assert np.isclose(fidelity, 1.0)
    
    def test_orthogonal_matrices(self):
        """Test fidelity of orthogonal matrices."""
        M1 = np.eye(10)
        M2 = np.diag([1, -1, 1, -1, 1, -1, 1, -1, 1, -1])
        
        fidelity = matrix_fidelity(M1, M2)
        
        # Should be low but not zero (trace overlap)
        assert 0 < fidelity < 1
    
    def test_scaled_matrices(self):
        """Test fidelity of scaled matrices."""
        M1 = np.random.randn(10, 10)
        M1 = (M1 + M1.T) / 2
        M2 = 2 * M1  # Same structure, different scale
        
        fidelity = matrix_fidelity(M1, M2)
        
        # Should be 1 (normalized inner product)
        assert np.isclose(fidelity, 1.0)
    
    def test_zero_matrix(self):
        """Test fidelity with zero matrix."""
        M1 = np.zeros((10, 10))
        M2 = np.random.randn(10, 10)
        
        fidelity = matrix_fidelity(M1, M2)
        
        # Zero vs non-zero should be 0
        assert np.isclose(fidelity, 0.0)


class TestSpectralFidelity:
    """Tests for spectral_fidelity function."""
    
    def test_identical_spectrum(self):
        """Test spectral fidelity of matrices with identical spectrum."""
        M = np.random.randn(10, 10)
        M = (M + M.T) / 2
        
        fidelity = spectral_fidelity(M, M)
        
        assert np.isclose(fidelity, 1.0)
    
    def test_similar_spectrum(self):
        """Test spectral fidelity of matrices with similar spectrum."""
        M1 = np.diag(np.arange(10, dtype=float))
        M2 = np.diag(np.arange(10, dtype=float) + 0.1)  # Small perturbation
        
        fidelity = spectral_fidelity(M1, M2)
        
        # Should be close to 1
        assert fidelity > 0.9


class TestTopologyFidelity:
    """Tests for topology_fidelity function."""
    
    def test_identical_topology(self):
        """Test topology fidelity of identical adjacencies."""
        A = np.random.rand(10, 10)
        A = (A + A.T) / 2
        np.fill_diagonal(A, 0)
        
        fidelity = topology_fidelity(A, A)
        
        assert np.isclose(fidelity, 1.0)
    
    def test_different_topology(self):
        """Test topology fidelity of different adjacencies."""
        A1 = np.random.rand(10, 10)
        A1 = (A1 + A1.T) / 2
        np.fill_diagonal(A1, 0)
        
        A2 = np.random.rand(10, 10)
        A2 = (A2 + A2.T) / 2
        np.fill_diagonal(A2, 0)
        
        fidelity = topology_fidelity(A1, A2)
        
        # Random matrices should have correlation near 0
        # Fidelity maps [-1, 1] to [0, 1], so should be near 0.5
        assert 0.2 < fidelity < 0.8


class TestClassifyReversibility:
    """Tests for classify_reversibility function."""
    
    def test_fully_reversible(self):
        """Test classification of fully reversible."""
        result = classify_reversibility(
            recovery_fidelity=0.99,
            trajectory_variance=0.001,
        )
        assert result == ReversibilityClass.FULLY_REVERSIBLE
    
    def test_irreversible(self):
        """Test classification of irreversible."""
        result = classify_reversibility(
            recovery_fidelity=0.3,
            trajectory_variance=0.01,
        )
        assert result == ReversibilityClass.IRREVERSIBLE
    
    def test_chaotic(self):
        """Test classification of chaotic."""
        result = classify_reversibility(
            recovery_fidelity=0.5,
            trajectory_variance=0.5,  # High variance
        )
        assert result == ReversibilityClass.CHAOTIC
    
    def test_unstable(self):
        """Test classification of unstable."""
        result = classify_reversibility(
            recovery_fidelity=0.05,  # Very low recovery
            trajectory_variance=0.01,
        )
        assert result == ReversibilityClass.UNSTABLE


class TestTimeReversalProbe:
    """Tests for TimeReversalProbe class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.n = 10
        self.M0 = np.random.randn(self.n, self.n)
        self.M0 = (self.M0 + self.M0.T) / 2
        
        self.A0 = np.random.rand(self.n, self.n)
        self.A0 = (self.A0 + self.A0.T) / 2
        np.fill_diagonal(self.A0, 0)
    
    def test_reversible_dynamics(self):
        """Test probe with perfectly reversible dynamics."""
        probe = TimeReversalProbe(verbose=False)
        
        # Simple linear dynamics (perfectly reversible)
        def forward(M, A):
            return -0.1 * M
        
        def reverse(M, A):
            return 0.1 * M  # Exact inverse
        
        result = probe.run(
            self.M0, self.A0,
            forward, reverse,
            n_steps=20,
        )
        
        # Should be fully or nearly reversible
        assert result.recovery_fidelity > 0.8
        assert result.reversibility_class in [
            ReversibilityClass.FULLY_REVERSIBLE,
            ReversibilityClass.PARTIALLY_REVERSIBLE,
        ]
    
    def test_irreversible_dynamics(self):
        """Test probe with irreversible dynamics."""
        probe = TimeReversalProbe(verbose=False)
        
        # Dynamics with noise (irreversible)
        rng = np.random.default_rng(42)
        
        def forward(M, A):
            noise = 0.1 * rng.normal(size=M.shape)
            noise = (noise + noise.T) / 2
            return -0.1 * M + noise
        
        def reverse(M, A):
            noise = 0.1 * rng.normal(size=M.shape)
            noise = (noise + noise.T) / 2
            return 0.1 * M + noise  # Can't undo the noise
        
        result = probe.run(
            self.M0, self.A0,
            forward, reverse,
            n_steps=30,
        )
        
        # Should not be fully reversible
        assert result.recovery_fidelity < 0.9
    
    def test_result_structure(self):
        """Test that result has expected structure."""
        probe = TimeReversalProbe()
        
        def forward(M, A):
            return -0.1 * M
        
        def reverse(M, A):
            return 0.1 * M
        
        result = probe.run(
            self.M0, self.A0,
            forward, reverse,
            n_steps=10,
        )
        
        assert isinstance(result, TimeReversalResult)
        assert result.forward_steps == 10
        assert result.reverse_steps == 10
        assert result.initial_state.shape == self.M0.shape
        assert 0 <= result.recovery_fidelity <= 1
        assert 0 <= result.spectral_recovery <= 1
        assert 0 <= result.topology_recovery <= 1
        assert isinstance(result.reversibility_class, ReversibilityClass)
    
    def test_trajectory_recording(self):
        """Test trajectory recording option."""
        probe = TimeReversalProbe()
        
        def forward(M, A):
            return -0.1 * M
        
        def reverse(M, A):
            return 0.1 * M
        
        result = probe.run(
            self.M0, self.A0,
            forward, reverse,
            n_steps=10,
            record_trajectory=True,
        )
        
        # Should have trajectory divergence data
        assert len(result.trajectory_divergence) > 0
    
    def test_no_trajectory_recording(self):
        """Test without trajectory recording."""
        probe = TimeReversalProbe()
        
        def forward(M, A):
            return -0.1 * M
        
        def reverse(M, A):
            return 0.1 * M
        
        result = probe.run(
            self.M0, self.A0,
            forward, reverse,
            n_steps=10,
            record_trajectory=False,
        )
        
        # Should still work but no trajectory data
        assert result.recovery_fidelity >= 0


class TestCreateReversibleDynamicsPair:
    """Tests for create_reversible_dynamics_pair function."""
    
    def test_creates_pair(self):
        """Test that function creates valid pair."""
        def base_update(M, A):
            return -0.1 * M
        
        forward, reverse = create_reversible_dynamics_pair(base_update)
        
        n = 10
        M = np.random.randn(n, n)
        M = (M + M.T) / 2
        A = np.eye(n)
        
        delta_forward = forward(M, A)
        delta_reverse = reverse(M, A)
        
        # Reverse should be negative of forward
        assert np.allclose(delta_forward, -delta_reverse)


class TestQuickReversibilityCheck:
    """Tests for quick_reversibility_check function."""
    
    def test_returns_tuple(self):
        """Test that function returns correct type."""
        n = 8
        M0 = np.random.randn(n, n)
        M0 = (M0 + M0.T) / 2
        
        A0 = np.random.rand(n, n)
        A0 = (A0 + A0.T) / 2
        np.fill_diagonal(A0, 0)
        
        def dynamics(M, A):
            return -0.1 * M
        
        fidelity, rev_class = quick_reversibility_check(M0, A0, dynamics, n_steps=10)
        
        assert isinstance(fidelity, float)
        assert isinstance(rev_class, ReversibilityClass)
        assert 0 <= fidelity <= 1


class TestPerturbationSensitivity:
    """Tests for perturbation sensitivity analysis."""
    
    def test_perturbation_analysis(self):
        """Test run_with_perturbation method."""
        n = 8
        M0 = np.random.randn(n, n)
        M0 = (M0 + M0.T) / 2
        
        A0 = np.random.rand(n, n)
        A0 = (A0 + A0.T) / 2
        np.fill_diagonal(A0, 0)
        
        probe = TimeReversalProbe(verbose=False)
        
        def forward(M, A):
            return -0.1 * M
        
        def reverse(M, A):
            return 0.1 * M
        
        result = probe.run_with_perturbation(
            M0, A0,
            forward, reverse,
            n_steps=10,
            perturbation_scale=0.01,
            n_trials=3,
        )
        
        assert 'baseline_fidelity' in result
        assert 'perturbed_fidelity_mean' in result
        assert 'sensitivity' in result
        assert result['sensitivity'] in ['low', 'moderate', 'high']
