"""Tests for physics hygiene module."""

import numpy as np
import pytest
import warnings

from analysis.physics_hygiene import (
    PhysicsHygiene,
    HygieneConfig,
    HygieneReport,
    PhysicsViolation,
    assert_physics_invariants,
    atomic_step,
    AtomicUpdateResult,
)


class TestPhysicsHygiene:
    """Tests for PhysicsHygiene class."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = HygieneConfig()
        assert config.check_shape
        assert config.check_hermiticity
        assert config.check_finite
        assert config.check_spectral_bounds
        assert not config.strict
    
    def test_valid_matrix_passes(self):
        """Test that a valid Hermitian matrix passes all checks."""
        hygiene = PhysicsHygiene()
        
        # Create valid Hermitian matrix
        M = np.random.randn(10, 10)
        M = (M + M.T) / 2  # Symmetrize
        
        hygiene.set_reference(M)
        report = hygiene.check(M)
        
        assert report.passed
        assert len(report.violations) == 0
    
    def test_shape_check_non_square(self):
        """Test shape check catches non-square matrices."""
        hygiene = PhysicsHygiene()
        
        M = np.random.randn(10, 8)  # Non-square
        
        report = hygiene.check(M)
        
        assert not report.passed
        assert any("not square" in v for v in report.violations)
    
    def test_shape_check_dimension_change(self):
        """Test shape check catches dimension changes."""
        hygiene = PhysicsHygiene()
        
        M0 = np.random.randn(10, 10)
        M0 = (M0 + M0.T) / 2
        hygiene.set_reference(M0)
        
        M1 = np.random.randn(8, 8)  # Different dimension
        M1 = (M1 + M1.T) / 2
        
        report = hygiene.check(M1)
        
        assert not report.passed
        assert any("Dimension changed" in v for v in report.violations)
    
    def test_finiteness_check_nan(self):
        """Test finiteness check catches NaN values."""
        hygiene = PhysicsHygiene()
        
        M = np.random.randn(10, 10)
        M[5, 5] = np.nan
        
        report = hygiene.check(M)
        
        assert not report.passed
        assert any("NaN" in v for v in report.violations)
    
    def test_finiteness_check_inf(self):
        """Test finiteness check catches Inf values."""
        hygiene = PhysicsHygiene()
        
        M = np.random.randn(10, 10)
        M[3, 4] = np.inf
        
        report = hygiene.check(M)
        
        assert not report.passed
        assert any("Inf" in v for v in report.violations)
    
    def test_hermiticity_check(self):
        """Test Hermiticity check catches asymmetric matrices."""
        config = HygieneConfig(auto_correct=False)
        hygiene = PhysicsHygiene(config)
        
        M = np.random.randn(10, 10)  # Not symmetric
        
        report = hygiene.check(M)
        
        assert not report.passed
        assert any("Hermiticity" in v for v in report.violations)
    
    def test_hermiticity_auto_correction(self):
        """Test Hermiticity auto-correction."""
        config = HygieneConfig(auto_correct=True)
        hygiene = PhysicsHygiene(config)
        
        M = np.random.randn(10, 10)  # Not symmetric
        M_copy = M.copy()
        
        report = hygiene.check(M)
        
        # Should have been corrected
        assert len(report.corrections) > 0
        assert np.allclose(M, M.T)  # Now symmetric
    
    def test_spectral_bounds(self):
        """Test spectral bounds check."""
        config = HygieneConfig(spectral_max=10.0)
        hygiene = PhysicsHygiene(config)
        
        # Create matrix with large eigenvalues
        M = 100 * np.eye(10)
        
        report = hygiene.check(M)
        
        assert not report.passed
        assert any("Spectral bound" in v for v in report.violations)
    
    def test_strict_mode_raises(self):
        """Test strict mode raises exception on violation."""
        config = HygieneConfig(strict=True, auto_correct=False)
        hygiene = PhysicsHygiene(config)
        
        M = np.random.randn(10, 10)  # Not symmetric
        
        with pytest.raises(PhysicsViolation):
            hygiene.check(M)
    
    def test_history_tracking(self):
        """Test history is tracked correctly."""
        hygiene = PhysicsHygiene()
        
        M = np.random.randn(10, 10)
        M = (M + M.T) / 2
        
        hygiene.check(M)
        hygiene.check(M)
        hygiene.check(M)
        
        assert len(hygiene.history) == 3
    
    def test_summary_statistics(self):
        """Test summary statistics."""
        hygiene = PhysicsHygiene()
        
        M = np.random.randn(10, 10)
        M = (M + M.T) / 2
        
        for _ in range(5):
            hygiene.check(M)
        
        summary = hygiene.summary()
        
        assert summary['n_checks'] == 5
        assert summary['n_passed'] == 5
        assert summary['pass_rate'] == 1.0


class TestAssertPhysicsInvariants:
    """Tests for assert_physics_invariants function."""
    
    def test_valid_matrix(self):
        """Test valid matrix passes."""
        M = np.random.randn(10, 10)
        M = (M + M.T) / 2
        
        # Should not raise
        assert_physics_invariants(M, N=10)
    
    def test_wrong_dimension(self):
        """Test wrong dimension raises."""
        M = np.random.randn(10, 10)
        M = (M + M.T) / 2
        
        with pytest.raises(AssertionError, match="Dimension changed"):
            assert_physics_invariants(M, N=8)
    
    def test_non_hermitian(self):
        """Test non-Hermitian raises."""
        M = np.random.randn(10, 10)  # Not symmetric
        
        with pytest.raises(AssertionError, match="Hermiticity"):
            assert_physics_invariants(M)
    
    def test_nan_values(self):
        """Test NaN values raise."""
        M = np.random.randn(10, 10)
        M = (M + M.T) / 2
        M[5, 5] = np.nan
        
        with pytest.raises(AssertionError, match="NaN or Inf"):
            assert_physics_invariants(M)


class TestAtomicStep:
    """Tests for atomic_step function."""
    
    def test_basic_update(self):
        """Test basic atomic update."""
        M = np.random.randn(10, 10)
        M = (M + M.T) / 2
        
        def compute_update(M_snapshot):
            return -0.1 * M_snapshot
        
        result = atomic_step(M, compute_update)
        
        assert isinstance(result, AtomicUpdateResult)
        assert result.M_new.shape == M.shape
        assert np.allclose(result.M_new, result.M_new.T)  # Symmetric
    
    def test_update_with_hygiene(self):
        """Test atomic update with hygiene checker."""
        hygiene = PhysicsHygiene()
        
        M = np.random.randn(10, 10)
        M = (M + M.T) / 2
        hygiene.set_reference(M)
        
        def compute_update(M_snapshot):
            return -0.1 * M_snapshot
        
        result = atomic_step(M, compute_update, hygiene=hygiene)
        
        assert result.hygiene_report is not None
        assert result.hygiene_report.passed
    
    def test_snapshot_immutability(self):
        """Test that snapshot is not modified during update."""
        M_original = np.random.randn(10, 10)
        M_original = (M_original + M_original.T) / 2
        M_copy = M_original.copy()
        
        def compute_update(M_snapshot):
            # Try to modify snapshot (should not affect original)
            M_snapshot[0, 0] = 999.0
            return -0.1 * M_snapshot
        
        result = atomic_step(M_original, compute_update)
        
        # Original should be unchanged
        assert np.allclose(M_original, M_copy)


class TestTraceConservation:
    """Tests for trace conservation checks."""
    
    def test_trace_conservation_passes(self):
        """Test trace conservation passes when trace is preserved."""
        config = HygieneConfig(check_trace=True, trace_rtol=0.01)
        hygiene = PhysicsHygiene(config)
        
        M0 = np.random.randn(10, 10)
        M0 = (M0 + M0.T) / 2
        hygiene.set_reference(M0)
        
        # Small perturbation preserving trace
        M1 = M0 + 0.001 * (np.random.randn(10, 10) + np.random.randn(10, 10).T) / 2
        # Adjust trace
        M1 = M1 - np.eye(10) * (np.trace(M1) - np.trace(M0)) / 10
        
        report = hygiene.check(M1)
        
        # Should pass (trace approximately conserved)
        trace_violation = any("Trace drift" in v for v in report.violations)
        assert not trace_violation
    
    def test_trace_conservation_fails(self):
        """Test trace conservation fails when trace changes."""
        config = HygieneConfig(check_trace=True, trace_rtol=0.01)
        hygiene = PhysicsHygiene(config)
        
        M0 = np.eye(10)  # Trace = 10
        hygiene.set_reference(M0)
        
        M1 = 2 * np.eye(10)  # Trace = 20 (100% change)
        
        report = hygiene.check(M1)
        
        assert any("Trace drift" in v for v in report.violations)
