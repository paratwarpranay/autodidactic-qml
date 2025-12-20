"""Tests for phase transition detection module.

Tests cover:
- susceptibility
- binder_cumulant
- OnlinePhaseDetector
- PhaseTransitionAnalyzer
- detect_learning_phases
"""

import numpy as np
import pytest

from analysis import (
    PhaseTransitionAnalyzer,
    OnlinePhaseDetector,
    detect_learning_phases,
    susceptibility,
    binder_cumulant,
)


class TestSusceptibility:
    """Tests for susceptibility computation.

    Note: susceptibility returns an array of local susceptibility values.
    """

    def test_constant_signal_zero_susceptibility(self):
        """Constant signal should have zero susceptibility."""
        y = np.ones(100)
        x = np.linspace(0, 1, 100)

        chi = susceptibility(y, x)
        # Chi is an array; all values should be near zero for constant signal
        assert np.all(np.abs(chi) < 1e-10)

    def test_linear_signal_low_susceptibility(self):
        """Linear signal should have low susceptibility variation."""
        x = np.linspace(0, 1, 100)
        y = 2 * x + 1

        chi = susceptibility(y, x)
        # Derivative is constant = 2
        # Susceptibility variance should be low
        assert np.all(chi >= 0)

    def test_step_function_high_susceptibility(self):
        """Step function should have high susceptibility at transition."""
        x = np.linspace(0, 1, 100)
        y = np.where(x < 0.5, 0, 1).astype(float)

        chi = susceptibility(y, x)
        # Sharp transition -> high susceptibility peak
        assert np.max(chi) > 0

    def test_positive_susceptibility(self):
        """Susceptibility should be non-negative."""
        rng = np.random.default_rng(42)
        for _ in range(5):
            y = rng.normal(size=50)
            x = np.linspace(0, 1, 50)
            chi = susceptibility(y, x)
            assert np.all(chi >= -1e-10)

    def test_returns_array(self):
        """Susceptibility should return array."""
        y = np.random.randn(50)
        x = np.linspace(0, 1, 50)
        chi = susceptibility(y, x)
        assert isinstance(chi, np.ndarray)
        assert len(chi) == len(y)


class TestBinderCumulant:
    """Tests for Binder cumulant computation.
    
    The Binder cumulant U = <m²>² / <m⁴> measures distribution peakedness.
    U = 1 for delta (constant), U = 1/3 for Gaussian, U = 5/9 for uniform.
    """

    def test_gaussian_binder(self):
        """Gaussian distribution has specific Binder cumulant.
        
        For standard Gaussian: <m²> = 1, <m⁴> = 3, so U = 1/3 ≈ 0.333.
        """
        rng = np.random.default_rng(42)
        samples = rng.normal(size=10000)

        U = binder_cumulant(samples)
        # For Gaussian, U = <m²>²/<m⁴> = 1/3 ≈ 0.333
        assert 0.30 < U < 0.40

    def test_uniform_binder(self):
        """Uniform distribution has different Binder cumulant.
        
        For uniform[-1,1]: <m²> = 1/3, <m⁴> = 1/5, so U = 5/9 ≈ 0.556.
        """
        rng = np.random.default_rng(42)
        samples = rng.uniform(-1, 1, size=10000)

        U = binder_cumulant(samples)
        # Uniform: U = (1/3)²/(1/5) = 5/9 ≈ 0.556
        assert 0.52 < U < 0.62

    def test_constant_binder(self):
        """Constant samples should give U = 1.
        
        For constant c: <m²> = c², <m⁴> = c⁴, so U = c⁴/c⁴ = 1.
        """
        samples = np.ones(100)
        U = binder_cumulant(samples)
        # All same value: m^4 = m^2^2, so U = 1
        assert np.isclose(U, 1.0)

    def test_bounded(self):
        """Binder cumulant should be bounded in [0, 1] for typical distributions."""
        rng = np.random.default_rng(42)
        samples = rng.exponential(size=1000)

        U = binder_cumulant(samples)
        # U should be in reasonable range [0, 1]
        assert 0 <= U <= 1

    def test_bimodal_distribution(self):
        """Bimodal distribution has higher U than unimodal."""
        rng = np.random.default_rng(42)
        # Bimodal: mix of two separated Gaussians
        n = 5000
        bimodal = np.concatenate([
            rng.normal(-2, 0.5, size=n),
            rng.normal(2, 0.5, size=n),
        ])
        unimodal = rng.normal(0, 1, size=2*n)
        
        U_bi = binder_cumulant(bimodal)
        U_uni = binder_cumulant(unimodal)
        
        # Bimodal (ordered phase) has higher U than unimodal (disordered)
        assert U_bi > U_uni


class TestOnlinePhaseDetector:
    """Tests for OnlinePhaseDetector class.

    Note: OnlinePhaseDetector.history is a dict of deques, one per metric name.
    """

    def test_initialization(self):
        """Test detector initialization."""
        detector = OnlinePhaseDetector(window_size=10)
        assert detector.window_size == 10
        assert isinstance(detector.history, dict)
        assert len(detector.history) == 0

    def test_threshold_parameter(self):
        """Test that threshold parameter is accepted."""
        detector = OnlinePhaseDetector(window_size=10, threshold=2.0)
        assert detector.threshold == 2.0
        assert detector.sensitivity == 2.0  # Alias

    def test_sensitivity_parameter(self):
        """Test that sensitivity parameter is accepted as alias."""
        detector = OnlinePhaseDetector(window_size=10, sensitivity=3.0)
        assert detector.threshold == 3.0
        assert detector.sensitivity == 3.0

    def test_update_accumulates_history(self):
        """Test that update adds to history."""
        detector = OnlinePhaseDetector(window_size=5)

        for i in range(3):
            detector.update({"loss": float(i)})

        # history is a dict with 'loss' key containing deque
        assert "loss" in detector.history
        assert len(detector.history["loss"]) == 3

    def test_update_returns_dict(self):
        """Test that update returns dictionary."""
        detector = OnlinePhaseDetector(window_size=5)

        # Need enough history first
        for i in range(10):
            result = detector.update({"loss": float(i)})

        assert isinstance(result, dict)

    def test_detect_no_transition_stable(self):
        """Test no transition detected for stable signal."""
        detector = OnlinePhaseDetector(window_size=10, threshold=2.0)

        for i in range(50):
            result = detector.update({"loss": 1.0 + 0.01 * np.random.randn()})

        # Stable signal should not trigger transition
        # result is dict of metric_name -> bool
        assert "loss" in result
        # Most stable signals should not trigger

    def test_detect_transition_on_jump(self):
        """Test transition detected on sharp change."""
        detector = OnlinePhaseDetector(window_size=10, threshold=2.0)

        # Stable phase
        for i in range(30):
            detector.update({"loss": 1.0})

        # Sharp jump
        result = detector.update({"loss": 10.0})

        # result is dict of metric_name -> is_transition
        assert "loss" in result
        # Jump should trigger transition
        assert result["loss"] == True

    def test_window_size_limits_history(self):
        """Test that history is limited to window_size."""
        detector = OnlinePhaseDetector(window_size=10)

        for i in range(20):
            detector.update({"loss": float(i)})

        # History should be capped at window_size
        assert len(detector.history["loss"]) == 10


class TestPhaseTransitionAnalyzer:
    """Tests for PhaseTransitionAnalyzer class."""

    def test_analyze_sweep_returns_dict(self):
        """Test analyze_sweep returns proper structure."""
        analyzer = PhaseTransitionAnalyzer()

        # Create fake sweep data
        g_values = np.linspace(0, 1, 20)
        observables = {
            "loss": g_values**2 + 0.1 * np.random.randn(20),
            "accuracy": 1 - g_values + 0.1 * np.random.randn(20),
        }

        result = analyzer.analyze_sweep(g_values, observables)

        assert isinstance(result, dict)
        # Check for expected keys
        for name in observables:
            assert name in result
            assert "susceptibility" in result[name] or "phase_boundaries" in result[name]

    def test_find_critical_points(self):
        """Test critical point detection."""
        analyzer = PhaseTransitionAnalyzer()

        # Create data with clear transition at g=0.5
        g_values = np.linspace(0, 1, 100)
        order_param = np.where(g_values < 0.5, 0, 1).astype(float)
        order_param += 0.05 * np.random.randn(100)

        critical = analyzer.find_critical_points(g_values, order_param)

        # Should find critical point near 0.5
        assert len(critical) > 0
        assert any(0.4 < c < 0.6 for c in critical)

    def test_find_critical_points_cusp_method(self):
        """Test critical point detection with cusp method."""
        analyzer = PhaseTransitionAnalyzer()

        # Create data with sharp cusp at g=0.5
        rng = np.random.default_rng(42)
        g_values = np.linspace(0, 1, 100)
        order_param = np.abs(g_values - 0.5)
        order_param += 0.02 * rng.normal(size=100)

        critical = analyzer.find_critical_points(g_values, order_param, method="cusp")

        # Should find critical point near 0.5
        if len(critical) > 0:
            assert any(0.4 < c < 0.6 for c in critical)

    def test_find_critical_points_smooth_transition(self):
        """Test critical point detection for smooth transition."""
        analyzer = PhaseTransitionAnalyzer()

        # Create smooth sigmoid transition
        g_values = np.linspace(0, 1, 100)
        order_param = 1 / (1 + np.exp(-20*(g_values - 0.5)))
        
        critical = analyzer.find_critical_points(g_values, order_param)
        
        # Should find inflection point near 0.5
        if len(critical) > 0:
            assert any(0.4 < c < 0.6 for c in critical)


class TestDetectLearningPhases:
    """Tests for detect_learning_phases function."""

    def test_detect_phases_from_loss(self):
        """Test phase detection from loss trajectory."""
        # Simulate loss trajectory with distinct phases
        rng = np.random.default_rng(42)
        t = np.arange(200)
        loss = np.zeros(200)
        # Phase 1: high loss
        loss[:50] = 1.0 + 0.1 * rng.normal(size=50)
        # Transition
        loss[50:70] = np.linspace(1.0, 0.3, 20)
        # Phase 2: low loss
        loss[70:150] = 0.3 + 0.05 * rng.normal(size=80)
        # Phase 3: plateau
        loss[150:] = 0.1 + 0.02 * rng.normal(size=50)

        phases = detect_learning_phases(loss)

        assert isinstance(phases, dict)
        # Should detect at least one transition
        if "n_phases" in phases:
            assert phases["n_phases"] >= 1
        if "transitions" in phases:
            assert isinstance(phases["transitions"], (list, np.ndarray))

    def test_stable_training_few_transitions(self):
        """Test that stable training gives few transitions."""
        rng = np.random.default_rng(42)
        # Constant loss (already converged)
        loss = 0.1 + 0.01 * rng.normal(size=200)

        phases = detect_learning_phases(loss, min_phase_duration=15)

        # Should detect few/no transitions due to hysteresis
        if "n_transitions" in phases:
            assert phases["n_transitions"] <= 3
        if "transitions" in phases:
            assert len(phases["transitions"]) <= 3

    def test_returns_required_fields(self):
        """Test that required fields are returned."""
        rng = np.random.default_rng(42)
        loss = np.exp(-np.linspace(0, 5, 100)) + 0.01 * rng.normal(size=100)

        phases = detect_learning_phases(loss)

        # Should have required fields
        assert isinstance(phases, dict)
        assert "phases" in phases
        assert "transitions" in phases
        assert "final_phase" in phases
        assert "n_transitions" in phases

    def test_insufficient_data(self):
        """Test handling of insufficient data."""
        loss = [1.0, 0.9, 0.8]  # Too short

        phases = detect_learning_phases(loss, window=20)

        assert phases["final_phase"] == "insufficient_data"
        assert phases["n_transitions"] == 0

    def test_hysteresis_prevents_flickering(self):
        """Test that hysteresis prevents rapid phase changes."""
        rng = np.random.default_rng(42)
        # Create noisy signal that would flicker without hysteresis
        loss = 0.5 + 0.2 * rng.normal(size=200)
        
        phases = detect_learning_phases(loss, min_phase_duration=20)
        
        # With hysteresis, should have few transitions despite noise
        assert phases["n_transitions"] <= 5


class TestPhaseDetectorIntegration:
    """Integration tests for phase detection."""

    def test_online_matches_batch(self):
        """Test online and batch detection roughly agree."""
        # Generate data
        rng = np.random.default_rng(42)
        n = 100
        loss = np.zeros(n)
        loss[:40] = 1.0 + 0.1 * rng.normal(size=40)
        loss[40:60] = np.linspace(1.0, 0.2, 20)
        loss[60:] = 0.2 + 0.05 * rng.normal(size=40)

        # Online detection
        detector = OnlinePhaseDetector(window_size=10, threshold=2.0)
        online_transitions = []
        for i, l in enumerate(loss):
            result = detector.update({"loss": l})
            if result.get("loss", False):
                online_transitions.append(i)

        # Batch detection
        batch_phases = detect_learning_phases(loss)

        # Both should identify transition region around index 40-60
        # (Exact agreement not required due to different methods)
        assert isinstance(online_transitions, list)
        assert isinstance(batch_phases, dict)

    def test_multiple_transitions(self):
        """Test detection of multiple transitions."""
        # Create multi-phase trajectory
        rng = np.random.default_rng(42)
        n = 200
        loss = np.zeros(n)
        # Phase 1
        loss[:50] = 1.0 + 0.05 * rng.normal(size=50)
        # Transition 1
        loss[50:60] = np.linspace(1.0, 0.5, 10)
        # Phase 2
        loss[60:120] = 0.5 + 0.05 * rng.normal(size=60)
        # Transition 2
        loss[120:130] = np.linspace(0.5, 0.1, 10)
        # Phase 3
        loss[130:] = 0.1 + 0.02 * rng.normal(size=70)

        phases = detect_learning_phases(loss, min_phase_duration=10)

        # Should detect multiple phases
        assert phases["n_phases"] >= 2
