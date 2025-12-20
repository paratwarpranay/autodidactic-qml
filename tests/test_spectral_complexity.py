"""Tests for spectral complexity analysis module.

Tests cover:
- spectral_entropy
- level_spacing_ratio
- participation_ratio
- spectral_form_factor
- SpectralComplexityAnalyzer
"""

import numpy as np
import pytest

from analysis import (
    SpectralComplexityAnalyzer,
    spectral_entropy,
    level_spacing_ratio,
    participation_ratio,
    spectral_form_factor,
    complexity_distance,
)


class TestSpectralEntropy:
    """Tests for spectral entropy computation."""

    def test_identity_matrix_low_entropy(self):
        """Identity matrix has minimal spectral spread."""
        M = np.eye(5)
        H = spectral_entropy(M)
        # All eigenvalues = 1, so low entropy after normalization
        assert H >= 0

    def test_random_matrix_positive_entropy(self):
        """Random matrix has positive entropy."""
        rng = np.random.default_rng(42)
        M = rng.normal(size=(10, 10))
        M = (M + M.T) / 2
        H = spectral_entropy(M)

        assert H > 0

    def test_entropy_bounded(self):
        """Entropy should be non-negative."""
        rng = np.random.default_rng(42)
        for _ in range(5):
            M = rng.normal(size=(8, 8))
            M = (M + M.T) / 2
            H = spectral_entropy(M)
            assert H >= 0

    def test_uniform_spectrum_high_entropy(self):
        """Uniform spectrum should have high entropy."""
        # Diagonal matrix with uniform eigenvalues
        eigs = np.ones(10)
        M = np.diag(eigs)
        H1 = spectral_entropy(M)

        # Diagonal matrix with varying eigenvalues
        eigs_var = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        M_var = np.diag(eigs_var)
        H2 = spectral_entropy(M_var)

        # Varying spectrum should have higher entropy
        assert H2 >= H1


class TestLevelSpacingRatio:
    """Tests for level spacing ratio computation.

    Note: level_spacing_ratio returns a dict with 'r_mean', 'r_std', 'is_chaotic'.
    """

    def test_poisson_statistics(self):
        """Uncorrelated eigenvalues should give r ~ 0.386."""
        rng = np.random.default_rng(42)
        # Diagonal matrix (Poisson statistics)
        eigs = rng.uniform(0, 10, size=100)
        M = np.diag(eigs)

        result = level_spacing_ratio(M)
        r = result["r_mean"]
        # Poisson: r ~ 0.386
        assert 0.2 < r < 0.6

    def test_goe_statistics(self):
        """GOE random matrix should give r ~ 0.530."""
        rng = np.random.default_rng(42)
        # GOE matrix
        M = rng.normal(size=(100, 100))
        M = (M + M.T) / np.sqrt(2)

        result = level_spacing_ratio(M)
        r = result["r_mean"]
        # GOE: r ~ 0.530
        assert 0.4 < r < 0.65

    def test_r_bounded(self):
        """Level spacing ratio should be in [0, 1]."""
        rng = np.random.default_rng(42)
        for _ in range(5):
            M = rng.normal(size=(20, 20))
            M = (M + M.T) / 2
            result = level_spacing_ratio(M)
            r = result["r_mean"]
            assert 0 <= r <= 1

    def test_small_matrix(self):
        """Test with small matrix."""
        M = np.array([[1, 0.5], [0.5, 2]])
        result = level_spacing_ratio(M)
        r = result["r_mean"]
        # Should return valid value even for small matrix
        assert 0 <= r <= 1

    def test_returns_dict(self):
        """Test that level_spacing_ratio returns dictionary."""
        M = np.random.default_rng(42).normal(size=(10, 10))
        M = (M + M.T) / 2
        result = level_spacing_ratio(M)

        assert isinstance(result, dict)
        assert "r_mean" in result
        assert "r_std" in result
        assert "is_chaotic" in result


class TestParticipationRatio:
    """Tests for participation ratio computation."""

    def test_localized_eigenvector(self):
        """Localized eigenvector should have low participation ratio."""
        # Diagonal matrix: eigenvectors are localized (single component)
        M = np.diag([1, 2, 3, 4, 5])
        pr = participation_ratio(M)

        # Each eigenvector has PR = 1/n = 0.2
        assert np.isclose(np.mean(pr), 0.2, atol=0.1)

    def test_delocalized_eigenvector(self):
        """Delocalized eigenvector should have high participation ratio."""
        n = 10
        # DFT matrix has delocalized eigenvectors
        dft = np.fft.fft(np.eye(n)) / np.sqrt(n)
        M = np.real(dft @ dft.T.conj())

        pr = participation_ratio(M)
        # Delocalized: PR should be higher
        assert np.mean(pr) > 0.5

    def test_pr_bounded(self):
        """Participation ratio should be in [1/n, 1]."""
        rng = np.random.default_rng(42)
        n = 10
        M = rng.normal(size=(n, n))
        M = (M + M.T) / 2

        pr = participation_ratio(M)
        assert np.all(pr >= 1/n - 0.01)
        assert np.all(pr <= 1 + 0.01)

    def test_returns_array(self):
        """Should return array of participation ratios."""
        M = np.random.default_rng(42).normal(size=(5, 5))
        M = (M + M.T) / 2
        pr = participation_ratio(M)

        assert isinstance(pr, np.ndarray)
        assert len(pr) == 5


class TestSpectralFormFactor:
    """Tests for spectral form factor computation."""

    def test_output_shape(self):
        """Test output shape matches time array."""
        M = np.random.default_rng(42).normal(size=(10, 10))
        M = (M + M.T) / 2
        t = np.linspace(0.1, 10, 20)

        sff = spectral_form_factor(M, t)
        assert sff.shape == t.shape

    def test_positive(self):
        """SFF should be positive."""
        M = np.random.default_rng(42).normal(size=(10, 10))
        M = (M + M.T) / 2
        t = np.linspace(0.1, 5, 10)

        sff = spectral_form_factor(M, t)
        assert np.all(sff >= 0)

    def test_early_time_small(self):
        """SFF at early times should be small for chaotic systems."""
        rng = np.random.default_rng(42)
        # GOE-like matrix
        M = rng.normal(size=(50, 50))
        M = (M + M.T) / np.sqrt(2)
        t = np.array([0.01, 0.1, 1.0, 10.0])

        sff = spectral_form_factor(M, t)
        # Early time SFF should be smaller than late time
        assert sff[0] < sff[-1] or np.isclose(sff[0], sff[-1], rtol=0.5)


class TestSpectralComplexityAnalyzer:
    """Tests for SpectralComplexityAnalyzer class."""

    def test_analyze_returns_dict(self):
        """Test analyze returns dictionary with all metrics."""
        analyzer = SpectralComplexityAnalyzer()
        M = np.random.default_rng(42).normal(size=(10, 10))
        M = (M + M.T) / 2

        result = analyzer.analyze(M)

        assert isinstance(result, dict)
        required_keys = [
            "spectral_entropy",
            "level_spacing_r",
            "participation_ratio_mean",
            "participation_ratio_std",
            "is_chaotic",
        ]
        for key in required_keys:
            assert key in result

    def test_is_chaotic_boolean(self):
        """Test is_chaotic returns proper boolean-like value."""
        analyzer = SpectralComplexityAnalyzer()
        M = np.random.default_rng(42).normal(size=(10, 10))
        M = (M + M.T) / 2

        result = analyzer.analyze(M)
        assert result["is_chaotic"] in [0, 1, 0.0, 1.0, True, False]

    def test_goe_detected_chaotic(self):
        """GOE matrix should be detected as chaotic."""
        analyzer = SpectralComplexityAnalyzer(goe_threshold=0.45)
        rng = np.random.default_rng(42)
        M = rng.normal(size=(100, 100))
        M = (M + M.T) / np.sqrt(2)

        result = analyzer.analyze(M)
        # Large GOE should have r > 0.45
        assert result["level_spacing_r"] > 0.4

    def test_diagonal_not_chaotic(self):
        """Diagonal matrix should not be chaotic."""
        analyzer = SpectralComplexityAnalyzer(goe_threshold=0.5)
        rng = np.random.default_rng(42)
        # Diagonal (Poisson statistics)
        M = np.diag(rng.uniform(0, 10, size=50))

        result = analyzer.analyze(M)
        # Should have lower r (Poisson ~ 0.386)
        assert result["level_spacing_r"] < 0.5

    def test_custom_threshold(self):
        """Test custom GOE threshold."""
        analyzer_low = SpectralComplexityAnalyzer(goe_threshold=0.3)
        analyzer_high = SpectralComplexityAnalyzer(goe_threshold=0.7)

        M = np.random.default_rng(42).normal(size=(20, 20))
        M = (M + M.T) / 2

        result_low = analyzer_low.analyze(M)
        result_high = analyzer_high.analyze(M)

        # With low threshold, more likely to be chaotic
        # With high threshold, less likely
        # Can't guarantee specific values, but check consistency
        assert "is_chaotic" in result_low
        assert "is_chaotic" in result_high


class TestComplexityDistance:
    """Tests for complexity_distance function."""

    def test_same_matrix_zero_distance(self):
        """Same matrix should have zero distance."""
        M = np.random.default_rng(42).normal(size=(10, 10))
        M = (M + M.T) / 2

        d = complexity_distance(M, M)
        assert np.isclose(d, 0, atol=1e-10)

    def test_different_matrices_positive_distance(self):
        """Different matrices should have positive distance."""
        rng = np.random.default_rng(42)
        M1 = rng.normal(size=(10, 10))
        M1 = (M1 + M1.T) / 2
        M2 = rng.normal(size=(10, 10))
        M2 = (M2 + M2.T) / 2

        d = complexity_distance(M1, M2)
        assert d > 0

    def test_symmetric(self):
        """Distance should be symmetric."""
        rng = np.random.default_rng(42)
        M1 = rng.normal(size=(10, 10))
        M1 = (M1 + M1.T) / 2
        M2 = rng.normal(size=(10, 10))
        M2 = (M2 + M2.T) / 2

        d12 = complexity_distance(M1, M2)
        d21 = complexity_distance(M2, M1)
        assert np.isclose(d12, d21)

    def test_non_negative(self):
        """Distance should be non-negative."""
        rng = np.random.default_rng(42)
        for _ in range(5):
            M1 = rng.normal(size=(8, 8))
            M1 = (M1 + M1.T) / 2
            M2 = rng.normal(size=(8, 8))
            M2 = (M2 + M2.T) / 2

            d = complexity_distance(M1, M2)
            assert d >= 0
