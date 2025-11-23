"""Tests for fuzzy membership functions."""

import numpy as np
import pytest

from fn_dbscan._fuzzy_functions import (
    linear_membership,
    exponential_membership,
    trapezoidal_membership,
    get_fuzzy_membership_function,
    calculate_fuzzy_cardinality,
)


class TestLinearMembership:
    """Tests for linear fuzzy membership function."""

    def test_at_zero_distance(self):
        """Test membership at distance 0."""
        assert linear_membership(0.0, 1.0) == 1.0

    def test_at_epsilon(self):
        """Test membership at distance epsilon."""
        assert linear_membership(1.0, 1.0) == 0.0

    def test_at_half_epsilon(self):
        """Test membership at distance epsilon/2."""
        assert linear_membership(0.5, 1.0) == 0.5

    def test_beyond_epsilon(self):
        """Test membership beyond epsilon."""
        assert linear_membership(2.0, 1.0) == 0.0

    def test_negative_distance(self):
        """Test membership with negative distance."""
        result = linear_membership(-0.5, 1.0)
        assert result == 1.0  # Clipped to maximum

    def test_invalid_epsilon(self):
        """Test with invalid epsilon."""
        with pytest.raises(ValueError):
            linear_membership(0.5, 0.0)
        with pytest.raises(ValueError):
            linear_membership(0.5, -1.0)

    def test_vectorized(self):
        """Test vectorized operation."""
        distances = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.5])
        expected = np.array([1.0, 0.75, 0.5, 0.25, 0.0, 0.0])
        result = linear_membership(distances, 1.0)
        np.testing.assert_array_almost_equal(result, expected)


class TestExponentialMembership:
    """Tests for exponential fuzzy membership function."""

    def test_at_zero_distance(self):
        """Test membership at distance 0."""
        assert exponential_membership(0.0, 1.0) == 1.0

    def test_at_epsilon(self):
        """Test membership at distance epsilon."""
        # With default k=20, result should be very close to 0
        result = exponential_membership(1.0, 1.0)
        assert result < 1e-10

    def test_at_epsilon_k1(self):
        """Test membership at distance epsilon with k=1."""
        result = exponential_membership(1.0, 1.0, k=1.0)
        expected = np.exp(-1.0)
        assert abs(result - expected) < 1e-10

    def test_beyond_epsilon(self):
        """Test membership beyond epsilon."""
        assert exponential_membership(2.0, 1.0) == 0.0

    def test_monotonic_decay(self):
        """Test that membership decreases with distance."""
        # Use k=1.0 for gentler decay to ensure distinct values
        d1 = exponential_membership(0.0, 1.0, k=1.0)
        d2 = exponential_membership(0.3, 1.0, k=1.0)
        d3 = exponential_membership(0.6, 1.0, k=1.0)
        assert d1 > d2 > d3

    def test_invalid_epsilon(self):
        """Test with invalid epsilon."""
        with pytest.raises(ValueError):
            exponential_membership(0.5, 0.0)

    def test_vectorized(self):
        """Test vectorized operation."""
        distances = np.array([0.0, 0.5, 1.0, 1.5])
        result = exponential_membership(distances, 1.0)
        assert result[0] == 1.0
        assert result[-1] == 0.0  # Beyond epsilon
        assert np.all(result >= 0.0) and np.all(result <= 1.0)


class TestTrapezoidalMembership:
    """Tests for trapezoidal fuzzy membership function."""

    def test_at_zero_distance(self):
        """Test membership at distance 0."""
        assert trapezoidal_membership(0.0, 1.0) == 1.0

    def test_plateau_region(self):
        """Test membership in plateau region."""
        assert trapezoidal_membership(0.25, 1.0) == 1.0
        assert trapezoidal_membership(0.5, 1.0) == 1.0

    def test_decay_region(self):
        """Test membership in decay region."""
        result = trapezoidal_membership(0.75, 1.0)
        expected = 2.0 * (1.0 - 0.75)
        assert abs(result - expected) < 1e-10

    def test_at_epsilon(self):
        """Test membership at distance epsilon."""
        assert trapezoidal_membership(1.0, 1.0) == 0.0

    def test_beyond_epsilon(self):
        """Test membership beyond epsilon."""
        assert trapezoidal_membership(1.5, 1.0) == 0.0

    def test_invalid_epsilon(self):
        """Test with invalid epsilon."""
        with pytest.raises(ValueError):
            trapezoidal_membership(0.5, 0.0)

    def test_vectorized(self):
        """Test vectorized operation."""
        distances = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.5])
        result = trapezoidal_membership(distances, 1.0)

        # Plateau region
        assert result[0] == 1.0
        assert result[1] == 1.0

        # Decay region
        assert 0.0 < result[3] < 1.0

        # Zero region
        assert result[4] == 0.0
        assert result[5] == 0.0


class TestGetFuzzyMembershipFunction:
    """Tests for fuzzy membership function getter."""

    def test_get_linear(self):
        """Test getting linear function."""
        func = get_fuzzy_membership_function('linear')
        assert func(0.5, 1.0) == 0.5

    def test_get_exponential(self):
        """Test getting exponential function."""
        func = get_fuzzy_membership_function('exponential')
        result = func(0.0, 1.0)
        assert result == 1.0

    def test_get_trapezoidal(self):
        """Test getting trapezoidal function."""
        func = get_fuzzy_membership_function('trapezoidal')
        assert func(0.25, 1.0) == 1.0

    def test_invalid_function(self):
        """Test with invalid function name."""
        with pytest.raises(ValueError, match="Unknown fuzzy function"):
            get_fuzzy_membership_function('invalid')


class TestCalculateFuzzyCardinality:
    """Tests for fuzzy cardinality calculation."""

    def test_single_point(self):
        """Test cardinality with single point (self)."""
        distances = np.array([0.0])
        cardinality = calculate_fuzzy_cardinality(distances, 1.0, 'linear')
        assert cardinality == 1.0

    def test_multiple_points_linear(self):
        """Test cardinality with multiple points using linear function."""
        distances = np.array([0.0, 0.5, 1.0, 1.5])
        cardinality = calculate_fuzzy_cardinality(distances, 1.0, 'linear')
        expected = 1.0 + 0.5 + 0.0 + 0.0  # 1.5
        assert abs(cardinality - expected) < 1e-10

    def test_empty_neighbors(self):
        """Test cardinality with no neighbors."""
        distances = np.array([])
        cardinality = calculate_fuzzy_cardinality(distances, 1.0, 'linear')
        assert cardinality == 0.0

    def test_all_fuzzy_functions(self):
        """Test cardinality with all fuzzy functions."""
        distances = np.array([0.0, 0.5])

        for func_type in ['linear', 'exponential', 'trapezoidal']:
            cardinality = calculate_fuzzy_cardinality(
                distances, 1.0, func_type
            )
            assert cardinality > 0.0
            assert isinstance(cardinality, float)

    def test_cardinality_decreases_with_distance(self):
        """Test that adding farther points increases cardinality less."""
        close_distances = np.array([0.0, 0.1, 0.2])
        far_distances = np.array([0.0, 0.7, 0.9])

        close_card = calculate_fuzzy_cardinality(close_distances, 1.0, 'linear')
        far_card = calculate_fuzzy_cardinality(far_distances, 1.0, 'linear')

        assert close_card > far_card
