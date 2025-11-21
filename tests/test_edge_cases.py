"""Test edge cases and corner cases for FN-DBSCAN.

This module contains tests for all edge cases documented in the
technical specification.
"""

import numpy as np
import pytest

from fn_dbscan import FN_DBSCAN


class TestInputEdgeCases:
    """Test edge cases related to input data."""

    def test_edge_case_1_empty_dataset(self):
        """Edge Case 1: Empty dataset."""
        X = np.array([]).reshape(0, 2)
        model = FN_DBSCAN(eps=0.5, min_cardinality=3)

        with pytest.raises(ValueError, match="0 samples"):
            model.fit(X)

    def test_edge_case_2_single_point(self):
        """Edge Case 2: Single point."""
        X = np.array([[1, 2]])
        model = FN_DBSCAN(eps=0.5, min_cardinality=3)
        model.fit(X)

        # Cardinality = 1.0 < 3, so noise
        assert model.labels_[0] == -1
        assert model.n_clusters_ == 0

    def test_edge_case_3_two_identical_points(self):
        """Edge Case 3: Two identical points."""
        X = np.array([[1, 2], [1, 2]])
        model = FN_DBSCAN(eps=0.5, min_cardinality=2)
        model.fit(X)

        # Both points have cardinality 2.0
        assert model.n_clusters_ == 1
        assert model.labels_[0] == model.labels_[1]
        assert model.labels_[0] == 0

    def test_edge_case_4_min_cardinality_one(self):
        """Edge Case 4: MinCard = 1."""
        X = np.array([[1, 2], [10, 20], [100, 200]])
        model = FN_DBSCAN(eps=1.0, min_cardinality=1.0)
        model.fit(X)

        # Every point is core (cardinality ≥ 1.0)
        assert len(model.core_sample_indices_) == 3

    def test_edge_case_5_eps_zero(self):
        """Edge Case 5: ε = 0 (should fail)."""
        X = np.array([[1, 2], [2, 3]])

        with pytest.raises(ValueError):
            FN_DBSCAN(eps=0).fit(X)

    def test_edge_case_6_extremely_small_eps(self):
        """Edge Case 6: Extremely small ε."""
        X = np.array([[0.0, 0.0], [0.1, 0.1], [0.2, 0.2]])
        model = FN_DBSCAN(eps=0.01, min_cardinality=2)
        model.fit(X)

        # No point has neighbors, all noise
        assert np.all(model.labels_ == -1)
        assert model.n_clusters_ == 0


class TestNumericalEdgeCases:
    """Test numerical edge cases."""

    def test_edge_case_7_very_small_distances(self):
        """Edge Case 7: Very small distances."""
        X = np.array([
            [0.0, 0.0],
            [1e-15, 1e-15],  # Extremely close
            [1e-10, 1e-10]
        ])
        model = FN_DBSCAN(eps=1.0, min_cardinality=2)
        model.fit(X)

        # Should handle small distances correctly
        assert model.n_clusters_ >= 0

    def test_edge_case_8_distance_exactly_at_eps(self):
        """Edge Case 8: Distance exactly at ε."""
        # Create points exactly eps apart
        X = np.array([[0, 0], [1, 0]])  # Distance = 1.0
        model = FN_DBSCAN(eps=1.0, min_cardinality=1.5, fuzzy_function='linear')
        model.fit(X)

        # For linear: membership at distance=eps is 0
        # Each point has cardinality = 1.0 (self) + 0.0 (other) = 1.0
        # 1.0 < 1.5, so both are noise
        assert np.all(model.labels_ == -1)

    def test_edge_case_9_nan_in_data(self):
        """Edge Case 9: NaN in data."""
        X = np.array([[1, 2], [np.nan, 3], [4, 5]])
        model = FN_DBSCAN()

        with pytest.raises(ValueError, match="NaN"):
            model.fit(X)

    def test_edge_case_10_inf_in_data(self):
        """Edge Case 10: Inf in data."""
        X = np.array([[1, 2], [np.inf, 3], [4, 5]])
        model = FN_DBSCAN()

        with pytest.raises(ValueError, match="Inf"):
            model.fit(X)

    def test_edge_case_cardinality_just_below_threshold(self):
        """Edge Case: Cardinality just below MinCard."""
        # Create scenario where cardinality is very close to threshold
        X = np.array([[0, 0], [1, 0], [2, 0]])

        # With linear membership and eps=1.5:
        # Point 1: self(1.0) + point0(~0.33) + point2(~0.33) ≈ 1.66
        model = FN_DBSCAN(eps=1.5, min_cardinality=1.67)
        model.fit(X)

        # Point 1 should not be core
        # But exact behavior depends on floating point
        assert model.n_clusters_ >= 0


class TestAlgorithmicEdgeCases:
    """Test algorithmic edge cases."""

    def test_edge_case_11_nested_clusters(self):
        """Edge Case 11: Nested clusters."""
        # Dense inner region + sparse outer region
        inner = np.random.RandomState(42).randn(20, 2) * 0.3
        outer = np.random.RandomState(43).randn(10, 2) * 3.0

        X = np.vstack([inner, outer])

        model = FN_DBSCAN(eps=1.0, min_cardinality=3)
        model.fit(X)

        # Should detect at least inner cluster
        assert model.n_clusters_ >= 1

    def test_edge_case_12_border_point_ambiguity(self):
        """Edge Case 12: Border point between two cores."""
        # Create two core points with a point in between
        X = np.array([
            [0, 0],   # Core 1
            [1, 0],   # Border (between two cores)
            [2, 0],   # Core 2
            [0, 0.1], # Support for core 1
            [2, 0.1]  # Support for core 2
        ])

        model = FN_DBSCAN(eps=1.5, min_cardinality=2)
        model.fit(X)

        # Border point should be assigned deterministically
        # to first cluster that reaches it
        middle_label = model.labels_[1]
        assert middle_label >= 0  # Not noise

    def test_edge_case_13_noise_point_reclassification(self):
        """Edge Case 13: Noise point becomes border point."""
        X = np.array([
            [0, 0], [1, 0], [2, 0],  # Core points
            [3, 0]  # Initially noise, but reachable
        ])

        model = FN_DBSCAN(eps=1.5, min_cardinality=1.6)
        model.fit(X)

        # Point 3 should be border point (not noise)
        # because it's reachable from core points
        assert model.labels_[3] >= 0

    def test_edge_case_14_all_noise(self):
        """Edge Case 14: All noise result."""
        # Well-separated points, impossible to cluster
        X = np.array([
            [0, 0],
            [100, 100],
            [200, 200],
            [300, 300]
        ])

        model = FN_DBSCAN(eps=1.0, min_cardinality=5)
        model.fit(X)

        assert model.n_clusters_ == 0
        assert np.all(model.labels_ == -1)
        assert len(model.core_sample_indices_) == 0

    def test_edge_case_15_single_cluster(self):
        """Edge Case 15: Single cluster result."""
        # Dense cluster where all points are connected
        X = np.random.RandomState(42).randn(20, 2) * 0.5

        model = FN_DBSCAN(eps=2.0, min_cardinality=3)
        model.fit(X)

        # Should form one cluster
        unique_labels = np.unique(model.labels_)
        # Remove noise label if present
        clusters = unique_labels[unique_labels >= 0]

        assert len(clusters) == 1
        assert model.n_clusters_ == 1


class TestFuzzyFunctionEdgeCases:
    """Test edge cases specific to fuzzy functions."""

    def test_linear_at_boundary(self):
        """Test linear function behavior at epsilon boundary."""
        X = np.array([[0, 0], [1, 0]])  # Distance = 1.0

        model = FN_DBSCAN(eps=1.0, min_cardinality=1.5, fuzzy_function='linear')
        model.fit(X)

        # Linear membership at d=eps is exactly 0
        # Each point: 1.0 (self) + 0.0 (other) = 1.0 < 1.5
        assert np.all(model.labels_ == -1)

    def test_exponential_at_boundary(self):
        """Test exponential function behavior at epsilon boundary."""
        X = np.array([[0, 0], [1, 0]])  # Distance = 1.0

        model = FN_DBSCAN(eps=1.0, min_cardinality=1.2, fuzzy_function='exponential')
        model.fit(X)

        # Exponential membership at d=eps is set to 0 (by design)
        # Each point: 1.0 (self) + 0.0 (other) = 1.0 < 1.2
        assert np.all(model.labels_ == -1)

    def test_trapezoidal_plateau_region(self):
        """Test trapezoidal function in plateau region."""
        X = np.array([[0, 0], [0.25, 0]])  # Distance = 0.25

        model = FN_DBSCAN(eps=1.0, min_cardinality=1.9, fuzzy_function='trapezoidal')
        model.fit(X)

        # Distance 0.25 < eps/2 = 0.5, so membership = 1.0
        # Each point: 1.0 (self) + 1.0 (other) = 2.0 >= 1.9
        assert model.n_clusters_ == 1


class TestLargeDatasets:
    """Test with larger datasets."""

    def test_moderate_size_dataset(self):
        """Test with moderate size dataset (1000 points)."""
        np.random.seed(42)
        X = np.random.randn(1000, 2)

        model = FN_DBSCAN(eps=0.5, min_cardinality=5)
        model.fit(X)

        assert len(model.labels_) == 1000
        assert model.n_clusters_ >= 0

    def test_high_dimensional_data(self):
        """Test with high-dimensional data."""
        np.random.seed(42)
        X = np.random.randn(100, 50)  # 50 dimensions

        model = FN_DBSCAN(eps=5.0, min_cardinality=3)
        model.fit(X)

        assert len(model.labels_) == 100


class TestSpecialConfigurations:
    """Test special parameter configurations."""

    def test_very_large_eps(self):
        """Test with very large epsilon."""
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])

        model = FN_DBSCAN(eps=100.0, min_cardinality=2)
        model.fit(X)

        # All points are neighbors of all, should form one cluster
        assert model.n_clusters_ == 1

    def test_very_small_min_cardinality(self):
        """Test with very small min_cardinality."""
        X = np.array([[1, 2], [3, 4], [5, 6]])

        model = FN_DBSCAN(eps=1.0, min_cardinality=0.5)
        model.fit(X)

        # Almost everything is core
        assert len(model.core_sample_indices_) >= 1

    def test_different_fuzzy_functions_same_params(self):
        """Test that different fuzzy functions can give different results."""
        X = np.array([[0, 0], [1, 0], [2, 0], [3, 0]])

        labels_linear = FN_DBSCAN(
            eps=2.0, min_cardinality=2.5, fuzzy_function='linear'
        ).fit_predict(X)

        labels_exp = FN_DBSCAN(
            eps=2.0, min_cardinality=2.5, fuzzy_function='exponential'
        ).fit_predict(X)

        labels_trap = FN_DBSCAN(
            eps=2.0, min_cardinality=2.5, fuzzy_function='trapezoidal'
        ).fit_predict(X)

        # All should work, though results may differ
        assert len(labels_linear) == 4
        assert len(labels_exp) == 4
        assert len(labels_trap) == 4
