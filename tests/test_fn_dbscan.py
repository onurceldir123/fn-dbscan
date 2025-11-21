"""Tests for FN_DBSCAN main algorithm."""

import numpy as np
import pytest
from sklearn.datasets import make_blobs, make_moons

from fn_dbscan import FN_DBSCAN


class TestFNDBSCANBasic:
    """Basic tests for FN_DBSCAN algorithm."""

    def test_initialization(self):
        """Test FN_DBSCAN initialization."""
        model = FN_DBSCAN(eps=0.5, min_cardinality=3, fuzzy_function='linear')
        assert model.eps == 0.5
        assert model.min_cardinality == 3
        assert model.fuzzy_function == 'linear'
        assert model.metric == 'euclidean'

    def test_repr(self):
        """Test string representation."""
        model = FN_DBSCAN(eps=0.5, min_cardinality=3)
        repr_str = repr(model)
        assert 'FN_DBSCAN' in repr_str
        assert 'eps=0.5' in repr_str
        assert 'min_cardinality=3' in repr_str

    def test_simple_clustering(self):
        """Test clustering on simple 2-cluster dataset."""
        # Create two well-separated blobs
        X, y_true = make_blobs(
            n_samples=50,
            centers=2,
            n_features=2,
            cluster_std=0.5,
            center_box=(0, 10),
            random_state=42
        )

        model = FN_DBSCAN(eps=2.0, min_cardinality=3)
        labels = model.fit_predict(X)

        # Should find 2 clusters
        assert model.n_clusters_ == 2
        assert len(labels) == 50
        assert set(labels) <= {-1, 0, 1}  # Only -1, 0, 1 should appear

    def test_fit_returns_self(self):
        """Test that fit returns self."""
        X = np.array([[1, 2], [2, 2], [2, 3]])
        model = FN_DBSCAN()
        result = model.fit(X)
        assert result is model

    def test_fit_predict_consistency(self):
        """Test that fit and fit_predict give same results."""
        X = np.array([[1, 2], [2, 2], [2, 3], [8, 7], [8, 8]])

        model1 = FN_DBSCAN(eps=3, min_cardinality=2)
        model1.fit(X)

        model2 = FN_DBSCAN(eps=3, min_cardinality=2)
        labels = model2.fit_predict(X)

        np.testing.assert_array_equal(model1.labels_, labels)

    def test_attributes_after_fit(self):
        """Test that required attributes exist after fit."""
        X = np.array([[1, 2], [2, 2], [2, 3]])
        model = FN_DBSCAN(eps=3, min_cardinality=2)
        model.fit(X)

        assert hasattr(model, 'labels_')
        assert hasattr(model, 'core_sample_indices_')
        assert hasattr(model, 'n_clusters_')

        assert len(model.labels_) == 3
        assert isinstance(model.core_sample_indices_, np.ndarray)
        assert isinstance(model.n_clusters_, (int, np.integer))


class TestFNDBSCANFuzzyFunctions:
    """Test FN_DBSCAN with different fuzzy functions."""

    def setup_method(self):
        """Set up test data."""
        self.X = np.array([
            [1, 2], [2, 2], [2, 3],  # Close cluster
            [8, 7], [8, 8], [25, 80]  # Another cluster + noise
        ])

    def test_linear_function(self):
        """Test with linear fuzzy function."""
        model = FN_DBSCAN(eps=3, min_cardinality=2, fuzzy_function='linear')
        model.fit(self.X)

        assert model.n_clusters_ >= 0
        assert len(model.labels_) == 6

    def test_exponential_function(self):
        """Test with exponential fuzzy function."""
        model = FN_DBSCAN(eps=3, min_cardinality=2, fuzzy_function='exponential')
        model.fit(self.X)

        assert model.n_clusters_ >= 0
        assert len(model.labels_) == 6

    def test_trapezoidal_function(self):
        """Test with trapezoidal fuzzy function."""
        model = FN_DBSCAN(eps=3, min_cardinality=2, fuzzy_function='trapezoidal')
        model.fit(self.X)

        assert model.n_clusters_ >= 0
        assert len(model.labels_) == 6

    def test_different_functions_may_differ(self):
        """Test that different fuzzy functions can produce different results."""
        labels_linear = FN_DBSCAN(
            eps=3, min_cardinality=2.5, fuzzy_function='linear'
        ).fit_predict(self.X)

        labels_exp = FN_DBSCAN(
            eps=3, min_cardinality=2.5, fuzzy_function='exponential'
        ).fit_predict(self.X)

        # Results may differ (but not necessarily)
        # Just check they both work
        assert len(labels_linear) == len(labels_exp) == 6


class TestFNDBSCANEdgeCases:
    """Test edge cases for FN_DBSCAN."""

    def test_empty_dataset(self):
        """Test with empty dataset."""
        X = np.array([]).reshape(0, 2)
        model = FN_DBSCAN()

        with pytest.raises(ValueError, match="0 samples"):
            model.fit(X)

    def test_single_point(self):
        """Test with single point."""
        X = np.array([[1, 2]])
        model = FN_DBSCAN(eps=0.5, min_cardinality=3)
        model.fit(X)

        # Single point cannot be core (cardinality = 1.0 < 3)
        assert model.labels_[0] == -1
        assert model.n_clusters_ == 0

    def test_two_identical_points(self):
        """Test with two identical points."""
        X = np.array([[1, 2], [1, 2]])
        model = FN_DBSCAN(eps=0.5, min_cardinality=2)
        model.fit(X)

        # Both points have cardinality 2.0, should form cluster
        assert model.n_clusters_ == 1
        assert model.labels_[0] == model.labels_[1]
        assert model.labels_[0] != -1

    def test_all_noise(self):
        """Test when all points are noise."""
        X = np.array([[1, 2], [10, 20], [100, 200]])
        model = FN_DBSCAN(eps=0.1, min_cardinality=5)
        model.fit(X)

        assert model.n_clusters_ == 0
        assert np.all(model.labels_ == -1)

    def test_single_cluster(self):
        """Test when all points form one cluster."""
        X = np.array([[1, 2], [1.5, 2.5], [2, 2], [2.5, 2.5]])
        model = FN_DBSCAN(eps=2.0, min_cardinality=2)
        model.fit(X)

        # All points should be in same cluster
        assert model.n_clusters_ == 1
        assert len(np.unique(model.labels_)) == 1
        assert model.labels_[0] == 0

    def test_min_cardinality_one(self):
        """Test with min_cardinality=1 (everything is core)."""
        X = np.array([[1, 2], [10, 20], [100, 200]])
        model = FN_DBSCAN(eps=1.0, min_cardinality=1.0)
        model.fit(X)

        # Each point has at least cardinality 1.0 (itself)
        # So all should be core, but may not be connected
        assert len(model.core_sample_indices_) == 3


class TestFNDBSCANNonConvex:
    """Test FN_DBSCAN on non-convex datasets."""

    def test_moons_dataset(self):
        """Test on sklearn's moons dataset."""
        X, y_true = make_moons(n_samples=100, noise=0.1, random_state=42)

        model = FN_DBSCAN(eps=0.3, min_cardinality=2.5)
        model.fit(X)

        # Should find approximately 2 clusters
        assert 1 <= model.n_clusters_ <= 3

    def test_deterministic(self):
        """Test that algorithm is deterministic."""
        X, _ = make_blobs(n_samples=50, random_state=42)

        model1 = FN_DBSCAN(eps=2.0, min_cardinality=3)
        labels1 = model1.fit_predict(X)

        model2 = FN_DBSCAN(eps=2.0, min_cardinality=3)
        labels2 = model2.fit_predict(X)

        np.testing.assert_array_equal(labels1, labels2)


class TestFNDBSCANInputValidation:
    """Test input validation for FN_DBSCAN."""

    def test_invalid_eps(self):
        """Test with invalid eps values."""
        X = np.array([[1, 2], [2, 3]])

        with pytest.raises((ValueError, TypeError)):
            FN_DBSCAN(eps=0).fit(X)

        with pytest.raises((ValueError, TypeError)):
            FN_DBSCAN(eps=-1).fit(X)

        with pytest.raises((ValueError, TypeError)):
            FN_DBSCAN(eps=np.inf).fit(X)

    def test_invalid_min_cardinality(self):
        """Test with invalid min_cardinality values."""
        X = np.array([[1, 2], [2, 3]])

        with pytest.raises((ValueError, TypeError)):
            FN_DBSCAN(min_cardinality=0).fit(X)

        with pytest.raises((ValueError, TypeError)):
            FN_DBSCAN(min_cardinality=-1).fit(X)

    def test_invalid_fuzzy_function(self):
        """Test with invalid fuzzy function."""
        X = np.array([[1, 2], [2, 3]])

        with pytest.raises((ValueError, TypeError)):
            FN_DBSCAN(fuzzy_function='invalid').fit(X)

        with pytest.raises((ValueError, TypeError)):
            FN_DBSCAN(fuzzy_function=123).fit(X)

    def test_nan_in_data(self):
        """Test with NaN in data."""
        X = np.array([[1, 2], [np.nan, 3]])
        model = FN_DBSCAN()

        with pytest.raises(ValueError, match="NaN"):
            model.fit(X)

    def test_inf_in_data(self):
        """Test with Inf in data."""
        X = np.array([[1, 2], [np.inf, 3]])
        model = FN_DBSCAN()

        with pytest.raises(ValueError, match="Inf"):
            model.fit(X)

    def test_1d_data(self):
        """Test with 1D data."""
        X = np.array([1, 2, 3, 4, 5])
        model = FN_DBSCAN()
        # Should reshape to 2D automatically
        model.fit(X)
        assert model.labels_.shape == (5,)


class TestFNDBSCANCore:
    """Test core point detection."""

    def test_core_sample_indices(self):
        """Test that core_sample_indices_ is correct."""
        X = np.array([
            [0, 0], [1, 0], [0, 1],  # Dense region
            [10, 10]  # Isolated point
        ])

        model = FN_DBSCAN(eps=2.0, min_cardinality=2)
        model.fit(X)

        # First 3 points should be core
        assert len(model.core_sample_indices_) >= 1

        # Core samples should not include noise
        for idx in model.core_sample_indices_:
            assert model.labels_[idx] != -1

    def test_noise_point_becomes_border(self):
        """Test that noise points can become border points."""
        # Create scenario where a point is initially noise but
        # gets absorbed by expanding cluster
        X = np.array([
            [0, 0], [1, 0], [2, 0],  # Core points
            [3, 0]  # Border point (not enough neighbors to be core)
        ])

        model = FN_DBSCAN(eps=1.5, min_cardinality=1.6)
        model.fit(X)

        # All points should be in same cluster (no noise)
        # because border point is reachable from core
        assert np.all(model.labels_ >= 0)
