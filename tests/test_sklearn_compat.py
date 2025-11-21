"""Tests for sklearn API compatibility."""

import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils.estimator_checks import parametrize_with_checks
from sklearn.datasets import make_blobs
import pandas as pd

from fn_dbscan import FN_DBSCAN


class TestSklearnCompatibility:
    """Test sklearn API compatibility."""

    def test_inherits_base_estimator(self):
        """Test that FN_DBSCAN inherits from BaseEstimator."""
        model = FN_DBSCAN()
        assert isinstance(model, BaseEstimator)

    def test_inherits_cluster_mixin(self):
        """Test that FN_DBSCAN inherits from ClusterMixin."""
        model = FN_DBSCAN()
        assert isinstance(model, ClusterMixin)

    def test_get_params(self):
        """Test get_params method from BaseEstimator."""
        model = FN_DBSCAN(eps=0.7, min_cardinality=4, fuzzy_function='exponential')
        params = model.get_params()

        assert params['eps'] == 0.7
        assert params['min_cardinality'] == 4
        assert params['fuzzy_function'] == 'exponential'
        assert 'metric' in params

    def test_set_params(self):
        """Test set_params method from BaseEstimator."""
        model = FN_DBSCAN()

        model.set_params(eps=1.5, min_cardinality=10)

        assert model.eps == 1.5
        assert model.min_cardinality == 10

    def test_set_params_returns_self(self):
        """Test that set_params returns self."""
        model = FN_DBSCAN()
        result = model.set_params(eps=0.5)
        assert result is model

    def test_clone_compatible(self):
        """Test compatibility with sklearn.clone."""
        from sklearn.base import clone

        model1 = FN_DBSCAN(eps=0.7, min_cardinality=4)
        model2 = clone(model1)

        assert model2.eps == model1.eps
        assert model2.min_cardinality == model1.min_cardinality
        assert model2 is not model1

    def test_fit_method_signature(self):
        """Test that fit has correct signature."""
        import inspect

        sig = inspect.signature(FN_DBSCAN.fit)
        params = list(sig.parameters.keys())

        assert 'self' in params
        assert 'X' in params
        assert 'y' in params

    def test_fit_predict_method_signature(self):
        """Test that fit_predict has correct signature."""
        import inspect

        sig = inspect.signature(FN_DBSCAN.fit_predict)
        params = list(sig.parameters.keys())

        assert 'self' in params
        assert 'X' in params
        assert 'y' in params

    def test_labels_attribute(self):
        """Test labels_ attribute exists after fit."""
        X = np.array([[1, 2], [2, 2], [2, 3]])
        model = FN_DBSCAN()
        model.fit(X)

        assert hasattr(model, 'labels_')
        assert isinstance(model.labels_, np.ndarray)
        assert model.labels_.shape == (3,)

    def test_no_labels_before_fit(self):
        """Test that labels_ doesn't exist before fit."""
        model = FN_DBSCAN()
        assert not hasattr(model, 'labels_')


class TestDataFormats:
    """Test different input data formats."""

    def test_numpy_array_input(self):
        """Test with numpy array input."""
        X = np.array([[1, 2], [2, 2], [2, 3]])
        model = FN_DBSCAN()
        model.fit(X)

        assert model.labels_.shape == (3,)

    def test_list_input(self):
        """Test with Python list input."""
        X = [[1, 2], [2, 2], [2, 3]]
        model = FN_DBSCAN()
        model.fit(X)

        assert model.labels_.shape == (3,)

    def test_pandas_dataframe_input(self):
        """Test with pandas DataFrame input."""
        X = pd.DataFrame({
            'feature1': [1, 2, 2],
            'feature2': [2, 2, 3]
        })
        model = FN_DBSCAN()
        model.fit(X)

        assert model.labels_.shape == (3,)

    def test_pandas_series_fails(self):
        """Test that pandas Series input is handled."""
        X = pd.Series([1, 2, 3, 4, 5])
        model = FN_DBSCAN()

        # Should either work (reshape to 2D) or fail gracefully
        try:
            model.fit(X)
            assert model.labels_.shape == (5,)
        except ValueError:
            pass  # Also acceptable

    def test_float32_input(self):
        """Test with float32 input."""
        X = np.array([[1, 2], [2, 2], [2, 3]], dtype=np.float32)
        model = FN_DBSCAN()
        model.fit(X)

        assert model.labels_.shape == (3,)

    def test_int_input(self):
        """Test with integer input."""
        X = np.array([[1, 2], [2, 2], [2, 3]], dtype=int)
        model = FN_DBSCAN()
        model.fit(X)

        assert model.labels_.shape == (3,)


class TestMetrics:
    """Test different distance metrics."""

    def test_euclidean_metric(self):
        """Test with Euclidean metric."""
        X, _ = make_blobs(n_samples=50, random_state=42)
        model = FN_DBSCAN(metric='euclidean')
        model.fit(X)

        assert model.n_clusters_ >= 0

    def test_manhattan_metric(self):
        """Test with Manhattan metric."""
        X, _ = make_blobs(n_samples=50, random_state=42)
        model = FN_DBSCAN(metric='manhattan')
        model.fit(X)

        assert model.n_clusters_ >= 0

    def test_chebyshev_metric(self):
        """Test with Chebyshev metric."""
        X, _ = make_blobs(n_samples=50, random_state=42)
        model = FN_DBSCAN(metric='chebyshev')
        model.fit(X)

        assert model.n_clusters_ >= 0


class TestReproducibility:
    """Test reproducibility of results."""

    def test_multiple_fits_same_result(self):
        """Test that multiple fits give same result."""
        X, _ = make_blobs(n_samples=50, random_state=42)

        model = FN_DBSCAN(eps=2.0, min_cardinality=3)

        labels1 = model.fit_predict(X)
        labels2 = model.fit_predict(X)

        np.testing.assert_array_equal(labels1, labels2)

    def test_same_params_same_result(self):
        """Test that same parameters give same result."""
        X, _ = make_blobs(n_samples=50, random_state=42)

        model1 = FN_DBSCAN(eps=2.0, min_cardinality=3, fuzzy_function='linear')
        labels1 = model1.fit_predict(X)

        model2 = FN_DBSCAN(eps=2.0, min_cardinality=3, fuzzy_function='linear')
        labels2 = model2.fit_predict(X)

        np.testing.assert_array_equal(labels1, labels2)
