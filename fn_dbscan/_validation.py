"""Input validation utilities for FN-DBSCAN algorithm.

This module provides functions to validate input parameters and data
for the FN-DBSCAN clustering algorithm.
"""

import numpy as np
from typing import Union, Tuple


def validate_eps(eps: float) -> None:
    """Validate epsilon parameter.

    Parameters
    ----------
    eps : float
        Epsilon (maximum neighborhood radius).

    Raises
    ------
    ValueError
        If eps is not positive.
    TypeError
        If eps is not a number.
    """
    if not isinstance(eps, (int, float, np.number)):
        raise TypeError(f"eps must be a number, got {type(eps)}")

    if not np.isfinite(eps):
        raise ValueError(f"eps must be finite, got {eps}")

    if eps <= 0:
        raise ValueError(f"eps must be > 0, got {eps}")


def validate_min_cardinality(min_cardinality: float) -> None:
    """Validate minimum cardinality parameter.

    Parameters
    ----------
    min_cardinality : float
        Minimum fuzzy cardinality for core points.

    Raises
    ------
    ValueError
        If min_cardinality is not positive.
    TypeError
        If min_cardinality is not a number.
    """
    if not isinstance(min_cardinality, (int, float, np.number)):
        raise TypeError(
            f"min_cardinality must be a number, got {type(min_cardinality)}"
        )

    if not np.isfinite(min_cardinality):
        raise ValueError(f"min_cardinality must be finite, got {min_cardinality}")

    if min_cardinality <= 0:
        raise ValueError(f"min_cardinality must be > 0, got {min_cardinality}")


def validate_fuzzy_function(fuzzy_function: str) -> None:
    """Validate fuzzy function parameter.

    Parameters
    ----------
    fuzzy_function : str
        Type of fuzzy membership function.

    Raises
    ------
    ValueError
        If fuzzy_function is not recognized.
    TypeError
        If fuzzy_function is not a string.
    """
    if not isinstance(fuzzy_function, str):
        raise TypeError(
            f"fuzzy_function must be a string, got {type(fuzzy_function)}"
        )

    valid_functions = {'linear', 'exponential', 'trapezoidal'}
    if fuzzy_function not in valid_functions:
        raise ValueError(
            f"fuzzy_function must be one of {valid_functions}, "
            f"got '{fuzzy_function}'"
        )


def validate_metric(metric: str) -> None:
    """Validate distance metric parameter.

    Parameters
    ----------
    metric : str
        Distance metric name.

    Raises
    ------
    TypeError
        If metric is not a string or callable.

    Notes
    -----
    This function performs basic validation. The actual metric validation
    is delegated to sklearn's NearestNeighbors class.
    """
    if not isinstance(metric, (str, callable)):
        raise TypeError(
            f"metric must be a string or callable, got {type(metric)}"
        )


def validate_data(X: np.ndarray) -> Tuple[np.ndarray, int, int]:
    """Validate input data array.

    Parameters
    ----------
    X : array-like
        Input data array.

    Returns
    -------
    X_validated : ndarray
        Validated and converted data array.
    n_samples : int
        Number of samples.
    n_features : int
        Number of features.

    Raises
    ------
    ValueError
        If X contains NaN or Inf values.
        If X is not 2D.
        If X has no samples.
    TypeError
        If X cannot be converted to ndarray.
    """
    # Convert to numpy array if needed
    try:
        X = np.asarray(X, dtype=float)
    except (ValueError, TypeError) as e:
        raise TypeError(f"X must be convertible to numpy array: {e}")

    # Check dimensionality
    if X.ndim == 1:
        # Single sample or single feature - reshape
        X = X.reshape(-1, 1) if len(X) > 1 else X.reshape(1, -1)

    if X.ndim != 2:
        raise ValueError(
            f"X must be a 2D array, got {X.ndim}D array with shape {X.shape}"
        )

    n_samples, n_features = X.shape

    # Check for empty data
    if n_samples == 0:
        raise ValueError("X has 0 samples")

    if n_features == 0:
        raise ValueError("X has 0 features")

    # Check for NaN/Inf
    if not np.all(np.isfinite(X)):
        raise ValueError("X contains NaN or Inf values")

    return X, n_samples, n_features


def validate_fit_params(eps: float, min_cardinality: float,
                        fuzzy_function: str, metric: str) -> None:
    """Validate all fit parameters.

    Parameters
    ----------
    eps : float
        Epsilon (maximum neighborhood radius).
    min_cardinality : float
        Minimum fuzzy cardinality for core points.
    fuzzy_function : str
        Type of fuzzy membership function.
    metric : str
        Distance metric name.

    Raises
    ------
    ValueError
        If any parameter is invalid.
    TypeError
        If any parameter has wrong type.
    """
    validate_eps(eps)
    validate_min_cardinality(min_cardinality)
    validate_fuzzy_function(fuzzy_function)
    validate_metric(metric)
