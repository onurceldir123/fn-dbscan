"""Fuzzy membership functions for FN-DBSCAN algorithm.

This module implements various fuzzy membership functions used to calculate
the fuzzy cardinality of neighborhoods in the FN-DBSCAN clustering algorithm.

References
----------
Nasibov, E. N., & Ulutagay, G. (2009). Robustness of density-based clustering
methods with various neighborhood relations. Fuzzy Sets and Systems, 160(24),
3601-3615.
"""

import numpy as np
from typing import Union


def linear_membership(distance: Union[float, np.ndarray], epsilon: float, k: float = 1.0, d_max: float = 1.0) -> Union[float, np.ndarray]:
    """Calculate linear fuzzy membership value.

    The linear membership function provides a simple linear decay from 1.0
    at distance 0 to 0.0 based on the k parameter.

    Formula from paper (Equation 5): μ(d) = max(0, 1 - k·d/d_max)
    where k = d_max / ε (automatically calculated from epsilon)

    Parameters
    ----------
    distance : float or ndarray
        Distance value(s) to calculate membership for.
    epsilon : float
        Maximum neighborhood radius (must be > 0).
    k : float, default=1.0
        Parameter controlling the steepness of the membership function.
    d_max : float, default=1.0
        Maximum distance in the dataset.

    Returns
    -------
    membership : float or ndarray
        Fuzzy membership value(s) in range [0, 1].

    Examples
    --------
    >>> linear_membership(0.0, 1.0, k=1.0, d_max=1.0)
    1.0
    >>> linear_membership(0.5, 1.0, k=1.0, d_max=1.0)
    0.5
    >>> linear_membership(1.0, 1.0, k=1.0, d_max=1.0)
    0.0
    """
    if epsilon <= 0:
        raise ValueError(f"epsilon must be > 0, got {epsilon}")
    if d_max <= 0:
        raise ValueError(f"d_max must be > 0, got {d_max}")

    # Formula: μ(d) = max(0, 1 - k·d/d_max)
    membership = 1.0 - k * (distance / d_max)
    return np.clip(membership, 0.0, 1.0)


def exponential_membership(distance: Union[float, np.ndarray], epsilon: float, k: float = 20.0, d_max: float = 1.0) -> Union[float, np.ndarray]:
    """Calculate exponential fuzzy membership value.

    The exponential membership function provides a smooth exponential decay
    from 1.0 at distance 0 towards 0.0 based on the k parameter.

    Formula from paper (Equation 6): μ(d) = exp(-(k·d/d_max)²)

    Parameters
    ----------
    distance : float or ndarray
        Distance value(s) to calculate membership for.
    epsilon : float
        Maximum neighborhood radius (must be > 0).
    k : float, default=20.0
        Parameter controlling the steepness of the membership function.
        Paper recommends k=20 for best results.
        However, dynamic calculation (k = d_max / eps) is used by default
        for better adaptability across different datasets and eps values.
    d_max : float, default=1.0
        Maximum distance in the dataset.

    Returns
    -------
    membership : float or ndarray
        Fuzzy membership value(s) in range [0, 1].

    Examples
    --------
    >>> exponential_membership(0.0, 1.0, k=1.0, d_max=1.0)
    1.0
    >>> exponential_membership(1.0, 1.0, k=1.0, d_max=1.0) < 0.5
    True
    """
    if epsilon <= 0:
        raise ValueError(f"epsilon must be > 0, got {epsilon}")
    if d_max <= 0:
        raise ValueError(f"d_max must be > 0, got {d_max}")

    # Formula: μ(d) = exp(-(k·d/d_max)²)
    membership = np.exp(-((k * distance / d_max) ** 2))

    return membership


def trapezoidal_membership(distance: Union[float, np.ndarray], epsilon: float, k: float = 1.0, d_max: float = 1.0) -> Union[float, np.ndarray]:
    """Calculate trapezoidal fuzzy membership value.

    The trapezoidal membership function has a plateau region with full
    membership (1.0) for distances up to ε/2, then linearly decays to 0.0
    at distance epsilon.

    Formula (from paper):
        μ(d) = 1.0,           if d ≤ ε/2
        μ(d) = 2(1 - d/ε),    if ε/2 < d ≤ ε
        μ(d) = 0.0,           if d > ε

    Parameters
    ----------
    distance : float or ndarray
        Distance value(s) to calculate membership for.
    epsilon : float
        Maximum neighborhood radius (must be > 0).
    k : float, default=1.0
        Parameter controlling the steepness (not used in trapezoidal).
    d_max : float, default=1.0
        Maximum distance in the dataset (not used in trapezoidal).

    Returns
    -------
    membership : float or ndarray
        Fuzzy membership value(s) in range [0, 1].

    Examples
    --------
    >>> trapezoidal_membership(0.0, 1.0)
    1.0
    >>> trapezoidal_membership(0.25, 1.0)
    1.0
    >>> trapezoidal_membership(0.75, 1.0)
    0.5
    >>> trapezoidal_membership(1.0, 1.0)
    0.0
    """
    if epsilon <= 0:
        raise ValueError(f"epsilon must be > 0, got {epsilon}")

    half_epsilon = epsilon / 2.0

    if isinstance(distance, np.ndarray):
        membership = np.ones_like(distance, dtype=float)

        # Linear decay region: ε/2 < d ≤ ε
        mask_decay = (distance > half_epsilon) & (distance <= epsilon)
        membership[mask_decay] = 2.0 * (1.0 - distance[mask_decay] / epsilon)

        # Zero region: d > ε
        mask_zero = distance > epsilon
        membership[mask_zero] = 0.0
    else:
        if distance <= half_epsilon:
            membership = 1.0
        elif distance <= epsilon:
            membership = 2.0 * (1.0 - distance / epsilon)
        else:
            membership = 0.0

    return np.clip(membership, 0.0, 1.0)


def get_fuzzy_membership_function(function_type: str):
    """Get the fuzzy membership function by name.

    Parameters
    ----------
    function_type : str
        Type of fuzzy function. Must be one of:
        - 'linear': Linear decay function
        - 'exponential': Exponential decay function
        - 'trapezoidal': Trapezoidal function with plateau

    Returns
    -------
    function : callable
        The fuzzy membership function.

    Raises
    ------
    ValueError
        If function_type is not recognized.

    Examples
    --------
    >>> func = get_fuzzy_membership_function('linear')
    >>> func(0.5, 1.0)
    0.5
    """
    functions = {
        'linear': linear_membership,
        'exponential': exponential_membership,
        'trapezoidal': trapezoidal_membership,
    }

    if function_type not in functions:
        raise ValueError(
            f"Unknown fuzzy function type: '{function_type}'. "
            f"Must be one of: {list(functions.keys())}"
        )

    return functions[function_type]


def calculate_fuzzy_cardinality(
    distances: np.ndarray,
    epsilon: float,
    fuzzy_function: str = 'linear'
) -> float:
    """Calculate the fuzzy cardinality for a set of distances.

    The fuzzy cardinality is the sum of fuzzy membership values for all
    neighbors within epsilon distance.

    Parameters
    ----------
    distances : ndarray
        Array of distances from a point to its neighbors.
    epsilon : float
        Maximum neighborhood radius (must be > 0).
    fuzzy_function : str, default='linear'
        Type of fuzzy membership function to use.

    Returns
    -------
    cardinality : float
        The fuzzy cardinality (sum of memberships).

    Examples
    --------
    >>> distances = np.array([0.0, 0.5, 1.0, 1.5])
    >>> calculate_fuzzy_cardinality(distances, epsilon=1.0, fuzzy_function='linear')
    1.5
    """
    membership_func = get_fuzzy_membership_function(fuzzy_function)
    memberships = membership_func(distances, epsilon)
    return float(np.sum(memberships))
