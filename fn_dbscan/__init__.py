"""FN-DBSCAN: Fuzzy Neighborhood DBSCAN clustering algorithm.

FN-DBSCAN is a density-based clustering algorithm that extends DBSCAN by
using fuzzy set theory to define neighborhood cardinality.

Classes
-------
FN_DBSCAN
    Main clustering class implementing the FN-DBSCAN algorithm.

Examples
--------
>>> from fn_dbscan import FN_DBSCAN
>>> import numpy as np
>>> X = np.array([[1, 2], [2, 2], [2, 3], [8, 7], [8, 8], [25, 80]])
>>> clustering = FN_DBSCAN(eps=3, min_cardinality=2).fit(X)
>>> clustering.labels_
array([0, 0, 0, 1, 1, -1])

References
----------
Nasibov, E. N., & Ulutagay, G. (2009). Robustness of density-based clustering
methods with various neighborhood relations. Fuzzy Sets and Systems, 160(24),
3601-3615.
"""

from .fn_dbscan import FN_DBSCAN
from ._fuzzy_functions import (
    linear_membership,
    exponential_membership,
    trapezoidal_membership,
    get_fuzzy_membership_function,
    calculate_fuzzy_cardinality,
)

__version__ = "0.1.0"

__all__ = [
    "FN_DBSCAN",
    "linear_membership",
    "exponential_membership",
    "trapezoidal_membership",
    "get_fuzzy_membership_function",
    "calculate_fuzzy_cardinality",
]
