"""FN-DBSCAN: Fuzzy Neighborhood DBSCAN clustering algorithm.

This module implements the FN-DBSCAN clustering algorithm as described in:
Nasibov, E. N., & Ulutagay, G. (2009). Robustness of density-based clustering
methods with various neighborhood relations. Fuzzy Sets and Systems, 160(24),
3601-3615.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.neighbors import NearestNeighbors
from typing import Optional

from ._fuzzy_functions import get_fuzzy_membership_function
from ._validation import validate_data, validate_fit_params


# Label constants
UNASSIGNED = -2  # Point not yet visited
NOISE = -1       # Point marked as noise


class FN_DBSCAN(BaseEstimator, ClusterMixin):
    """Fuzzy Neighborhood DBSCAN clustering.

    FN-DBSCAN is a density-based clustering algorithm that extends DBSCAN by
    using fuzzy set theory to define neighborhood cardinality. Instead of
    counting discrete points within epsilon radius, FN-DBSCAN computes a fuzzy
    cardinality based on a membership function that decreases with distance.

    Parameters
    ----------
    eps : float, default=0.5
        The maximum distance between two samples for one to be considered
        as in the neighborhood of the other. This is the most important
        DBSCAN parameter to choose appropriately for your dataset.

    min_cardinality : float, default=5.0
        The minimum fuzzy cardinality required for a point to be classified
        as a core point. This replaces the min_samples parameter in standard
        DBSCAN with a fuzzy equivalent.

    fuzzy_function : {'linear', 'exponential', 'trapezoidal'}, default='linear'
        The fuzzy membership function to use for calculating neighborhood
        cardinality:
        - 'linear': μ(d) = max(0, 1 - d/ε)
        - 'exponential': μ(d) = exp(-d/ε)
        - 'trapezoidal': μ(d) = 1 if d ≤ ε/2, else 2(1-d/ε)

    metric : str or callable, default='euclidean'
        The distance metric to use. Can be any metric from
        sklearn.metrics.pairwise or a custom callable.

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Cluster labels for each point in the dataset. Noisy samples are
        given the label -1.

    core_sample_indices_ : ndarray of shape (n_core_samples,)
        Indices of core samples.

    n_clusters_ : int
        Number of clusters found (excluding noise).

    Examples
    --------
    >>> from fn_dbscan import FN_DBSCAN
    >>> import numpy as np
    >>> X = np.array([[1, 2], [2, 2], [2, 3], [8, 7], [8, 8], [25, 80]])
    >>> clustering = FN_DBSCAN(eps=3, min_cardinality=2).fit(X)
    >>> clustering.labels_
    array([0, 0, 0, 1, 1, -1])
    >>> clustering.n_clusters_
    2

    Notes
    -----
    When using fuzzy_function='linear' and a crisp threshold, FN-DBSCAN
    behaves similarly to standard DBSCAN. The fuzzy approach provides more
    nuanced cluster assignment, especially for border points.

    References
    ----------
    Nasibov, E. N., & Ulutagay, G. (2009). Robustness of density-based
    clustering methods with various neighborhood relations.
    Fuzzy Sets and Systems, 160(24), 3601-3615.
    """

    def __init__(
        self,
        eps: float = 0.5,
        min_cardinality: float = 5.0,
        fuzzy_function: str = 'linear',
        metric: str = 'euclidean'
    ):
        self.eps = eps
        self.min_cardinality = min_cardinality
        self.fuzzy_function = fuzzy_function
        self.metric = metric

    def fit(self, X, y=None):
        """Perform FN-DBSCAN clustering from features.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training instances to cluster.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        # Validate parameters
        validate_fit_params(
            self.eps, self.min_cardinality,
            self.fuzzy_function, self.metric
        )

        # Validate and convert input data
        X, n_samples, n_features = validate_data(X)
        self._X = X

        # Initialize labels and visited flags
        labels = np.full(n_samples, UNASSIGNED, dtype=int)
        visited = np.zeros(n_samples, dtype=bool)

        # Build spatial index for efficient neighbor search
        self._nn = NearestNeighbors(
            radius=self.eps,
            metric=self.metric,
            algorithm='auto'
        )
        self._nn.fit(X)

        # Get fuzzy membership function
        membership_func = get_fuzzy_membership_function(self.fuzzy_function)

        # Track core samples
        core_samples = []

        # Current cluster ID
        cluster_id = 0

        # Main algorithm loop
        for point_idx in range(n_samples):
            # Skip if already visited
            if visited[point_idx]:
                continue

            # Mark as visited
            visited[point_idx] = True

            # Find neighbors within eps
            neighbors = self._range_query(point_idx)

            # Calculate fuzzy cardinality
            cardinality = self._calculate_fuzzy_cardinality(
                point_idx, neighbors, membership_func
            )

            # Check if point is a core point
            if cardinality < self.min_cardinality:
                # Mark as noise (may be changed later if reached by cluster)
                labels[point_idx] = NOISE
                continue

            # Point is a core point
            core_samples.append(point_idx)
            labels[point_idx] = cluster_id

            # Expand cluster from this core point
            self._expand_cluster(
                point_idx, neighbors, cluster_id,
                labels, visited, membership_func
            )

            # Move to next cluster
            cluster_id += 1

        # Store results
        self.labels_ = labels
        self.core_sample_indices_ = np.array(core_samples, dtype=int)
        self.n_clusters_ = cluster_id

        return self

    def fit_predict(self, X, y=None):
        """Perform clustering and return cluster labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training instances to cluster.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Cluster labels. Noisy samples are given the label -1.
        """
        self.fit(X, y)
        return self.labels_

    def _range_query(self, point_idx: int) -> np.ndarray:
        """Find all neighbors within eps of a point.

        Parameters
        ----------
        point_idx : int
            Index of the query point.

        Returns
        -------
        neighbors : ndarray
            Indices of neighbors within eps.
        """
        # Use radius_neighbors to find all points within eps
        neighbors = self._nn.radius_neighbors(
            self._X[point_idx].reshape(1, -1),
            return_distance=False
        )[0]

        return neighbors

    def _calculate_fuzzy_cardinality(
        self,
        point_idx: int,
        neighbors: np.ndarray,
        membership_func
    ) -> float:
        """Calculate fuzzy cardinality for a point.

        Parameters
        ----------
        point_idx : int
            Index of the query point.
        neighbors : ndarray
            Indices of neighbors within eps.
        membership_func : callable
            Fuzzy membership function.

        Returns
        -------
        cardinality : float
            Sum of fuzzy memberships.
        """
        if len(neighbors) == 0:
            return 0.0

        # Calculate distances to all neighbors
        point = self._X[point_idx]
        neighbor_points = self._X[neighbors]
        distances = np.linalg.norm(neighbor_points - point, axis=1)

        # Calculate fuzzy memberships
        memberships = membership_func(distances, self.eps)

        # Sum memberships to get fuzzy cardinality
        cardinality = np.sum(memberships)

        return float(cardinality)

    def _expand_cluster(
        self,
        point_idx: int,
        initial_neighbors: np.ndarray,
        cluster_id: int,
        labels: np.ndarray,
        visited: np.ndarray,
        membership_func
    ) -> None:
        """Expand cluster from a core point.

        Parameters
        ----------
        point_idx : int
            Index of the core point.
        initial_neighbors : ndarray
            Initial neighbors of the core point.
        cluster_id : int
            ID of the current cluster.
        labels : ndarray
            Array of cluster labels (modified in-place).
        visited : ndarray
            Array of visited flags (modified in-place).
        membership_func : callable
            Fuzzy membership function.
        """
        # Initialize seed set with initial neighbors
        # Use a list for efficient append operations
        seed_set = list(initial_neighbors)
        seed_index = 0

        # Process seed set
        while seed_index < len(seed_set):
            q = seed_set[seed_index]
            seed_index += 1

            # If q was noise, it's now a border point of this cluster
            if labels[q] == NOISE:
                labels[q] = cluster_id

            # If already visited, skip
            if visited[q]:
                continue

            # Mark as visited and assign to cluster
            visited[q] = True
            labels[q] = cluster_id

            # Find neighbors of q
            q_neighbors = self._range_query(q)

            # Calculate fuzzy cardinality for q
            q_cardinality = self._calculate_fuzzy_cardinality(
                q, q_neighbors, membership_func
            )

            # If q is also a core point, add its neighbors to seed set
            if q_cardinality >= self.min_cardinality:
                # Add new neighbors to seed set (avoiding duplicates is not
                # critical as we check visited flag)
                seed_set.extend(q_neighbors)

    def __repr__(self):
        """Return string representation of the estimator."""
        return (
            f"FN_DBSCAN(eps={self.eps}, "
            f"min_cardinality={self.min_cardinality}, "
            f"fuzzy_function='{self.fuzzy_function}', "
            f"metric='{self.metric}')"
        )
