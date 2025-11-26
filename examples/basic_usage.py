"""Basic usage example for FN-DBSCAN.

This example demonstrates the basic usage of the FN-DBSCAN clustering algorithm
on a simple synthetic dataset.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

from fn_dbscan import FN_DBSCAN


def main():
    """Run basic FN-DBSCAN example."""
    print("FN-DBSCAN Basic Usage Example")
    print("=" * 50)

    # Generate synthetic data with 3 clusters
    X, y_true = make_blobs(
        n_samples=300,
        centers=3,
        n_features=2,
        cluster_std=0.5,
        random_state=42
    )

    print(f"\nGenerated dataset with {len(X)} samples")
    print(f"True number of clusters: 3")

    # Create and fit FN-DBSCAN model
    model = FN_DBSCAN(
        eps=0.7,
        min_fuzzy_neighbors=5,
        fuzzy_function='linear',
        metric='euclidean'
    )

    print(f"\nFitting FN-DBSCAN with parameters:")
    print(f"  eps: {model.eps}")
    print(f"  min_fuzzy_neighbors: {model.min_fuzzy_neighbors}")
    print(f"  fuzzy_function: {model.fuzzy_function}")

    labels = model.fit_predict(X)

    # Print results
    print(f"\nClustering Results:")
    print(f"  Number of clusters found: {model.n_clusters_}")
    print(f"  Number of core samples: {len(model.core_sample_indices_)}")
    print(f"  Number of noise points: {np.sum(labels == -1)}")

    # Calculate cluster sizes
    unique_labels = np.unique(labels)
    print(f"\nCluster sizes:")
    for label in unique_labels:
        if label == -1:
            print(f"  Noise: {np.sum(labels == label)} points")
        else:
            print(f"  Cluster {label}: {np.sum(labels == label)} points")

    # Visualize results
    try:
        plt.figure(figsize=(12, 5))

        # Plot original data with true labels
        plt.subplot(1, 2, 1)
        plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', s=50, alpha=0.6)
        plt.title('Original Data (True Labels)')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.colorbar(label='True Cluster')

        # Plot clustered data
        plt.subplot(1, 2, 2)
        # Use different colors for clusters and noise
        colors = plt.cm.Spectral(np.linspace(0, 1, model.n_clusters_))

        for k, col in zip(range(model.n_clusters_), colors):
            class_member_mask = (labels == k)
            xy = X[class_member_mask]
            plt.scatter(xy[:, 0], xy[:, 1], c=[col], s=50, alpha=0.6,
                       label=f'Cluster {k}')

        # Plot noise points
        noise_mask = (labels == -1)
        if np.any(noise_mask):
            xy_noise = X[noise_mask]
            plt.scatter(xy_noise[:, 0], xy_noise[:, 1], c='black', s=50,
                       alpha=0.3, marker='x', label='Noise')

        # Mark core samples
        core_samples = X[model.core_sample_indices_]
        plt.scatter(core_samples[:, 0], core_samples[:, 1], c='red',
                   s=100, alpha=0.3, marker='o', linewidths=0,
                   label='Core samples')

        plt.title(f'FN-DBSCAN Clustering ({model.n_clusters_} clusters)')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()

        plt.tight_layout()
        plt.savefig('basic_usage_example.png', dpi=150, bbox_inches='tight')
        print("\nVisualization saved as 'basic_usage_example.png'")
        plt.show()

    except ImportError:
        print("\nMatplotlib not available, skipping visualization")


if __name__ == "__main__":
    main()
