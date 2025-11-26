"""Simple FN-DBSCAN clustering example with synthetic data.

This example demonstrates the basic usage of FN-DBSCAN on a simple
synthetic dataset with three well-separated clusters.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from fn_dbscan import FN_DBSCAN


def main():
    """Run basic FN-DBSCAN clustering example."""
    print("FN-DBSCAN Basic Example")
    print("=" * 60)

    # Create simple synthetic data with 3 clusters
    np.random.seed(42)

    # Cluster 1: around (0, 0)
    cluster1 = np.random.randn(30, 2) * 0.5 + [0, 0]

    # Cluster 2: around (5, 0)
    cluster2 = np.random.randn(30, 2) * 0.5 + [5, 0]

    # Cluster 3: around (2.5, 4)
    cluster3 = np.random.randn(30, 2) * 0.5 + [2.5, 4]

    # Add some noise points
    noise = np.random.randn(5, 2) * 2 + [2.5, 2]

    # Combine all data
    X = np.vstack([cluster1, cluster2, cluster3, noise])

    print(f"\nDataset created:")
    print(f"  Total samples: {len(X)}")
    print(f"  Features: {X.shape[1]}")
    print(f"  True clusters: 3")
    print(f"  Noise points: 5")

    # Create and fit FN-DBSCAN model
    print(f"\nFitting FN-DBSCAN...")
    model = FN_DBSCAN(
        eps=0.15,
        min_membership=0.0,
        min_fuzzy_neighbors=5,
        fuzzy_function='exponential',
        normalize=True
    )

    labels = model.fit_predict(X)

    # Print results
    print(f"\nClustering Results:")
    print(f"  Clusters found: {model.n_clusters_}")
    print(f"  Core samples: {len(model.core_sample_indices_)}")
    print(f"  Noise points: {np.sum(labels == -1)}")

    # Cluster distribution
    print(f"\nCluster distribution:")
    for i in range(model.n_clusters_):
        count = np.sum(labels == i)
        print(f"  Cluster {i}: {count} points")
    if np.any(labels == -1):
        print(f"  Noise: {np.sum(labels == -1)} points")

    # Visualize results
    print(f"\nGenerating visualization...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot original data
    ax1.scatter(X[:, 0], X[:, 1], c='gray', alpha=0.6, s=50)
    ax1.set_title('Original Data', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Feature 1')
    ax1.set_ylabel('Feature 2')
    ax1.grid(True, alpha=0.3)

    # Plot clustering results
    unique_labels = np.unique(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

    for label, color in zip(unique_labels, colors):
        if label == -1:
            # Noise points
            mask = (labels == label)
            ax2.scatter(X[mask, 0], X[mask, 1], c='black', marker='x',
                       s=100, alpha=0.5, label='Noise')
        else:
            # Cluster points
            mask = (labels == label)
            ax2.scatter(X[mask, 0], X[mask, 1], c=[color], s=50,
                       alpha=0.8, label=f'Cluster {label}')

    # Mark core samples
    core_samples = X[model.core_sample_indices_]
    ax2.scatter(core_samples[:, 0], core_samples[:, 1],
               facecolors='none', edgecolors='red', s=100,
               linewidths=2, label='Core samples')

    ax2.set_title(f'FN-DBSCAN Results ({model.n_clusters_} clusters)',
                 fontsize=12, fontweight='bold')
    ax2.set_xlabel('Feature 1')
    ax2.set_ylabel('Feature 2')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('examples/clustering_example.png', dpi=150, bbox_inches='tight')
    print(f"Visualization saved: examples/clustering_example.png")
    plt.show()

    print(f"\n{'=' * 60}")
    print("Example completed successfully!")


if __name__ == "__main__":
    main()
