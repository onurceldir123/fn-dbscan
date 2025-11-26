"""Compare FN-DBSCAN with standard DBSCAN.

This example compares the clustering results of FN-DBSCAN with standard
DBSCAN on the same dataset, highlighting the differences introduced by
fuzzy neighborhood cardinality.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN

from fn_dbscan import FN_DBSCAN


def main():
    """Compare FN-DBSCAN with DBSCAN."""
    print("FN-DBSCAN vs DBSCAN Comparison")
    print("=" * 50)

    # Generate non-convex dataset (moons)
    X, y_true = make_moons(n_samples=300, noise=0.05, random_state=42)

    print(f"\nGenerated 'moons' dataset with {len(X)} samples")

    # Parameters
    eps = 0.2
    min_samples = 5

    # Standard DBSCAN
    print(f"\nRunning standard DBSCAN...")
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels_dbscan = dbscan.fit_predict(X)

    n_clusters_dbscan = len(set(labels_dbscan)) - (1 if -1 in labels_dbscan else 0)
    n_noise_dbscan = list(labels_dbscan).count(-1)

    print(f"  Clusters found: {n_clusters_dbscan}")
    print(f"  Noise points: {n_noise_dbscan}")

    # FN-DBSCAN with different fuzzy functions
    fuzzy_functions = ['linear', 'exponential', 'trapezoidal']
    results = {}

    for fuzzy_func in fuzzy_functions:
        print(f"\nRunning FN-DBSCAN with {fuzzy_func} membership...")
        fn_dbscan = FN_DBSCAN(
            eps=eps,
            min_fuzzy_neighbors=min_samples,
            fuzzy_function=fuzzy_func
        )
        labels_fn = fn_dbscan.fit_predict(X)

        results[fuzzy_func] = {
            'labels': labels_fn,
            'n_clusters': fn_dbscan.n_clusters_,
            'n_noise': np.sum(labels_fn == -1),
            'n_core': len(fn_dbscan.core_sample_indices_)
        }

        print(f"  Clusters found: {results[fuzzy_func]['n_clusters']}")
        print(f"  Noise points: {results[fuzzy_func]['n_noise']}")
        print(f"  Core samples: {results[fuzzy_func]['n_core']}")

    # Visualize results
    try:
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # Plot standard DBSCAN
        ax = axes[0, 0]
        plot_clustering(X, labels_dbscan, ax, 'Standard DBSCAN')

        # Plot FN-DBSCAN variants
        for idx, fuzzy_func in enumerate(fuzzy_functions):
            row = (idx + 1) // 2
            col = (idx + 1) % 2
            ax = axes[row, col]

            labels = results[fuzzy_func]['labels']
            title = f'FN-DBSCAN ({fuzzy_func})'
            plot_clustering(X, labels, ax, title)

        plt.tight_layout()
        plt.savefig('comparison_dbscan.png', dpi=150, bbox_inches='tight')
        print("\nVisualization saved as 'comparison_dbscan.png'")
        plt.show()

    except ImportError:
        print("\nMatplotlib not available, skipping visualization")

    # Print comparison summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"{'Method':<25} {'Clusters':<10} {'Noise':<10} {'Core':<10}")
    print("-" * 50)
    print(f"{'DBSCAN':<25} {n_clusters_dbscan:<10} {n_noise_dbscan:<10} {'N/A':<10}")
    for fuzzy_func in fuzzy_functions:
        r = results[fuzzy_func]
        print(f"{'FN-DBSCAN (' + fuzzy_func + ')':<25} "
              f"{r['n_clusters']:<10} {r['n_noise']:<10} {r['n_core']:<10}")


def plot_clustering(X, labels, ax, title):
    """Plot clustering results."""
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)

    colors = plt.cm.Spectral(np.linspace(0, 1, max(n_clusters, 1)))

    for k, col in zip(range(n_clusters), colors):
        class_member_mask = (labels == k)
        xy = X[class_member_mask]
        ax.scatter(xy[:, 0], xy[:, 1], c=[col], s=30, alpha=0.7)

    # Plot noise
    if -1 in labels:
        noise_mask = (labels == -1)
        xy_noise = X[noise_mask]
        ax.scatter(xy_noise[:, 0], xy_noise[:, 1], c='black',
                  s=30, alpha=0.3, marker='x')

    ax.set_title(title)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')


if __name__ == "__main__":
    main()
