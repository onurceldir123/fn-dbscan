import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

# Add parent directory to path to import fn_dbscan
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fn_dbscan import FN_DBSCAN

def load_data(filepath):
    """Load dataset from CSV."""
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        return None
    
    df = pd.read_csv(filepath)
    return df[['x', 'y']].values, df['label'].values

def plot_clustering(X, labels, ax, title):
    """Plot clustering results."""
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    
    # Colors for clusters
    colors = plt.cm.Spectral(np.linspace(0, 1, max(n_clusters, 1)))
    
    # Plot noise first (so it's in background)
    if -1 in labels:
        noise_mask = (labels == -1)
        ax.scatter(X[noise_mask, 0], X[noise_mask, 1], c='black', s=10, alpha=0.3, marker='x', label='Noise')

    # Plot clusters
    for k, col in zip(range(n_clusters), colors):
        class_member_mask = (labels == k)
        ax.scatter(X[class_member_mask, 0], X[class_member_mask, 1], c=[col], s=20, alpha=0.8, label=f'Cluster {k}')
        
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])

def main():
    # Path to dataset
    dataset_path = os.path.join(os.path.dirname(__file__), '../data/pathbased_dataset.csv')
    
    print(f"Loading dataset from {dataset_path}...")
    data = load_data(dataset_path)
    if data is None:
        return
        
    X, y_true = data
    print(f"Dataset shape: {X.shape}")

    # User-specified configuration
    # FN-DBSCAN'i Klasik DBSCAN Gibi Çalıştırma Ayarları
    config = {
        'name': 'Simulation (Linear, k=1.0, eps1=0.5)',
        'params': {
            'epsilon1': 0.5,        # Klasik DBSCAN'deki 'eps' (Yarıçap) değeri gibi davranması hedefleniyor
            'epsilon2': 4.0,        # Klasik DBSCAN'deki 'MinPts'
            'fuzzy_function': 'linear', 
            'k': 1.0,               # Lineer fonksiyonun crisp davranışa yaklaşması için
            'normalize': True,
            'eps': 1.0              # Search radius must be large enough to include potential neighbors
        }
    }

    # Setup plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    print("\nRunning Simulation...")
    print("-" * 100)
    print(f"{'Method':<40} {'Clusters':<10} {'Noise':<10} {'ARI':<10} {'AMI':<10}")
    print("-" * 100)

    model = FN_DBSCAN(**config['params'])
    labels = model.fit_predict(X)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    # Calculate metrics
    ari = adjusted_rand_score(y_true, labels)
    ami = adjusted_mutual_info_score(y_true, labels)
    
    print(f"{config['name']:<40} {n_clusters:<10} {n_noise:<10} {ari:<10.4f} {ami:<10.4f}")
    
    # Add metrics to plot title
    title = f"{config['name']}\nARI: {ari:.3f}, AMI: {ami:.3f}, Noise: {n_noise}"
    plot_clustering(X, labels, ax, title)

    plt.tight_layout()
    output_file = 'simulation_result.png'
    plt.savefig(output_file)
    print("-" * 100)
    print(f"\nResult saved to {output_file}")

if __name__ == "__main__":
    main()
