import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (adjusted_rand_score, confusion_matrix, 
                             accuracy_score, precision_score, recall_score, f1_score)
from sklearn.datasets import make_moons, make_blobs
import seaborn as sns
import sys
import os

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fn_dbscan import FN_DBSCAN

def plot_confusion_matrix(cm, title, ax):
    """Plot confusion matrix heatmap"""
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=True)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')

def run_experiment(X, y_true, eps_dbscan, min_pts_dbscan, eps_fn, min_pts_fn, 
                   dataset_name, normalize=True):
    """Run clustering experiment and return results"""
    
    # Normalize
    if normalize:
        scaler = MinMaxScaler()
        X_norm = scaler.fit_transform(X)
    else:
        X_norm = X
    
    # DBSCAN
    dbscan = DBSCAN(eps=eps_dbscan, min_samples=min_pts_dbscan)
    labels_dbscan = dbscan.fit_predict(X_norm)
    
    # FN-DBSCAN
    fndbscan = FN_DBSCAN(eps=eps_fn, min_fuzzy_neighbors=min_pts_fn, 
                         fuzzy_function='exponential', k=20, normalize=False)
    labels_fndbscan = fndbscan.fit_predict(X_norm)
    
    # Calculate metrics
    n_clusters_dbscan = len(set(labels_dbscan)) - (1 if -1 in labels_dbscan else 0)
    n_noise_dbscan = list(labels_dbscan).count(-1)
    n_clusters_fndbscan = len(set(labels_fndbscan)) - (1 if -1 in labels_fndbscan else 0)
    n_noise_fndbscan = list(labels_fndbscan).count(-1)
    
    results = {
        'dataset': dataset_name,
        'X': X,
        'X_norm': X_norm,
        'y_true': y_true,
        'labels_dbscan': labels_dbscan,
        'labels_fndbscan': labels_fndbscan,
        'n_clusters_dbscan': n_clusters_dbscan,
        'n_noise_dbscan': n_noise_dbscan,
        'n_clusters_fndbscan': n_clusters_fndbscan,
        'n_noise_fndbscan': n_noise_fndbscan,
        'params_dbscan': {'eps': eps_dbscan, 'min_pts': min_pts_dbscan},
        'params_fndbscan': {'eps': eps_fn, 'min_pts': min_pts_fn}
    }
    
    # If ground truth exists, calculate supervised metrics
    if y_true is not None:
        results['ari_dbscan'] = adjusted_rand_score(y_true, labels_dbscan)
        results['ari_fndbscan'] = adjusted_rand_score(y_true, labels_fndbscan)
        
        # Confusion matrices (excluding noise)
        mask_dbscan = labels_dbscan != -1
        mask_fndbscan = labels_fndbscan != -1
        
        if mask_dbscan.sum() > 0:
            results['cm_dbscan'] = confusion_matrix(y_true[mask_dbscan], labels_dbscan[mask_dbscan])
        if mask_fndbscan.sum() > 0:
            results['cm_fndbscan'] = confusion_matrix(y_true[mask_fndbscan], labels_fndbscan[mask_fndbscan])
    
    return results

# Experiment 1: Make Moons
print("Running Experiment 1: Make Moons...")
X_moons, y_moons = make_moons(n_samples=300, noise=0.1, random_state=42)
results_moons = run_experiment(X_moons, y_moons, 0.08, 3, 0.15, 5, "Make Moons")

# Experiment 2: Make Blobs
print("Running Experiment 2: Make Blobs...")
X_blobs, y_blobs = make_blobs(n_samples=300, centers=3, n_features=2, 
                               cluster_std=0.5, random_state=42)
results_blobs = run_experiment(X_blobs, y_blobs, 0.3, 5, 0.3, 5, "Make Blobs")

# Experiment 3: Mall Customers
print("Running Experiment 3: Mall Customers...")
url = "https://raw.githubusercontent.com/tirthajyoti/Machine-Learning-with-Python/master/Datasets/Mall_Customers.csv"
df = pd.read_csv(url)
X_mall = df[['Annual Income (k$)', 'Spending Score (1-100)']].values
results_mall = run_experiment(X_mall, None, 0.08, 3, 0.1, 3, "Mall Customers")

# Experiment 4: Pathbased Dataset
print("Running Experiment 4: Pathbased Dataset...")
pathbased_path = os.path.join(os.path.dirname(__file__), '../data/pathbased_dataset.csv')
if os.path.exists(pathbased_path):
    df_path = pd.read_csv(pathbased_path)
    X_path = df_path[['x', 'y']].values
    y_path = df_path['label'].values
    results_path = run_experiment(X_path, y_path, 0.8, 4, 0.05, 4, "Pathbased")
else:
    results_path = None
    print("Pathbased dataset not found, skipping...")

# Create comprehensive visualizations
all_results = [results_moons, results_blobs, results_mall]
if results_path:
    all_results.append(results_path)

# Figure 1: Clustering Results Comparison
fig1 = plt.figure(figsize=(20, 5 * len(all_results)))
gs1 = fig1.add_gridspec(len(all_results), 3, hspace=0.3, wspace=0.3)

for idx, res in enumerate(all_results):
    # Ground Truth (if available)
    ax1 = fig1.add_subplot(gs1[idx, 0])
    if res['y_true'] is not None:
        ax1.scatter(res['X'][:, 0], res['X'][:, 1], c=res['y_true'], cmap='viridis', s=20, alpha=0.6)
        ax1.set_title(f"{res['dataset']}\nGround Truth", fontsize=12, fontweight='bold')
    else:
        ax1.scatter(res['X'][:, 0], res['X'][:, 1], c='gray', s=20, alpha=0.6)
        ax1.set_title(f"{res['dataset']}\n(No Ground Truth)", fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # DBSCAN Results
    ax2 = fig1.add_subplot(gs1[idx, 1])
    ax2.scatter(res['X'][:, 0], res['X'][:, 1], c=res['labels_dbscan'], cmap='plasma', s=20, alpha=0.6)
    title = f"DBSCAN\n{res['n_clusters_dbscan']} clusters, {res['n_noise_dbscan']} noise"
    if 'ari_dbscan' in res:
        title += f"\nARI: {res['ari_dbscan']:.3f}"
    ax2.set_title(title, fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # FN-DBSCAN Results
    ax3 = fig1.add_subplot(gs1[idx, 2])
    ax3.scatter(res['X'][:, 0], res['X'][:, 1], c=res['labels_fndbscan'], cmap='viridis', s=20, alpha=0.6)
    title = f"FN-DBSCAN\n{res['n_clusters_fndbscan']} clusters, {res['n_noise_fndbscan']} noise"
    if 'ari_fndbscan' in res:
        title += f"\nARI: {res['ari_fndbscan']:.3f}"
    ax3.set_title(title, fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)

plt.savefig('report_clustering_results.png', dpi=150, bbox_inches='tight')
print("Saved: report_clustering_results.png")

# Figure 2: Confusion Matrices (only for datasets with ground truth)
supervised_results = [r for r in all_results if r['y_true'] is not None]
if supervised_results:
    fig2 = plt.figure(figsize=(16, 5 * len(supervised_results)))
    gs2 = fig2.add_gridspec(len(supervised_results), 2, hspace=0.3, wspace=0.3)
    
    for idx, res in enumerate(supervised_results):
        # DBSCAN Confusion Matrix
        if 'cm_dbscan' in res:
            ax1 = fig2.add_subplot(gs2[idx, 0])
            plot_confusion_matrix(res['cm_dbscan'], 
                                f"{res['dataset']} - DBSCAN\nARI: {res['ari_dbscan']:.3f}", ax1)
        
        # FN-DBSCAN Confusion Matrix
        if 'cm_fndbscan' in res:
            ax2 = fig2.add_subplot(gs2[idx, 1])
            plot_confusion_matrix(res['cm_fndbscan'], 
                                f"{res['dataset']} - FN-DBSCAN\nARI: {res['ari_fndbscan']:.3f}", ax2)
    
    plt.savefig('report_confusion_matrices.png', dpi=150, bbox_inches='tight')
    print("Saved: report_confusion_matrices.png")

# Save results summary to CSV
summary_data = []
for res in all_results:
    row = {
        'Dataset': res['dataset'],
        'DBSCAN_Clusters': res['n_clusters_dbscan'],
        'DBSCAN_Noise': res['n_noise_dbscan'],
        'DBSCAN_eps': res['params_dbscan']['eps'],
        'DBSCAN_min_pts': res['params_dbscan']['min_pts'],
        'FN-DBSCAN_Clusters': res['n_clusters_fndbscan'],
        'FN-DBSCAN_Noise': res['n_noise_fndbscan'],
        'FN-DBSCAN_eps': res['params_fndbscan']['eps'],
        'FN-DBSCAN_min_pts': res['params_fndbscan']['min_pts'],
    }
    if 'ari_dbscan' in res:
        row['DBSCAN_ARI'] = res['ari_dbscan']
        row['FN-DBSCAN_ARI'] = res['ari_fndbscan']
    summary_data.append(row)

df_summary = pd.DataFrame(summary_data)
df_summary.to_csv('report_summary.csv', index=False)
print("Saved: report_summary.csv")

print("\nAll visualizations and summaries generated successfully!")
