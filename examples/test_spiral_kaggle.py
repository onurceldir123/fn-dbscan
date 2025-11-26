import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import adjusted_rand_score, silhouette_score, confusion_matrix
import seaborn as sns
import sys
import os

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fn_dbscan import FN_DBSCAN

# Download dataset from Kaggle
try:
    import kagglehub
    path = kagglehub.dataset_download("arushchillar/spiral-data")
    print(f"Dataset downloaded to: {path}")
except ImportError:
    print("Installing kagglehub...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "kagglehub"])
    import kagglehub
    path = kagglehub.dataset_download("arushchillar/spiral-data")
    print(f"Dataset downloaded to: {path}")

# Find data files in the downloaded path
import glob
data_files = glob.glob(os.path.join(path, "*.csv")) + glob.glob(os.path.join(path, "*.txt"))
print(f"\nFound data files: {data_files}")

# Load the first data file
if data_files:
    data_file = data_files[0]
    print(f"\nLoading: {data_file}")
    
    # Try different delimiters
    try:
        df = pd.read_csv(data_file, sep=',')
        if df.shape[1] == 1:  # Probably wrong delimiter
            raise ValueError("Only 1 column, trying other delimiters")
    except:
        try:
            df = pd.read_csv(data_file, sep='\t')
            if df.shape[1] == 1:
                raise ValueError("Only 1 column, trying other delimiters")
        except:
            # Space-separated, no header
            df = pd.read_csv(data_file, sep='\s+', header=None)
    
    # If no header, assign column names
    if df.columns[0] == 0 or isinstance(df.columns[0], int):
        if df.shape[1] == 2:
            df.columns = ['x', 'y']
        elif df.shape[1] == 3:
            df.columns = ['x', 'y', 'label']
        else:
            df.columns = [f'feature_{i}' for i in range(df.shape[1])]
    
    print(f"Dataset loaded: {df.shape[0]} samples, {df.shape[1]} features")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nFirst few rows:")
    print(df.head())
    
    # Extract features and labels
    if 'label' in df.columns or 'Label' in df.columns:
        label_col = 'label' if 'label' in df.columns else 'Label'
        y_true = df[label_col].values
        X = df.drop(columns=[label_col]).values
        has_labels = True
    else:
        X = df.values
        y_true = None
        has_labels = False
    
    print(f"\nFeatures shape: {X.shape}")
    if has_labels:
        print(f"Labels: {np.unique(y_true)}")
        print(f"Class distribution: {np.bincount(y_true)}")
else:
    print("No data files found!")
    sys.exit(1)

# Normalize data
scaler = MinMaxScaler()
X_norm = scaler.fit_transform(X)

# Parameter optimization
print("\n" + "="*60)
print("PARAMETER OPTIMIZATION")
print("="*60)

param_grid = {
    'eps': [0.05, 0.08, 0.1, 0.12, 0.15, 0.2],
    'min_samples': [3, 4, 5, 6, 7]
}

best_score_dbscan = -1
best_params_dbscan = {}
best_score_fndbscan = -1
best_params_fndbscan = {}

print("\nOptimizing DBSCAN parameters...")
for eps in param_grid['eps']:
    for min_samples in param_grid['min_samples']:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X_norm)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        # Skip if too few/many clusters or too much noise
        if n_clusters < 2 or n_clusters > 10 or n_noise > len(labels) * 0.3:
            continue
        
        # Use ARI if we have ground truth, otherwise silhouette
        if has_labels:
            score = adjusted_rand_score(y_true, labels)
        else:
            mask = labels != -1
            if mask.sum() > 1 and n_clusters > 1:
                score = silhouette_score(X_norm[mask], labels[mask])
            else:
                continue
        
        if score > best_score_dbscan:
            best_score_dbscan = score
            best_params_dbscan = {'eps': eps, 'min_samples': min_samples}
            metric_name = "ARI" if has_labels else "Silhouette"
            print(f"  New best: eps={eps}, min_samples={min_samples}, "
                  f"{metric_name}={score:.4f}, clusters={n_clusters}, noise={n_noise}")

print("\nOptimizing FN-DBSCAN parameters...")
for eps in param_grid['eps']:
    for min_samples in param_grid['min_samples']:
        fndbscan = FN_DBSCAN(eps=eps, min_fuzzy_neighbors=min_samples, 
                             fuzzy_function='exponential', k=20, normalize=False)
        labels = fndbscan.fit_predict(X_norm)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        # Skip if too few/many clusters or too much noise
        if n_clusters < 2 or n_clusters > 10 or n_noise > len(labels) * 0.3:
            continue
        
        # Use ARI if we have ground truth, otherwise silhouette
        if has_labels:
            score = adjusted_rand_score(y_true, labels)
        else:
            mask = labels != -1
            if mask.sum() > 1 and n_clusters > 1:
                score = silhouette_score(X_norm[mask], labels[mask])
            else:
                continue
        
        if score > best_score_fndbscan:
            best_score_fndbscan = score
            best_params_fndbscan = {'eps': eps, 'min_samples': min_samples}
            metric_name = "ARI" if has_labels else "Silhouette"
            print(f"  New best: eps={eps}, min_samples={min_samples}, "
                  f"{metric_name}={score:.4f}, clusters={n_clusters}, noise={n_noise}")

print("\n" + "="*60)
print("BEST PARAMETERS FOUND")
print("="*60)
metric_name = "ARI" if has_labels else "Silhouette"
print(f"DBSCAN: {best_params_dbscan}, {metric_name}={best_score_dbscan:.4f}")
print(f"FN-DBSCAN: {best_params_fndbscan}, {metric_name}={best_score_fndbscan:.4f}")

# Run with best parameters
print("\n" + "="*60)
print("FINAL CLUSTERING")
print("="*60)

dbscan_best = DBSCAN(**best_params_dbscan)
labels_dbscan = dbscan_best.fit_predict(X_norm)

fndbscan_best = FN_DBSCAN(eps=best_params_fndbscan['eps'], 
                          min_fuzzy_neighbors=best_params_fndbscan['min_samples'],
                          fuzzy_function='exponential', k=20, normalize=False)
labels_fndbscan = fndbscan_best.fit_predict(X_norm)

# Calculate metrics
n_clusters_dbscan = len(set(labels_dbscan)) - (1 if -1 in labels_dbscan else 0)
n_noise_dbscan = list(labels_dbscan).count(-1)
n_clusters_fndbscan = len(set(labels_fndbscan)) - (1 if -1 in labels_fndbscan else 0)
n_noise_fndbscan = list(labels_fndbscan).count(-1)

print(f"\nDBSCAN:")
print(f"  Clusters: {n_clusters_dbscan}")
print(f"  Noise: {n_noise_dbscan} ({n_noise_dbscan/len(labels_dbscan)*100:.1f}%)")
if has_labels:
    print(f"  ARI: {adjusted_rand_score(y_true, labels_dbscan):.4f}")

print(f"\nFN-DBSCAN:")
print(f"  Clusters: {n_clusters_fndbscan}")
print(f"  Noise: {n_noise_fndbscan} ({n_noise_fndbscan/len(labels_fndbscan)*100:.1f}%)")
if has_labels:
    print(f"  ARI: {adjusted_rand_score(y_true, labels_fndbscan):.4f}")

# Visualization
fig = plt.figure(figsize=(20, 8))
gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

# Row 1: Clustering results
if has_labels:
    ax1 = fig.add_subplot(gs[0, 0])
    scatter1 = ax1.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', s=20, alpha=0.6)
    ax1.set_title('Ground Truth', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter1, ax=ax1)

ax2 = fig.add_subplot(gs[0, 1] if has_labels else gs[0, 0])
scatter2 = ax2.scatter(X[:, 0], X[:, 1], c=labels_dbscan, cmap='plasma', s=20, alpha=0.6)
title = f'DBSCAN\n{n_clusters_dbscan} clusters, {n_noise_dbscan} noise'
if has_labels:
    title += f'\nARI: {adjusted_rand_score(y_true, labels_dbscan):.3f}'
ax2.set_title(title, fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
plt.colorbar(scatter2, ax=ax2)

ax3 = fig.add_subplot(gs[0, 2] if has_labels else gs[0, 1])
scatter3 = ax3.scatter(X[:, 0], X[:, 1], c=labels_fndbscan, cmap='viridis', s=20, alpha=0.6)
title = f'FN-DBSCAN\n{n_clusters_fndbscan} clusters, {n_noise_fndbscan} noise'
if has_labels:
    title += f'\nARI: {adjusted_rand_score(y_true, labels_fndbscan):.3f}'
ax3.set_title(title, fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)
plt.colorbar(scatter3, ax=ax3)

# Row 2: Confusion matrices (if ground truth available)
if has_labels:
    mask_dbscan = labels_dbscan != -1
    mask_fndbscan = labels_fndbscan != -1
    
    if mask_dbscan.sum() > 0:
        ax4 = fig.add_subplot(gs[1, 0])
        cm_dbscan = confusion_matrix(y_true[mask_dbscan], labels_dbscan[mask_dbscan])
        sns.heatmap(cm_dbscan, annot=True, fmt='d', cmap='Blues', ax=ax4, cbar=True)
        ax4.set_title('DBSCAN Confusion Matrix', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Predicted Cluster')
        ax4.set_ylabel('True Label')
    
    if mask_fndbscan.sum() > 0:
        ax5 = fig.add_subplot(gs[1, 1])
        cm_fndbscan = confusion_matrix(y_true[mask_fndbscan], labels_fndbscan[mask_fndbscan])
        sns.heatmap(cm_fndbscan, annot=True, fmt='d', cmap='Greens', ax=ax5, cbar=True)
        ax5.set_title('FN-DBSCAN Confusion Matrix', fontsize=12, fontweight='bold')
        ax5.set_xlabel('Predicted Cluster')
        ax5.set_ylabel('True Label')
    
    # Metrics comparison
    ax6 = fig.add_subplot(gs[1, 2])
    metrics_df = pd.DataFrame({
        'DBSCAN': [n_clusters_dbscan, n_noise_dbscan, 
                   adjusted_rand_score(y_true, labels_dbscan)],
        'FN-DBSCAN': [n_clusters_fndbscan, n_noise_fndbscan,
                      adjusted_rand_score(y_true, labels_fndbscan)]
    }, index=['Clusters', 'Noise', 'ARI'])
    sns.heatmap(metrics_df, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax6, cbar=True)
    ax6.set_title('Metrics Comparison', fontsize=12, fontweight='bold')

plt.savefig('spiral_clustering_results.png', dpi=150, bbox_inches='tight')
print("\n✅ Visualization saved: spiral_clustering_results.png")

# Save results
results_df = pd.DataFrame({
    'Algorithm': ['DBSCAN', 'FN-DBSCAN'],
    'Clusters': [n_clusters_dbscan, n_clusters_fndbscan],
    'Noise_Points': [n_noise_dbscan, n_noise_fndbscan],
    'Noise_Percentage': [n_noise_dbscan/len(X)*100, n_noise_fndbscan/len(X)*100],
    'eps': [best_params_dbscan['eps'], best_params_fndbscan['eps']],
    'min_samples': [best_params_dbscan['min_samples'], best_params_fndbscan['min_samples']]
})

if has_labels:
    results_df['ARI'] = [adjusted_rand_score(y_true, labels_dbscan),
                         adjusted_rand_score(y_true, labels_fndbscan)]

results_df.to_csv('spiral_clustering_summary.csv', index=False)
print("✅ Results saved: spiral_clustering_summary.csv")

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
