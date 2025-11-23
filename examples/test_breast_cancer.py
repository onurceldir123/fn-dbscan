import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import (adjusted_rand_score, silhouette_score, 
                             confusion_matrix, accuracy_score, 
                             precision_score, recall_score, f1_score)
from sklearn.decomposition import PCA
import seaborn as sns
import sys
import os

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fn_dbscan import FN_DBSCAN

# Fetch dataset
try:
    from ucimlrepo import fetch_ucirepo
    breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)
    X = breast_cancer_wisconsin_diagnostic.data.features
    y = breast_cancer_wisconsin_diagnostic.data.targets
    
    # Convert to numpy arrays
    X = X.values
    y_true = y.values.ravel()  # Flatten to 1D
    
    # Convert labels to binary (M=1, B=0)
    y_true = np.where(y_true == 'M', 1, 0)
    
    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Class distribution: Malignant={np.sum(y_true)}, Benign={len(y_true)-np.sum(y_true)}")
    
except ImportError:
    print("Installing ucimlrepo...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ucimlrepo"])
    from ucimlrepo import fetch_ucirepo
    breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)
    X = breast_cancer_wisconsin_diagnostic.data.features.values
    y = breast_cancer_wisconsin_diagnostic.data.targets.values.ravel()
    y_true = np.where(y == 'M', 1, 0)

# Normalize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA for visualization (reduce to 2D)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.2%}")

# Normalize PCA components for clustering
scaler_pca = MinMaxScaler()
X_pca_norm = scaler_pca.fit_transform(X_pca)

print("\n" + "="*60)
print("PARAMETER OPTIMIZATION")
print("="*60)

# Grid search for best parameters
param_grid = {
    'eps': [0.05, 0.08, 0.1, 0.12, 0.15],
    'min_samples': [3, 4, 5, 6, 7]
}

best_ari_dbscan = -1
best_params_dbscan = {}
best_ari_fndbscan = -1
best_params_fndbscan = {}

print("\nOptimizing DBSCAN parameters...")
for eps in param_grid['eps']:
    for min_samples in param_grid['min_samples']:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X_pca_norm)
        
        # Skip if all points are noise or all in one cluster
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        if n_clusters < 2 or n_clusters > 10:
            continue
            
        ari = adjusted_rand_score(y_true, labels)
        if ari > best_ari_dbscan:
            best_ari_dbscan = ari
            best_params_dbscan = {'eps': eps, 'min_samples': min_samples}
            print(f"  New best: eps={eps}, min_samples={min_samples}, ARI={ari:.4f}, clusters={n_clusters}")

print("\nOptimizing FN-DBSCAN parameters...")
for eps in param_grid['eps']:
    for min_samples in param_grid['min_samples']:
        fndbscan = FN_DBSCAN(eps=eps, epsilon2=min_samples, 
                             fuzzy_function='exponential', k=20, normalize=False)
        labels = fndbscan.fit_predict(X_pca_norm)
        
        # Skip if all points are noise or all in one cluster
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        if n_clusters < 2 or n_clusters > 10:
            continue
            
        ari = adjusted_rand_score(y_true, labels)
        if ari > best_ari_fndbscan:
            best_ari_fndbscan = ari
            best_params_fndbscan = {'eps': eps, 'min_samples': min_samples}
            print(f"  New best: eps={eps}, min_samples={min_samples}, ARI={ari:.4f}, clusters={n_clusters}")

print("\n" + "="*60)
print("BEST PARAMETERS FOUND")
print("="*60)
print(f"DBSCAN: {best_params_dbscan}, ARI={best_ari_dbscan:.4f}")
print(f"FN-DBSCAN: {best_params_fndbscan}, ARI={best_ari_fndbscan:.4f}")

# Run with best parameters
print("\n" + "="*60)
print("FINAL EVALUATION WITH BEST PARAMETERS")
print("="*60)

dbscan_best = DBSCAN(**best_params_dbscan)
labels_dbscan = dbscan_best.fit_predict(X_pca_norm)

fndbscan_best = FN_DBSCAN(eps=best_params_fndbscan['eps'], 
                          epsilon2=best_params_fndbscan['min_samples'],
                          fuzzy_function='exponential', k=20, normalize=False)
labels_fndbscan = fndbscan_best.fit_predict(X_pca_norm)

# Calculate metrics
def calculate_metrics(y_true, y_pred, algorithm_name):
    n_clusters = len(set(y_pred)) - (1 if -1 in y_pred else 0)
    n_noise = list(y_pred).count(-1)
    ari = adjusted_rand_score(y_true, y_pred)
    
    # For silhouette score, exclude noise points
    mask = y_pred != -1
    if mask.sum() > 1 and n_clusters > 1:
        silhouette = silhouette_score(X_pca_norm[mask], y_pred[mask])
    else:
        silhouette = -1
    
    # Confusion matrix (excluding noise)
    if mask.sum() > 0:
        cm = confusion_matrix(y_true[mask], y_pred[mask])
    else:
        cm = None
    
    print(f"\n{algorithm_name}:")
    print(f"  Clusters: {n_clusters}")
    print(f"  Noise: {n_noise} ({n_noise/len(y_pred)*100:.1f}%)")
    print(f"  ARI: {ari:.4f}")
    print(f"  Silhouette Score: {silhouette:.4f}")
    
    return {
        'labels': y_pred,
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'ari': ari,
        'silhouette': silhouette,
        'cm': cm
    }

results_dbscan = calculate_metrics(y_true, labels_dbscan, "DBSCAN")
results_fndbscan = calculate_metrics(y_true, labels_fndbscan, "FN-DBSCAN")

# Visualization
fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Row 1: Clustering results
ax1 = fig.add_subplot(gs[0, 0])
scatter1 = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=y_true, cmap='RdYlGn', s=30, alpha=0.6)
ax1.set_title('Ground Truth\n(Red=Malignant, Green=Benign)', fontsize=12, fontweight='bold')
ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
ax1.grid(True, alpha=0.3)
plt.colorbar(scatter1, ax=ax1)

ax2 = fig.add_subplot(gs[0, 1])
scatter2 = ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_dbscan, cmap='plasma', s=30, alpha=0.6)
ax2.set_title(f'DBSCAN\n{results_dbscan["n_clusters"]} clusters, ARI={results_dbscan["ari"]:.3f}', 
              fontsize=12, fontweight='bold')
ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
ax2.grid(True, alpha=0.3)
plt.colorbar(scatter2, ax=ax2)

ax3 = fig.add_subplot(gs[0, 2])
scatter3 = ax3.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_fndbscan, cmap='viridis', s=30, alpha=0.6)
ax3.set_title(f'FN-DBSCAN\n{results_fndbscan["n_clusters"]} clusters, ARI={results_fndbscan["ari"]:.3f}', 
              fontsize=12, fontweight='bold')
ax3.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
ax3.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
ax3.grid(True, alpha=0.3)
plt.colorbar(scatter3, ax=ax3)

# Row 2: Confusion Matrices
if results_dbscan['cm'] is not None:
    ax4 = fig.add_subplot(gs[1, 0])
    sns.heatmap(results_dbscan['cm'], annot=True, fmt='d', cmap='Blues', ax=ax4, cbar=True)
    ax4.set_title('DBSCAN Confusion Matrix', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Predicted Cluster')
    ax4.set_ylabel('True Label (0=Benign, 1=Malignant)')

if results_fndbscan['cm'] is not None:
    ax5 = fig.add_subplot(gs[1, 1])
    sns.heatmap(results_fndbscan['cm'], annot=True, fmt='d', cmap='Greens', ax=ax5, cbar=True)
    ax5.set_title('FN-DBSCAN Confusion Matrix', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Predicted Cluster')
    ax5.set_ylabel('True Label (0=Benign, 1=Malignant)')

# Row 2, Col 3: Metrics comparison
ax6 = fig.add_subplot(gs[1, 2])
metrics_comparison = pd.DataFrame({
    'DBSCAN': [results_dbscan['n_clusters'], results_dbscan['n_noise'], 
               results_dbscan['ari'], results_dbscan['silhouette']],
    'FN-DBSCAN': [results_fndbscan['n_clusters'], results_fndbscan['n_noise'],
                  results_fndbscan['ari'], results_fndbscan['silhouette']]
}, index=['Clusters', 'Noise', 'ARI', 'Silhouette'])
sns.heatmap(metrics_comparison, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax6, cbar=True)
ax6.set_title('Metrics Comparison', fontsize=12, fontweight='bold')

# Row 3: Parameter sensitivity analysis
ax7 = fig.add_subplot(gs[2, :])
param_text = f"""
OPTIMIZED PARAMETERS:

DBSCAN:
  • eps = {best_params_dbscan['eps']}
  • min_samples = {best_params_dbscan['min_samples']}
  • ARI Score = {best_ari_dbscan:.4f}

FN-DBSCAN:
  • eps = {best_params_fndbscan['eps']}
  • epsilon2 (min_samples) = {best_params_fndbscan['min_samples']}
  • fuzzy_function = exponential
  • k = 20
  • ARI Score = {best_ari_fndbscan:.4f}

DATASET INFO:
  • Total samples: {len(y_true)}
  • Features: {X.shape[1]} (reduced to 2 via PCA)
  • PCA variance explained: {pca.explained_variance_ratio_.sum():.2%}
  • Malignant: {np.sum(y_true)} ({np.sum(y_true)/len(y_true)*100:.1f}%)
  • Benign: {len(y_true)-np.sum(y_true)} ({(len(y_true)-np.sum(y_true))/len(y_true)*100:.1f}%)
"""
ax7.text(0.1, 0.5, param_text, fontsize=11, family='monospace', 
         verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax7.axis('off')

plt.savefig('breast_cancer_results.png', dpi=150, bbox_inches='tight')
print("\n✅ Visualization saved: breast_cancer_results.png")

# Save detailed results
results_df = pd.DataFrame({
    'Algorithm': ['DBSCAN', 'FN-DBSCAN'],
    'Clusters': [results_dbscan['n_clusters'], results_fndbscan['n_clusters']],
    'Noise_Points': [results_dbscan['n_noise'], results_fndbscan['n_noise']],
    'Noise_Percentage': [results_dbscan['n_noise']/len(y_true)*100, 
                         results_fndbscan['n_noise']/len(y_true)*100],
    'ARI': [results_dbscan['ari'], results_fndbscan['ari']],
    'Silhouette': [results_dbscan['silhouette'], results_fndbscan['silhouette']],
    'eps': [best_params_dbscan['eps'], best_params_fndbscan['eps']],
    'min_samples': [best_params_dbscan['min_samples'], best_params_fndbscan['min_samples']]
})

results_df.to_csv('breast_cancer_results.csv', index=False)
print("✅ Results saved: breast_cancer_results.csv")

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
