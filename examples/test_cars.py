import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import silhouette_score, confusion_matrix
from sklearn.decomposition import PCA
import seaborn as sns
import sys
import os

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fn_dbscan import FN_DBSCAN

# Load cars dataset
df = pd.read_csv('cars.csv', sep=';', encoding='latin1')
print(f"Dataset loaded: {df.shape[0]} cars, {df.shape[1]} features")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nFirst few rows:")
print(df.head())

# Data preprocessing
print("\n" + "="*60)
print("DATA PREPROCESSING")
print("="*60)

# Select numerical features for clustering
numerical_features = ['Year', 'Km', 'Price']
X_numerical = df[numerical_features].copy()

# Handle missing values
X_numerical = X_numerical.fillna(X_numerical.median())

print(f"\nNumerical features: {numerical_features}")
print(f"Shape: {X_numerical.shape}")
print(f"\nStatistics:")
print(X_numerical.describe())

# Normalize data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_numerical)

# Apply PCA for visualization (reduce to 2D)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print(f"\nPCA explained variance: {pca.explained_variance_ratio_.sum():.2%}")

# Normalize PCA components
scaler_pca = MinMaxScaler()
X_pca_norm = scaler_pca.fit_transform(X_pca)

# Parameter optimization
print("\n" + "="*60)
print("PARAMETER OPTIMIZATION")
print("="*60)

param_grid = {
    'eps': [0.05, 0.08, 0.1, 0.12, 0.15, 0.2],
    'min_samples': [3, 4, 5, 6, 7, 8]
}

best_silhouette_dbscan = -1
best_params_dbscan = {}
best_silhouette_fndbscan = -1
best_params_fndbscan = {}

print("\nOptimizing DBSCAN parameters...")
for eps in param_grid['eps']:
    for min_samples in param_grid['min_samples']:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X_pca_norm)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        # Skip if too few clusters or too much noise
        if n_clusters < 2 or n_clusters > 15 or n_noise > len(labels) * 0.3:
            continue
        
        # Calculate silhouette score (excluding noise)
        mask = labels != -1
        if mask.sum() > 1 and n_clusters > 1:
            silhouette = silhouette_score(X_pca_norm[mask], labels[mask])
            if silhouette > best_silhouette_dbscan:
                best_silhouette_dbscan = silhouette
                best_params_dbscan = {'eps': eps, 'min_samples': min_samples}
                print(f"  New best: eps={eps}, min_samples={min_samples}, "
                      f"Silhouette={silhouette:.4f}, clusters={n_clusters}, noise={n_noise}")

print("\nOptimizing FN-DBSCAN parameters...")
for eps in param_grid['eps']:
    for min_samples in param_grid['min_samples']:
        fndbscan = FN_DBSCAN(eps=eps, min_fuzzy_neighbors=min_samples, 
                             fuzzy_function='exponential', k=20, normalize=False)
        labels = fndbscan.fit_predict(X_pca_norm)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        # Skip if too few clusters or too much noise
        if n_clusters < 2 or n_clusters > 15 or n_noise > len(labels) * 0.3:
            continue
        
        # Calculate silhouette score (excluding noise)
        mask = labels != -1
        if mask.sum() > 1 and n_clusters > 1:
            silhouette = silhouette_score(X_pca_norm[mask], labels[mask])
            if silhouette > best_silhouette_fndbscan:
                best_silhouette_fndbscan = silhouette
                best_params_fndbscan = {'eps': eps, 'min_samples': min_samples}
                print(f"  New best: eps={eps}, min_samples={min_samples}, "
                      f"Silhouette={silhouette:.4f}, clusters={n_clusters}, noise={n_noise}")

print("\n" + "="*60)
print("BEST PARAMETERS FOUND")
print("="*60)
print(f"DBSCAN: {best_params_dbscan}, Silhouette={best_silhouette_dbscan:.4f}")
print(f"FN-DBSCAN: {best_params_fndbscan}, Silhouette={best_silhouette_fndbscan:.4f}")

# Run with best parameters
print("\n" + "="*60)
print("FINAL CLUSTERING WITH BEST PARAMETERS")
print("="*60)

dbscan_best = DBSCAN(**best_params_dbscan)
labels_dbscan = dbscan_best.fit_predict(X_pca_norm)

fndbscan_best = FN_DBSCAN(eps=best_params_fndbscan['eps'], 
                          min_fuzzy_neighbors=best_params_fndbscan['min_samples'],
                          fuzzy_function='exponential', k=20, normalize=False)
labels_fndbscan = fndbscan_best.fit_predict(X_pca_norm)

# Calculate metrics
n_clusters_dbscan = len(set(labels_dbscan)) - (1 if -1 in labels_dbscan else 0)
n_noise_dbscan = list(labels_dbscan).count(-1)
n_clusters_fndbscan = len(set(labels_fndbscan)) - (1 if -1 in labels_fndbscan else 0)
n_noise_fndbscan = list(labels_fndbscan).count(-1)

print(f"\nDBSCAN:")
print(f"  Clusters: {n_clusters_dbscan}")
print(f"  Noise: {n_noise_dbscan} ({n_noise_dbscan/len(labels_dbscan)*100:.1f}%)")
print(f"  Silhouette: {best_silhouette_dbscan:.4f}")

print(f"\nFN-DBSCAN:")
print(f"  Clusters: {n_clusters_fndbscan}")
print(f"  Noise: {n_noise_fndbscan} ({n_noise_fndbscan/len(labels_fndbscan)*100:.1f}%)")
print(f"  Silhouette: {best_silhouette_fndbscan:.4f}")

# Add cluster labels to dataframe
df['Cluster_DBSCAN'] = labels_dbscan
df['Cluster_FNDBSCAN'] = labels_fndbscan

# Analyze clusters
print("\n" + "="*60)
print("CLUSTER ANALYSIS")
print("="*60)

print("\nDBSCAN Cluster Statistics:")
for i in range(n_clusters_dbscan):
    cluster_data = df[df['Cluster_DBSCAN'] == i][numerical_features]
    print(f"\nCluster {i} ({len(cluster_data)} cars):")
    print(f"  Avg Year: {cluster_data['Year'].mean():.1f}")
    print(f"  Avg Km: {cluster_data['Km'].mean():.0f}")
    print(f"  Avg Price: {cluster_data['Price'].mean():.0f}")

print("\n" + "-"*60)
print("\nFN-DBSCAN Cluster Statistics:")
for i in range(n_clusters_fndbscan):
    cluster_data = df[df['Cluster_FNDBSCAN'] == i][numerical_features]
    print(f"\nCluster {i} ({len(cluster_data)} cars):")
    print(f"  Avg Year: {cluster_data['Year'].mean():.1f}")
    print(f"  Avg Km: {cluster_data['Km'].mean():.0f}")
    print(f"  Avg Price: {cluster_data['Price'].mean():.0f}")

# Visualization
fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Row 1: PCA visualizations
ax1 = fig.add_subplot(gs[0, 0])
scatter1 = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=df['Year'], cmap='viridis', s=20, alpha=0.6)
ax1.set_title('Cars by Year', fontsize=12, fontweight='bold')
ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
ax1.grid(True, alpha=0.3)
plt.colorbar(scatter1, ax=ax1, label='Year')

ax2 = fig.add_subplot(gs[0, 1])
scatter2 = ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_dbscan, cmap='plasma', s=20, alpha=0.6)
ax2.set_title(f'DBSCAN\n{n_clusters_dbscan} clusters, {n_noise_dbscan} noise, Silhouette={best_silhouette_dbscan:.3f}', 
              fontsize=12, fontweight='bold')
ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
ax2.grid(True, alpha=0.3)
plt.colorbar(scatter2, ax=ax2, label='Cluster')

ax3 = fig.add_subplot(gs[0, 2])
scatter3 = ax3.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_fndbscan, cmap='viridis', s=20, alpha=0.6)
ax3.set_title(f'FN-DBSCAN\n{n_clusters_fndbscan} clusters, {n_noise_fndbscan} noise, Silhouette={best_silhouette_fndbscan:.3f}', 
              fontsize=12, fontweight='bold')
ax3.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
ax3.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
ax3.grid(True, alpha=0.3)
plt.colorbar(scatter3, ax=ax3, label='Cluster')

# Row 2: Feature distributions by cluster
ax4 = fig.add_subplot(gs[1, 0])
for i in range(n_clusters_dbscan):
    cluster_data = df[df['Cluster_DBSCAN'] == i]
    ax4.scatter(cluster_data['Year'], cluster_data['Price'], label=f'Cluster {i}', alpha=0.6, s=15)
ax4.set_title('DBSCAN: Year vs Price', fontsize=12, fontweight='bold')
ax4.set_xlabel('Year')
ax4.set_ylabel('Price')
ax4.legend()
ax4.grid(True, alpha=0.3)

ax5 = fig.add_subplot(gs[1, 1])
for i in range(n_clusters_fndbscan):
    cluster_data = df[df['Cluster_FNDBSCAN'] == i]
    ax5.scatter(cluster_data['Year'], cluster_data['Price'], label=f'Cluster {i}', alpha=0.6, s=15)
ax5.set_title('FN-DBSCAN: Year vs Price', fontsize=12, fontweight='bold')
ax5.set_xlabel('Year')
ax5.set_ylabel('Price')
ax5.legend()
ax5.grid(True, alpha=0.3)

ax6 = fig.add_subplot(gs[1, 2])
metrics_comparison = pd.DataFrame({
    'DBSCAN': [n_clusters_dbscan, n_noise_dbscan, best_silhouette_dbscan],
    'FN-DBSCAN': [n_clusters_fndbscan, n_noise_fndbscan, best_silhouette_fndbscan]
}, index=['Clusters', 'Noise', 'Silhouette'])
sns.heatmap(metrics_comparison, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax6, cbar=True)
ax6.set_title('Metrics Comparison', fontsize=12, fontweight='bold')

# Row 3: Cluster size distributions
ax7 = fig.add_subplot(gs[2, 0])
cluster_sizes_dbscan = [list(labels_dbscan).count(i) for i in range(n_clusters_dbscan)]
ax7.bar(range(n_clusters_dbscan), cluster_sizes_dbscan, color='steelblue', alpha=0.7)
ax7.set_title('DBSCAN Cluster Sizes', fontsize=12, fontweight='bold')
ax7.set_xlabel('Cluster ID')
ax7.set_ylabel('Number of Cars')
ax7.grid(True, alpha=0.3, axis='y')

ax8 = fig.add_subplot(gs[2, 1])
cluster_sizes_fndbscan = [list(labels_fndbscan).count(i) for i in range(n_clusters_fndbscan)]
ax8.bar(range(n_clusters_fndbscan), cluster_sizes_fndbscan, color='seagreen', alpha=0.7)
ax8.set_title('FN-DBSCAN Cluster Sizes', fontsize=12, fontweight='bold')
ax8.set_xlabel('Cluster ID')
ax8.set_ylabel('Number of Cars')
ax8.grid(True, alpha=0.3, axis='y')

# Parameter info
ax9 = fig.add_subplot(gs[2, 2])
param_text = f"""
OPTIMIZED PARAMETERS:

DBSCAN:
  • eps = {best_params_dbscan['eps']}
  • min_samples = {best_params_dbscan['min_samples']}
  • Silhouette = {best_silhouette_dbscan:.4f}

FN-DBSCAN:
  • eps = {best_params_fndbscan['eps']}
  • min_fuzzy_neighbors = {best_params_fndbscan['min_samples']}
  • k = 20 (exponential)
  • Silhouette = {best_silhouette_fndbscan:.4f}

DATASET:
  • Total cars: {len(df)}
  • Features: Year, Km, Price
  • PCA variance: {pca.explained_variance_ratio_.sum():.2%}
"""
ax9.text(0.1, 0.5, param_text, fontsize=10, family='monospace', 
         verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax9.axis('off')

plt.savefig('cars_clustering_results.png', dpi=150, bbox_inches='tight')
print("\n✅ Visualization saved: cars_clustering_results.png")

# Save results
results_df = pd.DataFrame({
    'Algorithm': ['DBSCAN', 'FN-DBSCAN'],
    'Clusters': [n_clusters_dbscan, n_clusters_fndbscan],
    'Noise_Points': [n_noise_dbscan, n_noise_fndbscan],
    'Noise_Percentage': [n_noise_dbscan/len(df)*100, n_noise_fndbscan/len(df)*100],
    'Silhouette': [best_silhouette_dbscan, best_silhouette_fndbscan],
    'eps': [best_params_dbscan['eps'], best_params_fndbscan['eps']],
    'min_samples': [best_params_dbscan['min_samples'], best_params_fndbscan['min_samples']]
})
results_df.to_csv('cars_clustering_summary.csv', index=False)
print("✅ Results saved: cars_clustering_summary.csv")

# Save clustered data
df.to_csv('cars_with_clusters.csv', index=False, sep=';')
print("✅ Clustered data saved: cars_with_clusters.csv")

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
