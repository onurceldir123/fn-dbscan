import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
import seaborn as sns
import sys
import os

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fn_dbscan import FN_DBSCAN

# Load spiral dataset
path = "/Users/onurmertceldir/.cache/kagglehub/datasets/arushchillar/spiral-data/versions/1"
data_file = os.path.join(path, "ITdata.txt")

df = pd.read_csv(data_file, sep='\s+', header=None, names=['x', 'y'])
X = df.values

print(f"Dataset loaded: {X.shape[0]} samples")
print(f"Data range: X[{X[:, 0].min():.2f}, {X[:, 0].max():.2f}], Y[{X[:, 1].min():.2f}, {X[:, 1].max():.2f}]")

# Normalize data
scaler = MinMaxScaler()
X_norm = scaler.fit_transform(X)

print("\n" + "="*60)
print("COMPREHENSIVE PARAMETER OPTIMIZATION FOR SPIRAL DATA")
print("="*60)
print("Strategy: Wide parameter search to capture spiral structure")

# Geniş parametre taraması
param_grid_dbscan = {
    'eps': [0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12],
    'min_samples': [3, 4, 5, 6, 7]
}

param_grid_fndbscan = {
    'eps': [0.05, 0.07, 0.09, 0.11, 0.13, 0.15],
    'epsilon2': [3, 4, 5, 6],
    'k': [10, 15, 20, 25]
}

best_dbscan = {'silhouette': -1, 'params': None, 'labels': None, 'n_clusters': 0, 'n_noise': 0}
best_fndbscan = {'silhouette': -1, 'params': None, 'labels': None, 'n_clusters': 0, 'n_noise': 0}

print("\n" + "-"*60)
print("OPTIMIZING DBSCAN...")
print("-"*60)

total_configs = len(param_grid_dbscan['eps']) * len(param_grid_dbscan['min_samples'])
current = 0

for eps in param_grid_dbscan['eps']:
    for min_samples in param_grid_dbscan['min_samples']:
        current += 1
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X_norm)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        # Skip if only 1 cluster or too many clusters or too much noise
        if n_clusters < 2 or n_clusters > 10 or n_noise > len(X) * 0.2:
            continue
        
        mask = labels != -1
        if mask.sum() > 1 and n_clusters > 1:
            silhouette = silhouette_score(X_norm[mask], labels[mask])
            
            if silhouette > best_dbscan['silhouette']:
                best_dbscan = {
                    'silhouette': silhouette,
                    'params': {'eps': eps, 'min_samples': min_samples},
                    'labels': labels.copy(),
                    'n_clusters': n_clusters,
                    'n_noise': n_noise
                }
                print(f"  [{current}/{total_configs}] New best: eps={eps:.2f}, min_samples={min_samples}, "
                      f"clusters={n_clusters}, noise={n_noise}, Silhouette={silhouette:.4f}")

print("\n" + "-"*60)
print("OPTIMIZING FN-DBSCAN...")
print("-"*60)

total_configs = (len(param_grid_fndbscan['eps']) * len(param_grid_fndbscan['epsilon2']) * 
                 len(param_grid_fndbscan['k']))
current = 0

for eps in param_grid_fndbscan['eps']:
    for epsilon2 in param_grid_fndbscan['epsilon2']:
        for k in param_grid_fndbscan['k']:
            current += 1
            
            fndbscan = FN_DBSCAN(eps=eps, epsilon2=epsilon2, k=k, 
                                fuzzy_function='exponential', normalize=False)
            labels = fndbscan.fit_predict(X_norm)
            
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            
            # Skip if only 1 cluster or too many clusters or too much noise
            if n_clusters < 2 or n_clusters > 10 or n_noise > len(X) * 0.2:
                continue
            
            mask = labels != -1
            if mask.sum() > 1 and n_clusters > 1:
                silhouette = silhouette_score(X_norm[mask], labels[mask])
                
                if silhouette > best_fndbscan['silhouette']:
                    best_fndbscan = {
                        'silhouette': silhouette,
                        'params': {'eps': eps, 'epsilon2': epsilon2, 'k': k},
                        'labels': labels.copy(),
                        'n_clusters': n_clusters,
                        'n_noise': n_noise
                    }
                    print(f"  [{current}/{total_configs}] New best: eps={eps:.2f}, epsilon2={epsilon2}, k={k}, "
                          f"clusters={n_clusters}, noise={n_noise}, Silhouette={silhouette:.4f}")

print("\n" + "="*60)
print("OPTIMIZATION RESULTS")
print("="*60)

print(f"\n🏆 BEST DBSCAN:")
print(f"   Parameters: eps={best_dbscan['params']['eps']:.2f}, min_samples={best_dbscan['params']['min_samples']}")
print(f"   Clusters: {best_dbscan['n_clusters']}")
print(f"   Noise: {best_dbscan['n_noise']} ({best_dbscan['n_noise']/len(X)*100:.1f}%)")
print(f"   Silhouette Score: {best_dbscan['silhouette']:.4f}")

print(f"\n🏆 BEST FN-DBSCAN:")
print(f"   Parameters: eps={best_fndbscan['params']['eps']:.2f}, epsilon2={best_fndbscan['params']['epsilon2']}, k={best_fndbscan['params']['k']}")
print(f"   Clusters: {best_fndbscan['n_clusters']}")
print(f"   Noise: {best_fndbscan['n_noise']} ({best_fndbscan['n_noise']/len(X)*100:.1f}%)")
print(f"   Silhouette Score: {best_fndbscan['silhouette']:.4f}")

# Detailed visualization
fig = plt.figure(figsize=(18, 6))

# Original data distribution
ax1 = fig.add_subplot(131)
ax1.scatter(X[:, 0], X[:, 1], c='gray', s=15, alpha=0.5)
ax1.set_title('Original Spiral Data', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')

# Best DBSCAN
ax2 = fig.add_subplot(132)
scatter2 = ax2.scatter(X[:, 0], X[:, 1], c=best_dbscan['labels'], 
                       cmap='tab10', s=20, alpha=0.7, edgecolors='black', linewidth=0.3)
title2 = f"BEST DBSCAN\n"
title2 += f"eps={best_dbscan['params']['eps']:.2f}, min_samples={best_dbscan['params']['min_samples']}\n"
title2 += f"{best_dbscan['n_clusters']} clusters, {best_dbscan['n_noise']} noise\n"
title2 += f"Silhouette: {best_dbscan['silhouette']:.4f}"
ax2.set_title(title2, fontsize=11, fontweight='bold', color='darkblue')
ax2.grid(True, alpha=0.3)
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
plt.colorbar(scatter2, ax=ax2, label='Cluster')

# Best FN-DBSCAN
ax3 = fig.add_subplot(133)
scatter3 = ax3.scatter(X[:, 0], X[:, 1], c=best_fndbscan['labels'], 
                       cmap='tab10', s=20, alpha=0.7, edgecolors='black', linewidth=0.3)
title3 = f"BEST FN-DBSCAN\n"
title3 += f"eps={best_fndbscan['params']['eps']:.2f}, epsilon2={best_fndbscan['params']['epsilon2']}, k={best_fndbscan['params']['k']}\n"
title3 += f"{best_fndbscan['n_clusters']} clusters, {best_fndbscan['n_noise']} noise\n"
title3 += f"Silhouette: {best_fndbscan['silhouette']:.4f}"
ax3.set_title(title3, fontsize=11, fontweight='bold', color='darkgreen')
ax3.grid(True, alpha=0.3)
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
plt.colorbar(scatter3, ax=ax3, label='Cluster')

plt.suptitle('Spiral Dataset: Optimized Clustering Results', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('spiral_optimized_final.png', dpi=150, bbox_inches='tight')
print("\n✅ Visualization saved: spiral_optimized_final.png")

# Cluster analysis
print("\n" + "="*60)
print("CLUSTER ANALYSIS")
print("="*60)

print("\nDBSCAN Cluster Distribution:")
for i in range(best_dbscan['n_clusters']):
    count = list(best_dbscan['labels']).count(i)
    print(f"  Cluster {i}: {count} points ({count/len(X)*100:.1f}%)")
if best_dbscan['n_noise'] > 0:
    print(f"  Noise: {best_dbscan['n_noise']} points ({best_dbscan['n_noise']/len(X)*100:.1f}%)")

print("\nFN-DBSCAN Cluster Distribution:")
for i in range(best_fndbscan['n_clusters']):
    count = list(best_fndbscan['labels']).count(i)
    print(f"  Cluster {i}: {count} points ({count/len(X)*100:.1f}%)")
if best_fndbscan['n_noise'] > 0:
    print(f"  Noise: {best_fndbscan['n_noise']} points ({best_fndbscan['n_noise']/len(X)*100:.1f}%)")

# Save results
results_df = pd.DataFrame([
    {
        'Algorithm': 'DBSCAN',
        'eps': best_dbscan['params']['eps'],
        'min_samples/epsilon2': best_dbscan['params']['min_samples'],
        'k': 'N/A',
        'Clusters': best_dbscan['n_clusters'],
        'Noise': best_dbscan['n_noise'],
        'Noise_Pct': best_dbscan['n_noise']/len(X)*100,
        'Silhouette': best_dbscan['silhouette']
    },
    {
        'Algorithm': 'FN-DBSCAN',
        'eps': best_fndbscan['params']['eps'],
        'min_samples/epsilon2': best_fndbscan['params']['epsilon2'],
        'k': best_fndbscan['params']['k'],
        'Clusters': best_fndbscan['n_clusters'],
        'Noise': best_fndbscan['n_noise'],
        'Noise_Pct': best_fndbscan['n_noise']/len(X)*100,
        'Silhouette': best_fndbscan['silhouette']
    }
])

results_df.to_csv('spiral_optimized_results.csv', index=False)
print("\n✅ Results saved: spiral_optimized_results.csv")

# Winner determination
print("\n" + "="*60)
print("FINAL VERDICT")
print("="*60)

if best_dbscan['silhouette'] > best_fndbscan['silhouette']:
    diff = best_dbscan['silhouette'] - best_fndbscan['silhouette']
    print(f"\n🏆 Winner: DBSCAN")
    print(f"   Advantage: {diff:.4f} ({diff/best_fndbscan['silhouette']*100:.1f}% better)")
else:
    diff = best_fndbscan['silhouette'] - best_dbscan['silhouette']
    print(f"\n🏆 Winner: FN-DBSCAN")
    print(f"   Advantage: {diff:.4f} ({diff/best_dbscan['silhouette']*100:.1f}% better)")

print("\n" + "="*60)
