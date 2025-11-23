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

print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
print(f"\nData range:")
print(f"  X: [{X[:, 0].min():.3f}, {X[:, 0].max():.3f}]")
print(f"  Y: [{X[:, 1].min():.3f}, {X[:, 1].max():.3f}]")

# Normalize data
scaler = MinMaxScaler()
X_norm = scaler.fit_transform(X)

print("\n" + "="*60)
print("MANUAL PARAMETER TUNING FOR SPIRAL DATA")
print("="*60)

# Spiral veri için özel parametreler
# DBSCAN: Daha küçük eps ile spiral kolları ayırt etmeye çalış
dbscan_configs = [
    {'eps': 0.06, 'min_samples': 5, 'name': 'DBSCAN-1 (tight)'},
    {'eps': 0.08, 'min_samples': 5, 'name': 'DBSCAN-2 (medium)'},
    {'eps': 0.1, 'min_samples': 5, 'name': 'DBSCAN-3 (loose)'},
]

# FN-DBSCAN: Daha büyük eps ve farklı k değerleri ile spiral yapıyı yakalamaya çalış
fndbscan_configs = [
    {'eps': 0.1, 'epsilon2': 5, 'k': 15, 'name': 'FN-DBSCAN-1 (k=15)'},
    {'eps': 0.12, 'epsilon2': 5, 'k': 20, 'name': 'FN-DBSCAN-2 (k=20)'},
    {'eps': 0.1, 'epsilon2': 4, 'k': 25, 'name': 'FN-DBSCAN-3 (k=25)'},
]

# Test all configurations
results = []

print("\nTesting DBSCAN configurations...")
for config in dbscan_configs:
    name = config.pop('name')
    dbscan = DBSCAN(**config)
    labels = dbscan.fit_predict(X_norm)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    # Calculate silhouette if possible
    mask = labels != -1
    if mask.sum() > 1 and n_clusters > 1:
        silhouette = silhouette_score(X_norm[mask], labels[mask])
    else:
        silhouette = -1
    
    results.append({
        'algorithm': 'DBSCAN',
        'name': name,
        'labels': labels,
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'silhouette': silhouette,
        'params': config
    })
    
    print(f"  {name}: {n_clusters} clusters, {n_noise} noise, Silhouette={silhouette:.4f}")
    config['name'] = name  # Restore name

print("\nTesting FN-DBSCAN configurations...")
for config in fndbscan_configs:
    name = config.pop('name')
    fndbscan = FN_DBSCAN(fuzzy_function='exponential', normalize=False, **config)
    labels = fndbscan.fit_predict(X_norm)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    # Calculate silhouette if possible
    mask = labels != -1
    if mask.sum() > 1 and n_clusters > 1:
        silhouette = silhouette_score(X_norm[mask], labels[mask])
    else:
        silhouette = -1
    
    results.append({
        'algorithm': 'FN-DBSCAN',
        'name': name,
        'labels': labels,
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'silhouette': silhouette,
        'params': config
    })
    
    print(f"  {name}: {n_clusters} clusters, {n_noise} noise, Silhouette={silhouette:.4f}")
    config['name'] = name  # Restore name

# Find best configurations
best_dbscan = max([r for r in results if r['algorithm'] == 'DBSCAN'], 
                  key=lambda x: x['silhouette'])
best_fndbscan = max([r for r in results if r['algorithm'] == 'FN-DBSCAN'], 
                    key=lambda x: x['silhouette'])

print("\n" + "="*60)
print("BEST CONFIGURATIONS")
print("="*60)
print(f"\nBest DBSCAN: {best_dbscan['name']}")
print(f"  Parameters: {best_dbscan['params']}")
print(f"  Clusters: {best_dbscan['n_clusters']}, Noise: {best_dbscan['n_noise']}")
print(f"  Silhouette: {best_dbscan['silhouette']:.4f}")

print(f"\nBest FN-DBSCAN: {best_fndbscan['name']}")
print(f"  Parameters: {best_fndbscan['params']}")
print(f"  Clusters: {best_fndbscan['n_clusters']}, Noise: {best_fndbscan['n_noise']}")
print(f"  Silhouette: {best_fndbscan['silhouette']:.4f}")

# Comprehensive visualization
fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

# Row 1: DBSCAN configurations
for idx, config in enumerate(dbscan_configs):
    ax = fig.add_subplot(gs[0, idx])
    result = [r for r in results if r['name'] == config['name']][0]
    scatter = ax.scatter(X[:, 0], X[:, 1], c=result['labels'], cmap='tab10', s=15, alpha=0.7)
    ax.set_title(f"{config['name']}\n{result['n_clusters']} clusters, {result['n_noise']} noise\nSilhouette: {result['silhouette']:.3f}", 
                 fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

# Row 2: FN-DBSCAN configurations
for idx, config in enumerate(fndbscan_configs):
    ax = fig.add_subplot(gs[1, idx])
    result = [r for r in results if r['name'] == config['name']][0]
    scatter = ax.scatter(X[:, 0], X[:, 1], c=result['labels'], cmap='tab10', s=15, alpha=0.7)
    ax.set_title(f"{config['name']}\n{result['n_clusters']} clusters, {result['n_noise']} noise\nSilhouette: {result['silhouette']:.3f}", 
                 fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

# Row 3: Best comparison
ax_best1 = fig.add_subplot(gs[2, 0])
scatter1 = ax_best1.scatter(X[:, 0], X[:, 1], c=best_dbscan['labels'], cmap='tab10', s=20, alpha=0.7)
ax_best1.set_title(f"BEST DBSCAN: {best_dbscan['name']}\n{best_dbscan['n_clusters']} clusters, Silhouette: {best_dbscan['silhouette']:.3f}", 
                   fontsize=11, fontweight='bold', color='darkblue')
ax_best1.grid(True, alpha=0.3)
ax_best1.set_xlabel('X')
ax_best1.set_ylabel('Y')

ax_best2 = fig.add_subplot(gs[2, 1])
scatter2 = ax_best2.scatter(X[:, 0], X[:, 1], c=best_fndbscan['labels'], cmap='tab10', s=20, alpha=0.7)
ax_best2.set_title(f"BEST FN-DBSCAN: {best_fndbscan['name']}\n{best_fndbscan['n_clusters']} clusters, Silhouette: {best_fndbscan['silhouette']:.3f}", 
                   fontsize=11, fontweight='bold', color='darkgreen')
ax_best2.grid(True, alpha=0.3)
ax_best2.set_xlabel('X')
ax_best2.set_ylabel('Y')

# Metrics comparison
ax_metrics = fig.add_subplot(gs[2, 2])
comparison_data = []
for r in results:
    comparison_data.append({
        'Config': r['name'].split('-')[1],
        'Algorithm': r['algorithm'],
        'Clusters': r['n_clusters'],
        'Noise %': r['n_noise'] / len(X) * 100,
        'Silhouette': r['silhouette']
    })
df_comparison = pd.DataFrame(comparison_data)

# Create grouped bar chart
x_pos = np.arange(len(dbscan_configs))
width = 0.35

dbscan_sil = [r['silhouette'] for r in results if r['algorithm'] == 'DBSCAN']
fndbscan_sil = [r['silhouette'] for r in results if r['algorithm'] == 'FN-DBSCAN']

ax_metrics.bar(x_pos - width/2, dbscan_sil, width, label='DBSCAN', color='steelblue', alpha=0.8)
ax_metrics.bar(x_pos + width/2, fndbscan_sil, width, label='FN-DBSCAN', color='seagreen', alpha=0.8)

ax_metrics.set_xlabel('Configuration')
ax_metrics.set_ylabel('Silhouette Score')
ax_metrics.set_title('Silhouette Score Comparison', fontsize=11, fontweight='bold')
ax_metrics.set_xticks(x_pos)
ax_metrics.set_xticklabels(['Config 1', 'Config 2', 'Config 3'])
ax_metrics.legend()
ax_metrics.grid(True, alpha=0.3, axis='y')

plt.suptitle('Spiral Dataset: Parameter Comparison', fontsize=14, fontweight='bold', y=0.995)
plt.savefig('spiral_parameter_comparison.png', dpi=150, bbox_inches='tight')
print("\n✅ Visualization saved: spiral_parameter_comparison.png")

# Save detailed results
df_results = pd.DataFrame([{
    'Configuration': r['name'],
    'Algorithm': r['algorithm'],
    'Clusters': r['n_clusters'],
    'Noise_Points': r['n_noise'],
    'Noise_Percentage': r['n_noise'] / len(X) * 100,
    'Silhouette': r['silhouette'],
    'Parameters': str(r['params'])
} for r in results])

df_results.to_csv('spiral_parameter_results.csv', index=False)
print("✅ Results saved: spiral_parameter_results.csv")

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
print(f"\n🏆 Winner: {best_fndbscan['name'] if best_fndbscan['silhouette'] > best_dbscan['silhouette'] else best_dbscan['name']}")
print(f"   Silhouette Score: {max(best_fndbscan['silhouette'], best_dbscan['silhouette']):.4f}")
