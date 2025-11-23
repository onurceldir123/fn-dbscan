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

# Normalize data
scaler = MinMaxScaler()
X_norm = scaler.fit_transform(X)

print("\n" + "="*60)
print("TESTING RECOMMENDED PARAMETERS")
print("="*60)

# Önerilen parametreler
dbscan_configs = [
    {'eps': 0.06, 'min_samples': 4, 'name': 'DBSCAN (eps=0.06)'},
    {'eps': 0.07, 'min_samples': 4, 'name': 'DBSCAN (eps=0.07)'},
    {'eps': 0.08, 'min_samples': 4, 'name': 'DBSCAN (eps=0.08)'},
]

fndbscan_configs = [
    {'eps': 0.08, 'epsilon2': 4, 'k': 20, 'name': 'FN-DBSCAN (eps=0.08)'},
    {'eps': 0.09, 'epsilon2': 4, 'k': 20, 'name': 'FN-DBSCAN (eps=0.09)'},
    {'eps': 0.10, 'epsilon2': 4, 'k': 20, 'name': 'FN-DBSCAN (eps=0.10)'},
]

results = []

print("\nTesting DBSCAN configurations...")
print("Parameters: min_samples=4, eps varies")
for config in dbscan_configs:
    name = config.pop('name')
    dbscan = DBSCAN(**config)
    labels = dbscan.fit_predict(X_norm)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
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
    
    print(f"  {name}: {n_clusters} clusters, {n_noise} noise ({n_noise/len(X)*100:.1f}%), Silhouette={silhouette:.4f}")
    config['name'] = name

print("\nTesting FN-DBSCAN configurations...")
print("Parameters: epsilon2=4, k=20, function='exponential', eps varies")
for config in fndbscan_configs:
    name = config.pop('name')
    fndbscan = FN_DBSCAN(fuzzy_function='exponential', normalize=False, **config)
    labels = fndbscan.fit_predict(X_norm)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
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
    
    print(f"  {name}: {n_clusters} clusters, {n_noise} noise ({n_noise/len(X)*100:.1f}%), Silhouette={silhouette:.4f}")
    config['name'] = name

# Find best from each algorithm
best_dbscan = max([r for r in results if r['algorithm'] == 'DBSCAN'], 
                  key=lambda x: x['silhouette'])
best_fndbscan = max([r for r in results if r['algorithm'] == 'FN-DBSCAN'], 
                    key=lambda x: x['silhouette'])

print("\n" + "="*60)
print("BEST CONFIGURATIONS")
print("="*60)

print(f"\n🏆 Best DBSCAN: {best_dbscan['name']}")
print(f"   Parameters: {best_dbscan['params']}")
print(f"   Clusters: {best_dbscan['n_clusters']}, Noise: {best_dbscan['n_noise']} ({best_dbscan['n_noise']/len(X)*100:.1f}%)")
print(f"   Silhouette: {best_dbscan['silhouette']:.4f}")

print(f"\n🏆 Best FN-DBSCAN: {best_fndbscan['name']}")
print(f"   Parameters: {best_fndbscan['params']}")
print(f"   Clusters: {best_fndbscan['n_clusters']}, Noise: {best_fndbscan['n_noise']} ({best_fndbscan['n_noise']/len(X)*100:.1f}%)")
print(f"   Silhouette: {best_fndbscan['silhouette']:.4f}")

# Comprehensive visualization
fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

# Row 1: DBSCAN configurations
for idx, config in enumerate(dbscan_configs):
    ax = fig.add_subplot(gs[0, idx])
    result = [r for r in results if r['name'] == config['name']][0]
    
    # Color noise points differently
    colors = result['labels'].copy()
    scatter = ax.scatter(X[:, 0], X[:, 1], c=colors, cmap='tab10', s=20, alpha=0.7, 
                        edgecolors='black', linewidth=0.3)
    
    title = f"{config['name']}\n{result['n_clusters']} clusters, {result['n_noise']} noise ({result['n_noise']/len(X)*100:.1f}%)"
    if result['silhouette'] > 0:
        title += f"\nSilhouette: {result['silhouette']:.3f}"
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

# Row 2: FN-DBSCAN configurations
for idx, config in enumerate(fndbscan_configs):
    ax = fig.add_subplot(gs[1, idx])
    result = [r for r in results if r['name'] == config['name']][0]
    
    colors = result['labels'].copy()
    scatter = ax.scatter(X[:, 0], X[:, 1], c=colors, cmap='tab10', s=20, alpha=0.7,
                        edgecolors='black', linewidth=0.3)
    
    title = f"{config['name']}\n{result['n_clusters']} clusters, {result['n_noise']} noise ({result['n_noise']/len(X)*100:.1f}%)"
    if result['silhouette'] > 0:
        title += f"\nSilhouette: {result['silhouette']:.3f}"
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

# Row 3: Best comparison and metrics
ax_best1 = fig.add_subplot(gs[2, 0])
scatter1 = ax_best1.scatter(X[:, 0], X[:, 1], c=best_dbscan['labels'], cmap='tab10', 
                           s=25, alpha=0.8, edgecolors='black', linewidth=0.3)
ax_best1.set_title(f"🏆 BEST DBSCAN\neps={best_dbscan['params']['eps']}, min_samples={best_dbscan['params']['min_samples']}\n{best_dbscan['n_clusters']} clusters, Silhouette: {best_dbscan['silhouette']:.3f}", 
                   fontsize=11, fontweight='bold', color='darkblue')
ax_best1.grid(True, alpha=0.3)
ax_best1.set_xlabel('X', fontsize=10)
ax_best1.set_ylabel('Y', fontsize=10)

ax_best2 = fig.add_subplot(gs[2, 1])
scatter2 = ax_best2.scatter(X[:, 0], X[:, 1], c=best_fndbscan['labels'], cmap='tab10', 
                           s=25, alpha=0.8, edgecolors='black', linewidth=0.3)
ax_best2.set_title(f"🏆 BEST FN-DBSCAN\neps={best_fndbscan['params']['eps']}, epsilon2={best_fndbscan['params']['epsilon2']}, k={best_fndbscan['params']['k']}\n{best_fndbscan['n_clusters']} clusters, Silhouette: {best_fndbscan['silhouette']:.3f}", 
                   fontsize=11, fontweight='bold', color='darkgreen')
ax_best2.grid(True, alpha=0.3)
ax_best2.set_xlabel('X', fontsize=10)
ax_best2.set_ylabel('Y', fontsize=10)

# Comparison table
ax_table = fig.add_subplot(gs[2, 2])
ax_table.axis('tight')
ax_table.axis('off')

table_data = [
    ['Metric', 'DBSCAN', 'FN-DBSCAN'],
    ['eps', f"{best_dbscan['params']['eps']}", f"{best_fndbscan['params']['eps']}"],
    ['min_samples/ε2', f"{best_dbscan['params']['min_samples']}", f"{best_fndbscan['params']['epsilon2']}"],
    ['k', 'N/A', f"{best_fndbscan['params']['k']}"],
    ['Clusters', f"{best_dbscan['n_clusters']}", f"{best_fndbscan['n_clusters']}"],
    ['Noise', f"{best_dbscan['n_noise']} ({best_dbscan['n_noise']/len(X)*100:.1f}%)", 
              f"{best_fndbscan['n_noise']} ({best_fndbscan['n_noise']/len(X)*100:.1f}%)"],
    ['Silhouette', f"{best_dbscan['silhouette']:.4f}", f"{best_fndbscan['silhouette']:.4f}"],
]

table = ax_table.table(cellText=table_data, cellLoc='center', loc='center',
                       colWidths=[0.35, 0.325, 0.325])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2.2)

# Style header
for i in range(3):
    table[(0, i)].set_facecolor('#4472C4')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Alternate row colors
for i in range(1, len(table_data)):
    for j in range(3):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#E7E6E6')

ax_table.set_title('Best Configuration Comparison', fontsize=11, fontweight='bold', pad=15)

plt.suptitle('Spiral Dataset: Recommended Parameters Test\nDBSCAN: eps=0.06-0.08, min_samples=4 | FN-DBSCAN: eps=0.08-0.10, epsilon2=4, k=20', 
             fontsize=13, fontweight='bold', y=0.995)
plt.savefig('spiral_recommended_params.png', dpi=150, bbox_inches='tight')
print("\n✅ Visualization saved: spiral_recommended_params.png")

# Cluster analysis
print("\n" + "="*60)
print("CLUSTER DISTRIBUTION ANALYSIS")
print("="*60)

print(f"\nBest DBSCAN ({best_dbscan['name']}):")
for i in range(best_dbscan['n_clusters']):
    count = list(best_dbscan['labels']).count(i)
    print(f"  Cluster {i}: {count} points ({count/len(X)*100:.1f}%)")
if best_dbscan['n_noise'] > 0:
    print(f"  Noise (-1): {best_dbscan['n_noise']} points ({best_dbscan['n_noise']/len(X)*100:.1f}%)")

print(f"\nBest FN-DBSCAN ({best_fndbscan['name']}):")
for i in range(best_fndbscan['n_clusters']):
    count = list(best_fndbscan['labels']).count(i)
    print(f"  Cluster {i}: {count} points ({count/len(X)*100:.1f}%)")
if best_fndbscan['n_noise'] > 0:
    print(f"  Noise (-1): {best_fndbscan['n_noise']} points ({best_fndbscan['n_noise']/len(X)*100:.1f}%)")

# Save results
df_results = pd.DataFrame([{
    'Configuration': r['name'],
    'Algorithm': r['algorithm'],
    'Clusters': r['n_clusters'],
    'Noise': r['n_noise'],
    'Noise_Pct': r['n_noise']/len(X)*100,
    'Silhouette': r['silhouette'],
    'Parameters': str(r['params'])
} for r in results])

df_results.to_csv('spiral_recommended_results.csv', index=False)
print("\n✅ Results saved: spiral_recommended_results.csv")

# Final verdict
print("\n" + "="*60)
print("FINAL VERDICT")
print("="*60)

if best_dbscan['silhouette'] > best_fndbscan['silhouette']:
    diff = best_dbscan['silhouette'] - best_fndbscan['silhouette']
    pct = diff / best_fndbscan['silhouette'] * 100
    print(f"\n🏆 Winner: DBSCAN")
    print(f"   Silhouette: {best_dbscan['silhouette']:.4f} vs {best_fndbscan['silhouette']:.4f}")
    print(f"   Advantage: +{diff:.4f} ({pct:.1f}% better)")
else:
    diff = best_fndbscan['silhouette'] - best_dbscan['silhouette']
    pct = diff / best_dbscan['silhouette'] * 100
    print(f"\n🏆 Winner: FN-DBSCAN")
    print(f"   Silhouette: {best_fndbscan['silhouette']:.4f} vs {best_dbscan['silhouette']:.4f}")
    print(f"   Advantage: +{diff:.4f} ({pct:.1f}% better)")

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
