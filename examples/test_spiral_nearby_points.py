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
print("OPTIMIZED FOR CAPTURING NEARBY POINTS")
print("="*60)
print("Strategy: Larger epsilon values to connect nearby points better")

# Daha büyük epsilon değerleri - birbirine yakın noktaları yakalamak için
dbscan_configs = [
    {'eps': 0.12, 'min_samples': 4, 'name': 'DBSCAN-1 (eps=0.12)'},
    {'eps': 0.15, 'min_samples': 4, 'name': 'DBSCAN-2 (eps=0.15)'},
    {'eps': 0.18, 'min_samples': 4, 'name': 'DBSCAN-3 (eps=0.18)'},
]

# FN-DBSCAN için de daha büyük epsilon ve daha düşük min_samples
fndbscan_configs = [
    {'eps': 0.15, 'epsilon2': 3, 'k': 15, 'name': 'FN-DBSCAN-1 (eps=0.15, k=15)'},
    {'eps': 0.18, 'epsilon2': 3, 'k': 20, 'name': 'FN-DBSCAN-2 (eps=0.18, k=20)'},
    {'eps': 0.2, 'epsilon2': 3, 'k': 25, 'name': 'FN-DBSCAN-3 (eps=0.2, k=25)'},
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

# Find best configurations
best_dbscan = max([r for r in results if r['algorithm'] == 'DBSCAN'], 
                  key=lambda x: (x['silhouette'], -x['n_noise']))
best_fndbscan = max([r for r in results if r['algorithm'] == 'FN-DBSCAN'], 
                    key=lambda x: (x['silhouette'], -x['n_noise']))

print("\n" + "="*60)
print("BEST CONFIGURATIONS")
print("="*60)
print(f"\nBest DBSCAN: {best_dbscan['name']}")
print(f"  Parameters: eps={best_dbscan['params']['eps']}, min_samples={best_dbscan['params']['min_samples']}")
print(f"  Clusters: {best_dbscan['n_clusters']}, Noise: {best_dbscan['n_noise']} ({best_dbscan['n_noise']/len(X)*100:.1f}%)")
print(f"  Silhouette: {best_dbscan['silhouette']:.4f}")

print(f"\nBest FN-DBSCAN: {best_fndbscan['name']}")
print(f"  Parameters: eps={best_fndbscan['params']['eps']}, epsilon2={best_fndbscan['params']['epsilon2']}, k={best_fndbscan['params']['k']}")
print(f"  Clusters: {best_fndbscan['n_clusters']}, Noise: {best_fndbscan['n_noise']} ({best_fndbscan['n_noise']/len(X)*100:.1f}%)")
print(f"  Silhouette: {best_fndbscan['silhouette']:.4f}")

# Comprehensive visualization
fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

# Row 1: DBSCAN configurations
for idx, config in enumerate(dbscan_configs):
    ax = fig.add_subplot(gs[0, idx])
    result = [r for r in results if r['name'] == config['name']][0]
    scatter = ax.scatter(X[:, 0], X[:, 1], c=result['labels'], cmap='tab10', s=20, alpha=0.7, edgecolors='black', linewidth=0.5)
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
    scatter = ax.scatter(X[:, 0], X[:, 1], c=result['labels'], cmap='tab10', s=20, alpha=0.7, edgecolors='black', linewidth=0.5)
    title = f"{config['name']}\n{result['n_clusters']} clusters, {result['n_noise']} noise ({result['n_noise']/len(X)*100:.1f}%)"
    if result['silhouette'] > 0:
        title += f"\nSilhouette: {result['silhouette']:.3f}"
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

# Row 3: Best comparison
ax_best1 = fig.add_subplot(gs[2, 0])
scatter1 = ax_best1.scatter(X[:, 0], X[:, 1], c=best_dbscan['labels'], cmap='tab10', s=25, alpha=0.8, edgecolors='black', linewidth=0.5)
ax_best1.set_title(f"🏆 BEST DBSCAN\n{best_dbscan['n_clusters']} clusters, {best_dbscan['n_noise']} noise\nSilhouette: {best_dbscan['silhouette']:.3f}", 
                   fontsize=12, fontweight='bold', color='darkblue')
ax_best1.grid(True, alpha=0.3)
ax_best1.set_xlabel('X', fontsize=11)
ax_best1.set_ylabel('Y', fontsize=11)

ax_best2 = fig.add_subplot(gs[2, 1])
scatter2 = ax_best2.scatter(X[:, 0], X[:, 1], c=best_fndbscan['labels'], cmap='tab10', s=25, alpha=0.8, edgecolors='black', linewidth=0.5)
ax_best2.set_title(f"🏆 BEST FN-DBSCAN\n{best_fndbscan['n_clusters']} clusters, {best_fndbscan['n_noise']} noise\nSilhouette: {best_fndbscan['silhouette']:.3f}", 
                   fontsize=12, fontweight='bold', color='darkgreen')
ax_best2.grid(True, alpha=0.3)
ax_best2.set_xlabel('X', fontsize=11)
ax_best2.set_ylabel('Y', fontsize=11)

# Metrics comparison table
ax_table = fig.add_subplot(gs[2, 2])
ax_table.axis('tight')
ax_table.axis('off')

table_data = []
table_data.append(['Metric', 'DBSCAN', 'FN-DBSCAN'])
table_data.append(['Clusters', str(best_dbscan['n_clusters']), str(best_fndbscan['n_clusters'])])
table_data.append(['Noise', f"{best_dbscan['n_noise']} ({best_dbscan['n_noise']/len(X)*100:.1f}%)", 
                   f"{best_fndbscan['n_noise']} ({best_fndbscan['n_noise']/len(X)*100:.1f}%)"])
table_data.append(['Silhouette', f"{best_dbscan['silhouette']:.4f}", f"{best_fndbscan['silhouette']:.4f}"])
table_data.append(['eps', f"{best_dbscan['params']['eps']}", f"{best_fndbscan['params']['eps']}"])
table_data.append(['min_samples', f"{best_dbscan['params']['min_samples']}", 
                   f"{best_fndbscan['params']['epsilon2']}"])

table = ax_table.table(cellText=table_data, cellLoc='center', loc='center',
                       colWidths=[0.35, 0.325, 0.325])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Style header row
for i in range(3):
    table[(0, i)].set_facecolor('#4472C4')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Alternate row colors
for i in range(1, len(table_data)):
    for j in range(3):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#E7E6E6')

ax_table.set_title('Best Configuration Comparison', fontsize=12, fontweight='bold', pad=20)

plt.suptitle('Spiral Dataset: Optimized for Capturing Nearby Points\n(Larger Epsilon Values)', 
             fontsize=14, fontweight='bold', y=0.995)
plt.savefig('spiral_nearby_points_optimized.png', dpi=150, bbox_inches='tight')
print("\n✅ Visualization saved: spiral_nearby_points_optimized.png")

# Save results
df_results = pd.DataFrame([{
    'Configuration': r['name'],
    'Algorithm': r['algorithm'],
    'Clusters': r['n_clusters'],
    'Noise_Points': r['n_noise'],
    'Noise_Percentage': r['n_noise'] / len(X) * 100,
    'Silhouette': r['silhouette'],
    'Parameters': str(r['params'])
} for r in results])

df_results.to_csv('spiral_nearby_points_results.csv', index=False)
print("✅ Results saved: spiral_nearby_points_results.csv")

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)

# Determine winner
if best_dbscan['silhouette'] > best_fndbscan['silhouette']:
    winner = best_dbscan
    winner_algo = "DBSCAN"
else:
    winner = best_fndbscan
    winner_algo = "FN-DBSCAN"

print(f"\n🏆 Overall Winner: {winner_algo}")
print(f"   Configuration: {winner['name']}")
print(f"   Silhouette Score: {winner['silhouette']:.4f}")
print(f"   Clusters: {winner['n_clusters']}, Noise: {winner['n_noise']} ({winner['n_noise']/len(X)*100:.1f}%)")
