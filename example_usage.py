"""Basit FN-DBSCAN kullanım örneği"""
import numpy as np
import matplotlib.pyplot as plt
from fn_dbscan import FN_DBSCAN
from sklearn.datasets import make_moons, make_blobs

# Örnek 1: Basit veri
print("=" * 50)
print("Örnek 1: Basit Clustering")
print("=" * 50)

X = np.array([[1, 2], [2, 2], [2, 3], [8, 7], [8, 8], [25, 80]])
model = FN_DBSCAN(eps=3, min_fuzzy_neighbors=2, fuzzy_function='linear')
labels = model.fit_predict(X)

print(f"Veri noktaları:\n{X}")
print(f"Cluster etiketleri: {labels}")
print(f"Bulunan cluster sayısı: {model.n_clusters_}")
print(f"Gürültü noktası sayısı: {sum(labels == -1)}")

# Örnek 2: Moons dataset
print("\n" + "=" * 50)
print("Örnek 2: Moons Dataset (Non-convex)")
print("=" * 50)

X, _ = make_moons(n_samples=200, noise=0.1, random_state=42)
model = FN_DBSCAN(eps=0.3, min_fuzzy_neighbors=5)
labels = model.fit_predict(X)

print(f"Veri boyutu: {X.shape}")
print(f"Bulunan cluster sayısı: {model.n_clusters_}")
print(f"Core sample sayısı: {len(model.core_sample_indices_)}")
print(f"Gürültü noktası sayısı: {sum(labels == -1)}")

# Görselleştirme
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c='blue', alpha=0.5)
plt.title('Original Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.7)
plt.title(f'FN-DBSCAN Clustering ({model.n_clusters_} clusters)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar(label='Cluster ID')

plt.tight_layout()
plt.savefig('clustering_result.png')
print("\nGörselleştirme 'clustering_result.png' olarak kaydedildi")

# Örnek 3: Farklı fuzzy fonksiyonları karşılaştırma
print("\n" + "=" * 50)
print("Örnek 3: Farklı Fuzzy Fonksiyonları")
print("=" * 50)

X, _ = make_blobs(n_samples=300, centers=3, random_state=42)

for fuzzy_func in ['linear', 'exponential', 'trapezoidal']:
    model = FN_DBSCAN(eps=0.7, min_fuzzy_neighbors=5, fuzzy_function=fuzzy_func)
    labels = model.fit_predict(X)
    print(f"{fuzzy_func:12s}: {model.n_clusters_} cluster, {sum(labels == -1)} gürültü noktası")

print("\nTest tamamlandı!")
