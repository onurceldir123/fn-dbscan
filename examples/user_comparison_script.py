import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import adjusted_rand_score
import sys
import os

# Add parent directory to path to import fn_dbscan
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fn_dbscan import FN_DBSCAN as FNDBSCAN

from sklearn.datasets import make_blobs
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns

# --- 1. Veri Setini Yükle ve Hazırla ---
# Mall Customers veri seti (gerçek dünya verisi)
url = "https://raw.githubusercontent.com/tirthajyoti/Machine-Learning-with-Python/master/Datasets/Mall_Customers.csv"
df = pd.read_csv(url)

# Annual Income ve Spending Score özelliklerini kullanalım
X = df[['Annual Income (k$)', 'Spending Score (1-100)']].values
# Bu veri setinde gerçek etiket yok, bu yüzden unsupervised olarak değerlendireceğiz
labels_true = None  # Gerçek etiket yok

print(f"Veri seti yüklendi: {X.shape[0]} müşteri, {X.shape[1]} özellik")
print(f"Özellikler: Annual Income (k$), Spending Score (1-100)")

# Makale Tavsiyesi: Normalizasyon [0, 1] Aralığına
scaler = MinMaxScaler()
X_norm = scaler.fit_transform(X)

# --- 2. Parametre Ayarları ---
# DBSCAN için optimize edilmiş parametreler (Mall Customers için)
EPSILON_DBSCAN = 0.08    # Daha küçük yarıçap - müşteri segmentlerini ayırmak için
MIN_PTS_DBSCAN = 3       # Daha düşük eşik - küçük segmentleri de yakalamak için

# FN-DBSCAN için parametreler
EPSILON_FNDBSCAN = 0.1   # FN-DBSCAN için yarıçap
MIN_PTS_FNDBSCAN = 3     # FN-DBSCAN için eşik

# --- 3. Standart DBSCAN Çalıştır ---
dbscan = DBSCAN(eps=EPSILON_DBSCAN, min_samples=MIN_PTS_DBSCAN)
labels_dbscan = dbscan.fit_predict(X_norm)

# --- 4. FN-DBSCAN Çalıştır ---
try:
    fndbscan = FNDBSCAN(
        eps=EPSILON_FNDBSCAN,   # Yarıçap
        min_fuzzy_neighbors=MIN_PTS_FNDBSCAN,  # Min kardinalite
        fuzzy_function='exponential',
        k=20,
        normalize=False         # Zaten normalize ettik
    )
    labels_fndbscan = fndbscan.fit_predict(X_norm) 
except NameError:
    print("FNDBSCAN sınıfı tanımlı değil, sadece DBSCAN çalışıyor.")
    labels_fndbscan = np.zeros_like(labels_true)

# --- 5. Sonuçları Kıyasla ---
# Gerçek etiket olmadığı için sadece küme sayısı ve gürültü analizi yapacağız
n_clusters_dbscan = len(set(labels_dbscan)) - (1 if -1 in labels_dbscan else 0)
n_noise_dbscan = list(labels_dbscan).count(-1)
n_clusters_fndbscan = len(set(labels_fndbscan)) - (1 if -1 in labels_fndbscan else 0)
n_noise_fndbscan = list(labels_fndbscan).count(-1)

print(f"--- Sonuçlar (Unsupervised - Gerçek Etiket Yok) ---")
print(f"\nStandart DBSCAN (eps={EPSILON_DBSCAN}, min_pts={MIN_PTS_DBSCAN}):")
print(f"  Kümeler: {n_clusters_dbscan}, Gürültü: {n_noise_dbscan} ({n_noise_dbscan/len(X)*100:.1f}%)")

print(f"\nFN-DBSCAN (eps={EPSILON_FNDBSCAN}, min_pts={MIN_PTS_FNDBSCAN}):")
print(f"  Kümeler: {n_clusters_fndbscan}, Gürültü: {n_noise_fndbscan} ({n_noise_fndbscan/len(X)*100:.1f}%)")

# Küme dağılımlarını göster
print(f"\n--- Küme Dağılımları ---")
print(f"DBSCAN:")
for i in range(n_clusters_dbscan):
    count = list(labels_dbscan).count(i)
    print(f"  Küme {i}: {count} müşteri ({count/len(X)*100:.1f}%)")

print(f"\nFN-DBSCAN:")
for i in range(n_clusters_fndbscan):
    count = list(labels_fndbscan).count(i)
    print(f"  Küme {i}: {count} müşteri ({count/len(X)*100:.1f}%)")

# --- 6. Görselleştirme ---
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# DBSCAN Sonucu
axes[0].scatter(X[:, 0], X[:, 1], c=labels_dbscan, cmap='plasma', s=30, alpha=0.6)
axes[0].set_title(f"DBSCAN\n{n_clusters_dbscan} Küme, {n_noise_dbscan} Gürültü", 
                  fontsize=14, fontweight='bold')
axes[0].set_xlabel('Annual Income (k$)', fontsize=11)
axes[0].set_ylabel('Spending Score (1-100)', fontsize=11)
axes[0].grid(True, alpha=0.3)

# FN-DBSCAN Sonucu
axes[1].scatter(X[:, 0], X[:, 1], c=labels_fndbscan, cmap='viridis', s=30, alpha=0.6)
axes[1].set_title(f"FN-DBSCAN\n{n_clusters_fndbscan} Küme, {n_noise_fndbscan} Gürültü", 
                  fontsize=14, fontweight='bold')
axes[1].set_xlabel('Annual Income (k$)', fontsize=11)
axes[1].set_ylabel('Spending Score (1-100)', fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('user_comparison_result.png', dpi=150, bbox_inches='tight')
print("\nGörsel 'user_comparison_result.png' olarak kaydedildi.")
