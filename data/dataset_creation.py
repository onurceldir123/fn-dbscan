import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def generate_and_save_dataset(filename='pathbased_dataset.csv', n_samples_per_cluster=150, noise=0.15):
    """
    Kıvrımlı 3 kümeli veri setini oluşturur ve CSV dosyasına kaydeder.
    """
    np.random.seed(42)  # Tekrar üretilebilirlik için
    
    # Ortak parametre (t)
    t = np.linspace(0, 3 * np.pi, n_samples_per_cluster)
    
    # --- Veri Listeleri ---
    x_list = []
    y_list = []
    labels = []

    # --- 1. Küme (Üst) ---
    x1 = t
    y1 = 3 * np.cos(0.5 * t) + 8
    x1 += np.random.normal(scale=noise * 2, size=n_samples_per_cluster)
    y1 += np.random.normal(scale=noise * 3, size=n_samples_per_cluster)
    
    x_list.extend(x1)
    y_list.extend(y1)
    labels.extend([1] * n_samples_per_cluster) # Etiket: 1

    # --- 2. Küme (Orta) ---
    x2 = t
    y2 = 3 * np.cos(0.5 * t) + 4
    x2 += np.random.normal(scale=noise * 2, size=n_samples_per_cluster)
    y2 += np.random.normal(scale=noise * 3, size=n_samples_per_cluster)

    x_list.extend(x2)
    y_list.extend(y2)
    labels.extend([2] * n_samples_per_cluster) # Etiket: 2

    # --- 3. Küme (Alt) ---
    x3 = t
    y3 = 3 * np.cos(0.5 * t) + 0
    x3 += np.random.normal(scale=noise * 2, size=n_samples_per_cluster)
    y3 += np.random.normal(scale=noise * 3, size=n_samples_per_cluster)

    x_list.extend(x3)
    y_list.extend(y3)
    labels.extend([3] * n_samples_per_cluster) # Etiket: 3

    # --- DataFrame Oluşturma ve Kaydetme ---
    # Veriyi birleştir
    data = {'x': x_list, 'y': y_list, 'label': labels}
    df = pd.DataFrame(data)

    # CSV olarak kaydet (index numaralarını kaydetmiyoruz)
    df.to_csv(filename, index=False)
    print(f"Veri seti başarıyla '{filename}' adıyla kaydedildi.")
    
    return df

# Fonksiyonu çalıştır
df = generate_and_save_dataset(filename='pathbased_dataset.csv', n_samples_per_cluster=100, noise=0.3)

# --- Kontrol Amaçlı Görselleştirme ---
plt.figure(figsize=(8, 6), facecolor='#f0f0e6')
ax = plt.gca()
ax.set_facecolor='#f0f0e6'

# DataFrame içinden çizim yapalım (doğru kaydedildiğini teyit etmek için)
colors = {1: 'forestgreen', 2: 'blue', 3: 'tab:red'}
for label, color in colors.items():
    subset = df[df['label'] == label]
    plt.scatter(subset['x'], subset['y'], c=color, s=15, marker='s', label=f'Cluster {label}')

plt.title("Generated Dataset (Saved to CSV)", fontsize=14)
plt.legend()
plt.show()

# İlk 5 satırı göster
print("\nOluşturulan verinin ilk 5 satırı:")
print(df.head())