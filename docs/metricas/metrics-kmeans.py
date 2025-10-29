import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

plt.figure(figsize=(12, 10))

# 1) Carregar CarDekho e preparar features (sem alvo)
url = "https://raw.githubusercontent.com/LuisDegaspari/DataSet/refs/heads/main/cardekho_data.csv"
df = pd.read_csv(url)

df["Car_Age"] = 2025 - df["Year"]
df = df.drop(columns=["Year", "Car_Name"])          # removemos colunas irrelevantes

# X: todas as features EXCETO o preço de venda (K-Means é não supervisionado)
X = df.drop(columns=["Selling_Price"])

# One-Hot nas categóricas (estilo simples, sem sklearn encoders)
cat_cols = ["Fuel_Type", "Seller_Type", "Transmission"]
X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

# 2) Padronização simples (z-score)
X_np = X.to_numpy(dtype=float)
mu = X_np.mean(axis=0)
sigma = X_np.std(axis=0)
sigma[sigma == 0.0] = 1.0
X_std = (X_np - mu) / sigma

# 3) K-Means
kmeans = KMeans(n_clusters=3, init="k-means++", n_init=10, max_iter=300, random_state=42)
labels = kmeans.fit_predict(X_std)

# 4) Redução para 2D (PCA) só para visualização
pca = PCA(n_components=2, random_state=42)
X_2d = pca.fit_transform(X_std)
centroids_2d = pca.transform(kmeans.cluster_centers_)

# 5) Plot
scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap="viridis", s=50)
plt.scatter(centroids_2d[:, 0], centroids_2d[:, 1], c="red", marker="*", s=220, label="Centroids")
plt.title("K-Means (k=3) — CarDekho (PCA 2D)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()

# # Extras (opcional):
# print("Centroids (orig space):\n", kmeans.cluster_centers_)
# print("Inertia (WCSS):", kmeans.inertia_)

# 6) Exportar SVG no stdout (como você faz)
buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True, bbox_inches="tight")
print(buffer.getvalue())
