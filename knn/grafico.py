import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

plt.figure(figsize=(12, 10))

# 1) Carregar e preparar dataset (binário: barato=0 vs caro=1)
url = "https://raw.githubusercontent.com/LuisDegaspari/DataSet/refs/heads/main/cardekho_data.csv"
df = pd.read_csv(url)

df["Car_Age"] = 2025 - df["Year"]
df = df.drop(columns=["Year", "Car_Name"])

# alvo binário por mediana do preço
median_price = df["Selling_Price"].median()
y = (df["Selling_Price"] > median_price).astype(int)  # 1 = caro, 0 = barato

# 2 features para plotar fronteira
X = df[["Present_Price", "Car_Age"]].values

# split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# escalar
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std  = scaler.transform(X_test)

# procurar um k bom (3..21 ímpares)
best_k, best_acc = None, -1
for k in range(3, 22, 2):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_std, y_train)
    acc = accuracy_score(y_test, knn.predict(X_test_std))
    if acc > best_acc:
        best_acc, best_k = acc, k

# treinar final com melhor k
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train_std, y_train)
y_pred = knn.predict(X_test_std)
acc = accuracy_score(y_test, y_pred)

print(f"Melhor k: {best_k} | Accuracy: {acc:.2f}")

# ----- Fronteira de decisão -----
h = 0.02
x_min, x_max = X_train_std[:, 0].min() - 1, X_train_std[:, 0].max() + 1
y_min, y_max = X_train_std[:, 1].min() - 1, X_train_std[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = knn.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=0.3)

# scatter dos pontos de teste
sns.scatterplot(
    x=X_test_std[:, 0], y=X_test_std[:, 1],
    hue=y_test, style=y_test, palette="deep", s=110, edgecolor="k"
)

plt.title(f"KNN Decision Boundary (k={best_k}) — Accuracy={acc:.2f}")
plt.xlabel("Present_Price (padronizado)")
plt.ylabel("Car_Age (padronizado)")
plt.legend(title="Classe (0=barato, 1=caro)")

# Exportar SVG para embutir no HTML/MkDocs
buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=False)
print(buffer.getvalue())
