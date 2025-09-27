import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class KNNClassifier:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)

    def _predict(self, x):
        # Distância Euclidiana
        distances = [np.sqrt(np.sum((x - x_train)**2)) for x_train in self.X_train]
        # Índices dos k vizinhos
        k_indices = np.argsort(distances)[:self.k]
        # Rótulos dos vizinhos
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # Classe mais comum
        most_common = max(set(k_nearest_labels), key=k_nearest_labels.count)
        return most_common

# Carregar dataset CarDekho
url = "https://raw.githubusercontent.com/LuisDegaspari/DataSet/refs/heads/main/cardekho_data.csv"
df = pd.read_csv(url)

# Feature engineering
df["Car_Age"] = 2025 - df["Year"]
df = df.drop(columns=["Year", "Car_Name"])

# Alvo de classificação em 3 faixas
df["price_bucket"] = pd.qcut(df["Selling_Price"], q=3, labels=[0, 1, 2]).astype(int)

# One-Hot das categóricas
cat_cols = ["Fuel_Type", "Seller_Type", "Transmission"]
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# Matriz de features e alvo
X = df.drop(columns=["Selling_Price", "price_bucket"]).to_numpy(dtype=float)
y = df["price_bucket"].to_numpy()

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Garantir ndarray float (evita erro do std/sqrt)
X_train = np.asarray(X_train, dtype=float)
X_test  = np.asarray(X_test,  dtype=float)

# Normalização z-score (usando treino)
mu = X_train.mean(axis=0)
sigma = X_train.std(axis=0)
sigma[sigma == 0] = 1.0  # evita divisão por zero

X_train = (X_train - mu) / sigma
X_test  = (X_test  - mu) / sigma

# Treinar e avaliar
knn = KNNClassifier(k=5)  # ajuste k se quiser
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, predictions):.2f}")
