# decision_tree_classification_svg.py
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report

# 1) Dataset
url = "https://raw.githubusercontent.com/LuisDegaspari/DataSet/refs/heads/main/cardekho_data.csv"
# url = "https://raw.githubusercontent.com/LuisDegaspari/DataSet/main/cardekho_data.csv"  # forma "raw" clássica
df = pd.read_csv(url)

# 2) Feature engineering
df["Car_Age"] = 2025 - df["Year"]
df = df.drop(columns=["Year", "Car_Name"])

# 3) Criar CLASSES (tercis: 3 faixas com quantidades similares)
y = pd.qcut(df["Selling_Price"], q=3, labels=["baixo", "medio", "alto"])

# 4) Features (sem Selling_Price)
X = df.drop(columns=["Selling_Price"])

# 5) Pré-processamento (One-Hot nas categóricas; numéricas passam direto)
cat_cols = ["Fuel_Type", "Seller_Type", "Transmission"]
num_cols = ["Present_Price", "Kms_Driven", "Owner", "Car_Age"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols),
    ]
)

# 6) Modelo (classificação) — limitar profundidade para visualização melhor
clf = DecisionTreeClassifier(random_state=42, class_weight="balanced", max_depth=4)

# 7) Pipeline
pipe = Pipeline(steps=[
    ("prep", preprocessor),
    ("clf", clf),
])

# 8) Split estratificado (mantém proporção das classes)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 9) Treinar
pipe.fit(X_train, y_train)

# 10) Avaliar
y_pred = pipe.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f}\n")
print("Relatório de classificação:")
print(classification_report(y_test, y_pred, digits=4))

# 11) Obter nomes das features pós-one-hot para plot
#     (pega do ColumnTransformer dentro do pipeline já ajustado)
oh = pipe.named_steps["prep"].named_transformers_["cat"]
oh_names = list(oh.get_feature_names_out(cat_cols))
feature_names = oh_names + num_cols

# 12) Plotar a árvore
plt.figure(figsize=(16, 10))
plot_tree(
    pipe.named_steps["clf"],
    feature_names=feature_names,
    class_names=["baixo", "medio", "alto"],
    filled=True,
    rounded=True,
    proportion=False,
    fontsize=8
)

# 13) Exportar como SVG no stdout (como no seu exemplo) e salvar em arquivo
buffer = StringIO()
plt.savefig(buffer, format="svg", bbox_inches="tight")
svg_content = buffer.getvalue()
print(svg_content)                 # imprime o SVG (útil para incorporar em HTML)
with open("tree.svg", "w", encoding="utf-8") as f:
    f.write(svg_content)          # salva para usar no MkDocs: ![Árvore](tree.svg)

plt.show()