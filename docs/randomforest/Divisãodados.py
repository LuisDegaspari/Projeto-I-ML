import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# 1) Carregamento
url = "https://raw.githubusercontent.com/LuisDegaspari/DataSet/refs/heads/main/cardekho_data.csv"
df = pd.read_csv(url)

# 2) Preparação
df["Car_Age"] = 2025 - df["Year"]
df["Kms_Driven"] = np.log1p(df["Kms_Driven"])
df = df.drop(columns=["Year", "Car_Name"])

# 3) Buckets de preço (com fallback para empates)
tmp = pd.qcut(df["Selling_Price"], q=3, duplicates="drop")
n_bins = tmp.cat.categories.size
labels = ["baixo", "medio", "alto"][:n_bins]
df["price_bucket"] = pd.qcut(df["Selling_Price"], q=n_bins, labels=labels, duplicates="drop")

# 4) Features / Target
features = ["Present_Price", "Kms_Driven", "Owner", "Car_Age"]
target = "price_bucket"
X = df[features]
y = df[target]

# 5) Divisão (80/20 com estratificação)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 6) Tabelas de distribuição (contagem e proporção)
def dist_table(s: pd.Series, order):
    out = pd.concat([s.value_counts().reindex(order), 
                     s.value_counts(normalize=True).reindex(order)], axis=1)
    out.columns = ["count", "prop"]
    return out.fillna(0)

order = labels
train_dist = dist_table(y_train, order)
test_dist  = dist_table(y_test, order)

# 7) Saída formatada (igual ao estilo do RandomForest)
output = []
output.append("PREPARAÇÃO E DIVISÃO DOS DADOS")
output.append("-------------------------------")
output.append(f"Amostras totais: {len(X)}")
output.append(f"Treino: {len(X_train)} | Teste: {len(X_test)}")
output.append(f"Proporção: {len(X_train)/len(X):.1%} treino | {len(X_test)/len(X):.1%} teste\n")

output.append("Features utilizadas:")
output.append(", ".join(features) + "\n")

output.append("Distribuição das classes — Treino")
output.append(train_dist.to_string(float_format=lambda v: f"{v:.3f}"))
output.append("\nDistribuição das classes — Teste")
output.append(test_dist.to_string(float_format=lambda v: f"{v:.3f}"))

# 8) Imprimir com preservação de quebras no MkDocs/HTML
print(f"<pre>{'\n'.join(output)}</pre>")
