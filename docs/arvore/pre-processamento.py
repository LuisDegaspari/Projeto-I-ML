import pandas as pd

# ================================
# Carregar dados
# ================================
url = "https://raw.githubusercontent.com/LuisDegaspari/DataSet/refs/heads/main/cardekho_data.csv"
df = pd.read_csv(url)

# ================================
# 1) Feature engineering simples
#    - cria idade do carro
#    - normaliza (opcional) km para uma escala 0-1 só para ter freq/escala comparável
# ================================
df["Car_Age"] = 2025 - df["Year"]

# normalização simples (0-1) só como exemplo, não obrigatório
km_min, km_max = df["Kms_Driven"].min(), df["Kms_Driven"].max()
df["Kms_Scaled"] = (df["Kms_Driven"] - km_min) / (km_max - km_min)

# ================================
# 2) Selecionar colunas úteis
# ================================
cols = [
    "Selling_Price", "Present_Price", "Kms_Driven", "Kms_Scaled", "Owner",
    "Fuel_Type", "Seller_Type", "Transmission", "Car_Age"
]
df = df[cols]

# ================================
# 3) Tratamento de valores faltantes
# ================================
num_cols = ["Selling_Price", "Present_Price", "Kms_Driven", "Kms_Scaled", "Owner", "Car_Age"]
cat_cols = ["Fuel_Type", "Seller_Type", "Transmission"]

# numéricos -> mediana
for c in num_cols:
    df[c] = df[c].fillna(df[c].median())

# categóricos -> 'Unknown'
for c in cat_cols:
    df[c] = df[c].fillna("Unknown")

# ================================
# 4) Binarizações / dummies
#    - criar indicadores simples para presença de categorias
#    - e one-hot nas categóricas (drop_first para evitar colinearidade)
# ================================
# indicadores (exemplo simples)
df["is_Dealer"]   = (df["Seller_Type"] == "Dealer").astype(int)
df["is_Manual"]   = (df["Transmission"] == "Manual").astype(int)
df["is_Diesel"]   = (df["Fuel_Type"] == "Diesel").astype(int)

# one-hot completo (útil para modelos lineares)
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# ================================
# 5) (Opcional) tratar outliers simples
#    - winsorizar Kms_Driven e Selling_Price nos percentis 1% e 99%
# ================================
for c in ["Kms_Driven", "Selling_Price", "Present_Price"]:
    low, high = df[c].quantile(0.01), df[c].quantile(0.99)
    df[c] = df[c].clip(lower=low, upper=high)

# ================================
# 6) (Opcional) criar alvo de CLASSIFICAÇÃO a partir do preço
#    - 3 faixas: baixo / médio / alto
# ================================
df["price_bucket"] = pd.qcut(df["Selling_Price"], q=3, labels=["baixo", "medio", "alto"])

# ================================
# Exportar e prints de checagem
# ================================
# df.to_csv("data/cardekho_preprocessado.csv", index=False)

print(f"Valores ausentes após pré-processamento: {int(df.isnull().sum().sum())}\n")
print(f"Formato final: {df.shape}\n")
print("Distribuição do price_bucket (se criado):\n")
print(df["price_bucket"].value_counts(dropna=False).to_markdown())
