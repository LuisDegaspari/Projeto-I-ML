import pandas as pd
from sklearn.model_selection import train_test_split

# carregar direto do GitHub
url = "https://raw.githubusercontent.com/LuisDegaspari/DataSet/refs/heads/main/cardekho_data.csv"
df = pd.read_csv(url)

# feature engineering rápido
df["Car_Age"] = 2025 - df["Year"]
df = df.drop(columns=["Year", "Car_Name"])

# target categórico
df["price_bucket"] = pd.qcut(df["Selling_Price"], q=3, labels=["baixo", "medio", "alto"])

# features
features = ["Present_Price", "Kms_Driven", "Owner", "Car_Age"]
target = "price_bucket"

X = df[features]
y = df[target]

# divisão treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(
    f"Treino: {X_train.shape[0]} amostras<br>"
    f"Teste: {X_test.shape[0]} amostras<br>"
    f"Proporção: {X_train.shape[0]/X.shape[0]*100:.1f}% treino, "
    f"{X_test.shape[0]/X.shape[0]*100:.1f}% teste<br><br>"
)

print("<b>Distribuição das classes — Treino:</b><br>")
print(y_train.value_counts().to_frame("count").to_html())
print("<br><b>Distribuição das classes — Teste:</b><br>")
print(y_test.value_counts().to_frame("count").to_html())

