import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Carregar dataset
url = "https://raw.githubusercontent.com/LuisDegaspari/DataSet/refs/heads/main/cardekho_data.csv"
df = pd.read_csv(url)

# 2. Feature engineering
df["Car_Age"] = 2025 - df["Year"]   # idade do carro
df = df.drop(columns=["Year", "Car_Name"])  # descartar colunas irrelevantes

# 3. Separar X e y
X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]

# 4. Pré-processamento
cat_cols = ["Fuel_Type", "Seller_Type", "Transmission"]
num_cols = ["Present_Price", "Kms_Driven", "Owner", "Car_Age"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first"), cat_cols),
        ("num", "passthrough", num_cols)
    ]
)

# 5. Pipeline com RandomForest
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=200, random_state=42))
])

# 6. Treino/teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# 7. Avaliação
y_pred = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))
