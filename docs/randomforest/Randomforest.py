# Random Forest Regressor - CarDekho Dataset
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np
import textwrap

# 1) Carregar dataset
url = "https://raw.githubusercontent.com/LuisDegaspari/DataSet/main/cardekho_data.csv"
df = pd.read_csv(url)

# 2) Feature engineering
df["Car_Age"] = df["Year"].max() - df["Year"]
df = df.drop("Year", axis=1)

# 3) Encoding de variáveis categóricas
df = pd.get_dummies(df, columns=["Fuel_Type", "Seller_Type", "Transmission"], drop_first=True)
df["Brand"] = df["Car_Name"].str.split().str[0].str.lower()
df = df.drop("Car_Name", axis=1)
df = pd.get_dummies(df, columns=["Brand"], drop_first=True)

# 4) Definir features e target
X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]

# 5) Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6) Modelo Random Forest
rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

# 7) Avaliação
pred = rf.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)

# 8) Importância das features
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
top_features = importances.head(10).to_string()

# 9) Saída formatada
txt = textwrap.dedent(f"""
<pre>
RANDOM FOREST REGRESSOR — RESULTADOS
------------------------------------
RMSE: {rmse:.2f}
R²: {r2:.3f}

Top 10 Feature Importances:
---------------------------
{top_features}
</pre>
""").strip()

print(txt)
