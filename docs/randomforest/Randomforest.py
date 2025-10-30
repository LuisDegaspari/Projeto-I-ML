# Random Forest Regressor - CarDekho Dataset
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np

# Load dataset
url = "https://raw.githubusercontent.com/LuisDegaspari/DataSet/refs/heads/main/cardekho_data.csv"
df = pd.read_csv(url)

# Feature engineering
df["Car_Age"] = df["Year"].max() - df["Year"]  # Cria idade do carro
df = df.drop("Year", axis=1)                   # Remove o ano original

# Encode categorical variables
df = pd.get_dummies(df, columns=["Fuel_Type", "Seller_Type", "Transmission"], drop_first=True)

# Optionally extract car brand (reduz cardinalidade)
df["Brand"] = df["Car_Name"].str.split().str[0].str.lower()
df = df.drop("Car_Name", axis=1)
df = pd.get_dummies(df, columns=["Brand"], drop_first=True)

# Define features and target
X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
rf = RandomForestRegressor(
    n_estimators=100,     # Número de árvores
    max_depth=10,         # Profundidade máxima das árvores
    max_features='sqrt',  # Número de features por split
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

# Predict and evaluate
predictions = rf.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.3f}")

# Feature importances
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nFeature Importances:")
output = f"""
RMSE: {rmse:.2f}
R²: {r2:.3f}
    
Feature Importances:
--------------------
{importances.head(10).to_string()}
"""

print(f"<pre>{output}</pre>")

