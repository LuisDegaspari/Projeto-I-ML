import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Carregar dataset
url = "https://raw.githubusercontent.com/LuisDegaspari/DataSet/refs/heads/main/cardekho_data.csv"
df = pd.read_csv(url)

# Remover colunas irrelevantes
df = df.drop(columns=["Car_Name"])

# Definir colunas categóricas
categorical_cols = ["Fuel_Type", "Seller_Type", "Transmission"]

# Aplicar OneHotEncoder
encoder = OneHotEncoder(drop='first', sparse_output=False)
encoded_features = encoder.fit_transform(df[categorical_cols])

# Criar DataFrame com colunas novas
encoded_df = pd.DataFrame(
    encoded_features,
    columns=encoder.get_feature_names_out(categorical_cols),
    index=df.index
)

# Concatenar: dados numéricos + dummies
df_final = df.drop(categorical_cols, axis=1)
df_final = pd.concat([df_final, encoded_df], axis=1)

print(df_final.head())
