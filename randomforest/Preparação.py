import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline
import textwrap

url = "https://raw.githubusercontent.com/LuisDegaspari/DataSet/main/cardekho_data.csv"
df = pd.read_csv(url)

expected_cols = [
    "Car_Name", "Year", "Selling_Price", "Present_Price",
    "Kms_Driven", "Fuel_Type", "Seller_Type", "Transmission", "Owner"
]
missing_cols = [c for c in expected_cols if c not in df.columns]
if missing_cols:
    raise ValueError(f"Colunas ausentes no dataset: {missing_cols}")

df = df[expected_cols].copy()
df["Car_Age"] = df["Year"].max() - df["Year"]
df["Brand"] = df["Car_Name"].astype(str).str.split().str[0].str.lower()

def iqr_clip(series: pd.Series, lower_q=0.01, upper_q=0.99):
    q_low, q_hi = series.quantile([lower_q, upper_q])
    return series.clip(q_low, q_hi)

for col in ["Present_Price", "Kms_Driven"]:
    df[col] = iqr_clip(df[col], lower_q=0.01, upper_q=0.99)

y = df["Selling_Price"]
X = df.drop(columns=["Selling_Price", "Car_Name"])

numeric_cols = ["Present_Price", "Kms_Driven", "Owner", "Year", "Car_Age"]
low_card_cat = ["Fuel_Type", "Seller_Type", "Transmission", "Brand"]

def numeric_transformer_df(X_num: pd.DataFrame):
    X_num = X_num.copy()
    if "Kms_Driven" in X_num.columns:
        X_num["Kms_Driven"] = np.log1p(X_num["Kms_Driven"])
    return X_num

numeric_pipe = Pipeline([
    ("log_kms", FunctionTransformer(numeric_transformer_df, validate=False, feature_names_out="one-to-one"))
])

cat_pipe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

preprocessor = ColumnTransformer(
    [
        ("num", numeric_pipe, numeric_cols),
        ("cat", cat_pipe, low_card_cat),
    ],
    remainder="drop",
    verbose_feature_names_out=False
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

preprocessor.fit(X_train, y_train)
X_train_prepared = preprocessor.transform(X_train)
X_test_prepared = preprocessor.transform(X_test)
feature_names = preprocessor.get_feature_names_out()

txt = textwrap.dedent(f"""
<pre>
PREPARAÇÃO E DIVISÃO DOS DADOS
-------------------------------
Amostras totais: {len(X)}
Treino: {len(X_train)} | Teste: {len(X_test)}
Proporção: {len(X_train)/len(X):.1%} treino | {len(X_test)/len(X):.1%} teste

Colunas numéricas usadas:
{", ".join(numeric_cols)}

Colunas categóricas usadas:
{", ".join(low_card_cat)}

Matrizes pós-encoding:
X_train_prepared: {X_train_prepared.shape}
X_test_prepared : {X_test_prepared.shape}
Número total de features: {len(feature_names)}

Primeiras 15 features após encoding:
{", ".join(feature_names[:15]) + ("..." if len(feature_names) > 15 else "")}
</pre>
""").strip()

print(txt)
