import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from io import StringIO

# 1) Dataset
url = "https://raw.githubusercontent.com/LuisDegaspari/DataSet/refs/heads/main/cardekho_data.csv"
df = pd.read_csv(url)

# 2) Feature engineering
df = df.copy()
df["Car_Age"] = 2025 - df["Year"]
df = df.drop(columns=["Year", "Car_Name"])

# 3) Classes (tercis)
y = pd.qcut(df["Selling_Price"], q=3, labels=["baixo", "medio", "alto"])

# 4) Features (sem Selling_Price) + one-hot com pandas
X = df.drop(columns=["Selling_Price"])
cat_cols = ["Fuel_Type", "Seller_Type", "Transmission"]
X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

# 5) Split estratificado
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 6) Modelo
clf = tree.DecisionTreeClassifier(
    random_state=42,
    class_weight="balanced",
    max_depth=4
)
clf.fit(X_train, y_train)

# 7) Avaliação
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f}<br>")

# 8) Importância das features -> em %
fi = pd.DataFrame({
    "Feature": list(X_train.columns),
    "Importância (%)": (clf.feature_importances_ * 100)
})

# arredondar e ordenar
fi["Importância (%)"] = fi["Importância (%)"].round(2)
fi = fi.sort_values(by="Importância (%)", ascending=False)

print("<br><b>Importância das Features:</b>")
print(fi.to_html(index=False))  # tabela HTML pronta para o MkDocs

# 9) Plot da árvore -> SVG no stdout
plt.figure(figsize=(16, 10))
class_names = list(y.cat.categories) if hasattr(y, "cat") else sorted(pd.unique(y))

tree.plot_tree(
    clf,
    feature_names=list(X.columns),
    class_names=class_names,
    filled=True,
    rounded=True,
    fontsize=8
)

buffer = StringIO()
plt.savefig(buffer, format="svg", bbox_inches="tight")
print(buffer.getvalue())
