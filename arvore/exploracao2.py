import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO

# carregar dataset
url = "https://raw.githubusercontent.com/LuisDegaspari/DataSet/refs/heads/main/cardekho_data.csv"
df = pd.read_csv(url)

# scatter plot: Ano do carro x Preço de venda
plt.figure(figsize=(8,5))
plt.scatter(df["Year"], df["Selling_Price"], alpha=0.6, c="darkorange", edgecolors="k")
plt.title("Preço de Venda x Ano do Carro")
plt.xlabel("Ano do Carro")
plt.ylabel("Preço de Venda em mil")
plt.grid(True, linestyle="--", alpha=0.6)
buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=False)
print(buffer.getvalue())
