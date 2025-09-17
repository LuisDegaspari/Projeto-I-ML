Agora 

Nesse caso, utilizei todas as colunas relevantes do dataset após o pré-processamento e dividi o preço em três faixas com qcut. O modelo trabalhou com muito mais informações (preço presente, km rodado, idade do carro e variáveis categóricas codificadas) e teve de lidar com fronteiras de decisão mais complexas entre três classes. Por isso, a acurácia ficou em torno de 0.90, refletindo a maior dificuldade de separar corretamente as categorias.
=== "Saida"
    ``` python exec="1"

    --8<-- "docs\KNN\knn.py"

    ```
=== "Codigo"
    ``` python exec="0"

    --8<-- "docs\KNN\knn.py"


    ```


Para gerar o gráfico, simplifiquei o problema para duas variáveis (Present_Price e Car_Age) e apenas duas classes (caro ou barato, usando a mediana como limite). Essas variáveis têm forte relação com o preço e permitem uma separação mais clara entre as classes. O modelo conseguiu capturar bem essa divisão e atingiu uma acurácia mais alta, de 0.95, mas em um cenário mais simples e menos realista do que o anterior.
=== "Saida"
    ``` python exec="1" html="1"

    --8<-- "docs\KNN\grafico.py"

    ```
=== "Codigo"
    ``` python exec="0"

    --8<-- "docs\KNN\grafico.py"


    ```