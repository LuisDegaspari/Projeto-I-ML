### Preparação dos Dados

=== "Saida"
    ``` python exec="1" html="1"

    --8<-- "docs\randomforest\Preparação.py"

    ```
=== "Codigo"
    ``` python exec="0"

    --8<-- "docs\randomforest\Preparação.py"

    ```
Os dados foram tratados com foco em melhorar a qualidade das variáveis antes da modelagem.
Foi criada a variável Car_Age, representando a idade do veículo em relação ao ano de 2025.
A variável Kms_Driven passou por transformação logarítmica para reduzir a influência de outliers.
As colunas Year e Car_Name foram removidas por não contribuírem ao modelo.
O preço de venda (Selling_Price) foi discretizado em três faixas — baixo, médio e alto.
Essas etapas tornaram o conjunto mais consistente e adequado para classificação supervisionada.

### Divisão dos dados

=== "Saida"
    ``` python exec="1" html="1"

    --8<-- "docs\randomforest\Divisãodados.py"

    ```
=== "Codigo"
    ``` python exec="0"

    --8<-- "docs\randomforest\Divisãodados.py"

    ```

A base foi dividida em dois subconjuntos: 80% para treino e 20% para teste.
O método train_test_split foi utilizado com estratificação, garantindo equilíbrio entre as classes.
Essa abordagem assegura que as proporções das faixas de preço sejam semelhantes nos dois grupos.
O conjunto de treino é usado para ajustar os parâmetros do modelo, enquanto o teste avalia o desempenho.
A distribuição equilibrada reduz o risco de viés e melhora a generalização do algoritmo.
Essa divisão garante uma avaliação justa da capacidade preditiva da Random Forest.

### Random Forest

=== "Saida"
    ``` python exec="1" html="1"

    --8<-- "docs\randomforest\Randomforest.py"

    ```
=== "Codigo"
    ``` python exec="0"

    --8<-- "docs\randomforest\Randomforest.py"

    ```

O modelo utilizado foi o Random Forest Classifier, composto por múltiplas árvores de decisão.
Ele foi configurado com 100 árvores, profundidade máxima de 5 e amostragem aleatória de variáveis.
Essa estrutura aumenta a robustez e reduz a variância dos resultados.
O modelo apresentou boa acurácia na classificação das faixas de preço dos veículos.
As variáveis Present_Price e Car_Age se destacaram como as mais relevantes na previsão.
O uso da Random Forest garantiu previsões estáveis e interpretáveis dentro do conjunto de teste.