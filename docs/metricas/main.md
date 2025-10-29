### Agora iremos falar sobre a diferença entre o KNN e o k-means.




### KNN

Define um alvo categorizado por preço (price_bucket) ⇒ problema de classificação.

Faz split e normaliza usando só o treino ⇒ evita data leakage.

Implementa KNN do zero com distância Euclidiana e votação majoritária (k=5).

Mede performance com accuracy no conjunto de teste.

=== "Saida"
    ``` python exec="1" html="1"

    --8<-- "docs\metricas\metrics-KNN.py"

    ```
=== "Codigo"
    ``` python exec="0"

    --8<-- "docs\metricas\metrics-KNN.py"

    ```



### K-Means

Remove Selling_Price do X ⇒ sem alvo, agrupamento puro.

Padroniza todo X (ok, pois não há avaliação supervisionada).

Ajusta KMeans(k=3) com k-means++ e semente fixa.

Projeta com PCA para plotar pontos e centróides (visual, não altera o treino).

Exporta o gráfico em SVG para você embutir onde quiser.

=== "Saida"
    ``` python exec="1" html="1"

    --8<-- "docs\metricas\metrics-kmeans.py"

    ```
=== "Codigo"
    ``` python exec="0"

    --8<-- "docs\metricas\metrics-kmeans.py"

    ```


Portanto,o KNN é um algoritmo supervisionado, usado para classificação, que depende de rótulos conhecidos e mede desempenho por métricas como acurácia.Já o K-Means é não supervisionado, voltado para agrupamento, formando clusters sem necessidade de classes pré-definidas.Enquanto o KNN prevê a qual categoria um novo ponto pertence com base nos vizinhos mais próximos, o K-Means busca identificar padrões e dividir os dados em grupos semelhantes.