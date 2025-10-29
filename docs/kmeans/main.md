

=== "Saida"
    ``` python exec="1" html ="1"

    --8<-- "docs\kmeans\k-means.py"

    ```
=== "Codigo"
    ``` python exec="0"

    --8<-- "docs\kmeans\k-means.py"


    ```

No dataset CarDekho, apliquei o algoritmo K-Means para identificar agrupamentos de carros de forma não supervisionada. As variáveis categóricas foram transformadas por one-hot encoding, os dados foram normalizados e, em seguida, utilizei o PCA para reduzir as dimensões a dois componentes principais, permitindo a visualização. No gráfico, cada ponto representa um carro, colorido conforme o cluster ao qual pertence, enquanto as estrelas vermelhas indicam os centróides dos grupos. Esse resultado mostra como o K-Means conseguiu separar os carros em três perfis distintos com base em suas características.