

Esse Dataset é sobre o valor de carros usados e como podemos prever o valor deles.Durante a análise do dataset CarDekho, encontrei algumas complicações: a coluna Car_Name tinha muitos valores distintos e pouco úteis para a modelagem, sendo descartada. Também havia variáveis categóricas (Fuel_Type, Seller_Type, Transmission) que exigiram codificação, além da presença de outliers em Kms_Driven e Selling_Price, que distorciam a distribuição e precisaram ser tratados.




Exploração dos dados

Principais colunas do dataset

-Car_Name → Nome do modelo do carro (não utilizada na modelagem, pois possui alta cardinalidade).

-Year → Ano de fabricação do veículo (convertido em Car_Age no pré-processamento).

-Selling_Price → Preço efetivo de revenda do carro (variável alvo para regressão).

-Present_Price → Preço de mercado de um carro zero km do mesmo modelo.

-Kms_Driven → Quilometragem total rodada pelo veículo.

-Owner → Quantidade de donos anteriores.

-Fuel_Type → Tipo de combustível (ex.: Diesel, Petrol, CNG).

-Seller_Type → Tipo de vendedor (Particular ou Revendedor).

-Transmission → Tipo de transmissão (Manual ou Automática).

=== "Saida"
    ``` python exec="1" html="1"

    --8<-- "docs\arvore\exploracao.py"

    ```
=== "Codigo"
    ``` python exec="0"

    --8<-- "docs\arvore\exploracao.py"


    ```

Nesse gráfico podemos perceber que os Km_Rodado, tem relação com o preço dos veiculos vendidos, infelizmente esse Dataset tem muitos carros com quase 0 Km, e pouco usado.

=== "Saida"
    ``` python exec="1" html="1"

    --8<-- "docs\arvore\exploracao2.py"

    ```
=== "Codigo"
    ``` python exec="0"

    --8<-- "docs\arvore\exploracao2.py"


    ```
Gráfico de dispersão evidenciando que **carros mais novos** (anos recentes) tendem a ter preços de venda mais elevados, enquanto veículos mais antigos valem menos.  



=== "Saida"
    ``` python exec="1"

    --8<-- "docs\arvore\pre-processamento.py"


    ```
=== "Codigo"
    ``` python exec="0"

    --8<-- "docs\arvore\pre-processamento.py"


    ```
No pré-processamento do dataset CarDekho, criei a variável Car_Age a partir do ano de fabricação e normalizei a quilometragem. Removi colunas irrelevantes como Car_Name, tratei valores ausentes com mediana (numéricos) e “Unknown” (categóricos), e apliquei One-Hot Encoding para variáveis categóricas. Também limitei outliers e criei a variável alvo price_bucket (baixo, médio, alto), deixando a base pronta para regressão e classificação.

Após o pré-processamento, o dataset final possui **301 registros e 14 colunas**, sem valores ausentes.


=== "Saida"
    ``` python exec="1"

    --8<-- "docs\arvore\divisao-dados.py"


    ```
=== "Codigo"
    ``` python exec="0"

    --8<-- "docs\arvore\divisao-dados.py"


    ```
Com o dataset tratado, realizamos a separação em **conjunto de treino e teste**, garantindo reprodutibilidade e estratificação da variável alvo `price_bucket`.
Isso garante que os modelos sejam avaliados em dados nunca vistos, mantendo a proporção entre as classes.


=== "Saida"
    ``` python exec="1" html="1"

    --8<-- "docs\arvore\decision-tree.py"

    ```
=== "Codigo"
    ``` python exec="0"

    --8<-- "docs\arvore\decision-tree.py"


    ```

Aqui podemos perceber que nesse dataset o nível de importância do preço atual do carro e a idade (ano de fabricação) são os fatores que mais influenciam na decisão de compra ou no valor de revenda. Variáveis como tipo de combustível, número de donos e transmissão têm pouca relevância prática neste conjunto de dados.