

Esse Dataset é sobre o valor de carros usados e como podemos prever o valor deles




Exploração dos dados

Colunas mais importantes

Present_Price : preço atual de mercado de um carro zero km do mesmo modelo.

Car_Age : idade do carro (2025 – ano de fabricação).

Kms_Driven : total de quilômetros rodados.

Selling_Price : preço efetivo de revenda (variável alvo).

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
Aqui nessa vizualização vemos que quanto mais novos os carros mais caros são eles.


Vamos começar o pré-processamento dos dados

=== "Saida"
    ``` python exec="1"

    --8<-- "docs\arvore\pre-processamento.py"


    ```
=== "Codigo"
    ``` python exec="0"

    --8<-- "docs\arvore\pre-processamento.py"


    ```

Agora vai ter a divisão dos dados

=== "Saida"
    ``` python exec="1"

    --8<-- "docs\arvore\divisao-dados.py"


    ```
=== "Codigo"
    ``` python exec="0"

    --8<-- "docs\arvore\divisao-dados.py"


    ```









Começaremos com a Decision Tree

=== "Saida"
    ``` python exec="1" html="1"

    --8<-- "docs\arvore\decision-tree.py"

    ```
=== "Codigo"
    ``` python exec="0"

    --8<-- "docs\arvore\decision-tree.py"


    ```

Aqui podemos perceber que nesse dataset o nível de importância do preço atual do carro e a idade (ano de fabricação) são os fatores que mais influenciam na decisão de compra ou no valor de revenda. Variáveis como tipo de combustível, número de donos e transmissão têm pouca relevância prática neste conjunto de dados.