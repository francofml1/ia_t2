# -*- coding: utf-8 -*-

## Importações

from math import sqrt

# Manipulação de dados
import numpy as np
import pandas as pd


# Geração de números aleatórios
import random

# Geração de gráficos
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

"""## Funções Auxiliares

### Solução Aleatória
"""

# Cria uma solucao inicial com as cidades em um ordem aleatoria


def solucao_aleatoria(tsp):
    cidades = list(tsp.keys())
    solucao = []

    # as 3 linhas abaixo não são estritamente necessarias, servem
    # apenas para fixar a primeira cidade da lista na solução
    cidade = cidades[0]
    solucao.append(cidade)
    cidades.remove(cidade)

    for _ in range(0, len(cidades)):
        # print(_, cidades, solucao)
        cidade = random.choice(cidades)

        solucao.append(cidade)
        cidades.remove(cidade)

    return solucao


def solucao_aleatoria_prox(tsp):
    """Cria uma solucao inicial com as cidades em um ordem de PROXIMIDADE
    testes demonstraram que esta solução fica presa em um mínimo local
    """

    cidades = list(tsp.keys())
    solucao = []

    # as 3 linhas abaixo não são estritamente necessarias, servem
    # apenas para fixar a primeira cidade da lista na solução
    cidade = cidades[0]
    solucao.append(cidade)
    cidades.remove(cidade)

    for _ in range(0, len(cidades)):
        # print(_, cidades, solucao, ">>>")

        dist_cid = tsp.loc[cidade].items()
        idx_val = [
            (idx, val) for idx, val in dist_cid if val > 0 and idx not in solucao
        ]

        cidade = min(idx_val, key=lambda k: k[1])[0]

        # print(_,cidade, idx_val, list(dist_cid))

        solucao.append(cidade)
        cidades.remove(cidade)

    return solucao


"""### Calcula Custo"""


def calcula_custo(tsp, solucao):
    # Função Objetivo: calcula custo de uma dada solução.
    # Obs: Neste caso do problema do caixeiro viajante (TSP problem),
    # o custo é o comprimento da rota entre todas as cidades.
    N = len(solucao)
    custo = 0

    for i in range(N):
        # Quando chegar na última cidade, será necessário
        # voltar para o início para adicionar o
        # comprimento da rota da última cidade
        # até a primeira cidade, fechando o ciclo.
        #
        # Por isso, a linha abaixo:
        k = (i + 1) % N
        cidadeA = solucao[i]
        cidadeB = solucao[k]

        custo += tsp.loc[cidadeA, cidadeB]

        # print(tsp.loc[cidadeA, cidadeB], cidadeA,cidadeB)

    return custo


"""### Gera Vizinhos

Obs: a função `obtem_vizinhos` descrita abaixo foi gerada de forma simplificada, pois ela assume que todos os vizinhos possuem rota direta entre si. Isto tem caráter didático para simplifcar a solução. Observe que na prática isso nem sempre existe rotas diretas entre todas as cidades e, em tais casos, pode ser necessário modificar a função para corresponder a tais restrições.
"""


def gera_vizinhos(solucao):
    # A partir de uma dada solução, gera diversas variações (vizinhos)
    N = len(solucao)
    for i in range(1, N):  # deixa o primeiro fixo
        for j in range(i + 1, N):
            vizinho = solucao.copy()
            vizinho[i] = solucao[j]
            vizinho[j] = solucao[i]

            yield (vizinho)


def obtem_melhor_vizinho(tsp, solucao):
    # Seleciona Melhor Vizinho
    melhor_custo = calcula_custo(tsp, solucao)
    melhor_vizinho = solucao

    for vizinho in gera_vizinhos(solucao):
        custo_atual = calcula_custo(tsp, vizinho)
        if custo_atual < melhor_custo:
            melhor_custo = custo_atual
            melhor_vizinho = vizinho

    return melhor_vizinho, melhor_custo


### Random-Walk - clássico
def obtem_vizinho_aleatorio(tsp, solucao):
    vizinhos = list(gera_vizinhos(solucao))

    aleatorio_vizinho = random.choice(vizinhos)
    aleatorio_custo = calcula_custo(tsp, aleatorio_vizinho)

    return aleatorio_vizinho, aleatorio_custo


def random_walk(tsp):
    solucao_inicial = solucao_aleatoria(tsp)

    atual_solucao, atual_custo = obtem_vizinho_aleatorio(tsp, solucao_inicial)

    for _ in range(30):
        atual_solucao, atual_custo = obtem_vizinho_aleatorio(tsp, atual_solucao)

    return atual_custo, atual_solucao


"""### Hill-Climbing - clássico"""
# def hill_climbing(tsp):
#     solucao_inicial = solucao_aleatoria(tsp)

#     melhor_solucao, melhor_custo = obtem_melhor_vizinho(tsp, solucao_inicial)

#     while True:
#         vizinho_atual = melhor_solucao
#         custo_atual   = melhor_custo

#         melhor_solucao, melhor_custo = obtem_melhor_vizinho(tsp, vizinho_atual)
#         #print(melhor_custo)

#         if custo_atual <= melhor_custo:
#             break

#     return custo_atual, melhor_solucao


def hill_climbing(tsp):
    # solucao inicial
    solucao_inicial = solucao_aleatoria(tsp)
    # melhor solucao ate o momento
    solucao_melhor, custo_melhor = obtem_melhor_vizinho(tsp, solucao_inicial)

    while True:
        # tenta obter um candidato melhor
        candidato_atual, custo_atual = obtem_melhor_vizinho(tsp, solucao_melhor)
        # print(custo_melhor, custo_atual)

        if custo_atual < custo_melhor:
            custo_melhor = custo_atual
            solucao_melhor = candidato_atual
        else:
            break  # custo nao melhorou, entao sai do while

    return custo_melhor, solucao_melhor


def hill_climbing_restart(tsp):
    for _ in range(50):
        # solucao inicial
        solucao_inicial = solucao_aleatoria(tsp)
        # melhor solucao ate o momento
        solucao_melhor, custo_melhor = obtem_melhor_vizinho(tsp, solucao_inicial)

        while True:
            # tenta obter um candidato melhor
            candidato_atual, custo_atual = obtem_melhor_vizinho(tsp, solucao_melhor)
            # print(custo_melhor, custo_atual)

            if custo_atual < custo_melhor:
                custo_melhor = custo_atual
                solucao_melhor = candidato_atual
            else:
                break  # custo nao melhorou, entao sai do while

    return custo_melhor, solucao_melhor


"""### Cálculo da Matriz de Distâncias"""


def distancia(x1, y1, x2, y2):
    """Distancia Euclidiana entre dois pontos"""
    dx = x2 - x1
    dy = y2 - y1
    return sqrt(dx**2 + dy**2)


def gera_matriz_distancias(Coordenadas):
    """### Calcula matriz de distancias.

    OBS:  Não é estritamente necessario calculá-las a priori.
          Foi feito assim apenas para fins didáticos.
          Ao invés, as distâncias podem ser calculadas sob demanda.
    """
    n_cidades = len(Coordenadas)
    dist = np.zeros((n_cidades, n_cidades), dtype=float)

    for i in range(0, n_cidades):
        for j in range(i + 1, n_cidades):
            x1, y1 = Coordenadas.iloc[i]
            x2, y2 = Coordenadas.iloc[j]

            dist[i, j] = distancia(x1, y1, x2, y2)
            dist[j, i] = dist[i, j]

    return dist


def gera_coordenadas_aleatorias(n_cidades: int):
    """### Gerador de Problemas Aleatórios

    Gera aleatoriamente as coordenadas de N cidades.
    Obs: esta informação geralmente é fornecida como entrada do problema.
    """
    minimo = 10
    maximo = 90
    escala = (maximo - minimo) - 1

    # gera n coordenadas (x,y) aleatorias entre [min, max]
    X = minimo + escala * np.random.rand(n_cidades)
    Y = minimo + escala * np.random.rand(n_cidades)
    coordenadas = {"X": X, "Y": Y}

    cidades = ["A" + str(i) for i in range(n_cidades)]

    df_cidades = pd.DataFrame(coordenadas, index=cidades)
    df_cidades.index.name = "CIDADE"

    return df_cidades


def gera_problema_tsp(df_cidades: list):
    """Recebe uma lista com as coordenadas reais de uma cidade e
    gera uma matriz de distancias entre as cidades.
    Obs: a matriz é simetrica e com diagonal nula
    """
    # nomes ficticios das cidades
    cidades = df_cidades.index

    # calcula matriz de distancias
    distancias = gera_matriz_distancias(df_cidades)

    # cria estrutura para armazena as distâncias entre todas as cidades
    tsp = pd.DataFrame(distancias, columns=cidades, index=cidades)

    return tsp


"""### Plota Rotas"""

# # Plota Rotas usando a biblioteca SEABORN
# def plota_rotas_sns(df_cidades, ordem_cidades):
#     # Plota a solução do roteamento das cidades
#     df_solucao = df_cidades.copy()
#     df_solucao = df_solucao.reindex(ordem_cidades)

#     sns.scatterplot(data = df_solucao, x = 'X', y = 'Y', s=50)
#     sns.lineplot(data = df_solucao, x = 'X', y = 'Y', sort=False, estimator=None)

#     # liga a última à primeira cidade para fechar o ciclo
#     sns.lineplot(data = df_solucao.iloc[[-1,0]], x = 'X', y = 'Y', sort=False)

#     n_lin = df_solucao.shape[0] # numero de linhas do df
#     X = df_solucao['X']
#     Y = df_solucao['Y']

#     # loop para adicionar anotações uma a uma
#     for i in range(0, n_lin):
#         plt.text(X.iloc[i], Y.iloc[i], df_solucao.index[i],
#                 horizontalalignment='left', size='medium',
#                 color='black', weight='semibold')

#     plt.show()


def plota_rotas(df_cidades, ordem_cidades):
    """Plota a solução do roteamento das cidades
    usando a biblioteca PLOTLY
    """
    df_solucao = df_cidades.copy()
    df_solucao = df_solucao.reindex(ordem_cidades)

    X = df_solucao["X"]
    Y = df_solucao["Y"]
    cidades = list(df_solucao.index)

    # cria objeto gráfico
    fig = go.Figure()

    fig.update_layout(autosize=False, width=500, height=500, showlegend=False)

    # gera linhas com as rotas da primeira ate a ultima cidade
    fig.add_trace(
        go.Scatter(
            x=X,
            y=Y,
            text=cidades,
            textposition="bottom center",
            mode="lines+markers+text",
            name="",
        )
    )

    # acrescenta linha da última para a primeira para fechar o ciclo
    fig.add_trace(
        go.Scatter(x=X.iloc[[-1, 0]], y=Y.iloc[[-1, 0]], mode="lines+markers", name="")
    )

    fig.show()


"""### Boxplots"""


def boxplot_sorted(df, rot=90, figsize=(12, 6), fontsize=20):
    df2 = df.T
    meds = df2.median().sort_values(ascending=False)
    axes = df2[meds.index].boxplot(
        figsize=figsize,
        rot=rot,
        fontsize=fontsize,
        boxprops=dict(linewidth=4, color="cornflowerblue"),
        whiskerprops=dict(linewidth=4, color="cornflowerblue"),
        medianprops=dict(linewidth=4, color="firebrick"),
        capprops=dict(linewidth=4, color="cornflowerblue"),
        flierprops=dict(
            marker="o",
            markerfacecolor="dimgray",
            markersize=12,
            markeredgecolor="black",
        ),
        return_type="axes",
    )

    axes.set_title("Cost of Algorithms", fontsize=fontsize)


"""## Execução"""


def run_once():
    """Executa 1 vez"""

    # Simula a criação de N cidades
    # com suas respectivas distâncias

    n_cidades = 10
    df_coordenadas = gera_coordenadas_aleatorias(n_cidades)
    df_coordenadas

    tsp = gera_problema_tsp(df_coordenadas)
    tsp

    solucao = ["A" + str(i) for i in range(n_cidades)]
    solucao
    plota_rotas(df_coordenadas, solucao)

    solucao = solucao_aleatoria(tsp)
    print(solucao)
    plota_rotas(df_coordenadas, solucao)

    # busca local da melhor solução e o seu custo
    custo, solucao = hill_climbing(tsp)

    print(f"{custo:7.3f}    {solucao}")

    plota_rotas(df_coordenadas, solucao)

    for _ in range(10):
        custo, solucao = hill_climbing(tsp)

        print(f"{custo:7.3f}    {solucao}")

        plota_rotas(df_coordenadas, solucao)


def run_n_times(tsp, n_vezes=30):
    """Executa N vezes - SIMPLES"""

    for _ in range(n_vezes):
        # solucao, custo = random_walk(tsp)
        custo, solucao = hill_climbing(tsp)
        print(f"{custo:7.3f}, {solucao}")

    """Observe que há uma certa variabilidade na soluções acima. Isto se deve à característica estocástica do algoritmo de solução."""


def cria_df_custos(algoritmos, n_vezes):
    """Cria estruta de dados (DataFrame) para armazenar vários resultados
    diferentes e visualizá-los através de estatísticas
    """
    nomes_algoritmos = algoritmos.keys()

    n_lin = len(nomes_algoritmos)
    n_col = n_vezes

    df_results = pd.DataFrame(np.zeros((n_lin, n_col)), index=nomes_algoritmos)
    df_results.index.name = "ALGORITMO"

    return df_results


def executa_n_vezes(tsp, algoritmos, n_vezes):
    """Executa N vezes para gerar estatísticas da variável custo"""
    # Cria DataFrame para armazenar os resultados
    df_custo = cria_df_custos(algoritmos, n_vezes)

    for algoritmo, funcao_algoritmo in algoritmos.items():
        print(algoritmo)

        for i in range(n_vezes):
            custo, solucao = funcao_algoritmo(tsp)
            df_custo.loc[algoritmo, i] = custo

            print(f"{custo:10.3f}  {solucao}")

    return df_custo


def run_n_times_dataframe():
    """
    ### Executa N vezes - ESTRUTURADA com DataFrame

    A seguir, é apresentada uma forma mais estruturada de se rodar várias vezes usando a estrutura de dados **`DataFrame`** para armazenar os resultados e permitir visualização de box-plots
    """

    # Dicionario com Nomes dos modelos e suas respectivas variantes
    # Tuple: (Algoritmo, Variante): funcao_algoritmo
    algoritmos = {"Random Walk": random_walk, "Hill-Climbing": hill_climbing}

    """#### PROBLEMA GERADO ALEATORIAMENTE"""

    ###################################
    # PROBLEMA GERADO ALEATORIAMENTE  #
    ###################################

    # cria instancia do problema com n cidades

    n_cidades = 10
    df_coordenadas = gera_coordenadas_aleatorias(n_cidades)

    tsp = gera_problema_tsp(df_coordenadas)

    # numero de vezes que executará cada algoritmo
    n_vezes = 100

    # Executa N vezes para gerar estatísticas da variável custo
    df_custo = executa_n_vezes(tsp, algoritmos, n_vezes)

    """##### Box Plots"""

    boxplot_sorted(df_custo, rot=90, figsize=(12, 6), fontsize=20)

    """Observe no gráfico acima que a variabilidade do Hill-climbing é bem menor que a do Random-Walk.

    Como você justifica isso?
    """

    df_custo.T.describe()

    """---

#### PROBLEMAS REAIS

A seguir, são apresentados alguns links de problemas reais.

Basta escolher um link de **`download de dados`**, setar a variável URL abaixo, e rodar o algoritmo.

Western Sahara
    29 cidades
    The optimal tour has length 27.603
    Foto dos Pontos: http://www.math.uwaterloo.ca/tsp/world/wipoints.html
    Foto da Solução: http://www.math.uwaterloo.ca/tsp/world/witour.html
    Download dos Dados: http://www.math.uwaterloo.ca/tsp/world/wi29.tsp

    Djibouti dataset
    38 cidades
    The optimal tour has length 6.656
    Foto dos Pontos: http://www.math.uwaterloo.ca/tsp/world/djpoints.html
    Foto da Solução: http://www.math.uwaterloo.ca/tsp/world/djtour.html
    Download dos Dados: http://www.math.uwaterloo.ca/tsp/world/dj38.tsp

    Qatar
    194 cidades
    The optimal tour has length 9.352
    Foto dos Pontos: http://www.math.uwaterloo.ca/tsp/world/qapoints.html
    Foto da Solução: http://www.math.uwaterloo.ca/tsp/world/qatour.html
    Download dos Dados: http://www.math.uwaterloo.ca/tsp/world/qa194.tsp

    Uruguay
    734 cidades
    The optimal tour has length 79.114
    Foto dos Pontos: http://www.math.uwaterloo.ca/tsp/world/uypoints.html
    Foto da Solução: http://www.math.uwaterloo.ca/tsp/world/uytour.html
    Download dos Dados: http://www.math.uwaterloo.ca/tsp/world/uy734.tsp
"""


def real_problem(algoritmos):
    url_coordenadas_cidade = "https://www.math.uwaterloo.ca/tsp/world/wi29.tsp"

    df_coordenadas = pd.read_table(
        url_coordenadas_cidade,
        skiprows=7,  # ignora as 7 primeiras linhas com informações
        names=["X", "Y"],  # nomes das colunas
        sep=" ",  # separador das colunas
        index_col=0,  # usar col=0 como index (nome das cidades)
        skipfooter=1,  # ignora a última linha (EOF)
        engine="python",  # para o parser usar skipfooter sem warning
    )

    # descomente a linha abaixo para conferir se os dados foram lidos corretamente

    df_coordenadas

    tsp = gera_problema_tsp(df_coordenadas)

    # solucao, custo = random_walk(tsp)
    custo, solucao = hill_climbing(tsp)
    print(f"{custo:7.3f}, {solucao}")
    plota_rotas(df_coordenadas, solucao)

    ###################################
    # PROBLEMA REAL #
    ###################################

    tsp = gera_problema_tsp(df_coordenadas)

    # tsp

    n_vezes = 30

    # Executa N vezes para gerar estatísticas da variável custo
    df_custo = executa_n_vezes(tsp, algoritmos, n_vezes)

    """### Plota a rota da melhor solução obtida"""

    # Hill-climbing

    # Melhor solucao encontrada
    # 28340.563  [1, 2, 6, 5, 4, 3, 7, 9, 8, 13, 14, 16, 24, 27, 25, 20, 26, 28, 29, 23, 22, 21, 17, 18, 19, 15, 12, 10, 11]

    solucao = [
        1,
        2,
        6,
        5,
        4,
        3,
        7,
        9,
        8,
        13,
        14,
        16,
        24,
        27,
        25,
        20,
        26,
        28,
        29,
        23,
        22,
        21,
        17,
        18,
        19,
        15,
        12,
        10,
        11,
    ]

    plota_rotas(df_coordenadas, solucao)

    # Solucao otima
    solucao = [
        1,
        6,
        10,
        11,
        12,
        15,
        19,
        18,
        17,
        21,
        22,
        23,
        29,
        28,
        26,
        20,
        25,
        27,
        24,
        16,
        14,
        13,
        9,
        7,
        3,
        4,
        8,
        5,
        2,
    ]

    plota_rotas(df_coordenadas, solucao)

    """##### Box Plots"""

    boxplot_sorted(df_custo, rot=90, figsize=(12, 6), fontsize=20)

    df_custo.T.describe()


# Stochastic hill-­climbing
def hill_climbing_stochastic(tsp):
    pass
    # ponha seu código aqui


# First­-choice hill-­climbing
def hill_climbing_firstchoice(tsp):
    pass
    # ponha seu código aqui


# Random­-restart hill­-climbing
def hill_climbing_randomrestart(tsp):
    pass
    # ponha seu código aqui


def run_all(df_coordenadas):
    # Para rodar tudo junto:

    algoritmos = {
        #'Random Walk - classic': solucao_aleatoria,
        "Hill-Climbing": hill_climbing,
        "Hill-Climbing - stochastic": hill_climbing_stochastic,
        "Hill-Climbing - first-choice": hill_climbing_firstchoice,
        "Hill-Climbing - random-restart": hill_climbing_randomrestart
        # ...
    }

    tsp = gera_problema_tsp(df_coordenadas)

    # numero de vezes que executará cada algoritmo
    n_vezes = 30

    # Executa N vezes para gerar estatísticas da variável custo
    # DataFrame df_custo serve para armazenar todos os resultados
    df_custo = executa_n_vezes(tsp, algoritmos, n_vezes)

    boxplot_sorted(df_custo, rot=90, figsize=(12, 6), fontsize=20)

    df_custo.T.describe()
