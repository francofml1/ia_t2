# Inteligência Artificial - Trabalho 2: 

Respositório desenvolvido para execução do Trabalho 2 da disciplina de Inteligência Artificial.

Métodos implementados:
- [HC-C] - Hill-Blimbing (clássico)
- [HC-R] - Hill-Blimbing with Restart
- [SA] - Simulated Annealing
- [GA] - Genetic Algorithm
---
- Na descrição dos problemas são deginidas a função objetivo e as funções para geração de vizinhos;
- Também são definidosos parâmetros que devem ser usados pelos métodos;
- Executar cada algoritmo no mínimo 10 vezes (talvez 30)
- [Implemenrtação do HC-C para o TSP](https://colab.research.google.com/drive/1Gnhhp0GX8140lAYv-uCg2qnR6E92gutX?usp=sharing)
- [Implementação do GA para o problema das 8 rainhas](https://colab.research.google.com/drive/1N6lVlrffNY3Gy_yDupReT50MRZfVHGqt?usp=sharing)
- [Implementação do SA para o TSP](https://colab.research.google.com/drive/1MS0-4QF76dfQrulwdOqIkJPNi8U5SHdG?usp=sharing)
- [notebook_IA_cap4_SimulatedAnnealing](https://colab.research.google.com/drive/173NM-mgBzf8ptPzXUFnuD8JjwHNKBPCF?usp=sharing)

---
# Referências:
[1] [Use a função numpy.random.randn da biblioteca numpy para amostrar o valor aleatório](https://numpy.org/doc/stable/reference/random/generated/numpy.random.randn.html?highlight=randn#numpy.random.randn)

[2] [Use a função numpy.random.uniform da biblioteca numpy para amostrar o valor aleatório](https://numpy.org/doc/stable/reference/random/generated/numpy.random.uniform.html?highlight=uniform#numpy.random.uniform)

[3] [Biblioteca argparse para criação de interfaces de linha de comando](https://docs.python.org/3/library/argparse.html)

[4] [Use a função seaborn.lineplot (ou equivalente) para produzir o gráfico das curvas médias da função objetivo com os intervalos de variação](https://seaborn.pydata.org/examples/errorband_lineplots.html)

[5] [Vídeo - Genetic Algorithms 18/30: Order One Crossover](https://www.youtube.com/watch?v=HATPHZ6P7c4)

[6] [Seção Order crossover (OX)](http://creationwiki.org/pt/Algoritmo_gen%C3%A9tico)

---
# Get start:

```bash

# Cria venv
python -m venv "venv"
.\venv\Scripts\activate
pip install -r requirements.txt
```

## Execução da questão 1:
```bash
.\venv\Scripts\activate
python ./questao_1.py
```
## Execução da questão 2:
```bash
.\venv\Scripts\activate
python ./questao_2.py
```

## Salvar dependências:
```bash
pip freeze > requirements.txt
```
