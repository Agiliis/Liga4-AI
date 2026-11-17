# Liga4-AI
Jogo de Liga 4 contra a máquina. Projeto da disciplina de Inteligência Artificial, pela POLI UPE

## Databases
1. UCI: https://archive.ics.uci.edu/dataset/26/connect+4

This database contains all legal 8-ply positions in the game of connect-4 in which neither player has won yet, and in which the next move is not forced.</br>
x is the first player; o the second.</br>
The outcome class is the game theoretical value for the first player.

**Instâncias: 67557**

2. Kaggle: https://www.kaggle.com/datasets/tbrewer/connect-4

This data set is the current result of training a Neural Network to play Connect-4.</br>
Each row represents the end results of a Connect-4 game.</br>
Columns 1-42 are the positions on the grid from left to right, top to bottom. Each element in these columns represent to player's piece : 1, and -1, 0 marks an empty cell.</br>
Column 43 marks the winner of the game : -1, 1, and 0 for tie games.</br>

**Instâncias: 376641**

## Algoritmos:

1. **1-NN**

Para cada jogada possível, mede a menor distância entre o quadro gerado pela jogada e algum quadro em que a máquina vença. Então escolhe a jogada com a menor das distâncias.

**Database**: [./db/df_jogador_2.csv](./db/df_jogador_2.csv)</br>
**1**: 1º jogador</br>
**-1**: 2º jogador (máquina)</br>
**0**: vazio</br>
**Forma (shape): (169025, 42)**</br>
*Armazena todos os jogos onde o 2º jogador ganha. Processado do DB do Kaggle*