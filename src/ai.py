"""
ai.py - Molde da CPU do modo "Player VS CPU".

É aqui que a sua IA vai morar. O contrato é um só: receber a partida (`game`)
e devolver o número da coluna escolhida (0 .. cols-1).

STATUS ATUAL: PLACEHOLDER. Joga em uma coluna aleatória só para o modo
Player VS CPU funcionar de ponta a ponta enquanto a IA de verdade não existe.
Apague o `random.choice` quando for implementar a sua lógica.

O que a IA pode usar (tudo vem do backend):

    game.turn                     peça de quem está jogando agora (Piece.P1 ou P2)
    game.turn.opponent            peça do adversário
    game.valid_moves()            colunas que ainda aceitam peça
    game.snapshot()               estado completo em tipos simples (listas/ints)
    game.board.copy()             cópia do tabuleiro para simular sem afetar a partida
    board.drop(col, peca)         solta uma peça (devolve a linha onde caiu)
    board.undo(col)               desfaz a última peça da coluna
    board.winning_line(row, col)  sequência vencedora passando por essa casa, ou None

O frontend chama `choose_move` na vez da CPU, depois de uma pequena pausa
(CPU_THINK_DELAY em frontend.py). Se a sua IA demorar vários segundos, ela
precisa rodar numa thread/processo, senão a janela congela enquanto ela pensa.
"""

from __future__ import annotations

import random
import numpy as np
import pandas as pd

from src.backend import Game, Player, Piece
from scipy.spatial import distance


class CPUPlayer(Player):
    """Jogador controlado pelo computador (molde)."""

    # is_human já é False na classe Player: o frontend sabe que deve pedir a jogada.

    def __init__(self, name: str = "CPU"):
        super().__init__(name)

        df = pd.read_csv("db/df_jogador_2.csv", index_col=0)

        self.data = df.to_numpy(dtype=np.float32)

        self._sq_norms = (self.data ** 2).sum(axis=1) 


    def KNN_1(self, quadro_atual, dist_fun="euclidean"):
        q = np.asarray(quadro_atual, dtype=np.float32)

        match dist_fun:
            case "cityblock":
                return float(np.abs(self.data - q).sum(axis=1).min())
            case _:
                d2 = self._sq_norms - 2.0 * (self.data @ q) + float(q @ q)

                return float(np.sqrt(max(float(d2.min()), 0.0)))


    def choose_move(self, game: Game) -> int:
        """Devolve a coluna em que a CPU quer jogar."""

        centro = (game.board.cols - 1) / 2
        melhor_chave, melhor_coluna = None, None

        for coluna in game.valid_moves():
            quadro_simulado = game.board.copy()
            quadro_simulado.drop(coluna, Piece.P2)

            dist = self.KNN_1(quadro_atual=quadro_simulado.to_flat())

            chave = (round(dist, 4), abs(coluna - centro), random.random())
            if melhor_chave is None or chave < melhor_chave:
                melhor_chave, melhor_coluna = chave, coluna
        
        return melhor_coluna
