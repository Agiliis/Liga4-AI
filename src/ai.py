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

from src.backend import Game, Player


class CPUPlayer(Player):
    """Jogador controlado pelo computador (molde)."""

    # is_human já é False na classe Player: o frontend sabe que deve pedir a jogada.

    def __init__(self, name: str = "CPU"):
        super().__init__(name)
        # Espaço para configurações da sua IA (dificuldade, profundidade de busca...):
        # self.depth = 4

    def choose_move(self, game: Game) -> int:
        """Devolve a coluna em que a CPU quer jogar."""
        # TODO: trocar pelo algoritmo da IA. Exemplo de ideia, passo a passo:
        #   1. se eu consigo vencer jogando numa coluna, jogo nela;
        #   2. se o adversário venceria na próxima, bloqueio essa coluna;
        #   3. senão, avalio as jogadas (minimax, MCTS, heurística...) e escolho a melhor.
        return random.choice(game.valid_moves())
