"""Baseline 1: vence se puder, bloqueia se precisar, senão joga ao acaso."""

from __future__ import annotations

import random

from src.backend import Game, Player


class HeuristicPlayer(Player):
    """Vence se puder, bloqueia se precisar, senão joga ao acaso.

    Serve para saber se o modelo aprendeu mais que essas duas regras óbvias.
    """

    def choose_move(self, game: Game) -> int:
        me, opp = game.turn, game.turn.opponent
        moves = game.valid_moves()
        for piece in (me, opp):  # 1º: vencer; 2º: bloquear
            for col in moves:
                b = game.board.copy()
                row = b.drop(col, piece)
                if b.winning_line(row, col):
                    return col
        return random.choice(moves)
