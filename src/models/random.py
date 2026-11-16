"""Baseline 0: jogador que escolhe uma coluna válida ao acaso.

Obs.: dentro do pacote, `import random` abaixo pega o módulo da biblioteca
padrão (imports absolutos), não este arquivo.
"""

from __future__ import annotations

import random

from src.backend import Game, Player


class RandomPlayer(Player):
    """Escolhe uma coluna válida ao acaso. É o grupo de controle da avaliação."""

    def choose_move(self, game: Game) -> int:
        return random.choice(game.valid_moves())
