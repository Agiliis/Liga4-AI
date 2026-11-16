"""Registro único dos modelos (jogadores não humanos) do projeto.

Cada valor de MODELOS é uma *fábrica*: chamá-la cria o Player. Assim nada
pesado é carregado na importação (o KNN1 lê um CSV grande no construtor).

    from src.models import MODELOS, criar_modelo
    jogador = criar_modelo("KNN1")
"""

from __future__ import annotations

from typing import Callable

from src.backend import Player
from src.models.heuristica import HeuristicPlayer
from src.models.random import RandomPlayer


def _criar_knn1() -> Player:
    from src.ai import CPUPlayer  # import preguiçoso: pandas/scipy + leitura do CSV

    return CPUPlayer("KNN1")


MODELOS: dict[str, Callable[[], Player]] = {
    "Aleatório": lambda: RandomPlayer("Aleatório"),
    "Heurística": lambda: HeuristicPlayer("Heurística"),
    "KNN1": _criar_knn1,
}

# TODO: o KNN1 (src/ai.py) usa Piece.P2 fixo em choose_move, então só sabe
# jogar como 2º jogador. Quando ele passar a usar game.turn, remova-o daqui.
SO_JOGA_COMO_P2: frozenset[str] = frozenset({"KNN1"})


def criar_modelo(nome: str) -> Player:
    """Cria uma nova instância do modelo `nome` (KeyError se não existir)."""
    try:
        fabrica = MODELOS[nome]
    except KeyError:
        raise KeyError(f"Modelo desconhecido: {nome!r}. Opções: {list(MODELOS)}") from None
    return fabrica()


__all__ = ["MODELOS", "SO_JOGA_COMO_P2", "criar_modelo", "RandomPlayer", "HeuristicPlayer"]
