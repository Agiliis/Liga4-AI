"""Ponto de entrada do Conecta-4.  Execute:  python main.py"""

from src.ai import CPUPlayer
from src.models import TreeKNNPlayer
from src.backend import Game, HumanPlayer, Piece
from src.frontend import App


def new_pvp() -> Game:
    """Player VS Player: dois humanos no mesmo computador."""
    return Game(players={
        Piece.P1: HumanPlayer("Vermelho"),
        Piece.P2: HumanPlayer("Amarelo"),
    })


def new_pvc() -> Game:
    """Player VS CPU: um humano e a IA (molde em ai.py)."""
    return Game(players={
        Piece.P1: HumanPlayer("Você"),
        Piece.P2: TreeKNNPlayer("Tree+KNN"),
    })


def main() -> None:
    App(new_pvp, new_pvc).run()


if __name__ == "__main__":
    main()
