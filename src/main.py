"""Ponto de entrada do Conecta-4.  Execute:  python main.py"""

from src.backend import Game
from src.frontend import App


def main() -> None:
    # Dois humanos no mesmo computador (jogadores padrão: Vermelho e Amarelo).
    game = Game()

    # Quando a IA existir, é só trocar o segundo jogador, por exemplo:
    #
    #   from backend import HumanPlayer, Piece
    #   game = Game(players={
    #       Piece.P1: HumanPlayer("Você"),
    #       Piece.P2: MinhaIA("CPU"),      # classe que herda de backend.Player
    #   })

    App(game).run()


if __name__ == "__main__":
    main()
