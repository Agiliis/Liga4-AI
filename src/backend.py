"""
backend.py - Regras e estado do Conecta-4 (sem nenhuma dependência de pygame).

Tudo o que a interface, uma IA ou até um servidor de rede precisam está aqui:

    Board    grade do jogo: soltar/desfazer peças, achar sequências vencedoras
    Game     partida: de quem é a vez, status, vencedor, histórico, snapshot
    Player   interface que qualquer jogador implementa (humano, CPU, IA, rede...)

Como plugar uma IA depois
-------------------------
Herde de `Player`, implemente `choose_move(game)` devolvendo o índice da coluna
(0 .. cols-1) e passe o jogador ao criar a partida:

    game = Game(players={Piece.P1: HumanPlayer("Você"), Piece.P2: MinhaIA("CPU")})

O frontend já chama `choose_move` sempre que for a vez de um jogador que não é
humano. A IA pode usar `game.snapshot()` (dados simples, fáceis de serializar)
ou `game.board.copy()` + `board.drop()` / `board.undo()` para simular jogadas
sem mexer na partida real.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Optional

# --------------------------------------------------------------------------- #
# Configuração padrão da grade (troque aqui, o frontend se adapta sozinho)
# --------------------------------------------------------------------------- #
ROWS = 7      # linhas
COLS = 6      # colunas
CONNECT = 4   # peças em sequência necessárias para vencer

Cell = tuple[int, int]  # (linha, coluna); a linha 0 é o topo do tabuleiro


class Piece(IntEnum):
    """Conteúdo de uma casa. Os valores inteiros facilitam usar em IA/numpy."""

    EMPTY = 0
    P1 = 1
    P2 = 2

    @property
    def opponent(self) -> "Piece":
        if self is Piece.EMPTY:
            raise ValueError("Casa vazia não tem oponente.")
        return Piece.P2 if self is Piece.P1 else Piece.P1


class Status(Enum):
    PLAYING = "playing"
    WON = "won"
    DRAW = "draw"


class InvalidMoveError(ValueError):
    """Jogada impossível: coluna cheia, fora da grade ou partida encerrada."""


@dataclass(frozen=True)
class Move:
    piece: Piece
    row: int
    col: int


# --------------------------------------------------------------------------- #
# Tabuleiro
# --------------------------------------------------------------------------- #
class Board:
    """Grade do jogo. Só conhece peças e geometria, não sabe de turnos."""

    DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))  # -, |, \, /

    def __init__(self, rows: int = ROWS, cols: int = COLS, connect: int = CONNECT):
        if connect > max(rows, cols):
            raise ValueError("`connect` maior que a grade: ninguém conseguiria vencer.")
        self.rows = rows
        self.cols = cols
        self.connect = connect
        self._grid: list[list[Piece]] = [[Piece.EMPTY] * cols for _ in range(rows)]
        self._heights: list[int] = [0] * cols  # quantas peças há em cada coluna
        self.move_count = 0

    # ---- consulta ---------------------------------------------------------- #
    def get(self, row: int, col: int) -> Piece:
        return self._grid[row][col]

    def column_height(self, col: int) -> int:
        return self._heights[col]

    def is_valid_move(self, col: int) -> bool:
        return 0 <= col < self.cols and self._heights[col] < self.rows

    def valid_moves(self) -> list[int]:
        return [c for c in range(self.cols) if self._heights[c] < self.rows]

    def landing_row(self, col: int) -> int:
        """Linha onde uma peça solta nesta coluna vai parar."""
        if not self.is_valid_move(col):
            raise InvalidMoveError(f"Coluna {col} indisponível.")
        return self.rows - 1 - self._heights[col]

    def is_full(self) -> bool:
        return self.move_count == self.rows * self.cols

    # ---- modificação ------------------------------------------------------- #
    def drop(self, col: int, piece: Piece) -> int:
        """Solta `piece` na coluna `col` e devolve a linha onde ela caiu."""
        row = self.landing_row(col)
        self._grid[row][col] = Piece(piece)
        self._heights[col] += 1
        self.move_count += 1
        return row

    def undo(self, col: int) -> None:
        """Remove a peça do topo da coluna (útil para buscas de IA)."""
        if not (0 <= col < self.cols) or self._heights[col] == 0:
            raise InvalidMoveError(f"Coluna {col} não tem peças para remover.")
        row = self.rows - self._heights[col]
        self._grid[row][col] = Piece.EMPTY
        self._heights[col] -= 1
        self.move_count -= 1

    # ---- vitória ----------------------------------------------------------- #
    def winning_line(self, row: int, col: int) -> Optional[list[Cell]]:
        """Sequência vencedora que passa pela casa (row, col), ou None.

        Devolve a sequência inteira (pode ter mais de `connect` casas).
        """
        piece = self._grid[row][col]
        if piece == Piece.EMPTY:
            return None
        for dr, dc in self.DIRECTIONS:
            forward = self._run(row, col, dr, dc, piece)
            backward = self._run(row, col, -dr, -dc, piece)
            line = backward[::-1] + [(row, col)] + forward
            if len(line) >= self.connect:
                return line
        return None

    def _run(self, row: int, col: int, dr: int, dc: int, piece: Piece) -> list[Cell]:
        cells: list[Cell] = []
        r, c = row + dr, col + dc
        while 0 <= r < self.rows and 0 <= c < self.cols and self._grid[r][c] == piece:
            cells.append((r, c))
            r += dr
            c += dc
        return cells

    # ---- utilidades -------------------------------------------------------- #
    def copy(self) -> "Board":
        clone = Board(self.rows, self.cols, self.connect)
        clone._grid = [row[:] for row in self._grid]
        clone._heights = self._heights[:]
        clone.move_count = self.move_count
        return clone

    def to_list(self) -> list[list[int]]:
        return [[int(p) for p in row] for row in self._grid]

    def __str__(self) -> str:
        symbols = {Piece.EMPTY: ".", Piece.P1: "X", Piece.P2: "O"}
        return "\n".join(" ".join(symbols[p] for p in row) for row in self._grid)


# --------------------------------------------------------------------------- #
# Jogadores
# --------------------------------------------------------------------------- #
class Player(ABC):
    """Contrato de qualquer jogador. É aqui que a IA vai se conectar."""

    is_human = False

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def choose_move(self, game: "Game") -> int:
        """Devolve a coluna escolhida (0 .. cols-1) para o estado atual de `game`."""


class HumanPlayer(Player):
    """Jogador humano: a jogada vem do mouse/teclado, não deste método."""

    is_human = True

    def choose_move(self, game: "Game") -> int:
        raise NotImplementedError("Jogadas humanas chegam pela interface.")


# --------------------------------------------------------------------------- #
# Partida
# --------------------------------------------------------------------------- #
class Game:
    """Uma partida: tabuleiro + jogadores + turnos + resultado."""

    def __init__(
        self,
        players: Optional[dict[Piece, Player]] = None,
        rows: int = ROWS,
        cols: int = COLS,
        connect: int = CONNECT,
        first_player: Piece = Piece.P1,
    ):
        self.players: dict[Piece, Player] = players or {
            Piece.P1: HumanPlayer("Vermelho"),
            Piece.P2: HumanPlayer("Amarelo"),
        }
        if set(self.players) != {Piece.P1, Piece.P2}:
            raise ValueError("`players` precisa ter exatamente Piece.P1 e Piece.P2.")
        self._dims = (rows, cols, connect)
        self.first_player = first_player
        self.reset()

    # ---- ciclo de vida ----------------------------------------------------- #
    def reset(self, first_player: Optional[Piece] = None) -> None:
        if first_player is not None:
            self.first_player = first_player
        self.board = Board(*self._dims)
        self.turn: Piece = self.first_player
        self.status: Status = Status.PLAYING
        self.winner: Optional[Piece] = None
        self.winning_line: list[Cell] = []
        self.history: list[Move] = []

    # ---- consulta ---------------------------------------------------------- #
    @property
    def is_over(self) -> bool:
        return self.status is not Status.PLAYING

    @property
    def current_player(self) -> Player:
        return self.players[self.turn]

    @property
    def last_move(self) -> Optional[Move]:
        return self.history[-1] if self.history else None

    def valid_moves(self) -> list[int]:
        return [] if self.is_over else self.board.valid_moves()

    # ---- jogadas ----------------------------------------------------------- #
    def play(self, col: int) -> int:
        """Joga na coluna `col` para quem está na vez. Devolve a linha da peça.

        Levanta InvalidMoveError se a partida acabou ou a coluna não aceita peças.
        """
        if self.is_over:
            raise InvalidMoveError("A partida já terminou.")
        piece = self.turn
        row = self.board.drop(col, piece)
        self.history.append(Move(piece, row, col))

        line = self.board.winning_line(row, col)
        if line:
            self.status = Status.WON
            self.winner = piece
            self.winning_line = line
        elif self.board.is_full():
            self.status = Status.DRAW
        else:
            self.turn = piece.opponent
        return row

    def undo(self) -> Optional[Move]:
        """Desfaz a última jogada (devolve-a) ou None se não houver."""
        if not self.history:
            return None
        move = self.history.pop()
        self.board.undo(move.col)
        self.turn = move.piece
        self.status = Status.PLAYING
        self.winner = None
        self.winning_line = []
        return move

    # ---- exportação (IA, rede, logs) --------------------------------------- #
    def snapshot(self) -> dict:
        """Estado completo em tipos simples (listas, ints, strings)."""
        return {
            "rows": self.board.rows,
            "cols": self.board.cols,
            "connect": self.board.connect,
            "board": self.board.to_list(),  # 0 vazio, 1 = P1, 2 = P2; linha 0 = topo
            "turn": int(self.turn),
            "status": self.status.value,
            "winner": int(self.winner) if self.winner else None,
            "valid_moves": self.valid_moves(),
            "history": [(int(m.piece), m.row, m.col) for m in self.history],
        }
