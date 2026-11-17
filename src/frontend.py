"""
frontend.py - Tela do jogo (pygame) e o `App`, que navega entre as telas.

Este módulo só desenha e lê entradas. Toda regra do jogo vive no backend:
o frontend chama `game.play(col)`, lê `game.board`, `game.status`, etc.

Telas
-----
    MenuScreen / CreditsScreen   em menu.py
    DevModeScreen                em dev_mode.py (avaliação dos modelos)
    GameScreen                   aqui: o tabuleiro em si

O `App` recebe duas "fábricas" de partida (funções sem argumentos que devolvem
um `Game`), uma para cada modo. Quem decide quais jogadores existem (humanos,
CPU...) é o main.py, não o frontend.

Controles na partida
--------------------
    Mouse          mover = escolher coluna | clique = soltar a peça
    Setas / A, D   mover a seleção
    Espaço/Enter   soltar a peça (também Seta para baixo / S)
    R              novo jogo
    Esc            voltar ao menu

Jogadores que não são humanos (Player.is_human == False) são consultados
automaticamente via `choose_move(game)`: é por aqui que a IA joga.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import pygame

from src.backend import Game, InvalidMoveError, Piece, Status
from src.dev_mode import DevModeScreen
from src.menu import CreditsScreen, MenuScreen
from src.ui import (
    GO_CREDITS, GO_DEV, GO_MENU, GO_PVC, GO_PVP, MUTED, PIECE_STYLE, QUIT, TEXT,
    Button, Screen, make_disc_sprite, make_gradient, mix, pick_font,
)

# --------------------------------------------------------------------------- #
# Aparência
# --------------------------------------------------------------------------- #
FPS = 60
CPU_THINK_DELAY = 0.45  # pausa antes de um jogador não humano jogar (segundos)

BOARD_COLOR = (36, 82, 214)
BOARD_LIGHT = (84, 136, 250)
BOARD_DARK = (18, 44, 140)
BACKPLATE = (10, 14, 34)


# --------------------------------------------------------------------------- #
# Geometria da janela (calculada a partir do tamanho da grade)
# --------------------------------------------------------------------------- #
class Layout:
    PAD = 14          # moldura interna do tabuleiro
    MARGIN_X = 40
    HEADER_H = 170    # título, status e a peça que "paira" sobre o tabuleiro
    FOOTER_H = 120    # placar, botões e dicas
    MIN_W = 560

    def __init__(self, rows: int, cols: int, screen_size=None, max_cell: int = 84):
        screen_w, screen_h = screen_size or (1920, 1080)
        by_height = (screen_h * 0.90 - self.HEADER_H - self.FOOTER_H - 2 * self.PAD) / rows
        by_width = (screen_w * 0.90 - 2 * self.MARGIN_X - 2 * self.PAD) / cols
        self.rows, self.cols = rows, cols
        self.cell = int(max(40, min(max_cell, by_height, by_width)))
        self.radius = int(self.cell * 0.40)

        board_w = cols * self.cell + 2 * self.PAD
        board_h = rows * self.cell + 2 * self.PAD
        self.win_w = max(self.MIN_W, board_w + 2 * self.MARGIN_X)
        self.win_h = self.HEADER_H + board_h + self.FOOTER_H
        self.board_rect = pygame.Rect((self.win_w - board_w) // 2, self.HEADER_H, board_w, board_h)

        self.drop_y = self.board_rect.top - self.radius - 8
        self.score_y = self.board_rect.bottom + 24

        bw, bh, gap = 170, 38, 14
        cy = self.board_rect.bottom + 62
        self.new_rect = pygame.Rect(0, 0, bw, bh)
        self.new_rect.center = (self.win_w // 2 - (bw + gap) // 2, cy)
        self.menu_rect = pygame.Rect(0, 0, bw, bh)
        self.menu_rect.center = (self.win_w // 2 + (bw + gap) // 2, cy)
        self.hint_y = self.win_h - 14

    def cell_center(self, row: int, col: int) -> tuple[float, float]:
        x = self.board_rect.left + self.PAD + col * self.cell + self.cell / 2
        y = self.board_rect.top + self.PAD + row * self.cell + self.cell / 2
        return x, y

    def col_at(self, pos) -> Optional[int]:
        """Coluna sob a posição do mouse (só vale até a borda de baixo do tabuleiro)."""
        x, y = pos
        if y > self.board_rect.bottom:
            return None
        col = int((x - (self.board_rect.left + self.PAD)) // self.cell)
        return col if 0 <= col < self.cols else None


# --------------------------------------------------------------------------- #
# Sprites pré-renderizados (supersampling => bordas suaves, sem custo por frame)
# --------------------------------------------------------------------------- #
def make_ring_sprite(radius: int) -> pygame.Surface:
    ss = 4
    outer = radius + 10
    r = outer * ss
    surf = pygame.Surface((2 * r, 2 * r), pygame.SRCALPHA)
    surf.fill((255, 255, 255, 0))
    pygame.draw.circle(surf, (255, 255, 255, 80), (r, r), (radius + 6) * ss, width=9 * ss)
    pygame.draw.circle(surf, (255, 255, 255, 255), (r, r), (radius + 5) * ss, width=3 * ss)
    return pygame.transform.smoothscale(surf, (2 * outer, 2 * outer))


def make_board_sprite(layout: Layout) -> pygame.Surface:
    """Moldura azul com furos transparentes (as peças aparecem por trás dela)."""
    ss = 3
    w, h = layout.board_rect.size
    surf = pygame.Surface((w * ss, h * ss), pygame.SRCALPHA)
    surf.fill((*BOARD_DARK, 0))
    corner = 28 * ss
    inset = 4 * ss
    inner = pygame.Rect(inset, inset, (w - 8) * ss, (h - 8) * ss)
    pygame.draw.rect(surf, BOARD_DARK, (0, 0, w * ss, h * ss), border_radius=corner)
    pygame.draw.rect(surf, BOARD_COLOR, inner, border_radius=corner - inset)
    pygame.draw.rect(surf, BOARD_LIGHT, inner, width=2 * ss, border_radius=corner - inset)

    ring_outer = mix(BOARD_COLOR, BOARD_DARK, 0.8)
    ring_inner = mix(BOARD_COLOR, BOARD_DARK, 0.35)
    for row in range(layout.rows):
        for col in range(layout.cols):
            cx = int((layout.PAD + col * layout.cell + layout.cell / 2) * ss)
            cy = int((layout.PAD + row * layout.cell + layout.cell / 2) * ss)
            r = layout.radius * ss
            pygame.draw.circle(surf, ring_outer, (cx, cy), r + 6 * ss)
            pygame.draw.circle(surf, ring_inner, (cx, cy), r + 3 * ss)
            pygame.draw.circle(surf, (*ring_inner, 0), (cx, cy), r)  # furo
    return pygame.transform.smoothscale(surf, (w, h))


def make_background(layout: Layout) -> pygame.Surface:
    w, h = layout.win_w, layout.win_h
    surf = make_gradient(w, h)

    # sombra suave sob o tabuleiro (retângulos concêntricos, do maior ao menor)
    shadow = pygame.Surface((w, h), pygame.SRCALPHA)
    steps = 14
    for i in range(steps, 0, -1):
        grow = i * 3
        pygame.draw.rect(
            shadow, (0, 0, 0, 7 + (steps - i) * 3),
            layout.board_rect.inflate(grow, grow).move(0, 14),
            border_radius=28 + grow // 2,
        )
    surf.blit(shadow, (0, 0))

    # fundo escuro atrás dos furos: dá profundidade às casas vazias
    pygame.draw.rect(surf, BACKPLATE, layout.board_rect.inflate(-6, -6), border_radius=26)
    return surf


# --------------------------------------------------------------------------- #
# Componentes
# --------------------------------------------------------------------------- #
@dataclass
class FallingDisc:
    """Peça em queda livre, com quique ao bater no fundo."""

    col: int
    row: int
    piece: Piece
    x: float
    y: float
    target_y: float
    gravity: float
    vy: float = 0.0
    done: bool = False

    def update(self, dt: float) -> None:
        self.vy += self.gravity * dt
        self.y += self.vy * dt
        if self.y >= self.target_y:
            self.y = self.target_y
            if self.vy > 140:
                self.vy = -self.vy * 0.32
            else:
                self.vy = 0.0
                self.done = True


# --------------------------------------------------------------------------- #
# Tela da partida
# --------------------------------------------------------------------------- #
class GameScreen(Screen):
    """O tabuleiro. Os assets são criados uma vez; `start(game)` começa cada partida."""

    def __init__(self, layout: Layout):
        self.layout = layout
        self.screen: Optional[pygame.Surface] = None  # superfície do frame atual (vem do draw)
        self.game: Optional[Game] = None
        self.time = 0.0
        self._build_assets()
        self._clear_state()

    # ---- inicialização ----------------------------------------------------- #
    def _build_assets(self) -> None:
        L = self.layout
        self.bg = make_background(L)
        self.board_sprite = make_board_sprite(L)
        pieces = (Piece.P1, Piece.P2)
        self.discs = {p: make_disc_sprite(L.radius, PIECE_STYLE[p]) for p in pieces}
        self.previews = {p: s.copy() for p, s in self.discs.items()}
        self.previews_dim = {p: s.copy() for p, s in self.discs.items()}
        for p in pieces:
            self.previews[p].set_alpha(175)
            self.previews_dim[p].set_alpha(55)
        self.icons = {p: make_disc_sprite(11, PIECE_STYLE[p]) for p in pieces}
        self.ring = make_ring_sprite(L.radius)

        self.column_glow = pygame.Surface((L.cell, L.rows * L.cell), pygame.SRCALPHA)
        pygame.draw.rect(self.column_glow, (255, 255, 255, 40), self.column_glow.get_rect(), border_radius=14)
        self.overlay = pygame.Surface((L.win_w, L.win_h), pygame.SRCALPHA)

        self.f_title = pick_font(44)
        self.f_big = pick_font(34)
        self.f_status = pick_font(26)
        self.f_small = pick_font(19)
        self.f_hint = pick_font(15, bold=False)
        f_button = pick_font(18)
        self.new_button = Button(L.new_rect, "Novo jogo  (R)", f_button)
        self.menu_button = Button(L.menu_rect, "Menu  (Esc)", f_button)

    def _clear_state(self) -> None:
        self.hover_col: Optional[int] = None
        self.anim: Optional[FallingDisc] = None
        self.cpu_timer = 0.0
        self.scores = {Piece.P1: 0, Piece.P2: 0, "draws": 0}
        self.result_recorded = False

    # ---- ações ------------------------------------------------------------- #
    def start(self, game: Game) -> None:
        """Começa uma partida nova (zera o placar)."""
        board = game.board
        if (board.rows, board.cols) != (self.layout.rows, self.layout.cols):
            raise ValueError(
                "Todos os modos precisam usar a mesma grade "
                f"({self.layout.rows}x{self.layout.cols}); recebi {board.rows}x{board.cols}."
            )
        self.game = game
        self._clear_state()
        self.hover_col = self.layout.col_at(pygame.mouse.get_pos())

    def new_game(self) -> None:
        """Reinicia mantendo o placar. Se a partida anterior terminou, quem começa se alterna."""
        first = self.game.first_player
        if self.game.is_over:
            first = first.opponent
        self.game.reset(first)
        self.anim = None
        self.cpu_timer = 0.0
        self.result_recorded = False

    def _can_human_play(self) -> bool:
        g = self.game
        return self.anim is None and not g.is_over and g.current_player.is_human

    def _play(self, col: int) -> None:
        """Executa a jogada no backend e inicia a animação (levanta InvalidMoveError)."""
        piece = self.game.turn
        row = self.game.play(col)
        L = self.layout
        x, target_y = L.cell_center(row, col)
        self.anim = FallingDisc(
            col=col, row=row, piece=piece, x=x, y=L.drop_y,
            target_y=target_y, gravity=L.cell * 30.0,
        )

    def _human_drop(self, col: Optional[int]) -> None:
        if col is None or not self._can_human_play():
            return
        try:
            self._play(col)
        except InvalidMoveError:
            pass  # coluna cheia: ignora o clique

    def _move_hover(self, delta: int) -> None:
        cols = self.game.board.cols
        if self.hover_col is None:
            self.hover_col = cols // 2
        else:
            self.hover_col = max(0, min(cols - 1, self.hover_col + delta))

    # ---- eventos ----------------------------------------------------------- #
    def handle_event(self, event: pygame.event.Event) -> Optional[str]:
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                return GO_MENU
            if event.key == pygame.K_r:
                self.new_game()
            elif event.key in (pygame.K_LEFT, pygame.K_a):
                self._move_hover(-1)
            elif event.key in (pygame.K_RIGHT, pygame.K_d):
                self._move_hover(+1)
            elif event.key in (pygame.K_SPACE, pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_DOWN, pygame.K_s):
                self._human_drop(self.hover_col)

        elif event.type == pygame.MOUSEMOTION:
            self.hover_col = self.layout.col_at(event.pos)

        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.new_button.hit(event.pos):
                self.new_game()
            elif self.menu_button.hit(event.pos):
                return GO_MENU
            else:
                col = self.layout.col_at(event.pos)
                self.hover_col = col if col is not None else self.hover_col
                self._human_drop(col)
        return None

    # ---- atualização ------------------------------------------------------- #
    def update(self, dt: float) -> None:
        self.time += dt

        if self.anim is not None:
            self.anim.update(dt)
            if self.anim.done:
                self.anim = None
            return

        g = self.game
        if g.is_over:
            if not self.result_recorded:
                self.result_recorded = True
                if g.status is Status.WON:
                    self.scores[g.winner] += 1
                else:
                    self.scores["draws"] += 1
            return

        player = g.current_player
        if not player.is_human:
            self.cpu_timer += dt
            if self.cpu_timer >= CPU_THINK_DELAY:
                self.cpu_timer = 0.0
                self._play(player.choose_move(g))

    # ---- desenho ----------------------------------------------------------- #
    def draw(self, surface: pygame.Surface) -> None:
        self.screen = surface
        L = self.layout
        self.screen.blit(self.bg, (0, 0))
        self._draw_title()
        self._draw_status()
        self._draw_column_glow()
        self._draw_discs()
        self.screen.blit(self.board_sprite, L.board_rect)
        self._draw_hover_preview()
        self._draw_win_effect()
        self._draw_footer()

    def _draw_title(self) -> None:
        L = self.layout
        yellow = PIECE_STYLE[Piece.P2]["base"]
        a = self.f_title.render("CONECTA", True, TEXT)
        b = self.f_title.render("4", True, yellow)
        gap = 12
        total = a.get_width() + gap + b.get_width()
        left = (L.win_w - total) // 2
        y = 36
        self.screen.blit(a, a.get_rect(midleft=(left, y)))
        self.screen.blit(b, b.get_rect(midleft=(left + a.get_width() + gap, y)))
        for piece, x in ((Piece.P1, left - 30), (Piece.P2, left + total + 30)):
            icon = self.icons[piece]
            self.screen.blit(icon, icon.get_rect(center=(x, y + 2)))

    def _draw_status(self) -> None:
        g, L = self.game, self.layout
        y = 82
        busy = self.anim is not None

        if g.status is Status.WON and not busy:
            text = self.f_big.render(f"{g.players[g.winner].name} venceu!", True, PIECE_STYLE[g.winner]["base"])
            text = pygame.transform.rotozoom(text, 0, 1 + 0.05 * math.sin(self.time * 4))
            self.screen.blit(text, text.get_rect(center=(L.win_w // 2, y)))
            return
        if g.status is Status.DRAW and not busy:
            text = self.f_big.render("Empate!", True, TEXT)
            text = pygame.transform.rotozoom(text, 0, 1 + 0.05 * math.sin(self.time * 4))
            self.screen.blit(text, text.get_rect(center=(L.win_w // 2, y)))
            return

        piece = self.anim.piece if busy else g.turn
        prefix = self.f_status.render("Vez de ", True, TEXT)
        name = self.f_status.render(g.players[piece].name, True, PIECE_STYLE[piece]["base"])
        icon = self.icons[piece]
        total = icon.get_width() + 10 + prefix.get_width() + name.get_width()
        x = (L.win_w - total) // 2
        self.screen.blit(icon, icon.get_rect(midleft=(x, y)))
        x += icon.get_width() + 10
        self.screen.blit(prefix, prefix.get_rect(midleft=(x, y)))
        self.screen.blit(name, name.get_rect(midleft=(x + prefix.get_width(), y)))

    def _draw_column_glow(self) -> None:
        col = self.hover_col
        if col is None or not self._can_human_play() or not self.game.board.is_valid_move(col):
            return
        L = self.layout
        x = L.board_rect.left + L.PAD + col * L.cell
        self.screen.blit(self.column_glow, (x, L.board_rect.top + L.PAD))

    def _draw_discs(self) -> None:
        L, g = self.layout, self.game
        skip = (self.anim.row, self.anim.col) if self.anim else None
        for row in range(g.board.rows):
            for col in range(g.board.cols):
                piece = g.board.get(row, col)
                if piece == Piece.EMPTY or (row, col) == skip:
                    continue
                sprite = self.discs[piece]
                self.screen.blit(sprite, sprite.get_rect(center=L.cell_center(row, col)))
        if self.anim:
            sprite = self.discs[self.anim.piece]
            self.screen.blit(sprite, sprite.get_rect(center=(self.anim.x, self.anim.y)))

    def _draw_hover_preview(self) -> None:
        col = self.hover_col
        if col is None or not self._can_human_play():
            return
        x = self.layout.cell_center(0, col)[0]
        y = self.layout.drop_y + 3 * math.sin(self.time * 4)
        sprites = self.previews if self.game.board.is_valid_move(col) else self.previews_dim
        sprite = sprites[self.game.turn]
        self.screen.blit(sprite, sprite.get_rect(center=(x, y)))

    def _draw_win_effect(self) -> None:
        g, L = self.game, self.layout
        if g.status is not Status.WON or self.anim is not None:
            return
        pulse = 0.5 + 0.5 * math.sin(self.time * 5)
        cells = g.winning_line
        p1 = L.cell_center(*cells[0])
        p2 = L.cell_center(*cells[-1])
        width = max(6, L.radius // 3)
        color = (255, 255, 255, int(55 + 55 * pulse))

        self.overlay.fill((0, 0, 0, 0))
        pygame.draw.line(self.overlay, color, p1, p2, width)
        for p in (p1, p2):
            pygame.draw.circle(self.overlay, color, (int(p[0]), int(p[1])), width // 2)
        self.screen.blit(self.overlay, (0, 0))

        self.ring.set_alpha(int(150 + 105 * pulse))
        for cell in cells:
            self.screen.blit(self.ring, self.ring.get_rect(center=L.cell_center(*cell)))

    def _draw_footer(self) -> None:
        g, L = self.game, self.layout
        cx = L.win_w / 2
        spacing = L.win_w * 0.31
        items = (
            (cx - spacing, Piece.P1, f"{g.players[Piece.P1].name}  {self.scores[Piece.P1]}"),
            (cx, None, f"Empates  {self.scores['draws']}"),
            (cx + spacing, Piece.P2, f"{g.players[Piece.P2].name}  {self.scores[Piece.P2]}"),
        )
        for x, piece, label in items:
            text = self.f_small.render(label, True, TEXT if piece else MUTED)
            icon = self.icons[piece] if piece else None
            icon_w = icon.get_width() + 8 if icon else 0
            left = x - (icon_w + text.get_width()) / 2
            if icon:
                self.screen.blit(icon, icon.get_rect(midleft=(left, L.score_y)))
            self.screen.blit(text, text.get_rect(midleft=(left + icon_w, L.score_y)))

        mouse = pygame.mouse.get_pos()
        pulse = 0.5 + 0.5 * math.sin(self.time * 4) if g.is_over and self.anim is None else 0.0
        self.new_button.draw(self.screen, mouse, pulse)
        self.menu_button.draw(self.screen, mouse)

        hint = self.f_hint.render(
            "Clique ou use as setas + espaço   |   R: novo jogo   |   Esc: menu", True, MUTED
        )
        self.screen.blit(hint, hint.get_rect(center=(L.win_w // 2, L.hint_y)))


# --------------------------------------------------------------------------- #
# Aplicação
# --------------------------------------------------------------------------- #
GameFactory = Callable[[], Game]


class App:
    """Abre a janela e navega entre menu, créditos e partida.

    new_pvp / new_pvc: funções sem argumentos que devolvem um `Game` novo para
    cada modo (quem monta os jogadores é o main.py). Os dois modos precisam usar
    a mesma grade, porque a janela é dimensionada uma única vez.
    """

    def __init__(self, new_pvp: GameFactory, new_pvc: GameFactory):
        pygame.init()
        pygame.display.set_caption("Conecta 4")

        self.new_pvp = new_pvp
        self.new_pvc = new_pvc

        info = pygame.display.Info()
        screen_size = (info.current_w, info.current_h) if info.current_w > 0 and info.current_h > 0 else None
        sample = new_pvp().board
        self.layout = Layout(sample.rows, sample.cols, screen_size)
        self.screen = self._open_window()
        self.clock = pygame.time.Clock()

        size = (self.layout.win_w, self.layout.win_h)
        background = make_gradient(*size)
        self.menu = MenuScreen(size, background)
        self.credits = CreditsScreen(size, background)
        self.dev_mode = DevModeScreen(size, background)
        self.game_screen = GameScreen(self.layout)
        self.current: Screen = self.menu

    def _open_window(self) -> pygame.Surface:
        size = (self.layout.win_w, self.layout.win_h)
        try:
            # SCALED: a janela pode ser redimensionada mantendo a proporção
            return pygame.display.set_mode(size, pygame.SCALED | pygame.RESIZABLE)
        except pygame.error:
            return pygame.display.set_mode(size)

    def _navigate(self, action: str) -> bool:
        """Troca de tela. Devolve False quando é para fechar o jogo."""
        if action == QUIT:
            return False
        if action == GO_MENU:
            self.current = self.menu
        elif action == GO_CREDITS:
            self.current = self.credits
        elif action == GO_DEV:
            self.current = self.dev_mode
        elif action == GO_PVP:
            self.game_screen.start(self.new_pvp())
            self.current = self.game_screen
        elif action == GO_PVC:
            self.game_screen.start(self.new_pvc())
            self.current = self.game_screen
        return True

    def run(self) -> None:
        running = True
        while running:
            dt = min(self.clock.tick(FPS) / 1000.0, 0.05)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
                action = self.current.handle_event(event)
                if action and not self._navigate(action):
                    running = False
                    break
            self.current.update(dt)
            self.current.draw(self.screen)
            pygame.display.flip()
        pygame.quit()
