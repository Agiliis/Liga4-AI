"""
ui.py - Peças visuais compartilhadas por todas as telas (menu, créditos, jogo).

Nada aqui conhece regras do jogo; só cores, fontes, botão e o contrato `Screen`.

Como as telas se comunicam
--------------------------
Toda tela herda de `Screen`. O método `handle_event` devolve `None` (continua na
tela) ou o nome de uma ação, e o `App` (em frontend.py) decide para onde ir:

    GO_PVP       começa uma partida Player VS Player
    GO_PVC       começa uma partida Player VS CPU
    GO_CREDITS   abre os créditos
    GO_MENU      volta ao menu
    QUIT         fecha o jogo
"""

from __future__ import annotations

from typing import Optional

import pygame

from src.backend import Piece

# --------------------------------------------------------------------------- #
# Ações de navegação
# --------------------------------------------------------------------------- #
GO_PVP = "pvp"
GO_PVC = "pvc"
GO_CREDITS = "credits"
GO_MENU = "menu"
QUIT = "quit"

# --------------------------------------------------------------------------- #
# Cores
# --------------------------------------------------------------------------- #
BG_TOP = (16, 20, 46)
BG_BOTTOM = (56, 30, 96)
TEXT = (240, 243, 255)
MUTED = (146, 154, 192)
BUTTON = (56, 66, 128)
BUTTON_HOVER = (80, 94, 172)

PIECE_STYLE = {
    Piece.P1: {"base": (233, 66, 84), "dark": (148, 26, 48), "light": (255, 142, 152)},
    Piece.P2: {"base": (255, 204, 54), "dark": (192, 136, 6), "light": (255, 242, 164)},
}


def mix(a, b, t: float):
    """Mistura linear entre duas cores RGB."""
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def pick_font(size: int, bold: bool = True) -> pygame.font.Font:
    return pygame.font.SysFont(
        "segoeui,helveticaneue,helvetica,arial,dejavusans,freesans", size, bold=bold
    )


# --------------------------------------------------------------------------- #
# Desenho reaproveitável
# --------------------------------------------------------------------------- #
def make_gradient(w: int, h: int) -> pygame.Surface:
    """Fundo em degradê vertical."""
    surf = pygame.Surface((w, h))
    for y in range(h):
        pygame.draw.line(surf, mix(BG_TOP, BG_BOTTOM, y / max(1, h - 1)), (0, y), (w, y))
    return surf


def make_disc_sprite(radius: int, style: dict) -> pygame.Surface:
    """Peça com brilho. Renderizada em 4x e reduzida, para bordas suaves."""
    ss = 4
    r = radius * ss
    surf = pygame.Surface((2 * r, 2 * r), pygame.SRCALPHA)
    surf.fill((*style["dark"], 0))
    c = (r, r)
    pygame.draw.circle(surf, style["dark"], c, r)
    pygame.draw.circle(surf, style["base"], c, int(r * 0.92))
    pygame.draw.circle(surf, mix(style["base"], style["dark"], 0.45), c, int(r * 0.66))
    pygame.draw.circle(surf, mix(style["base"], style["light"], 0.22), c, int(r * 0.58))
    shine = pygame.Surface((2 * r, 2 * r), pygame.SRCALPHA)
    pygame.draw.ellipse(
        shine, (255, 255, 255, 75),
        pygame.Rect(int(r * 0.38), int(r * 0.24), int(r * 0.60), int(r * 0.28)),
    )
    surf.blit(shine, (0, 0))
    return pygame.transform.smoothscale(surf, (2 * radius, 2 * radius))


class Button:
    def __init__(self, rect: pygame.Rect, label: str, font: pygame.font.Font, radius: int = 12):
        self.rect = rect
        self.label = label
        self.font = font
        self.radius = radius

    def hit(self, pos) -> bool:
        return self.rect.collidepoint(pos)

    def draw(self, surface: pygame.Surface, mouse_pos, pulse: float = 0.0, selected: bool = False) -> None:
        hovered = self.rect.collidepoint(mouse_pos)
        color = BUTTON_HOVER if (hovered or selected) else BUTTON
        color = mix(color, PIECE_STYLE[Piece.P2]["base"], 0.35 * pulse)
        pygame.draw.rect(surface, color, self.rect, border_radius=self.radius)
        border = mix(color, (255, 255, 255), 0.6 if selected else 0.3)
        pygame.draw.rect(surface, border, self.rect, width=3 if selected else 2, border_radius=self.radius)
        text = self.font.render(self.label, True, TEXT)
        surface.blit(text, text.get_rect(center=self.rect.center))


# --------------------------------------------------------------------------- #
# Contrato das telas
# --------------------------------------------------------------------------- #
class Screen:
    """Base de qualquer tela. O App chama estes três métodos a cada frame."""

    def handle_event(self, event: pygame.event.Event) -> Optional[str]:
        """Processa um evento. Devolve uma ação (GO_*, QUIT) ou None."""
        return None

    def update(self, dt: float) -> None:
        pass

    def draw(self, surface: pygame.Surface) -> None:
        pass
