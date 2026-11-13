"""
menu.py - Telas de menu e créditos.

    MenuScreen      "Player VS Player", "Player VS CPU" e "Créditos"
    CreditsScreen   os 4 desenvolvedores, com foto e nome

Fotos dos desenvolvedores
-------------------------
Por enquanto cada um aparece com um avatar genérico. Para usar fotos de verdade,
coloque os arquivos em  assets/devs/  (ao lado deste arquivo) com os nomes
listados em DEVELOPERS (dev1.png ... dev4.png). Se o arquivo existir, ele é usado
(recortado em círculo); se não, o avatar genérico continua valendo.
Os nomes também são editados em DEVELOPERS.
"""

from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass
from typing import Optional

import pygame

from src.backend import Piece
from src.ui import (
    GO_CREDITS, GO_MENU, GO_PVC, GO_PVP, MUTED, PIECE_STYLE, QUIT, TEXT,
    Button, Screen, make_disc_sprite, mix, pick_font,
)

ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "devs")

# (nome, arquivo da foto dentro de assets/devs/)  -> nomes inventados por enquanto
DEVELOPERS = (
    ("Lucas Almeida", "dev1.png"),
    ("Marina Costa", "dev2.png"),
    ("Rafael Lima", "dev3.png"),
    ("Beatriz Nogueira", "dev4.png"),
)
AVATAR_COLORS = ((92, 124, 250), (150, 96, 230), (46, 184, 150), (236, 120, 90))


# --------------------------------------------------------------------------- #
# Avatar redondo (foto real ou silhueta genérica)
# --------------------------------------------------------------------------- #
def make_avatar(size: int, color, ring_color, photo_path: Optional[str] = None) -> pygame.Surface:
    ss = 3
    big = size * ss
    surf = pygame.Surface((big, big), pygame.SRCALPHA)

    loaded = False
    if photo_path and os.path.isfile(photo_path):
        try:
            img = pygame.image.load(photo_path).convert_alpha()
            iw, ih = img.get_size()
            scale = max(big / iw, big / ih)  # "cover": preenche o quadrado e corta o excesso
            img = pygame.transform.smoothscale(img, (max(1, int(iw * scale)), max(1, int(ih * scale))))
            surf.blit(img, img.get_rect(center=(big // 2, big // 2)))
            loaded = True
        except (pygame.error, OSError):
            loaded = False  # arquivo corrompido: cai no avatar genérico

    if not loaded:
        top, bottom = mix(color, (255, 255, 255), 0.25), mix(color, (0, 0, 0), 0.35)
        for y in range(big):
            pygame.draw.line(surf, mix(top, bottom, y / (big - 1)), (0, y), (big, y))
        silhouette = mix(color, (255, 255, 255), 0.82)
        pygame.draw.circle(surf, silhouette, (big // 2, int(big * 0.40)), int(big * 0.19))
        pygame.draw.ellipse(
            surf, silhouette,
            pygame.Rect(int(big * 0.17), int(big * 0.66), int(big * 0.66), int(big * 0.62)),
        )

    # recorte circular: onde a máscara é transparente, o avatar também fica
    mask = pygame.Surface((big, big), pygame.SRCALPHA)
    mask.fill((0, 0, 0, 0))
    pygame.draw.circle(mask, (255, 255, 255, 255), (big // 2, big // 2), big // 2)
    surf.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)

    pygame.draw.circle(surf, ring_color, (big // 2, big // 2), big // 2, width=4 * ss)
    return pygame.transform.smoothscale(surf, (size, size))


# --------------------------------------------------------------------------- #
# Peças flutuando ao fundo
# --------------------------------------------------------------------------- #
@dataclass
class _Floater:
    x: float
    y: float
    speed: float
    sway: float
    phase: float
    radius: int
    sprite: pygame.Surface


class FloatingDiscs:
    """Decoração: peças translúcidas subindo devagar atrás do conteúdo."""

    def __init__(self, w: int, h: int, count: int = 14, seed: int = 7):
        rng = random.Random(seed)
        self.w, self.h = w, h
        self.items: list[_Floater] = []
        for i in range(count):
            radius = rng.randint(12, 34)
            piece = Piece.P1 if i % 2 == 0 else Piece.P2
            sprite = make_disc_sprite(radius, PIECE_STYLE[piece])
            sprite.set_alpha(rng.randint(40, 80))
            self.items.append(_Floater(
                x=rng.uniform(0, w), y=rng.uniform(0, h), speed=rng.uniform(10, 28),
                sway=rng.uniform(6, 18), phase=rng.uniform(0, math.tau),
                radius=radius, sprite=sprite,
            ))

    def update(self, dt: float) -> None:
        for it in self.items:
            it.y -= it.speed * dt
            if it.y < -it.radius:
                it.y = self.h + it.radius

    def draw(self, surface: pygame.Surface, t: float) -> None:
        for it in self.items:
            x = it.x + it.sway * math.sin(t * 0.8 + it.phase)
            surface.blit(it.sprite, it.sprite.get_rect(center=(x, it.y)))


# --------------------------------------------------------------------------- #
# Menu principal
# --------------------------------------------------------------------------- #
class MenuScreen(Screen):
    ENTRIES = (
        ("Player VS Player", GO_PVP),
        ("Player VS CPU", GO_PVC),
        ("Créditos", GO_CREDITS),
    )

    def __init__(self, size: tuple[int, int], background: pygame.Surface):
        w, h = size
        self.w, self.h = w, h
        self.bg = background
        self.time = 0.0
        self.selected = 0
        self.floaters = FloatingDiscs(w, h)

        self.f_title = pick_font(66)
        self.f_sub = pick_font(20, bold=False)
        self.f_hint = pick_font(15, bold=False)
        f_btn = pick_font(25)

        self.title_y = int(h * 0.16)
        bw, bh, gap = 320, 60, 18
        top = int(h * 0.45)
        self.buttons: list[tuple[Button, str]] = []
        for i, (label, action) in enumerate(self.ENTRIES):
            rect = pygame.Rect(0, 0, bw, bh)
            rect.midtop = (w // 2, top + i * (bh + gap))
            self.buttons.append((Button(rect, label, f_btn, radius=18), action))

        self.wave = [make_disc_sprite(24, PIECE_STYLE[Piece.P1 if i % 2 == 0 else Piece.P2]) for i in range(4)]

    def handle_event(self, event: pygame.event.Event) -> Optional[str]:
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                return QUIT
            if event.key in (pygame.K_UP, pygame.K_w):
                self.selected = (self.selected - 1) % len(self.buttons)
            elif event.key in (pygame.K_DOWN, pygame.K_s):
                self.selected = (self.selected + 1) % len(self.buttons)
            elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
                return self.buttons[self.selected][1]

        elif event.type == pygame.MOUSEMOTION:
            for i, (button, _) in enumerate(self.buttons):
                if button.hit(event.pos):
                    self.selected = i

        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for button, action in self.buttons:
                if button.hit(event.pos):
                    return action
        return None

    def update(self, dt: float) -> None:
        self.time += dt
        self.floaters.update(dt)

    def draw(self, surface: pygame.Surface) -> None:
        surface.blit(self.bg, (0, 0))
        self.floaters.draw(surface, self.time)
        self._draw_title(surface)

        mouse = pygame.mouse.get_pos()
        for i, (button, _) in enumerate(self.buttons):
            button.draw(surface, mouse, selected=(i == self.selected))

        hint = self.f_hint.render("Setas + Enter, ou use o mouse   |   Esc: sair", True, MUTED)
        surface.blit(hint, hint.get_rect(center=(self.w // 2, self.h - 28)))

    def _draw_title(self, surface: pygame.Surface) -> None:
        yellow = PIECE_STYLE[Piece.P2]["base"]
        a = self.f_title.render("CONECTA", True, TEXT)
        b = self.f_title.render("4", True, yellow)
        gap = 16
        total = a.get_width() + gap + b.get_width()
        left = (self.w - total) // 2
        y = self.title_y

        for text, x in ((a, left), (b, left + a.get_width() + gap)):
            shadow = text.copy()
            shadow.fill((0, 0, 0, 255), special_flags=pygame.BLEND_RGBA_MULT)
            shadow.set_alpha(90)
            surface.blit(shadow, shadow.get_rect(midleft=(x, y + 4)))
            surface.blit(text, text.get_rect(midleft=(x, y)))

        sub = self.f_sub.render("Alinhe 4 peças para vencer", True, MUTED)
        surface.blit(sub, sub.get_rect(center=(self.w // 2, y + 52)))

        for i, sprite in enumerate(self.wave):  # quatro peças "ondulando" em linha
            x = self.w / 2 + (i - 1.5) * 62
            wy = y + 108 + 7 * math.sin(self.time * 3 + i * 0.9)
            surface.blit(sprite, sprite.get_rect(center=(x, wy)))


# --------------------------------------------------------------------------- #
# Créditos
# --------------------------------------------------------------------------- #
class CreditsScreen(Screen):
    def __init__(self, size: tuple[int, int], background: pygame.Surface):
        w, h = size
        self.w, self.h = w, h
        self.bg = background
        self.time = 0.0
        self.floaters = FloatingDiscs(w, h, count=10, seed=3)

        self.f_title = pick_font(44)
        self.f_sub = pick_font(18, bold=False)
        self.f_name = pick_font(21)

        top, bottom = 150, h - 110
        pitch = (bottom - top) / 2
        diameter = int(min(150, pitch * 0.62, w * 0.26))

        self.slots: list[tuple[pygame.Surface, str, tuple[int, int]]] = []
        for i, (name, filename) in enumerate(DEVELOPERS):
            col, row = i % 2, i // 2
            cx = int(w * (0.30 if col == 0 else 0.70))
            cy = int(top + pitch * row + diameter / 2)
            ring = PIECE_STYLE[Piece.P1 if i % 2 == 0 else Piece.P2]["base"]
            avatar = make_avatar(
                diameter, AVATAR_COLORS[i % len(AVATAR_COLORS)], ring,
                os.path.join(ASSETS_DIR, filename),
            )
            self.slots.append((avatar, name, (cx, cy)))
        self.diameter = diameter

        rect = pygame.Rect(0, 0, 190, 38)
        rect.center = (w // 2, h - 62)
        self.back = Button(rect, "Voltar  (Esc)", pick_font(18))

    def handle_event(self, event: pygame.event.Event) -> Optional[str]:
        if event.type == pygame.KEYDOWN and event.key in (
            pygame.K_ESCAPE, pygame.K_BACKSPACE, pygame.K_RETURN, pygame.K_KP_ENTER
        ):
            return GO_MENU
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and self.back.hit(event.pos):
            return GO_MENU
        return None

    def update(self, dt: float) -> None:
        self.time += dt
        self.floaters.update(dt)

    def draw(self, surface: pygame.Surface) -> None:
        surface.blit(self.bg, (0, 0))
        self.floaters.draw(surface, self.time)

        title = self.f_title.render("CRÉDITOS", True, TEXT)
        surface.blit(title, title.get_rect(center=(self.w // 2, 56)))
        sub = self.f_sub.render("Os desenvolvedores do jogo", True, MUTED)
        surface.blit(sub, sub.get_rect(center=(self.w // 2, 102)))

        for avatar, name, (cx, cy) in self.slots:
            surface.blit(avatar, avatar.get_rect(center=(cx, cy)))
            label = self.f_name.render(name, True, TEXT)
            surface.blit(label, label.get_rect(center=(cx, cy + self.diameter // 2 + 24)))

        self.back.draw(surface, pygame.mouse.get_pos())
