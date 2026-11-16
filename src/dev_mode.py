"""
dev_mode.py - Modo desenvolvedor: avaliação dos modelos dentro do jogo.

Duas abas:

    Avaliar modelo     um modelo (P2) contra um adversário (P1), comparado com o
                       jogador aleatório na mesma posição (src.avaliacao.avaliar_modelo)
    Modelo vs modelo   dois modelos frente a frente (src.avaliacao.confronto)

As partidas rodam numa thread separada para a janela não congelar; a tela só
lê o progresso e o resultado a cada frame. Sair da tela (Esc) cancela a
avaliação em andamento.

Controles: mouse para escolher as opções | Enter: rodar/cancelar |
Tab: trocar de aba | Esc: voltar ao menu
"""

from __future__ import annotations

import threading
import time
from typing import Callable, Optional

import pygame

from src.avaliacao import avaliar_modelo, confronto, posicoes_confronto
from src.backend import Piece
from src.models import MODELOS, SO_JOGA_COMO_P2
from src.ui import (
    BG_TOP, BUTTON, GO_MENU, MUTED, PIECE_STYLE, TEXT, Button, Screen, mix, pick_font,
)

SEED = 0                      # semente fixa: o mesmo clique dá o mesmo resultado
N_OPCOES = (50, 100, 200, 500)
# Só pode ser adversário (P1) quem sabe jogar como 1º jogador.
ADVERSARIOS = tuple(m for m in MODELOS if m not in SO_JOGA_COMO_P2)

MARGIN = 24
OK = (46, 184, 150)
BAD = PIECE_STYLE[Piece.P1]["base"]
ACCENT = PIECE_STYLE[Piece.P2]["base"]
PANEL = (10, 14, 34)


# --------------------------------------------------------------------------- #
# Avaliação em segundo plano
# --------------------------------------------------------------------------- #
class _Cancelado(Exception):
    """Levantada dentro do callback de progresso para interromper a avaliação."""


class Tarefa:
    """Roda `func(..., callback_progresso=...)` numa thread daemon.

    A thread só escreve nos atributos deste objeto; a tela só lê. Cada clique
    em "Rodar" cria uma Tarefa nova, então uma thread cancelada que ainda está
    terminando não sobrescreve o resultado de outra.
    """

    def __init__(self, func: Callable[..., dict], *args):
        self.feitas = 0
        self.total = 0
        self.resultado: Optional[dict] = None
        self.erro: Optional[str] = None
        self.cancelada = False
        self._parar = threading.Event()
        self.inicio = time.perf_counter()
        self._thread = threading.Thread(target=self._rodar, args=(func, args), daemon=True)
        self._thread.start()

    @property
    def rodando(self) -> bool:
        return self._thread.is_alive()

    @property
    def cancelando(self) -> bool:
        return self._parar.is_set() and self.rodando

    def cancelar(self) -> None:
        self._parar.set()

    def _progresso(self, feitas: int, total: int) -> None:
        if self._parar.is_set():
            raise _Cancelado
        self.feitas, self.total = feitas, total
        time.sleep(0)  # cede o GIL para a janela continuar fluida

    def _rodar(self, func, args) -> None:
        try:
            self.resultado = func(*args, callback_progresso=self._progresso)
        except _Cancelado:
            self.cancelada = True
        except Exception as exc:  # mostra na tela em vez de matar a thread em silêncio
            self.erro = f"{type(exc).__name__}: {exc}"


# --------------------------------------------------------------------------- #
# Widgets
# --------------------------------------------------------------------------- #
class Seletor:
    """Rótulo + uma fileira de botões, dos quais um fica selecionado."""

    LABEL_W = 128

    def __init__(self, rotulo: str, opcoes, valor, rect: pygame.Rect,
                 f_rotulo: pygame.font.Font, f_botao: pygame.font.Font):
        self.rotulo = rotulo
        self.opcoes = list(opcoes)
        self.indice = self.opcoes.index(valor) if valor in self.opcoes else 0
        self.rect = rect
        self.f_rotulo = f_rotulo
        x0, gap = rect.left + self.LABEL_W, 8
        bw = (rect.right - x0 - gap * (len(self.opcoes) - 1)) / len(self.opcoes)
        self.botoes = [
            Button(pygame.Rect(int(x0 + i * (bw + gap)), rect.top, int(bw), rect.height),
                   str(op), f_botao, radius=10)
            for i, op in enumerate(self.opcoes)
        ]

    @property
    def valor(self):
        return self.opcoes[self.indice]

    def clicar(self, pos) -> bool:
        for i, b in enumerate(self.botoes):
            if b.hit(pos):
                self.indice = i
                return True
        return False

    def draw(self, surface: pygame.Surface, mouse) -> None:
        txt = self.f_rotulo.render(self.rotulo, True, MUTED)
        surface.blit(txt, txt.get_rect(midleft=(self.rect.left, self.rect.centery)))
        for i, b in enumerate(self.botoes):
            b.draw(surface, mouse, selected=(i == self.indice))


def _encaixar(texto: str, fonte: pygame.font.Font, largura: int) -> str:
    """Corta o texto com reticências se ele não couber em `largura` pixels."""
    if fonte.size(texto)[0] <= largura:
        return texto
    while texto and fonte.size(texto + "…")[0] > largura:
        texto = texto[:-1]
    return texto + "…"


def _segmentos(surface, partes: list[tuple[str, pygame.font.Font, tuple]], centro) -> None:
    """Desenha trechos com fontes/cores diferentes numa mesma linha centralizada."""
    imgs = [f.render(t, True, c) for t, f, c in partes]
    x = centro[0] - sum(i.get_width() for i in imgs) // 2
    for img in imgs:
        surface.blit(img, img.get_rect(midleft=(x, centro[1])))
        x += img.get_width()


def _painel(surface, rect: pygame.Rect, alpha: int = 150) -> None:
    s = pygame.Surface(rect.size, pygame.SRCALPHA)
    pygame.draw.rect(s, (*PANEL, alpha), s.get_rect(), border_radius=12)
    surface.blit(s, rect.topleft)


def _tabela(surface, top: int, cabecalho: list[str], linhas: list[list[str]],
            rh: int, f_cab, f_cel, largura_total: int) -> int:
    """Tabela métrica x jogador. Devolve a coordenada y logo abaixo dela."""
    x, w = MARGIN, largura_total - 2 * MARGIN
    rect = pygame.Rect(x, top, w, rh * (1 + len(linhas)) + 8)
    _painel(surface, rect)
    fracs = (0.34, 0.33, 0.33)
    cols, acc = [], x
    for f in fracs:
        cols.append((acc, int(w * f)))
        acc += int(w * f)

    for i, linha in enumerate([cabecalho] + linhas):
        cy = top + 4 + i * rh + rh // 2
        if i > 0 and i % 2 == 0:
            faixa = pygame.Surface((w - 8, rh), pygame.SRCALPHA)
            faixa.fill((255, 255, 255, 14))
            surface.blit(faixa, (x + 4, cy - rh // 2))
        fonte = f_cab if i == 0 else f_cel
        for j, (texto, (cx, cw)) in enumerate(zip(linha, cols)):
            cor = MUTED if (i == 0 or j == 0) else TEXT
            img = fonte.render(_encaixar(texto, fonte, cw - 16), True, cor)
            if j == 0:
                surface.blit(img, img.get_rect(midleft=(cx + 12, cy)))
            else:
                surface.blit(img, img.get_rect(center=(cx + cw // 2, cy)))
        if i == 0:
            y = top + 4 + rh
            pygame.draw.line(surface, mix(PANEL, TEXT, 0.25), (x + 10, y), (x + w - 10, y))
    return rect.bottom


# ---- formatação ------------------------------------------------------------ #
def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _num(x: Optional[float], sufixo: str = "") -> str:
    return "—" if x is None else f"{x:.1f}{sufixo}"


def _p(p: float) -> str:
    return f"{p:.2g}" if p >= 0.001 else f"{p:.1e}"


def _linhas_metricas(*ms: dict) -> list[list[str]]:
    def vit(m):
        lo, hi = m["ic95_vitoria"]
        return f"{_pct(m['taxa_vitoria'])} [{100 * lo:.0f}–{100 * hi:.0f}]"

    return [
        ["Vitórias (IC 95%)"] + [vit(m) for m in ms],
        ["Derrotas"] + [_pct(m["taxa_derrota"]) for m in ms],
        ["Empates"] + [_pct(m["taxa_empate"]) for m in ms],
        ["Jogadas até vencer"] + [_num(m["jogadas_ate_vencer"]) for m in ms],
        ["Jogadas até perder"] + [_num(m["jogadas_ate_perder"]) for m in ms],
        ["Tempo por jogada"] + [_num(m["ms_por_jogada"], " ms") for m in ms],
    ]


# --------------------------------------------------------------------------- #
# Abas
# --------------------------------------------------------------------------- #
class _Aba:
    """Base das duas abas: seletores, botão Rodar/Cancelar, progresso e resultado."""

    titulo = ""
    EXTRA_RESULTADO = 66  # px do resultado que não são linhas da tabela

    def __init__(self, tela: "DevModeScreen"):
        self.tela = tela
        self.tarefa: Optional[Tarefa] = None
        self.resultado: Optional[dict] = None
        self.mensagem: Optional[tuple[str, tuple]] = None  # (texto, cor) do último fim
        w = tela.w
        self.seletores = self._criar_seletores(tela.y_conteudo)
        y = self.seletores[-1].rect.bottom + 12
        self.botao = Button(pygame.Rect(MARGIN, y, 150, 36), "Rodar", tela.f_botao, radius=12)
        self.barra = pygame.Rect(MARGIN + 162, y + 3, w - 2 * MARGIN - 162, 30)
        self.y_resultado = self.botao.rect.bottom + 10
        # altura das linhas da tabela: o que sobrar até o rodapé, entre 18 e 30 px
        livre = tela.h - 28 - self.y_resultado - self.EXTRA_RESULTADO
        self.rh = int(max(18, min(30, livre / 7)))

    # ---- a implementar nas abas ----------------------------------------- #
    def _criar_seletores(self, top: int) -> list[Seletor]:
        raise NotImplementedError

    def _iniciar(self) -> Tarefa:
        raise NotImplementedError

    def _aviso(self) -> Optional[tuple[str, tuple]]:
        """Mensagem mostrada na barra enquanto nada roda (prévia das opções)."""
        return None

    def _desenhar_resultado(self, surface, r: dict) -> None:
        raise NotImplementedError

    def _texto_progresso(self, t: Tarefa) -> str:
        return f"{t.feitas}/{t.total} partidas"

    # ---- comum ------------------------------------------------------------ #
    def _seletor(self, i: int, rotulo: str, opcoes, valor, top: int) -> Seletor:
        rect = pygame.Rect(MARGIN, top + i * 40, self.tela.w - 2 * MARGIN, 32)
        return Seletor(rotulo, opcoes, valor, rect, self.tela.f_rotulo, self.tela.f_opcao)

    @property
    def ocupada(self) -> bool:
        return self.tarefa is not None and self.tarefa.rodando

    def cancelar(self) -> None:
        if self.ocupada:
            self.tarefa.cancelar()

    def rodar_ou_cancelar(self) -> None:
        if self.ocupada:
            self.tarefa.cancelar()
            return
        try:
            self.tarefa = self._iniciar()
        except (ValueError, KeyError) as exc:
            self.mensagem = (str(exc), BAD)
            return
        self.mensagem = None

    def handle_click(self, pos) -> None:
        if self.botao.hit(pos):
            self.rodar_ou_cancelar()
        elif not self.ocupada:
            for s in self.seletores:
                if s.clicar(pos):
                    self.mensagem = None
                    break

    def update(self) -> None:
        t = self.tarefa
        if t is None or t.rodando:
            return
        # a thread terminou: recolhe o que ela deixou e esquece a tarefa
        if t.resultado is not None:
            self.resultado = t.resultado
            self.mensagem = (f"Concluído em {t.resultado['duracao_s']:.1f} s.", OK)
        elif t.erro:
            self.mensagem = (t.erro, BAD)
        elif t.cancelada:
            self.mensagem = ("Cancelado.", MUTED)
        self.tarefa = None

    def draw(self, surface, mouse) -> None:
        for s in self.seletores:
            s.draw(surface, mouse)
        if self.ocupada:  # opções "apagadas" enquanto a avaliação roda
            top = self.seletores[0].rect.top - 4
            area = pygame.Rect(MARGIN - 4, top, self.tela.w - 2 * MARGIN + 8,
                               self.seletores[-1].rect.bottom + 4 - top)
            veu = pygame.Surface(area.size, pygame.SRCALPHA)
            veu.fill((*BG_TOP, 140))
            surface.blit(veu, area.topleft)

        t = self.tarefa
        if t is not None and t.cancelando:
            self.botao.label = "Cancelando…"
        else:
            self.botao.label = "Cancelar" if self.ocupada else "Rodar"
        self.botao.draw(surface, mouse, pulse=0.0 if self.ocupada else 0.25)
        self._desenhar_barra(surface)

        if self.resultado is not None:
            self._desenhar_resultado(surface, self.resultado)
        elif not self.ocupada:
            dica = self.tela.f_info.render(
                "Escolha as opções e clique em Rodar (ou Enter).", True, MUTED)
            surface.blit(dica, dica.get_rect(midtop=(self.tela.w // 2, self.y_resultado + 20)))

    def _desenhar_barra(self, surface) -> None:
        r = self.barra
        pygame.draw.rect(surface, mix(PANEL, BUTTON, 0.5), r, border_radius=10)
        t = self.tarefa
        if t is not None:
            frac = t.feitas / t.total if t.total else 0.0
            if frac > 0:
                cheio = r.copy()
                cheio.width = max(20, int(r.width * frac))
                pygame.draw.rect(surface, mix(BUTTON, ACCENT, 0.55), cheio, border_radius=10)
            if t.cancelando:
                texto, cor = "Cancelando…", TEXT
            elif t.total == 0:
                texto, cor = "Carregando modelos…", TEXT
            else:
                texto, cor = f"{self._texto_progresso(t)}  ·  {time.perf_counter() - t.inicio:.0f} s", TEXT
        elif self.mensagem is not None:
            texto, cor = self.mensagem
        else:
            texto, cor = self._aviso() or ("Pronto.", MUTED)
        pygame.draw.rect(surface, mix(PANEL, TEXT, 0.2), r, width=1, border_radius=10)
        img = self.tela.f_info.render(_encaixar(texto, self.tela.f_info, r.width - 16), True, cor)
        surface.blit(img, img.get_rect(center=r.center))


class AbaAvaliar(_Aba):
    titulo = "Avaliar modelo"

    def _criar_seletores(self, top):
        modelos = list(MODELOS)
        padrao = "KNN1" if "KNN1" in MODELOS else modelos[0]
        return [
            self._seletor(0, "Modelo", modelos, padrao, top),
            self._seletor(1, "Adversário", ADVERSARIOS, ADVERSARIOS[0], top),
            self._seletor(2, "Partidas", N_OPCOES, 100, top),
        ]

    def _iniciar(self):
        modelo, adv, n = (s.valor for s in self.seletores)
        return Tarefa(avaliar_modelo, modelo, adv, n, SEED)

    def _aviso(self):
        modelo, adv, n = (s.valor for s in self.seletores)
        return (f"{modelo} joga como P2 contra {adv} (P1), mais {n} partidas do controle aleatório.", MUTED)

    def _texto_progresso(self, t):
        n = t.total // 2
        fase = "controle aleatório" if t.feitas <= n else "modelo"
        return f"{t.feitas}/{t.total} partidas ({fase})"

    def _desenhar_resultado(self, surface, r):
        tela = self.tela
        y = self.y_resultado
        info = (f"{r['modelo']} (P2) vs {r['adversario']} (P1)  ·  {r['n']} partidas "
                f"+ {r['n']} do controle  ·  {r['duracao_s']:.1f} s")
        img = tela.f_info.render(_encaixar(info, tela.f_info, tela.w - 2 * MARGIN), True, MUTED)
        surface.blit(img, img.get_rect(midtop=(tela.w // 2, y)))
        y += img.get_height() + 6

        linhas = _linhas_metricas(r["metricas_modelo"], r["metricas_controle"])
        y = _tabela(surface, y, ["Métrica", r["modelo"], f"{r['controle']} (controle)"],
                    linhas, self.rh, tela.f_cab, tela.f_cel, tela.w)

        t = r["teste_vs_controle"]
        sim = t["melhor_que_acaso"]
        _segmentos(surface, [
            ("Melhor que o acaso?  ", tela.f_veredito, TEXT),
            ("SIM" if sim else "NÃO", tela.f_veredito, OK if sim else BAD),
            (f"   p = {_p(t['p_valor'])}  (z = {t['z']:.2f}, α = 0,05)", tela.f_info, MUTED),
        ], (tela.w // 2, y + 16))


class AbaVersus(_Aba):
    titulo = "Modelo vs modelo"
    EXTRA_RESULTADO = 98  # + linha do placar

    def _criar_seletores(self, top):
        modelos = list(MODELOS)
        a = "Heurística" if "Heurística" in MODELOS else modelos[0]
        b = "KNN1" if "KNN1" in MODELOS else modelos[-1]
        return [
            self._seletor(0, "Modelo A", modelos, a, top),
            self._seletor(1, "Modelo B", modelos, b, top),
            self._seletor(2, "Partidas", N_OPCOES, 100, top),
        ]

    def _iniciar(self):
        a, b, n = (s.valor for s in self.seletores)
        posicoes_confronto(a, b)  # valida já (ValueError vira mensagem na tela)
        return Tarefa(confronto, a, b, n, SEED)

    def _aviso(self):
        a, b, _ = (s.valor for s in self.seletores)
        try:
            p1, p2 = posicoes_confronto(a, b)
        except ValueError:
            return (f"{a} e {b} só jogam como P2 (limitação atual): escolha outro par.", BAD)
        texto = f"P1 (começa): {p1}   ·   P2: {p2}"
        if p1 != a:
            texto += f"   ({p2} só joga como P2)"
        return (texto, MUTED)

    def _desenhar_resultado(self, surface, r):
        tela = self.tela
        y = self.y_resultado
        placar = r["placar"]
        c1, c2 = PIECE_STYLE[Piece.P1]["light"], PIECE_STYLE[Piece.P2]["base"]
        _segmentos(surface, [
            (f"{r['p1']} (P1)  ", tela.f_placar_nome, c1),
            (f"{placar['p1']} × {placar['p2']}", tela.f_placar, TEXT),
            (f"  {r['p2']} (P2)", tela.f_placar_nome, c2),
        ], (tela.w // 2, y + 14))
        y += 32
        info = (f"{placar['empates']} empate(s)  ·  {r['n']} partidas  ·  {r['duracao_s']:.1f} s"
                f"  ·  P1 começa (já é vantagem)")
        img = tela.f_info.render(_encaixar(info, tela.f_info, tela.w - 2 * MARGIN), True, MUTED)
        surface.blit(img, img.get_rect(midtop=(tela.w // 2, y)))
        y += img.get_height() + 6

        linhas = _linhas_metricas(r["metricas_p1"], r["metricas_p2"])
        y = _tabela(surface, y, ["Métrica", f"{r['p1']} (P1)", f"{r['p2']} (P2)"],
                    linhas, self.rh, tela.f_cab, tela.f_cel, tela.w)

        t = r["teste_diferenca"]
        sim = t["diferenca_significativa"]
        _segmentos(surface, [
            ("Diferença significativa?  ", tela.f_veredito, TEXT),
            ("SIM" if sim else "NÃO", tela.f_veredito, OK if sim else MUTED),
            (f"   p = {_p(t['p_valor'])}  ({t['decididas']} partidas decididas)", tela.f_info, MUTED),
        ], (tela.w // 2, y + 16))


# --------------------------------------------------------------------------- #
# Tela
# --------------------------------------------------------------------------- #
class DevModeScreen(Screen):
    def __init__(self, size: tuple[int, int], background: pygame.Surface):
        w, h = size
        self.w, self.h = w, h
        self.bg = background

        tamanho = 34  # encolhe o título se a janela for estreita (não pode encostar no "← Menu")
        while tamanho > 20 and pick_font(tamanho).size("MODO DESENVOLVEDOR")[0] > w - 2 * (MARGIN + 104):
            tamanho -= 2
        self.f_titulo = pick_font(tamanho)
        self.f_aba = pick_font(19)
        self.f_botao = pick_font(19)
        self.f_rotulo = pick_font(17, bold=False)
        self.f_opcao = pick_font(16)
        self.f_info = pick_font(14, bold=False)
        self.f_veredito = pick_font(18)
        self.f_placar = pick_font(26)
        self.f_placar_nome = pick_font(19)
        self.f_hint = pick_font(15, bold=False)

        # topo: título e "voltar"
        self.voltar = Button(pygame.Rect(MARGIN, 16, 96, 30), "← Menu", pick_font(15), radius=10)
        # abas
        gap = 10
        tw = (w - 2 * MARGIN - gap) // 2
        self.rects_abas = [pygame.Rect(MARGIN + i * (tw + gap), 58, tw, 36) for i in range(2)]
        self.y_conteudo = 106

        self.abas: list[_Aba] = [AbaAvaliar(self), AbaVersus(self)]
        # fonte da tabela conforme a altura das linhas (a janela menor tem ~560 px)
        pequena = min(a.rh for a in self.abas) < 25
        self.f_cab = pick_font(14 if pequena else 16)
        self.f_cel = pick_font(14 if pequena else 16, bold=False)
        self.botoes_abas = [Button(r, a.titulo, self.f_aba, radius=12)
                            for r, a in zip(self.rects_abas, self.abas)]
        self.ativa = 0

    @property
    def _ocupada(self) -> bool:
        return any(a.ocupada for a in self.abas)

    def _sair(self) -> str:
        for a in self.abas:
            a.cancelar()
        return GO_MENU

    def handle_event(self, event: pygame.event.Event) -> Optional[str]:
        aba = self.abas[self.ativa]
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                return self._sair()
            if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                aba.rodar_ou_cancelar()
            elif event.key == pygame.K_TAB and not self._ocupada:
                self.ativa = (self.ativa + 1) % len(self.abas)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.voltar.hit(event.pos):
                return self._sair()
            for i, b in enumerate(self.botoes_abas):
                if b.hit(event.pos):
                    if not self._ocupada:  # não troca de aba no meio de uma avaliação
                        self.ativa = i
                    return None
            aba.handle_click(event.pos)
        return None

    def update(self, dt: float) -> None:
        for a in self.abas:
            a.update()

    def draw(self, surface: pygame.Surface) -> None:
        surface.blit(self.bg, (0, 0))
        mouse = pygame.mouse.get_pos()

        titulo = self.f_titulo.render("MODO DESENVOLVEDOR", True, TEXT)
        surface.blit(titulo, titulo.get_rect(center=(self.w // 2, 31)))
        self.voltar.draw(surface, mouse)

        for i, b in enumerate(self.botoes_abas):
            b.draw(surface, mouse, selected=(i == self.ativa))

        self.abas[self.ativa].draw(surface, mouse)

        dica = "Enter: rodar/cancelar   |   Tab: trocar de aba   |   Esc: voltar ao menu"
        img = self.f_hint.render(_encaixar(dica, self.f_hint, self.w - 2 * MARGIN), True, MUTED)
        surface.blit(img, img.get_rect(center=(self.w // 2, self.h - 14)))
