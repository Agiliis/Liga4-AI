"""
avaliacao.py - Avaliação dos modelos sem interface (headless).

Roda N partidas entre dois jogadores usando o próprio backend e mede:
  - % de vitórias / derrotas / empates (com intervalo de confiança de 95%, Wilson)
  - nº médio de jogadas até vencer / perder
  - teste estatístico: o modelo é melhor que o acaso? (compara com um
    jogador aleatório na MESMA posição, que é o grupo de controle)

Uso (na raiz do repositório):
    python -m src.avaliacao                       # KNN1 vs aleatório, 200 partidas
    python -m src.avaliacao --n 500 --seed 42
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from dataclasses import dataclass, field
from functools import cache
from typing import Callable, Optional

from src.backend import Game, Piece, Player, Status
from src.models import MODELOS, SO_JOGA_COMO_P2, criar_modelo

# Nome do modelo usado como grupo de controle no teste estatístico.
CONTROLE = "Aleatório"

# TODO: o KNN1 (src/ai.py) tem Piece.P2 fixo em choose_move, então só joga como
# 2º jogador. Por isso o modelo avaliado (e o controle) sempre joga como P2 e o
# adversário como P1. Quando o KNN usar game.turn, dá para avaliar nas duas
# posições (e alternar quem começa).
POSICAO_AVALIADA = Piece.P2


# --------------------------------------------------------------------------- #
# Partidas
# --------------------------------------------------------------------------- #
@dataclass
class Resultado:
    vitorias: int = 0
    derrotas: int = 0
    empates: int = 0
    jogadas_vitoria: list[int] = field(default_factory=list)
    jogadas_derrota: list[int] = field(default_factory=list)
    tempo_por_jogada: list[float] = field(default_factory=list)

    @property
    def n(self) -> int:
        return self.vitorias + self.derrotas + self.empates


def _jogar(p1: Player, p2: Player, resultados: dict[Piece, Resultado]) -> None:
    """Joga uma partida e anota o resultado do ponto de vista de cada peça em
    `resultados` (pode ser só uma, ou as duas)."""
    game = Game(players={Piece.P1: p1, Piece.P2: p2})
    while not game.is_over:
        jogador = game.current_player
        t0 = time.perf_counter()
        col = jogador.choose_move(game)
        res = resultados.get(game.turn)
        if res is not None:
            res.tempo_por_jogada.append(time.perf_counter() - t0)
        game.play(col)

    jogadas = len(game.history)
    for alvo, res in resultados.items():
        if game.status is Status.DRAW:
            res.empates += 1
        elif game.winner == alvo:
            res.vitorias += 1
            res.jogadas_vitoria.append(jogadas)
        else:
            res.derrotas += 1
            res.jogadas_derrota.append(jogadas)


def jogar_partida(p1: Player, p2: Player, alvo: Piece, res: Resultado) -> None:
    """Joga uma partida e anota o resultado do ponto de vista de `alvo`."""
    _jogar(p1, p2, {alvo: res})


def avaliar(
    p1: Player,
    p2: Player,
    alvo: Piece,
    n: int,
    ao_terminar_partida: Optional[Callable[[], None]] = None,
) -> Resultado:
    res = Resultado()
    for _ in range(n):
        jogar_partida(p1, p2, alvo, res)
        if ao_terminar_partida is not None:
            ao_terminar_partida()
    return res


# --------------------------------------------------------------------------- #
# Estatística
# --------------------------------------------------------------------------- #
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Intervalo de confiança de 95% para uma proporção (Wilson)."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    centro = (p + z * z / (2 * n)) / den
    meia = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (centro - meia, centro + meia)


def teste_duas_proporcoes(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float]:
    """Teste z unilateral: a taxa 1 é MAIOR que a taxa 2? Devolve (z, p-valor)."""
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return (0.0, 1.0)
    z = (k1 / n1 - k2 / n2) / se
    pval = 0.5 * math.erfc(z / math.sqrt(2))
    return (z, pval)


def teste_binomial(k: int, n: int) -> float:
    """Teste binomial exato bilateral de H0: p = 0,5. Devolve o p-valor.

    Usado no confronto direto: entre as partidas decididas (sem empate), os
    dois modelos vencem com a mesma frequência?
    """
    if n == 0:
        return 1.0
    obs = math.comb(n, k)
    pval = sum(math.comb(n, i) for i in range(n + 1) if math.comb(n, i) <= obs) / 2**n
    return min(1.0, pval)


def media(xs: list[float]) -> Optional[float]:
    return sum(xs) / len(xs) if xs else None


def metricas(r: Resultado) -> dict:
    """Converte um Resultado em dicionário de tipos simples (bom para UI/JSON)."""
    n = r.n
    lo, hi = wilson(r.vitorias, n)
    t = media(r.tempo_por_jogada)
    return {
        "n": n,
        "vitorias": r.vitorias,
        "derrotas": r.derrotas,
        "empates": r.empates,
        "taxa_vitoria": r.vitorias / n if n else 0.0,
        "taxa_derrota": r.derrotas / n if n else 0.0,
        "taxa_empate": r.empates / n if n else 0.0,
        "ic95_vitoria": (lo, hi),
        "jogadas_ate_vencer": media(r.jogadas_vitoria),  # None se nunca venceu
        "jogadas_ate_perder": media(r.jogadas_derrota),  # None se nunca perdeu
        "ms_por_jogada": 1000 * t if t is not None else None,
    }


# --------------------------------------------------------------------------- #
# API para a interface
# --------------------------------------------------------------------------- #
@cache
def _instancia(nome: str) -> Player:
    """Uma instância por modelo, reaproveitada entre avaliações.

    Os jogadores atuais não guardam estado entre jogadas, então reaproveitar é
    seguro e evita recarregar o CSV do KNN1 a cada chamada.
    """
    return criar_modelo(nome)


def avaliar_modelo(
    nome_modelo: str,
    nome_adversario: str,
    n: int,
    seed: Optional[int] = None,
    callback_progresso: Optional[Callable[[int, int], None]] = None,
) -> dict:
    """Avalia `nome_modelo` contra `nome_adversario` em `n` partidas.

    Roda também `n` partidas do grupo de controle (jogador aleatório na mesma
    posição, contra o mesmo adversário) e faz o teste z "modelo vence mais que
    o acaso?". Não imprime nada.

    `callback_progresso(feitas, total)` é chamado após cada partida
    (total = 2 * n: controle + modelo).

    O modelo avaliado sempre joga como P2 (ver TODO em POSICAO_AVALIADA).
    """
    for nome in (nome_modelo, nome_adversario):
        if nome not in MODELOS:
            raise KeyError(f"Modelo desconhecido: {nome!r}. Opções: {list(MODELOS)}")
    if nome_adversario in SO_JOGA_COMO_P2:
        raise ValueError(
            f"{nome_adversario!r} só joga como 2º jogador e não pode ser o adversário (P1)."
        )
    if n <= 0:
        raise ValueError("n precisa ser positivo.")

    if seed is not None:
        random.seed(seed)

    modelo = _instancia(nome_modelo)
    adversario = _instancia(nome_adversario)
    controle = _instancia(CONTROLE)

    total = 2 * n
    feitas = 0

    def _progresso() -> None:
        nonlocal feitas
        feitas += 1
        if callback_progresso is not None:
            callback_progresso(feitas, total)

    t0 = time.perf_counter()
    res_controle = avaliar(adversario, controle, POSICAO_AVALIADA, n, _progresso)
    res_modelo = avaliar(adversario, modelo, POSICAO_AVALIADA, n, _progresso)
    duracao = time.perf_counter() - t0

    z, p = teste_duas_proporcoes(
        res_modelo.vitorias, res_modelo.n, res_controle.vitorias, res_controle.n
    )
    return {
        "modelo": nome_modelo,
        "adversario": nome_adversario,
        "controle": CONTROLE,
        "posicao_modelo": POSICAO_AVALIADA.name,  # "P2"
        "n": n,
        "seed": seed,
        "metricas_modelo": metricas(res_modelo),
        "metricas_controle": metricas(res_controle),
        "teste_vs_controle": {
            "z": z,
            "p_valor": p,
            "alfa": 0.05,
            "melhor_que_acaso": p < 0.05,
        },
        "duracao_s": duracao,
    }


def posicoes_confronto(nome_a: str, nome_b: str) -> tuple[str, str]:
    """Decide quem é P1 e quem é P2 num confronto. Devolve (nome_p1, nome_p2).

    Por padrão A é P1 e B é P2; se A só souber jogar como P2, inverte.
    """
    # TODO: remover quando o KNN1 deixar de ter Piece.P2 fixo (ver POSICAO_AVALIADA).
    if nome_a in SO_JOGA_COMO_P2 and nome_b in SO_JOGA_COMO_P2:
        raise ValueError(
            f"{nome_a!r} e {nome_b!r} só jogam como 2º jogador; não dá para colocá-los frente a frente."
        )
    if nome_a in SO_JOGA_COMO_P2:
        return nome_b, nome_a
    return nome_a, nome_b


def confronto(
    nome_a: str,
    nome_b: str,
    n: int,
    seed: Optional[int] = None,
    callback_progresso: Optional[Callable[[int, int], None]] = None,
) -> dict:
    """Modelo contra modelo: `n` partidas entre `nome_a` e `nome_b`.

    As posições vêm de `posicoes_confronto` (um modelo que só joga como P2 fica
    como P2). Devolve o placar, as métricas de cada lado e um teste binomial
    sobre as partidas decididas ("algum dos dois vence mais que o outro?").
    `callback_progresso(feitas, n)` é chamado após cada partida. Não imprime nada.
    """
    for nome in (nome_a, nome_b):
        if nome not in MODELOS:
            raise KeyError(f"Modelo desconhecido: {nome!r}. Opções: {list(MODELOS)}")
    if n <= 0:
        raise ValueError("n precisa ser positivo.")
    nome_p1, nome_p2 = posicoes_confronto(nome_a, nome_b)

    if seed is not None:
        random.seed(seed)
    p1, p2 = _instancia(nome_p1), _instancia(nome_p2)

    res = {Piece.P1: Resultado(), Piece.P2: Resultado()}
    t0 = time.perf_counter()
    for i in range(n):
        _jogar(p1, p2, res)
        if callback_progresso is not None:
            callback_progresso(i + 1, n)
    duracao = time.perf_counter() - t0

    v1, v2 = res[Piece.P1].vitorias, res[Piece.P2].vitorias
    decididas = v1 + v2
    pval = teste_binomial(v1, decididas)
    return {
        "p1": nome_p1,
        "p2": nome_p2,
        "n": n,
        "seed": seed,
        "placar": {"p1": v1, "p2": v2, "empates": res[Piece.P1].empates},
        "metricas_p1": metricas(res[Piece.P1]),
        "metricas_p2": metricas(res[Piece.P2]),
        "teste_diferenca": {
            "decididas": decididas,
            "p_valor": pval,
            "alfa": 0.05,
            "diferenca_significativa": pval < 0.05,
            # Lembrete: em Conecta-4 começar jogando é vantagem, então parte da
            # diferença pode ser da posição, não do modelo.
        },
        "duracao_s": duracao,
    }


# --------------------------------------------------------------------------- #
# Linha de comando
# --------------------------------------------------------------------------- #
def _fmt(x: Optional[float], spec: str) -> str:
    return format(x, spec) if x is not None else format(float("nan"), spec)


def resumo(nome: str, m: dict) -> str:
    lo, hi = m["ic95_vitoria"]
    return (
        f"{nome:<34} n={m['n']:<4} "
        f"V={m['taxa_vitoria']:6.1%} [{lo:5.1%}–{hi:5.1%}]  "
        f"D={m['taxa_derrota']:6.1%}  E={m['taxa_empate']:5.1%}  "
        f"jog.até vencer={_fmt(m['jogadas_ate_vencer'], '5.1f')}  "
        f"jog.até perder={_fmt(m['jogadas_ate_perder'], '5.1f')}  "
        f"{_fmt(m['ms_por_jogada'], '6.1f')} ms/jogada"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=200, help="partidas por confronto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # No Windows, com a saída redirecionada (pipe/arquivo), o padrão é cp1252,
    # que não tem "α" nem "–" e derrubaria o print.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("Carregando KNN1...")
    _instancia("KNN1")  # carrega o CSV antes, para não contar no tempo da 1ª jogada

    print(f"\nTodos jogam como 2º jogador (P2). {args.n} partidas por linha.\n")
    for adv in ("Aleatório", "Heurística"):
        print(f"--- Adversário (P1): {adv} ---")
        knn = avaliar_modelo("KNN1", adv, args.n, args.seed)
        ref = avaliar_modelo("Heurística", adv, args.n, args.seed)
        print(resumo(f"{CONTROLE} (controle)", knn["metricas_controle"]))
        print(resumo("KNN1", knn["metricas_modelo"]))
        print(resumo("Heurística vence/bloqueia (ref.)", ref["metricas_modelo"]))
        t = knn["teste_vs_controle"]
        veredito = "SIM" if t["melhor_que_acaso"] else "NÃO"
        print(f"KNN1 vence mais que o acaso? z={t['z']:.2f}, p={t['p_valor']:.2g} -> {veredito} (α=0,05)\n")


if __name__ == "__main__":
    main()
