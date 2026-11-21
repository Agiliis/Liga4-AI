"""
JOGADOR HÍBRIDO: ÁRVORE (MINIMAX + ALPHA-BETA) + KNN COMO AVALIADOR DE FOLHAS
============================================================================

Pré-requisito: rodar ANTES `python limpar_dados_treino.py` e colocar o `tree_knn_dataset.npz`
gerado na pasta `db/` (na raiz do projeto, onde o jogo é executado).

Convenções internas:
    - Dentro da árvore, o tabuleiro está SEMPRE na visão da IA: +1 = peça da IA, -1 = adversário.
    - Para consultar o KNN/cache/lookup, o tabuleiro é convertido para a visão dos dados de treino:
      +1 = quem começou a partida, -1 = quem jogou em segundo.
    - Linha 0 = topo do tabuleiro, linha 5 = base.

Novidades em relação ao `knn_new.py`:
    1. Treina com os dados limpos (sem terminais, convenção única, meio de jogo reconstruído).
    2. Lookup exato: tabuleiros de 8 peças existentes no UCI respondem na hora, sem KNN.
    3. Cache de avaliações do KNN (chave = tabuleiro, tratando espelhos como iguais).
       Os espelhos dos tabuleiros de treino já vêm prontos no .npz (feitos em limpar_dados_treino.py).
    4. Limpeza da cache: tabuleiros com menos peças que o jogo atual nunca mais ocorrem e são apagados.
    5. A árvore joga e desfaz jogadas no MESMO tabuleiro (sem copiar) e checa vitória só ao redor da peça nova.
    6. Descobre se a IA é o primeiro ou o segundo jogador pela paridade de peças a cada jogada
       e usa `game.turn` para saber a COR da própria peça (a CPU é sempre Piece.P2, mesmo quando começa).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.neighbors import KNeighborsRegressor

from src.backend import Game, Player

# =====================================================================
# CONFIGURAÇÕES
# =====================================================================
CAMINHO_DADOS_LIMPOS = Path("db") / "tree_knn_dataset.npz"

NUMERO_DE_LINHAS = 6
NUMERO_DE_COLUNAS = 7
NUMERO_DE_CASAS = NUMERO_DE_LINHAS * NUMERO_DE_COLUNAS

QUANTIDADE_DE_VIZINHOS_DO_KNN = 9
PROFUNDIDADE_DA_ARVORE = 4
ORDEM_DAS_COLUNAS_DO_CENTRO_PARA_AS_BORDAS = (3, 2, 4, 1, 5, 0, 6)

PONTUACAO_BASE_DE_VITORIA = 100.0  # bem maior que qualquer nota do KNN (que fica entre -1 e +1)
CODIGO_FONTE_UCI = 0
NUMERO_DE_PECAS_DOS_TABULEIROS_DO_UCI = 8

# Trava de segurança de memória: se a cache passar disso, ela é esvaziada.
# (Cada entrada ocupa algo como 200 bytes, então 300 mil entradas ≈ 60 MB.)
LIMITE_MAXIMO_DE_ENTRADAS_NA_CACHE = 300_000

DIRECOES_PARA_CONFERIR_QUATRO_EM_LINHA = ((0, 1), (1, 0), (1, 1), (1, -1))


# =====================================================================
# TREINO DO KNN
# =====================================================================
def carregar_dados_limpos() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not CAMINHO_DADOS_LIMPOS.exists():
        raise FileNotFoundError(
            f"'{CAMINHO_DADOS_LIMPOS.name}' não encontrado. Rode primeiro: python limpar_dados_treino.py"
        )
    arquivo = np.load(CAMINHO_DADOS_LIMPOS)
    if "ja_inclui_espelhos" not in arquivo.files:
        raise RuntimeError(
            f"'{CAMINHO_DADOS_LIMPOS.name}' é de uma versão antiga (sem os espelhos prontos). "
            "Rode de novo: python limpar_dados_treino.py"
        )
    return arquivo["tabuleiros"], arquivo["notas"], arquivo["fontes"]


def treinar_knn(tabuleiros: np.ndarray, notas: np.ndarray) -> KNeighborsRegressor:
    """Treina o KNN com os dados limpos (que já incluem os espelhos e não têm duplicatas)."""
    modelo_knn = KNeighborsRegressor(
        n_neighbors=QUANTIDADE_DE_VIZINHOS_DO_KNN,
        metric="manhattan",
        weights="distance",
        # Busca bruta: com 42 dimensões a Ball-Tree quase não consegue podar nada e acabou MAIS LENTA
        # nos testes. Os vizinhos e as notas são exatamente os mesmos; muda só o jeito de procurar.
        algorithm="brute",
        n_jobs=1,  # consultas são uma folha por vez: criar várias threads a cada chamada só atrasa
    )
    modelo_knn.fit(tabuleiros.astype(np.float32), notas)
    return modelo_knn


def gerar_chave_canonica(tabuleiro_na_visao_do_primeiro: np.ndarray) -> bytes:
    """Chave de dicionário que é a MESMA para um tabuleiro e para o seu espelho."""
    bytes_do_tabuleiro = tabuleiro_na_visao_do_primeiro.tobytes()
    bytes_do_espelho = tabuleiro_na_visao_do_primeiro[:, ::-1].tobytes()
    return min(bytes_do_tabuleiro, bytes_do_espelho)


def montar_dicionario_de_valores_exatos_do_uci(
    tabuleiros: np.ndarray, notas: np.ndarray, fontes: np.ndarray
) -> dict[bytes, float]:
    """Lookup: tabuleiro de 8 peças -> valor teórico exato (vitória/derrota/empate do 1º jogador)."""
    dicionario: dict[bytes, float] = {}
    for tabuleiro, nota in zip(tabuleiros[fontes == CODIGO_FONTE_UCI], notas[fontes == CODIGO_FONTE_UCI]):
        grade = tabuleiro.reshape(NUMERO_DE_LINHAS, NUMERO_DE_COLUNAS)
        dicionario[gerar_chave_canonica(grade)] = float(nota)
    return dicionario


# =====================================================================
# JOGADOR
# =====================================================================
class TreeKNNPlayer(Player):
    def __init__(self, name: str = "Tree+KNN", profundidade: int = PROFUNDIDADE_DA_ARVORE):
        super().__init__(name)
        tabuleiros, notas, fontes = carregar_dados_limpos()
        self.modelo_knn = treinar_knn(tabuleiros, notas)
        self.valores_exatos_do_uci = montar_dicionario_de_valores_exatos_do_uci(tabuleiros, notas, fontes)
        self.profundidade_da_arvore = profundidade

        # Cache organizada por quantidade de peças: cache[n] guarda só tabuleiros com n peças.
        # Isso permite apagar de uma vez todos os que ficaram impossíveis (menos peças que o jogo atual).
        self.cache_de_avaliacoes_por_numero_de_pecas: list[dict[bytes, float]] = [
            {} for _ in range(NUMERO_DE_CASAS + 1)
        ]
        self.total_de_entradas_na_cache = 0

        # Contadores só para você acompanhar o desempenho.
        self.quantidade_de_consultas_ao_knn = 0
        self.quantidade_de_acertos_na_cache = 0
        self.quantidade_de_valores_exatos_usados = 0

        # Estado da busca atual (preenchido a cada jogada em choose_move).
        self._tabuleiro_na_visao_da_ia = np.zeros((NUMERO_DE_LINHAS, NUMERO_DE_COLUNAS), dtype=np.int8)
        self._proxima_linha_livre_de_cada_coluna = [NUMERO_DE_LINHAS - 1] * NUMERO_DE_COLUNAS
        self._numero_de_pecas_no_tabuleiro = 0
        self._ia_e_o_primeiro_jogador = True

    # -----------------------------------------------------------------
    # Cache
    # -----------------------------------------------------------------
    def _apagar_da_cache_os_tabuleiros_impossiveis(self, numero_de_pecas_atual: int) -> None:
        """Peças nunca saem do tabuleiro: com N peças no jogo, tabuleiros com menos de N nunca mais voltam."""
        for numero_de_pecas in range(numero_de_pecas_atual):
            cache_dessa_quantidade = self.cache_de_avaliacoes_por_numero_de_pecas[numero_de_pecas]
            if cache_dessa_quantidade:
                self.total_de_entradas_na_cache -= len(cache_dessa_quantidade)
                cache_dessa_quantidade.clear()

    def _esvaziar_cache_se_passou_do_limite(self) -> None:
        if self.total_de_entradas_na_cache > LIMITE_MAXIMO_DE_ENTRADAS_NA_CACHE:
            for cache_dessa_quantidade in self.cache_de_avaliacoes_por_numero_de_pecas:
                cache_dessa_quantidade.clear()
            self.total_de_entradas_na_cache = 0

    # -----------------------------------------------------------------
    # Avaliação de folhas (valor exato -> cache -> KNN)
    # -----------------------------------------------------------------
    def _avaliar_folha(self) -> float:
        """Nota estratégica (entre -1 e +1) do tabuleiro atual, do ponto de vista da IA."""
        if self._ia_e_o_primeiro_jogador:
            tabuleiro_na_visao_do_primeiro = self._tabuleiro_na_visao_da_ia
        else:
            tabuleiro_na_visao_do_primeiro = -self._tabuleiro_na_visao_da_ia

        chave = gerar_chave_canonica(tabuleiro_na_visao_do_primeiro)
        numero_de_pecas = self._numero_de_pecas_no_tabuleiro
        nota_do_ponto_de_vista_do_primeiro = None

        if numero_de_pecas == NUMERO_DE_PECAS_DOS_TABULEIROS_DO_UCI:
            nota_do_ponto_de_vista_do_primeiro = self.valores_exatos_do_uci.get(chave)
            if nota_do_ponto_de_vista_do_primeiro is not None:
                self.quantidade_de_valores_exatos_usados += 1

        if nota_do_ponto_de_vista_do_primeiro is None:
            cache_dessa_quantidade = self.cache_de_avaliacoes_por_numero_de_pecas[numero_de_pecas]
            nota_do_ponto_de_vista_do_primeiro = cache_dessa_quantidade.get(chave)

            if nota_do_ponto_de_vista_do_primeiro is not None:
                self.quantidade_de_acertos_na_cache += 1
            else:
                vetor_de_consulta = tabuleiro_na_visao_do_primeiro.reshape(1, -1).astype(np.float32)
                nota_do_ponto_de_vista_do_primeiro = float(self.modelo_knn.predict(vetor_de_consulta)[0])
                self.quantidade_de_consultas_ao_knn += 1
                cache_dessa_quantidade[chave] = nota_do_ponto_de_vista_do_primeiro
                self.total_de_entradas_na_cache += 1

        if self._ia_e_o_primeiro_jogador:
            return nota_do_ponto_de_vista_do_primeiro
        return -nota_do_ponto_de_vista_do_primeiro

    # -----------------------------------------------------------------
    # Mecânica do tabuleiro (joga e desfaz no mesmo array, sem copiar)
    # -----------------------------------------------------------------
    def _colunas_jogaveis(self) -> list[int]:
        return [
            coluna for coluna in ORDEM_DAS_COLUNAS_DO_CENTRO_PARA_AS_BORDAS
            if self._proxima_linha_livre_de_cada_coluna[coluna] >= 0
        ]

    def _colocar_peca(self, coluna: int, peca: int) -> int:
        linha = self._proxima_linha_livre_de_cada_coluna[coluna]
        self._tabuleiro_na_visao_da_ia[linha, coluna] = peca
        self._proxima_linha_livre_de_cada_coluna[coluna] -= 1
        self._numero_de_pecas_no_tabuleiro += 1
        return linha

    def _remover_peca(self, coluna: int) -> None:
        self._proxima_linha_livre_de_cada_coluna[coluna] += 1
        linha = self._proxima_linha_livre_de_cada_coluna[coluna]
        self._tabuleiro_na_visao_da_ia[linha, coluna] = 0
        self._numero_de_pecas_no_tabuleiro -= 1

    def _ultima_jogada_completou_quatro_em_linha(self, linha: int, coluna: int, peca: int) -> bool:
        """Só a peça recém-colocada pode ter criado um 4 em linha: conferimos apenas ao redor dela."""
        tabuleiro = self._tabuleiro_na_visao_da_ia
        for passo_linha, passo_coluna in DIRECOES_PARA_CONFERIR_QUATRO_EM_LINHA:
            pecas_seguidas = 1
            for sentido in (1, -1):
                linha_vizinha = linha + sentido * passo_linha
                coluna_vizinha = coluna + sentido * passo_coluna
                while (
                    0 <= linha_vizinha < NUMERO_DE_LINHAS
                    and 0 <= coluna_vizinha < NUMERO_DE_COLUNAS
                    and tabuleiro[linha_vizinha, coluna_vizinha] == peca
                ):
                    pecas_seguidas += 1
                    linha_vizinha += sentido * passo_linha
                    coluna_vizinha += sentido * passo_coluna
            if pecas_seguidas >= 4:
                return True
        return False

    # -----------------------------------------------------------------
    # Árvore Minimax com poda Alpha-Beta
    # -----------------------------------------------------------------
    def _minimax(self, profundidade_restante: int, alpha: float, beta: float, e_a_vez_da_ia: bool) -> float:
        colunas_jogaveis = self._colunas_jogaveis()
        if not colunas_jogaveis:
            return 0.0  # tabuleiro cheio: empate
        if profundidade_restante == 0:
            return self._avaliar_folha()

        peca_de_quem_joga = 1 if e_a_vez_da_ia else -1
        # Vitória encontrada mais cedo vale mais (e derrota mais tardia vale menos ruim).
        pontuacao_se_essa_jogada_vencer = PONTUACAO_BASE_DE_VITORIA + profundidade_restante - 1

        if e_a_vez_da_ia:
            melhor_nota = -np.inf
            for coluna in colunas_jogaveis:
                linha = self._colocar_peca(coluna, peca_de_quem_joga)
                if self._ultima_jogada_completou_quatro_em_linha(linha, coluna, peca_de_quem_joga):
                    nota = pontuacao_se_essa_jogada_vencer
                else:
                    nota = self._minimax(profundidade_restante - 1, alpha, beta, False)
                self._remover_peca(coluna)

                melhor_nota = max(melhor_nota, nota)
                alpha = max(alpha, melhor_nota)
                if beta <= alpha:
                    break
            return melhor_nota

        pior_nota = np.inf
        for coluna in colunas_jogaveis:
            linha = self._colocar_peca(coluna, peca_de_quem_joga)
            if self._ultima_jogada_completou_quatro_em_linha(linha, coluna, peca_de_quem_joga):
                nota = -pontuacao_se_essa_jogada_vencer
            else:
                nota = self._minimax(profundidade_restante - 1, alpha, beta, True)
            self._remover_peca(coluna)

            pior_nota = min(pior_nota, nota)
            beta = min(beta, pior_nota)
            if beta <= alpha:
                break
        return pior_nota

    # -----------------------------------------------------------------
    # Interface com o backend do grupo
    # -----------------------------------------------------------------
    def _preparar_estado_da_busca(self, tabuleiro_real: np.ndarray, peca_da_ia_no_backend: int) -> None:
        numero_de_pecas = int(np.count_nonzero(tabuleiro_real))

        # QUEM COMEÇOU e QUAL É A COR DA MINHA PEÇA são coisas independentes:
        #   - quem começou: pela paridade. Número par de peças no tabuleiro = é a vez de quem começou.
        #   - cor da minha peça: vem do backend (game.turn). O humano é sempre Piece.P1 (+1) e a CPU
        #     sempre Piece.P2 (-1), mesmo nas partidas em que a CPU começa (o frontend alterna quem começa).
        self._ia_e_o_primeiro_jogador = (numero_de_pecas % 2 == 0)

        # Visão da IA: a peça da IA vira +1 e a do adversário vira -1, qualquer que seja a cor real dela.
        self._tabuleiro_na_visao_da_ia = (tabuleiro_real * np.int8(peca_da_ia_no_backend)).astype(np.int8)
        self._numero_de_pecas_no_tabuleiro = numero_de_pecas
        self._proxima_linha_livre_de_cada_coluna = [
            NUMERO_DE_LINHAS - 1 - int(np.count_nonzero(self._tabuleiro_na_visao_da_ia[:, coluna]))
            for coluna in range(NUMERO_DE_COLUNAS)
        ]

    def choose_move(self, game: Game) -> int:
        tabuleiro_real = np.array(game.board.to_flat(), dtype=np.int8).reshape(NUMERO_DE_LINHAS, NUMERO_DE_COLUNAS)
        peca_da_ia_no_backend = int(game.turn)  # +1 (Piece.P1) ou -1 (Piece.P2): a cor de quem vai jogar agora
        self._preparar_estado_da_busca(tabuleiro_real, peca_da_ia_no_backend)

        self._apagar_da_cache_os_tabuleiros_impossiveis(self._numero_de_pecas_no_tabuleiro)
        self._esvaziar_cache_se_passou_do_limite()

        colunas_jogaveis = self._colunas_jogaveis()
        if len(colunas_jogaveis) == 1:
            return colunas_jogaveis[0]
        if self._numero_de_pecas_no_tabuleiro == 0:
            return 3  # abertura: a coluna central é a jogada vencedora conhecida para quem começa

        melhor_coluna = colunas_jogaveis[0]
        melhor_nota = -np.inf
        alpha, beta = -np.inf, np.inf

        for coluna in colunas_jogaveis:
            linha = self._colocar_peca(coluna, 1)
            if self._ultima_jogada_completou_quatro_em_linha(linha, coluna, 1):
                self._remover_peca(coluna)
                return coluna  # vitória imediata: nem precisa olhar o resto
            nota = self._minimax(self.profundidade_da_arvore - 1, alpha, beta, False)
            self._remover_peca(coluna)

            if nota > melhor_nota:
                melhor_nota, melhor_coluna = nota, coluna
            alpha = max(alpha, melhor_nota)

        return melhor_coluna
