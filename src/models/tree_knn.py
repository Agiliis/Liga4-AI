#=============================================================================
#JOGADOR HÍBRIDO: ÁRVORE (MINIMAX + ALPHA-BETA) + KNN
#=============================================================================
#Combina uma busca em árvore com um KNN: a árvore olha 4 jogadas à frente e garante que a IA não deixe passar
#uma vitória ou um bloqueio óbvio, enquanto o KNN entra apenas nas folhas em que a árvore não achou vencedor, dando
#uma nota entre -1 e +1 que indica quem está melhor naquela posição.
#Para rodar, o tree_knn_dataset.npz (gerado pelo limpar_dados_treino.py) precisa estar na pasta db/.
#Dentro da árvore o tabuleiro fica sempre na visão da IA (+1 = peça da IA, -1 = adversário), mas para consultar
#o lookup, a cache e o KNN o tabuleiro é convertido para a visão dos dados de treino (+1 = quem começou a partida,
#-1 = quem jogou em segundo). A linha 0 é o topo do tabuleiro e a linha 5 é a base.

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.neighbors import KNeighborsRegressor

from src.backend import Game, Player

#Parâmetros que mais alteram o comportamento da IA (K do KNN, profundidade da árvore e
#ordem das colunas), facilitando ajustes sem buscar no meio do código. A ordem vai do centro para as bordas
#porque a coluna central participa de mais combinações de vitória, então testá-la primeiro faz a poda
#Alpha-Beta cortar mais ramos. A pontuação de vitória (100) é bem maior que qualquer nota do KNN, assim uma
#estimativa estatística nunca se sobrepõe a uma vitória ou derrota real. 
#O limite da cache funciona como trava de memória: cada entrada ocupa cerca de 150 bytes, então 300 mil 
#entradas ficam em torno de 45 MB.
CAMINHO_DADOS_LIMPOS = Path("db") / "tree_knn_dataset.npz"

NUMERO_DE_LINHAS = 6
NUMERO_DE_COLUNAS = 7
NUMERO_DE_CASAS = NUMERO_DE_LINHAS * NUMERO_DE_COLUNAS

QUANTIDADE_DE_VIZINHOS_DO_KNN = 9
PROFUNDIDADE_DA_ARVORE = 4
ORDEM_DAS_COLUNAS_DO_CENTRO_PARA_AS_BORDAS = (3, 2, 4, 1, 5, 0, 6)

PONTUACAO_BASE_DE_VITORIA = 100.0
CODIGO_FONTE_UCI = 0
NUMERO_DE_PECAS_DOS_TABULEIROS_DO_UCI = 8

LIMITE_MAXIMO_DE_ENTRADAS_NA_CACHE = 300_000

DIRECOES_PARA_CONFERIR_QUATRO_EM_LINHA = ((0, 1), (1, 0), (1, 1), (1, -1))


#=============================================================================
#TREINO DO KNN
#=============================================================================
def carregar_dados_limpos() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not CAMINHO_DADOS_LIMPOS.exists():
        raise FileNotFoundError(
            f"'{CAMINHO_DADOS_LIMPOS.name}' não encontrado. Rode primeiro: python limpar_dados_treino.py"
        )
    arquivo = np.load(CAMINHO_DADOS_LIMPOS)
    if "ja_inclui_espelhos" not in arquivo.files:#flag pra não usar arquivo antigo
        raise RuntimeError(
            f"'{CAMINHO_DADOS_LIMPOS.name}' é de uma versão antiga (sem os espelhos prontos). "
            "Rode de novo: python limpar_dados_treino.py"
        )
    return arquivo["tabuleiros"], arquivo["notas"], arquivo["fontes"]

def treinar_knn(tabuleiros: np.ndarray, notas: np.ndarray) -> KNeighborsRegressor:
    modelo_knn = KNeighborsRegressor(
        n_neighbors=QUANTIDADE_DE_VIZINHOS_DO_KNN,
        metric="manhattan",
        weights="distance",
        algorithm="brute",
        n_jobs=1,
    )
    modelo_knn.fit(tabuleiros.astype(np.float32), notas)
    return modelo_knn


#Transforma o tabuleiro em uma chave de dicionário (bytes) que é a mesma para ele e para o seu espelho
#horizontal, mantendo sempre a menor das duas versões. Assim, o lookup e a cache tratam os dois como o
#mesmo tabuleiro, o que dobra a chance de reaproveitar uma nota já calculada.
def gerar_chave_canonica(tabuleiro_na_visao_do_primeiro: np.ndarray) -> bytes:
    bytes_do_tabuleiro = tabuleiro_na_visao_do_primeiro.tobytes()
    bytes_do_espelho = tabuleiro_na_visao_do_primeiro[:, ::-1].tobytes()
    return min(bytes_do_tabuleiro, bytes_do_espelho)


#Monta o dicionário (lookup) apenas com as linhas vindas do UCI, que são tabuleiros de 8 peças com o resultado
#teórico exato, sem ruído. Quando uma folha da árvore tem 8 peças e consta nesse dicionário, a nota é devolvida
#na hora, sem aproximação e sem precisar chamar o KNN.
def montar_dicionario_de_valores_exatos_do_uci(
    tabuleiros: np.ndarray, notas: np.ndarray, fontes: np.ndarray
) -> dict[bytes, float]:
    dicionario: dict[bytes, float] = {}
    for tabuleiro, nota in zip(tabuleiros[fontes == CODIGO_FONTE_UCI], notas[fontes == CODIGO_FONTE_UCI]):
        grade = tabuleiro.reshape(NUMERO_DE_LINHAS, NUMERO_DE_COLUNAS)
        dicionario[gerar_chave_canonica(grade)] = float(nota)
    return dicionario


#O __init__ carrega os dados, treina o KNN uma única vez, monta o lookup do UCI e cria a cache vazia (uma "gaveta" 
#por número de peças), além dos contadores usados para acompanhar quantas notas vieram do KNN, da cache ou do
#lookup. Também armazena o estado da busca (tabuleiro na visão da IA, próxima linha livre de cada coluna etc.),
#que é refeito a cada jogada.
class TreeKNNPlayer(Player):
    def __init__(self, name: str = "Tree+KNN", profundidade: int = PROFUNDIDADE_DA_ARVORE):
        super().__init__(name)
        tabuleiros, notas, fontes = carregar_dados_limpos()
        self.modelo_knn = treinar_knn(tabuleiros, notas)
        self.valores_exatos_do_uci = montar_dicionario_de_valores_exatos_do_uci(tabuleiros, notas, fontes)
        self.profundidade_da_arvore = profundidade

        self.cache_de_avaliacoes_por_numero_de_pecas: list[dict[bytes, float]] = [
            {} for _ in range(NUMERO_DE_CASAS + 1)
        ]
        self.total_de_entradas_na_cache = 0

        self.quantidade_de_consultas_ao_knn = 0
        self.quantidade_de_acertos_na_cache = 0
        self.quantidade_de_valores_exatos_usados = 0

        self._tabuleiro_na_visao_da_ia = np.zeros((NUMERO_DE_LINHAS, NUMERO_DE_COLUNAS), dtype=np.int8)
        self._proxima_linha_livre_de_cada_coluna = [NUMERO_DE_LINHAS - 1] * NUMERO_DE_COLUNAS
        self._numero_de_pecas_no_tabuleiro = 0
        self._ia_e_o_primeiro_jogador = True

    #=========================================================================
    #CACHE
    #=========================================================================
    #Guarda as notas que o KNN já calculou, organizadas por número de peças. Como as peças nunca saem
    #do tabuleiro, quando o jogo chega em N peças os tabuleiros com menos de N nunca mais voltam. Assim,
    #essas "gavetas" inteiras são apagadas de uma vez, operação barata que libera memória sem conferir item a item.
    def _apagar_da_cache_os_tabuleiros_impossiveis(self, numero_de_pecas_atual: int) -> None:
        for numero_de_pecas in range(numero_de_pecas_atual):
            cache_dessa_quantidade = self.cache_de_avaliacoes_por_numero_de_pecas[numero_de_pecas]
            if cache_dessa_quantidade:
                self.total_de_entradas_na_cache -= len(cache_dessa_quantidade)
                cache_dessa_quantidade.clear()

    #Trava de segurança(completamente opcional, mas me deu na telha): 
    #se a cache passar do limite ela é totalmente esvaziada, evitando crescimento excessivo
    #que possa pesar na memória do computador durante a partida.
    def _esvaziar_cache_se_passou_do_limite(self) -> None:
        if self.total_de_entradas_na_cache > LIMITE_MAXIMO_DE_ENTRADAS_NA_CACHE:
            for cache_dessa_quantidade in self.cache_de_avaliacoes_por_numero_de_pecas:
                cache_dessa_quantidade.clear()
            self.total_de_entradas_na_cache = 0

    #=========================================================================
    #AVALIAÇÃO DE FOLHAS
    #=========================================================================
    #Calcula a nota estratégica (entre -1 e +1) de uma folha da árvore, do ponto de vista da IA, buscando da forma
    #mais barata para a mais cara: primeiro o lookup do UCI (válido apenas com 8 peças), depois a cache e, só se
    #for um tabuleiro novo, o KNN (que já salva o resultado na cache). Como os dados de treino estão na visão de
    #quem começou, o tabuleiro é convertido para essa perspectiva antes da consulta e o sinal da nota é invertido
    #no final quando a IA não é o primeiro jogador.
    def _avaliar_folha(self) -> float:
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

    #=========================================================================
    #MECÂNICA DO TABULEIRO
    #=========================================================================
    #Em vez de copiar o tabuleiro a cada nó da árvore, a peça é colocada e desfeita no mesmo array
    #(_colocar_peca e _remover_peca), mantendo o registro de qual é a próxima linha livre de cada coluna. Isso
    #acelera a execução e evita criar milhares de cópias durante a busca.

    #Devolve só as colunas que ainda aceitam peça, já na ordem do centro para as bordas.
    def _colunas_jogaveis(self) -> list[int]:
        return [
            coluna for coluna in ORDEM_DAS_COLUNAS_DO_CENTRO_PARA_AS_BORDAS
            if self._proxima_linha_livre_de_cada_coluna[coluna] >= 0
        ]

    #Solta a peça na coluna e devolve a linha onde ela caiu, atualizando a próxima linha livre e a contagem de peças.
    def _colocar_peca(self, coluna: int, peca: int) -> int:
        linha = self._proxima_linha_livre_de_cada_coluna[coluna]
        self._tabuleiro_na_visao_da_ia[linha, coluna] = peca
        self._proxima_linha_livre_de_cada_coluna[coluna] -= 1
        self._numero_de_pecas_no_tabuleiro += 1
        return linha

    #Faz o caminho inverso do _colocar_peca, tirando a peça do topo da coluna para a árvore voltar um passo.
    def _remover_peca(self, coluna: int) -> None:
        self._proxima_linha_livre_de_cada_coluna[coluna] += 1
        linha = self._proxima_linha_livre_de_cada_coluna[coluna]
        self._tabuleiro_na_visao_da_ia[linha, coluna] = 0
        self._numero_de_pecas_no_tabuleiro -= 1

    #Confere se a peça que acabou de cair fechou 4 em linha, olhando apenas ao redor dela nas 4 direções (horizontal,
    #vertical e as duas diagonais) e contando as peças iguais para os dois lados. Isso funciona porque só a
    #última peça colocada pode ter criado uma sequência nova, dispensando a varredura do tabuleiro inteiro.
    def _ultima_jogada_completou_quatro_em_linha(self, linha: int, coluna: int, peca: int) -> bool:
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

    #=========================================================================
    #ÁRVORE MINIMAX COM PODA ALPHA-BETA
    #=========================================================================
    #O Minimax alterna entre a vez da IA (que maximiza a nota) e a do adversário (que minimiza,
    #assumindo que ele joga o melhor possível), enquanto a poda Alpha-Beta interrompe um ramo assim que fica provado
    #que ele não muda a decisão final. Se uma jogada fecha 4 em linha, a nota é +-100 somada à profundidade
    #restante, o que faz a IA preferir vencer o mais rápido possível e adiar ao máximo uma derrota. Tabuleiro
    #cheio vale empate (0) e, ao chegar na profundidade 0 sem vencedor, a nota vem do avaliador de folhas.
    def _minimax(self, profundidade_restante: int, alpha: float, beta: float, e_a_vez_da_ia: bool) -> float:
        colunas_jogaveis = self._colunas_jogaveis()
        if not colunas_jogaveis:
            return 0.0
        if profundidade_restante == 0:
            return self._avaliar_folha()

        peca_de_quem_joga = 1 if e_a_vez_da_ia else -1
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

    #=========================================================================
    #INTERFACE COM O BACKEND
    #=========================================================================
    def _preparar_estado_da_busca(self, tabuleiro_real: np.ndarray, peca_da_ia_no_backend: int) -> None:
        numero_de_pecas = int(np.count_nonzero(tabuleiro_real))

        self._ia_e_o_primeiro_jogador = (numero_de_pecas % 2 == 0)

        self._tabuleiro_na_visao_da_ia = (tabuleiro_real * np.int8(peca_da_ia_no_backend)).astype(np.int8)
        self._numero_de_pecas_no_tabuleiro = numero_de_pecas
        self._proxima_linha_livre_de_cada_coluna = [
            NUMERO_DE_LINHAS - 1 - int(np.count_nonzero(self._tabuleiro_na_visao_da_ia[:, coluna]))
            for coluna in range(NUMERO_DE_COLUNAS)
        ]

    #Único método chamado pelo frontend: recebe a partida e devolve a coluna escolhida. Antes de buscar, remove
    #da cache os estados que se tornaram impossíveis. Estatisticamente, a coluna central (3) é a melhor jogada
    #para o primeiro jogador. Nos demais casos, simula cada coluna: se alguma já vence na hora,
    #retorna ela sem avaliar o resto; senão, chama o Minimax para as respostas do adversário e escolhe a coluna
    #de maior nota (em caso de empate prevalece a mais central, devido à ordem das colunas).
    def choose_move(self, game: Game) -> int:
        tabuleiro_real = np.array(game.board.to_flat(), dtype=np.int8).reshape(NUMERO_DE_LINHAS, NUMERO_DE_COLUNAS)
        peca_da_ia_no_backend = int(game.turn)
        self._preparar_estado_da_busca(tabuleiro_real, peca_da_ia_no_backend)

        self._apagar_da_cache_os_tabuleiros_impossiveis(self._numero_de_pecas_no_tabuleiro)
        self._esvaziar_cache_se_passou_do_limite()

        colunas_jogaveis = self._colunas_jogaveis()
        if len(colunas_jogaveis) == 1:
            return colunas_jogaveis[0]
        if self._numero_de_pecas_no_tabuleiro == 0:
            return 3

        melhor_coluna = colunas_jogaveis[0]
        melhor_nota = -np.inf
        alpha, beta = -np.inf, np.inf

        for coluna in colunas_jogaveis:
            linha = self._colocar_peca(coluna, 1)
            if self._ultima_jogada_completou_quatro_em_linha(linha, coluna, 1):
                self._remover_peca(coluna)
                return coluna
            nota = self._minimax(self.profundidade_da_arvore - 1, alpha, beta, False)
            self._remover_peca(coluna)

            if nota > melhor_nota:
                melhor_nota, melhor_coluna = nota, coluna
            alpha = max(alpha, melhor_nota)

        return melhor_coluna