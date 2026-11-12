# Conecta 4 (Python + pygame)

Jogo para dois jogadores no mesmo computador, com a lógica separada da interface
para facilitar a entrada de uma IA/CPU depois.

## Como rodar

```bash
pip install -r requirements.txt
python main.py
```

**Controles:** mouse (mover e clicar) ou setas + espaço/enter · `R` novo jogo · `Esc` sair.

## Estrutura

| Arquivo        | Papel                                                                 |
|----------------|-----------------------------------------------------------------------|
| `backend.py`   | Regras e estado. **Não usa pygame.** `Board`, `Game`, `Player`.       |
| `frontend.py`  | Tudo que é visual: janela, animações, placar, entrada de mouse/teclado. |
| `main.py`      | Junta as duas partes e inicia o jogo.                                 |

O frontend nunca decide regras: ele chama `game.play(col)` e desenha o que está em
`game.board`, `game.status`, `game.winner` e `game.winning_line`.

## Tamanho da grade

Em `backend.py`:

```python
ROWS = 7      # linhas
COLS = 6      # colunas
CONNECT = 4   # peças em sequência para vencer
```

O Conecta 4 tradicional é 6 linhas x 7 colunas; basta inverter os dois números se
preferir. A janela se ajusta sozinha ao tamanho da grade e ao tamanho da sua tela.

## Como conectar a IA depois

1. Crie uma classe que herda de `backend.Player` e implemente `choose_move`:

   ```python
   from backend import Player

   class MinhaIA(Player):
       def choose_move(self, game) -> int:
           estado = game.snapshot()        # dict simples: tabuleiro, vez, jogadas válidas...
           # ou: sim = game.board.copy(); sim.drop(col, peca); sim.undo(col)
           return coluna_escolhida          # 0 .. cols-1
   ```

2. Passe-a ao criar a partida em `main.py`:

   ```python
   game = Game(players={Piece.P1: HumanPlayer("Você"), Piece.P2: MinhaIA("CPU")})
   ```

O frontend já pergunta a jogada a qualquer jogador que não seja humano, com uma
pequena pausa para dar a impressão de "pensar".

Referência rápida do backend para IA:

- `game.snapshot()` devolve o estado em tipos simples (listas/ints/strings), fácil de enviar por rede.
- `board.copy()`, `board.drop(col, peca)` e `board.undo(col)` permitem simular jogadas (minimax, MCTS).
- `board.winning_line(row, col)` e `board.valid_moves()` ajudam a avaliar posições.
- `Piece.P1` / `Piece.P2` valem 1 e 2 (0 é casa vazia); a linha 0 é o topo do tabuleiro.

Se a IA demorar (vários segundos), rode `choose_move` em uma thread/processo para a
janela não congelar. Hoje a chamada é síncrona, o que basta para uma IA rápida.
