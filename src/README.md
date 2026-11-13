# Conecta 4 (Python + pygame)

Jogo com menu, modo Player VS Player, modo Player VS CPU e créditos. A lógica fica
separada da interface para facilitar a entrada de uma IA/CPU.

## Como rodar

```bash
pip install -r requirements.txt
python main.py
```

**Menu:** mouse ou setas + Enter · `Esc` sai.
**Partida:** mouse (mover e clicar) ou setas + espaço/enter · `R` novo jogo · `Esc` volta ao menu.

## Estrutura

| Arquivo        | Papel                                                                        |
|----------------|------------------------------------------------------------------------------|
| `backend.py`   | Regras e estado. **Não usa pygame.** `Board`, `Game`, `Player`.              |
| `ai.py`        | `CPUPlayer`: **molde da IA** (hoje joga aleatório, é um placeholder).         |
| `frontend.py`  | Tela da partida (tabuleiro, animações, placar) e o `App` que navega entre telas. |
| `menu.py`      | Menu principal e créditos (nomes e fotos dos desenvolvedores).               |
| `ui.py`        | Peças visuais compartilhadas: cores, fontes, botão, contrato `Screen`.       |
| `main.py`      | Define os dois modos (quais jogadores cada um usa) e inicia o jogo.          |

O frontend nunca decide regras: ele chama `game.play(col)` e desenha o que está em
`game.board`, `game.status`, `game.winner` e `game.winning_line`.

## Modos de jogo

Estão em `main.py`, cada um é uma função que devolve um `Game`:

```python
def new_pvp():   # dois humanos
    return Game(players={Piece.P1: HumanPlayer("Vermelho"), Piece.P2: HumanPlayer("Amarelo")})

def new_pvc():   # humano contra a CPU
    return Game(players={Piece.P1: HumanPlayer("Você"), Piece.P2: CPUPlayer("CPU")})
```

Os dois modos precisam usar a mesma grade (o `App` valida isso).

## Tamanho da grade

Em `backend.py`:

```python
ROWS = 7      # linhas
COLS = 6      # colunas
CONNECT = 4   # peças em sequência para vencer
```

O Conecta 4 tradicional é 6 linhas x 7 colunas; basta inverter os dois números se
preferir. A janela se ajusta sozinha ao tamanho da grade e ao tamanho da sua tela.

## Como preencher a IA

Abra `ai.py` e implemente `choose_move` da classe `CPUPlayer`. Ela recebe a partida
e devolve o número da coluna (0 .. cols-1):

```python
class CPUPlayer(Player):
    def choose_move(self, game) -> int:
        estado = game.snapshot()        # dict simples: tabuleiro, vez, jogadas válidas...
        # ou: sim = game.board.copy(); sim.drop(col, peca); sim.undo(col)
        return coluna_escolhida
```

O frontend já pergunta a jogada a qualquer jogador que não seja humano, com uma
pequena pausa (`CPU_THINK_DELAY` em `frontend.py`) para dar a impressão de "pensar".

Referência rápida do backend para IA:

- `game.turn` é a peça de quem joga agora; `game.turn.opponent` é a do adversário.
- `game.snapshot()` devolve o estado em tipos simples (listas/ints/strings), fácil de enviar por rede.
- `board.copy()`, `board.drop(col, peca)` e `board.undo(col)` permitem simular jogadas (minimax, MCTS).
- `board.winning_line(row, col)` e `board.valid_moves()` ajudam a avaliar posições.
- `Piece.P1` / `Piece.P2` valem 1 e 2 (0 é casa vazia); a linha 0 é o topo do tabuleiro.

Se a IA demorar (vários segundos), rode `choose_move` em uma thread/processo para a
janela não congelar. Hoje a chamada é síncrona, o que basta para uma IA rápida.

## Créditos: nomes e fotos

Os nomes (inventados por enquanto) estão em `DEVELOPERS`, no topo de `menu.py`.
Cada desenvolvedor aparece com um avatar genérico. Para usar fotos de verdade, crie a
pasta `assets/devs/` ao lado de `menu.py` e coloque lá `dev1.png`, `dev2.png`,
`dev3.png` e `dev4.png` (ou ajuste os nomes em `DEVELOPERS`). Qualquer proporção
funciona: a foto é recortada em círculo automaticamente.
