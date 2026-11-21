# Simulação 6⁴ e 8⁴ (com 4⁴ de referência) — como ler estes arquivos

## O que é
Reimplementação **independente** da camada base (banho térmico, fixação de Landau, Hessiana de Faddeev–Popov,
fluxo de Wilson) e da camada ultravioleta, com a mesma interface do `cl6.py`. O zip original não continha
`colorlattice4`/`cl5`; este código **não é** o ColorLattice 4/5 e deve ser descrito assim no artigo.
Protocolo: 300 varreduras de termalização, uma configuração a cada 40 varreduras, 2 cadeias × 3 configurações
(6⁴ e 4⁴) ou 1 cadeia × 2 configurações (8⁴); 8 partidas de calibre (4 em 8⁴); τ = 0,5, T = 0,2, 3 quadros de fluxo.

| rede | β | configurações |
|---|---|---|
| 4⁴ (denso, com referência exata) | 3, 5, 6, 8, 12 | 6 cada |
| 6⁴ (sem matriz densa) | 3, 5, 6, 8, 12 | 6 cada |
| 8⁴ (sem matriz densa) | 3, 6, 12 | 2 cada |

## Pastas
- `metricas/` — CSV: `metricas_por_configuracao` (plaquete, F, λ₁…λ₆ de Faddeev–Popov, λ₁ do Laplaciano, Tr X/dim p, f₁…f₆, referência exata em 4⁴),
  `copias_de_gribov`, `funcao_de_contagem_N_sobre_V` (Λ de 0 a 8), `algebra_de_regimes_L4`, `serie_monte_carlo_plaquete` (por varredura, inclui termalização).
  `tabela_resultados.md` traz as médias por (rede, β).
- `campos/` — `.npy`: `L{L}_beta{β}_rho_modes.npy` forma (configurações, 6, L,L,L,L): densidade Σₐ v_k(x,a)² dos 6 modos mais baixos;
  `..._plaq_field.npy` forma (configurações, L,L,L,L): plaquete local; `..._chain_index.npy`: (cadeia, índice) de cada configuração.
  O eixo das configurações segue a ordem da cadeia de Monte Carlo (espaçadas de 40 varreduras: são amostras, não uma evolução dinâmica contínua).
  `converter_npy_para_h5.py` cria um `campos.h5` na sua máquina (o `h5py` não estava disponível aqui).
- `figuras/` — 12 PNG (fig01–fig12), mesmo estilo do artigo.
- `codigo/`, `verificacao/` — código, scripts, e resultados da bateria de testes.

## Limites que precisam constar no artigo
1. Verificação: 18 testes; 17 passaram de imediato. Dois critérios foram **revisados depois de ver o resultado**: simetria da Hessiana (zero exato → 1e-14; a diferença real era 2e-16)
   e contagem na rede fria (primeiro avaliada exatamente nos degraus, onde um filtro suave perde meio degrau; depois nos pontos médios entre níveis, erro ≤ 0,75% da dimensão). Esse segundo teste está em `verify7_staircase.py`.
2. Estimador sem matriz de Tr X/dim p (filtro suave + Hutchinson): contra a diagonalização exata em 4⁴ (30 configurações por Λ), o desvio médio é +2,7% (Λ=2, intervalo −3,4% a +9,8%) e +1,8% (Λ=4, −4,0% a +4,9%).
   f₁ concorda com o exato a 0,03 (Λ=2) e 0,006 (Λ=4) no máximo. Em 6⁴ e 8⁴ não há referência exata.
3. Estatística pequena (6 configurações por β, 2 em 8⁴, uma semente). Mínimos de Gribov contados com 8 partidas (4 em 8⁴) são um limite inferior.
4. ‖X‖ em 6⁴ e 8⁴ aparece só como cota inferior (Rayleigh com o modo mais baixo); em 4⁴ a razão exata λ₁‖X‖/f₁ ficou entre 1,0000 e 1,039.
5. A álgebra de regimes (Pc, pΛ) foi calculada só em 4⁴ (exige diagonalização densa).
