# TDLeaf(λ) Training — Run 1: `nn-fresh-260309`

## Overview

This documents the first complete TDLeaf(λ) training run on Leaf, carried out on 9 March 2026.  The goal was to train a randomly-initialised NNUE network from scratch via self-play and measure how much strength it gains as a function of game count.

---

## Network Initialisation

The starting network, **`nn-fresh-260309.nnue`**, was created by Leaf's `--init-nnue` facility.  It is not derived from any pre-trained chess data; all weights are drawn from Gaussian distributions whose parameters (mean, σ) were measured empirically from the Stockfish 15.1 release network (`nn-ad9b42354671.nnue`).  See `docs/NNUE.md` for the per-layer distributions.

PSQT weights are initialised to piece-value priors rather than random values.  The prior
assigns each piece type a uniform signed value chosen so that one extra own piece of that type
scores the standard centipawn equivalent: pawn = 100 cp, knight = 300 cp, bishop = 300 cp,
rook = 500 cp, queen = 900 cp.  Kings are set to zero.  The network therefore begins with a
crude but sensible material prior for positional scoring, while the FC layers start from
random noise.

*(Note: this Run 1 init differed from the current `--init-nnue` in two ways:*
*— **FC/FT weights:** means were copied from Stockfish 15.1 measurements (FT: −0.71, FC0: +0.24,
FC1: −1.10, FC2: +1.10) rather than set to zero.  The current init uses zero means (He/Kaiming
principle) to eliminate directional neuron bias from game 1, and reduces FC0 std from 8.43
(SF15.1 measured) to 3.0 (He-adjusted for 1024-input fan-in), cutting initial FC0 saturation
from ~24% to ~3%.*
*— **PSQT:** used uniform material values (100/300/300/500/900 cp, game-stage-independent).
The current init maps each of the 8 PSQT buckets to an interpolated classical game stage via
Leaf's own piece-square tables — a richer starting point that gives the network opening/endgame
awareness from game 1.  See `docs/NNUE.md` for full details.)*

The network is a statistically plausible but chess-naïve starting point — it has the right weight magnitudes but no learned positional chess knowledge.

---

## Training Procedure

| Parameter | Value |
|-----------|-------|
| Algorithm | TDLeaf(λ), online, all layers updated |
| Training format | Self-play: `Leaf_vtrain` (learning) vs `Leaf_vtrain_ro` (read-only); superseded by symmetric self-play (`_a` vs `_b`) as of 2026-03-12 |
| Positions | Fischer Random (Chess960), random starting position each game |
| Opening book | None |
| Tablebases | Disabled |
| Time control | 3+0.05 s/move |
| Concurrency | 5 simultaneous games |
| Total training games | 8,000 |

The read-only opponent (`_ro`) reloads the latest `.tdleaf.bin` weights at the start of each 500-game training iteration.  This means the learning engine's weights are periodically adopted as the new baseline opponent, providing a gradually strengthening training signal without the instability of per-game opponent updates.

Network snapshots were saved at **500, 1000, 2000, 4000, and 8000 games**.

---

## Testing Procedure

After training, a test tournament was run among all network snapshots plus two reference engines, producing `pgn/fresh-260309-testing.pgn`.

| Parameter | Value |
|-----------|-------|
| Games per pair | 500 (250 each colour) |
| Positions | Fischer Random, no opening book |
| Tablebases | Disabled |
| Time control | 10+0.1 s/move |
| Total games | 6,500 |

**Reference engines:**

- **`EXchess_classic`** — the classical hand-crafted Leaf eval; the strong-reference ceiling for this run
- **`EXchess_classic_material`** — a material-only classical build (`MATERIAL_ONLY=1`); the weak-reference floor

---

## Results

### Bayesian Elo Ratings

Computed with `scripts/bayeselo_ratings.py` (BayesElo, maximum-likelihood).  Ratings are relative within this pool.

| Rank | Engine                        |  Elo  |  ±  | Games | Score |  Oppo | Draws |
|-----:|-------------------------------|------:|----:|------:|------:|------:|------:|
|    1 | EXchess_classic               | +1055 | 152 | 1,000 |  100% |   +76 |    0% |
|    2 | Leaf_vnn-fresh-260309-8000g  |  +135 |  14 | 3,500 |   71% |   −19 |   11% |
|    3 | EXchess_classic_material      |   +40 |  19 | 1,000 |   45% |   +76 |   22% |
|    4 | Leaf_vnn-fresh-260309-4000g  |   +18 |  13 | 3,500 |   60% |    −3 |   14% |
|    5 | Leaf_vnn-fresh-260309-2000g  |  −140 |  21 | 1,000 |   22% |   +76 |   21% |
|    6 | Leaf_vnn-fresh-260309-1000g  |  −271 |  26 | 1,000 |   12% |   +76 |   11% |
|    7 | Leaf_vnn-fresh-260309-500g   |  −353 |  32 | 1,000 |    8% |   +76 |    7% |
|    8 | Leaf_vnn-fresh-260309 (0g)   |  −484 |  44 | 1,000 |    4% |   +76 |    3% |

### Progress by Game Count

| Snapshot | Elo | Gain vs previous | Elo/game |
|----------|----:|-----------------:|---------:|
| 0g (fresh init) | −484 | — | — |
| 500g | −353 | +131 | 0.26 |
| 1000g | −271 | +82 | 0.16 |
| 2000g | −139 | +132 | 0.13 |
| 4000g | +18 | +157 | 0.08 |
| 8000g | +135 | +117 | 0.03 |
| **Total gain** | | **+619** | |

The rate of improvement is declining with game count, falling from ~0.26 Elo/game early to ~0.03 Elo/game between 4000 and 8000 games.  Improvement is still positive at 8000 games, however.

### Key Pairwise Results (8000g network)

| Opponent | W | D | L | Score |
|----------|--:|--:|--:|------:|
| Leaf_vnn-fresh-260309 (0g) | 479 | 9 | 12 | 96.7% |
| Leaf_vnn-fresh-260309-500g | 446 | 34 | 20 | 92.6% |
| Leaf_vnn-fresh-260309-1000g | 447 | 33 | 20 | 92.7% |
| Leaf_vnn-fresh-260309-2000g | 380 | 77 | 43 | 83.7% |
| Leaf_vnn-fresh-260309-4000g | 295 | 107 | 98 | 69.7% |
| EXchess_classic_material | 246 | 113 | 141 | 60.5% |
| EXchess_classic | 1 | 1 | 498 | 0.3% |

The 8000g network is above the material-only classical eval (60.5%) and dominates all earlier snapshots.  It remains far below the classical eval — losing 498 of 500 games against `EXchess_classic`.


---

## Observations

1. **Monotonic improvement, but decelerating.**  Every snapshot is stronger than the last, but the Elo gain per game drops sharply: from ~0.26 at the start to ~0.03 between 4000 and 8000 games.  The network is still improving at 8000 games but may be approaching a ceiling under the current training conditions (3+0.05 TC, Fischer Random, no hyperparameter tuning).

2. **Crossing the material-only threshold.**  The 4000g network is statistically equal to the material-only eval (49.5%); the 8000g network is clearly above it (60.5%).  The crossing of this threshold occurs between 4000 and 8000 games, meaning the network begins learning meaningful positional patterns beyond raw material sometime in that window.

3. **Large gap to classical eval.**  The 8000g net is ~919 Elo below `EXchess_classic` in this pool.  This is expected: the classical eval encodes decades of chess knowledge; a self-trained network at this game count is not expected to match it.  Closing this gap is the long-term goal.

4. **Fischer Random as training distribution.**  Using random starting positions removes opening-book effects and ensures the network is exposed to a wide variety of piece configurations from move 1.  It is not yet known whether a network trained on Fischer Random positions will transfer well to standard chess.

5. **Draw rate increases with strength.**  The fresh network draws only 3% of games (mostly losing); the 8000g network draws 11% overall.  This is qualitatively consistent with a stronger evaluation keeping games competitive longer.

---

## Next Steps

- Continue training beyond 8000 games to assess whether improvement continues or plateaus.
- Run a test match against standard-chess opponents (not Fischer Random) to check transferability.
- ~~Investigate learning-rate tuning for PSQT~~ — Addressed (2026-03-15): Adam optimizer with
  separate `TDLEAF_ADAM_PSQT_LR0=20.0` gives ~5% PSQT baseline change per 5,000 games vs ~0.3%
  previously.  See `docs/TDLEAF.md`.
- Consider a training run starting from the SF15.1 network (`nn-ad9b42354671.nnue`) rather than
  a fresh initialisation, as a comparison point.
- ~~Run a second full training run with the improved init (classical PSQ PSQT buckets) and Adam
  optimizer to establish a new Elo baseline curve.~~ — In progress (2026-03-16): preliminary
  result at 5,000 games shows ~+300 Elo vs. fresh init; formal test tournament pending.
