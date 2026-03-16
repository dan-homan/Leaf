# Leaf TODO

Planned investigations, improvements, and open questions.

---

## TDLeaf(λ) Training

### Adam hyperparameter tuning

The Adam + per-weight LR decay optimizer (`TDLEAF_ADAM_LR0=0.5`, `TDLEAF_ADAM_PSQT_LR0=20`,
`TDLEAF_ADAM_C=500`) uses initial guesses that have not been systematically tuned.  A grid
search varying each independently across 500–1000-game runs would establish better defaults.

Key questions:

- **FC LR0 (0.5):** After 5,000 games the FC0/FC1 float shadows spread to std≈30/50, filling
  the int8 range.  Float-shadow clamping prevents zombie weights but further training will be
  limited at the ±127 boundary.  Monitor whether the network continues to improve or plateaus
  as the distribution saturates.
- **PSQT LR0 (20):** Set to give ~5% baseline change per feature over 5,000 games (vs ~0.3%
  with the old unified LR0=0.5).  Tune empirically; 10–50 is a reasonable search range.
- **C (500):** LR half-life in per-weight updates.  `lr(cnt) = LR0 / (1 + cnt/C)`.  Larger C
  extends the fast-learning phase; smaller C converges more aggressively.  500–2000 is the
  practical range for ~5,000-game sessions.
- **LR_SCALE multipliers:** `NNUE_PSQT_LR_SCALE`, `NNUE_FC_BIAS_LR_SCALE`, and
  `NNUE_FT_BIAS_LR_SCALE` are no longer the primary controls for per-step size when Adam is
  active — Adam normalises gradient magnitude, making these pre-scalings irrelevant to actual
  step size.  They can be removed in a future cleanup (or left as plain-GD fallback controls
  for `TDLEAF_ADAM_LR0 = 0.0`).

### TDLEAF_MAX_UPDATE_FRAC

`TDLEAF_MAX_UPDATE_FRAC` is currently 1.0 (disabled).  Under Adam the effective per-update
magnitude is bounded by `TDLEAF_ADAM_LR0`, so the fraction-of-weight clamping is less urgent
than under plain GD.  Still, a tighter cap (e.g. 0.20) could prevent any single pathological
game from dominating the accumulated gradient during concurrent two-engine training.  Consider
enabling after confirming stable Adam behaviour over a longer run.

### Horizon noise mitigation — ablation testing plan

Two mechanisms reduce the influence of tactics-beyond-horizon on TD errors:
score-change clipping (`TDLEAF_SCORE_CLIP_CP`) and ID-stability weighting
(`TDLEAF_ID_VAR_SIGMA2`).  Their individual contributions should be isolated.

**Recommended ablation (500 games per arm, same starting network):**

| Arm | TDLEAF_SCORE_CLIP_CP | TDLEAF_ID_VAR_SIGMA2 | Description |
|-----|---------------------|---------------------|-------------|
| A (baseline) | 1e6 (disabled) | 1e6 (disabled) | Original algorithm |
| B (clip only) | 200 cp | 1e6 (disabled) | Approach 1 only |
| C (ID weight only) | 1e6 (disabled) | 10 000 cp² | Approach 2 only |
| D (combined) | 200 cp | 10 000 cp² | Both active (current default) |

**Metric:** Elo gain per game vs. the starting network (use `bayeselo_ratings.py` on
a 100-game test match against the starting network after each 500-game training run).

**Override at build time:**

```sh
# Arm A — no mitigation
perl comp.pl train_arm_a NNUE=1 NNUE_NET=nn-start.nnue TDLEAF=1 \
  -D TDLEAF_SCORE_CLIP_CP=1000000.0f -D TDLEAF_ID_VAR_SIGMA2=1000000.0f

# Arm B — clip only
perl comp.pl train_arm_b NNUE=1 NNUE_NET=nn-start.nnue TDLEAF=1 \
  -D TDLEAF_SCORE_CLIP_CP=200.0f -D TDLEAF_ID_VAR_SIGMA2=1000000.0f
```

(Similarly for arms C and D.)

**After the ablation:** if one approach dominates, drop the other to reduce complexity.

### Search parameter tuning
The search's pruning parameters (null-move margins, futility thresholds, aspiration
windows, LMR reduction tables) were tuned for the classical eval.  The NNUE eval has a
different score distribution and may benefit from re-tuning these constants.  CLOP or
a self-play tournament with systematic variation would be the appropriate approach.

---

## NNUE Infrastructure

### Pawn hash under NNUE
The classical eval stores pawn structure scores in a pawn hash table.  The NNUE eval
bypasses classical eval entirely, so `pawn hash hits` is always 0 in NNUE mode and the
pawn hash memory (≈19 MB) is wasted.  Disabling or shrinking it at build time when
`NNUE=1` would recover that memory (no effect on playing strength).

### search.par revision
Review whether `search.par` is still the right mechanism for runtime configuration —
consider what parameters are still relevant, whether the file format should be updated,
and how it interacts with xboard/cutechess invocation.

### Multi-thread accumulator correctness
The SMP search allocates one `ts_thread_data` per thread, each with its own
`search_node n[MAXD+1]` stack including per-node accumulators.  Each thread's root
accumulator is independently initialised.  Thread interactions have not been tested
under NNUE; correctness is expected but unverified with `THREADS > 1`.

---

## Resolved / Implemented

### ~~Adam + per-weight LR decay~~ ✓ Implemented (2026-03-15)
Adam optimizer with per-weight LR decay `lr(cnt) = LR0 / (1 + cnt/C)` is live.
FC/FT: `TDLEAF_ADAM_LR0=0.5`; PSQT: `TDLEAF_ADAM_PSQT_LR0=20.0`; C=500.
FC0/FC1 float shadows clamped to ±127 to prevent zombie weights.  See `docs/TDLEAF.md`.

### ~~Epoch-based replay~~ ✓ Implemented (2026-03-11)
Flavor B is live with `TDLEAF_REPLAY_K=1` (default) and `TDLEAF_REPLAY_BUF_N=8`.
Ablation: K=1 is the current conservative default (K=2 marginal gain; K=6 large regression).

### ~~Bias initialisation~~ ✓ Implemented (2026-03-11)
FC biases (FC0/FC1/FC2) and FT biases are zero-initialised in `--init-nnue` mode.
Random N(μ,σ) from SF15.1 was removed — it added noise TDLeaf must first cancel.

### ~~Linux compilation~~ ✓ Resolved (2026-03-11)
AVX2 x86-64 SIMD paths added to `nnue.cpp` for SqrCReLU, FC0, and FC1.
Fallback chain: NEON (ARM) → AVX2 (x86-64 with `-mavx2`) → scalar.
Default build uses `-march=x86-64-v3`; use `NATIVE=1` for CPU-native tuning.
