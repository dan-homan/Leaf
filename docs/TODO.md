# Leaf TODO

Planned investigations, improvements, and open questions.

---

## TDLeaf(λ) Training

### Adam hyperparameter tuning

The Adam + per-weight LR decay optimizer (`TDLEAF_ADAM_LR0=0.2`, `TDLEAF_ADAM_PSQT_LR0=2.0`,
`TDLEAF_ADAM_C=5000`) uses initial guesses that have not been systematically tuned.  A grid
search varying each independently across 500–1000-game runs would establish better defaults.

Key questions:

- **FC LR0 (0.2):** After 5,000 games the FC0/FC1 float shadows spread to std≈30/50, filling
  the int8 range.  Float-shadow clamping prevents zombie weights but further training will be
  limited at the ±127 boundary.  Monitor whether the network continues to improve or plateaus
  as the distribution saturates.
- **PSQT LR0 (2.0):** Tuned separately from FC LR0 since Adam normalises gradient magnitude
  and PSQT operates at int32 scale.  Tune empirically.
- **C (5000):** LR half-life in per-weight updates.  Larger C extends the fast-learning phase;
  smaller C converges more aggressively.

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

### ~~FT bias training disabled~~ ✓ Fixed (2026-03-20)
FT bias gradient accumulation and Adam updates disabled during online TDLeaf learning.
The shared bias gradient is structurally tiny (shifts both perspectives equally) but
Adam normalizes it into full LR-sized steps, causing universal negative drift (mean -371
after 5000 games, gating off ~93% of SqrCReLU dimensions).  Per-feature FT weights
absorb any needed offset.  Biases stay at baseline .nnue values.

### ~~Asymmetric lambda~~ ✓ Implemented (2026-03-20)
`TDLEAF_LAMBDA_DECISIVE=0.8` for wins/losses, `TDLEAF_LAMBDA_DRAW=0.5` for draws.
Decisive games get longer eligibility traces; draws use shorter traces to reduce
balanced-position noise.  Set both to the same value for symmetric behaviour.

### ~~Per-weight bias correction~~ ✓ Implemented (2026-03-20)
Adam bias correction now uses per-weight update count (`cnt+1`) instead of global
`t_adam`.  For weights with ≥20 updates, bias correction is skipped (≈1.0).
Fixes under-correction for sparse FT/PSQT features.

### ~~Gradient clipping by global norm~~ ✓ Implemented (2026-03-20)
`TDLEAF_GRAD_CLIP_NORM=1.0` clips the global L2 gradient norm before each Adam step.
Applied in `tdleaf_update_after_game`, `tdleaf_replay`, and `tdleaf_flush_batch`.
Set to 0 to disable.  Logs pre-clip norm to stderr when clipping occurs.

### ~~Prioritized experience replay~~ ✓ Implemented (2026-03-20)
`TDLEAF_REPLAY_MIN_ERROR=0.0` (disabled by default).  Replay games with cumulative
|e[t]| below the threshold are skipped.  TD error sums are updated after each replay
pass for accurate future prioritization.

### ~~Flavor A replay~~ ✓ Implemented (2026-03-20, FC-only fix 2026-03-20)
Replay rebuilds accumulators from stored leaf positions using current FT weights
via `nnue_init_accumulator()`, then re-enumerates features and re-evaluates scores.
**FC-only replay gradients:** FT/PSQT gradient backprop is suppressed during replay
(`fc_only=true`) to prevent a positive feedback loop where rebuilt accumulators
amplify live-pass FT updates, causing eval divergence within ~130 games.  FC layers
still receive correct gradients from refreshed accumulators during replay.
`TDRecord` stores the leaf `position` (~256 bytes/ply).

### ~~AdamW decoupled weight decay~~ ✓ Implemented (2026-03-19)
`TDLEAF_WEIGHT_DECAY=1e-4` applied to FC weights and FT weights after each Adam step.
Skipped for biases (no benefit) and PSQT (would fight classical prior).

### ~~Mini-batch gradient accumulation~~ ✓ Implemented (2026-03-19)
Gradients accumulated across `TDLEAF_BATCH_SIZE=4` games before each Adam step.
Reduces single-game gradient noise and file I/O by ~4×.  `tdleaf_flush_batch()`
applies any pending partial batch at session end.  Set `TDLEAF_BATCH_SIZE=1` to restore
per-game updates.

### ~~Per-weight FT second moment~~ ✓ Implemented (2026-03-19)
FT weights upgraded from per-row RMSProp v (~88 KB) to per-weight v (~92 MB, OS lazy-paged).
Each of the 1024 dimensions within a feature row now has its own variance estimate,
allowing the optimizer to adapt step sizes per-dimension rather than using a coarse
per-row average.

### ~~LR warmup~~ ✓ Implemented (2026-03-19)
Linear warmup over first `TDLEAF_ADAM_WARMUP=50` Adam steps.  Prevents early-training
instability from cold-start v estimates.  Set `TDLEAF_ADAM_WARMUP=0` to disable.

### ~~Adam + per-weight LR decay~~ ✓ Implemented (2026-03-15)
Adam optimizer with per-weight LR decay `lr(cnt) = LR0×(floor+(1−floor)/(1+cnt/C))` is live.
FC/FT: `TDLEAF_ADAM_LR0=0.2`; PSQT: `TDLEAF_ADAM_PSQT_LR0=1.0`; C=5000; floor=0.05.
FC0/FC1 float shadows clamped to ±127 to prevent zombie weights.  See `docs/TDLEAF.md`.

### ~~Epoch-based replay~~ ✓ Implemented (2026-03-11, upgraded to Flavor A 2026-03-20)
Flavor A is live with `TDLEAF_REPLAY_K=1` (default) and `TDLEAF_REPLAY_BUF_N=8`.
Ablation: K=1 is the current conservative default (K=2 marginal gain; K=6 large regression).

### ~~Bias initialisation~~ ✓ Implemented (2026-03-11)
FC biases (FC0/FC1/FC2) and FT biases are zero-initialised in `--init-nnue` mode.
Random N(μ,σ) from SF15.1 was removed — it added noise TDLeaf must first cancel.

### ~~Linux compilation~~ ✓ Resolved (2026-03-11)
AVX2 x86-64 SIMD paths added to `nnue.cpp` for SqrCReLU, FC0, and FC1.
Fallback chain: NEON (ARM) → AVX2 (x86-64 with `-mavx2`) → scalar.
Default build uses `-march=x86-64-v3`; use `NATIVE=1` for CPU-native tuning.
