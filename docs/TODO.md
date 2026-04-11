# Leaf TODO

Planned investigations, improvements, and open questions.

---

## TDLeaf(λ) Training

### Training depth curriculum (active area — 2026-04-11)

Empirical findings from nn-fresh-260410 training (~1.4M games total):

**Phase 1 — pure self-play, timed, 500k games:**
Elo vs classic_eval (timed, ~40 Elo speed penalty for NNUE):
0g → -1000, 10k → -590, 50k → -380, 200k → -240, 500k → -170.
Rapid early gain followed by strong diminishing returns.

**Phase 2 — mixed self-play + vs classic_eval, depth=6 fixed, ~930k games:**
Elo vs classic_eval at depth=6 (no speed offset):
Start → -37, after 300k games → -18, then essentially flat through 930k games.
The first 300k games gained ~19 Elo; the next 630k gained ~2-3 Elo total.
Conclusion: training stalled at depth=6. The engine quickly exhausts the
learnable signal at any fixed depth and reaches a local equilibrium.

**Phase 3 — depth=8, 50k games:**
Virtually all improvement occurred in the first 10k games, then flat again.
Pattern: switching depth gives a one-time "calibration kick" as the network
corrects its weights for the deeper signal, then stalls at the new equilibrium.

**Key insight — Adam v is NOT the cause of plateaus:**
After ~1.4M games, FC max v = 4.5e-6 → effective FC LR = 0.01/sqrt(4.5e-6) ≈ 4.7.
Adam v is negligibly small; the FC plateau is caused by small TD errors
(gradients are tiny when the engine is near its depth-equilibrium), not by
accumulated v damping updates. Optimizer reset would give a brief burst but
not sustained improvement.

**Recommended strategy going forward:**
Depth curriculum: spend ~10-15k games per depth step, then advance.
e.g. d8 → d10 → d12 → d14. Each transition gives one calibration kick.
Staying at any depth beyond ~10-15k games past the initial burst yields
diminishing returns. The depth itself is the primary lever, not game volume.

**Future experiments to try:**
- Depth asymmetry: NNUE at d8, opponent at d10 (stronger signal without
  proportionally more NNUE search cost).
- Adam reset at depth transitions: zero v/m via `reset_adam.py` when
  switching depth to allow full-sized steps into the new signal. Given
  v is already near zero for FC, this matters most for FT v (sparse rows)
  and piece_val v. Low priority given small FC v, but worth testing at d10+.
- Increasing λ toward 1.0 at higher depths: longer-range credit assignment
  helps with positional learning once tactical signal is cleaner.
- Larger/more varied opening EPD set: position diversity creates fresh
  gradients even at fixed depth, potentially slowing the plateau.

### Adam hyperparameter tuning

The Adam optimizer uses five separate LRs tuned from 190k-game weight distribution
analysis.  Key monitoring points:

- **FC LR0 (0.1):** FC1 saturation at 0.5% after 190k games; stacks 5,6 at 1.2–1.7%.
  Reduced from 0.13 to extend runway before saturation becomes problematic.
- **FT LR0 (1.0):** FT weights barely changed (std 44.006 vs 44.0 init).  With only
  3–50 updates per weight, FT learning is very slow; high LR compensates.
- **FT bias LR0 (0.01):** Separate LR prevents dying-ReLU from update frequency asymmetry.
- **PSQT LR0 (1.6):** PSQT barely moves (std change -44 from 35642 over 190k games).
  Correct behavior — dense piece_val handles material corrections.

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

### Prioritized experience replay

The replay buffer currently iterates over all buffered games with equal weight.  Games
with larger total TD error (`Σ|e[t]|`) contain more learning signal and should be
replayed with higher priority.  Simplest variant: weight each game by its cumulative
absolute TD error, or skip games where total error is near zero.

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

### Win-only .tdleaf.bin writes (`TDLEAF_WIN_ONLY_WRITE`)

Compile-time flag that suppresses writing `.tdleaf.bin` after draws and losses.
Gradients still applied to in-memory weights and other-process deltas still merged from disk on
all games; only the disk write is gated on `td_result >= 1.0`.  Requires refactoring
`nnue_save_fc_weights` to split its read+merge phase from its write phase.
See memory for full implementation plan.

---

## Known Issues

(none currently open)

---

## Resolved / Implemented

### ~~Persistent Adam v~~ ✓ Implemented (2026-04-02)
Adam second-moment (v) arrays and t_adam persisted to .tdleaf.bin v6.  Multi-writer
merge uses max(v_file, v_local) per element.  FT weight v (~92 MB) excluded.  Momentum
(m) not persisted — recovers in ~10 steps.  ~564 KB file size increase.

### ~~Separate FT bias LR~~ ✓ Implemented (2026-04-01)
`TDLEAF_ADAM_FT_BIAS_LR0 = 0.01` (10× slower than FC) prevents dying-ReLU from update
frequency asymmetry.  FT biases update ~200×/game vs FT weights ~8/5000g.

### ~~--init-nnue-noprior~~ ✓ Implemented (2026-04-01)
All piece PSQT values initialised at 100cp (uniform) instead of classical material.
Forces material value learning from scratch.  training_run.py offers the choice.

### ~~FC-only replay~~ ✓ Implemented (2026-03-29)
Replay with FT/PSQT gradients causes eval divergence; `fc_only=true` suppresses
FT/PSQT/FT-bias/piece_val gradients during replay.  +20 Elo over 5000 games.
Subsequently disabled (TDLEAF_REPLAY_K=0) as benefit faded after first 5000 games.

### ~~Init-nnue redesign~~ ✓ Implemented (2026-03-23)
Weight initialization redesigned for TDLeaf training (decoupled from SF15.1 statistics).
FT weights N(0,5), FC weights N(0,{1,3,2}), all means zero, PSQT pure material (no
piece-square bonuses).  Separate `TDLEAF_ADAM_FT_LR0=0.2` for sparse FT weights.

### ~~Flavor A replay~~ ✓ Implemented (2026-03-21)
Replay now rebuilds accumulators from stored leaf positions using current FT weights,
ensuring FT gradients during replay are self-consistent with the current network.
`TDRecord` stores the leaf `position` (~300 bytes/ply, ~6% size increase).

### ~~Per-weight bias correction~~ ✓ Implemented (2026-03-21)
FC and PSQT Adam steps use per-weight bias correction (`eff_t = cnt + 1`) instead of
global `t_adam`.  bc1 skipped at cnt≥20 (negligible); bc2 always applied.  FT RMSProp
retains global bc2 (sparse features need growing global correction).

### ~~Per-weight LR decay removed~~ ✓ Removed (2026-03-22)
Per-weight LR decay (`TDLEAF_ADAM_C`, `TDLEAF_ADAM_LR_FLOOR`) removed.  AdamW weight
decay now handles regularization; LR0 tuned directly to the right value (0.13 FC, 1.6
PSQT) instead of starting high and decaying.  `--set-cnt` and `_prompt_init_cnt` also
removed as they existed only to prime the LR decay schedule.

### ~~AdamW decoupled weight decay~~ ✓ Implemented (2026-03-21)
`TDLEAF_WEIGHT_DECAY=1e-4` applied to FC weights and FT weights after each Adam step.
Skipped for biases (no benefit) and PSQT (would fight classical prior).

### ~~Gradient clipping by global norm~~ ✓ Implemented (2026-03-21)
`TDLEAF_GRAD_CLIP_NORM=1.0` clips the global L2 gradient norm before each Adam step.
Applied in `tdleaf_update_after_game`, `tdleaf_replay`, and `tdleaf_flush_batch`.
Set to 0 to disable.

### ~~Asymmetric lambda~~ ✓ Implemented (2026-03-21)
`TDLEAF_LAMBDA_DECISIVE=0.8` for wins/losses, `TDLEAF_LAMBDA_DRAW=0.5` for draws.
Decisive games get longer eligibility traces; draws use shorter traces to reduce
balanced-position noise.  Set both to the same value for symmetric behaviour.

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

### ~~Adam optimizer~~ ✓ Implemented (2026-03-15), LR decay removed (2026-03-22)
Adam optimizer with fixed LR (constant after warmup).  Per-weight LR decay was removed
in favour of direct LR tuning + AdamW weight decay.
FC/FT: `TDLEAF_ADAM_LR0=0.13`; PSQT: `TDLEAF_ADAM_PSQT_LR0=1.6`.
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
