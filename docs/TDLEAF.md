# TDLeaf(λ) NNUE Learning — Implementation Reference

## Overview

TDLeaf(λ) is a temporal-difference reinforcement learning algorithm adapted for minimax
search (Baxter, Tridgell, and Weaver, 2000).  It uses the game result and the sequence
of NNUE evaluations at PV leaf positions to form TD errors, then backpropagates those
errors through the NNUE network to update weights.

Build with:

```sh
perl comp.pl 2026_03_09a NNUE=1 TDLEAF=1
```

All learning code is gated by `#if TDLEAF`; when `TDLEAF=0` (default) no overhead is added.

> **Protocol requirement — TDLeaf only works under xboard/CECP.**
> The learning hooks (`tdleaf_record_ply`, `tdleaf_update_after_game`) are called
> from inside `make_move()`, which is part of the xboard game loop.  When Leaf runs
> under the UCI protocol the GUI drives the game externally — `make_move()` is never
> called — so **no weights are updated and no `.tdleaf.bin` is written**, even if the
> binary was compiled with `TDLEAF=1`.  Always use xboard-protocol matches (via
> `match.py` or direct xboard GUI invocation) for training.

---

## Algorithm

For a game of T half-moves:

- `d_t` = sigmoid of the **NNUE static evaluation at the PV leaf position** at ply t,
  from White's perspective:
  `d_t = 1 / (1 + exp(-score_white_t / K))`, K ≈ 400 cp.
- `z` = game result from White's perspective: 1.0 = White wins, 0.5 = draw, 0.0 = Black wins.

**Temporal difference errors (backward view):**

```
e_{T-1} = z - d_{T-1}
e_t     = clip(d_{t+1} - d_t) + λ * e_{t+1}     for t = T-2 … 0
```

where `clip(d_{t+1} - d_t)` applies proportional scaling when the white-POV score
change between consecutive moves exceeds `TDLEAF_SCORE_CLIP_CP` centipawns — see
[Horizon Noise Mitigation](#horizon-noise-mitigation) below.

**Weight update (gradient ascent on prediction accuracy):**

```
Δw = α * Σ_t  e_t * ∇_w d_t
```

where `∇_w d_t = d_t * (1 - d_t) / K * ∇_w score_t`.

Defaults: `λ_decisive = 0.8` (wins/losses), `λ_draw = 0.5` (draws), `K = 400`.
Gradient updates use Adam with per-weight LR decay; see [Adam Optimizer](#adam-optimizer-with-per-weight-lr-decay) below.

**Key design choice:** `d_t` is computed from `nnue_evaluate()` (direct static eval of the
PV leaf), not from the search score propagated from the root.  This ensures the sigmoid
value and the forward-pass gradient are computed from the same NNUE evaluation of the same
position, making the gradient self-consistent.

---

## Scope: All NNUE Weights Are Trained

| Layer | Parameters | Notes |
|-------|-----------|-------|
| FC0 weights/biases | 1,024×16 int8 + 16 int32, ×8 stacks | Quantized int8, float shadow |
| FC1 weights/biases | 32×32 int8 + 32 int32, ×8 stacks | Same |
| FC2 weights/bias   | 32 int8 + 1 int32, ×8 stacks | Same |
| FT biases          | 1,024 int16 | Dense update; static float shadow (4 KB) |
| FT weights         | 22,528×1,024 int16 | Sparse update; float shadow on heap (~92 MB) |
| PSQT weights       | 22,528×8 int32 | Sparse update; float shadow (~720 KB) |

FT weights and PSQT are updated sparsely: only the ~30–60 feature rows active at each
leaf position are touched.  `ft_dirty[FT_INPUTS]` tracks which rows received gradient
during the game; only dirty rows are scanned in `nnue_apply_gradients`.

FT biases are updated densely every game (all 1,024 values): the gradient is the sum of
`g_acc[persp][d]` across both perspectives.

---

## File Structure

| File | Contents |
|------|----------|
| `src/tdleaf.h` | Hyperparameters, `TDRecord`, `TDGameRecord`, function declarations |
| `src/tdleaf.cpp` | PV walking, TD error computation, gradient backprop hooks |
| `src/nnue.cpp` | FP32 shadow arrays, forward pass, gradient accumulation, weight save/load |
| `src/nnue.h` | `NNUEActivations` struct, TDLeaf function declarations |

---

## Data Structures

### `TDRecord` (per-ply snapshot, stored in `TDGameRecord`)

```cpp
struct TDRecord {
    int16_t acc[2][NNUE_HALF_DIMS];          // accumulator at PV leaf (int16)
    int32_t psqt[2][NNUE_PSQT_BKTS];        // PSQT at PV leaf
    int     score_stm;                        // NNUE static eval at leaf, STM POV (cp)
    int     stack;                            // layer-stack index (piece_count-1)/4
    bool    wtm;                              // White to move at leaf
    float   id_score_variance;               // variance of last TD_ID_HIST ID-depth scores (cp²)
    int     ft_idx[2][NNUE_MAX_FT_PER_PERSP]; // active FT feature indices
    int8_t  n_ft[2];                          // active feature count per perspective
    position pos;                             // leaf position for Flavor A replay
};
```

Memory: ≈ (2×2048 + 8×4 + 4+4+1+4 + 2×64×4 + 2 + ~300) bytes × 400 plies ≈ 2.4 MB.

### `TDGameRecord`

```cpp
struct TDGameRecord {
    TDRecord plies[MAX_GAME_PLY];   // one per half-move (max 400)
    int      n_plies;               // plies recorded so far
};
```

One global instance in `game_rec`; `n_plies` reset to 0 at game start.

---

## PV Leaf Score

`tdleaf_record_ply` walks the PV from the root accumulator using `nnue_record_delta` /
`nnue_apply_delta`, then calls `nnue_evaluate(leaf_acc, leaf_wtm, pc)` to get the leaf
score.  `leaf_wtm = root_wtm XOR (pv_len & 1)` — the side to move at the leaf flips once
per ply walked.

The score is always stored from the leaf's side-to-move perspective (`score_stm`).
`tdleaf_update_after_game` converts to White's perspective as:
`score_white = leaf_wtm ? score_stm : -score_stm`.

To verify correctness at build time:

```sh
perl comp.pl 2026_03_09a NNUE=1 TDLEAF=1 TDLEAF_CHECK_SCORE=1
```

This prints `direct` (NNUE leaf eval) vs `propagated` (root score with per-ply sign flip)
for every ply, flagging differences > 300 cp.

---

## Gradient Flow

```
FC2 output (positional)
  → ∂/∂(positional) via cp_factor and wtm_sign
  → FC2 weights/bias
  → CReLU backward → FC1 pre-activation
  → FC1 weights/bias
  → dual-activation backward (SqrCReLU + CReLU on FC0 outputs 0–14)
  → FC0 pre-activation
  → FC0 weights/bias
  → SqrCReLU backward on accumulator pairs
  → g_acc[2][1024]  (gradient w.r.t. each accumulator unit per perspective)
  → FT weight rows for each active feature index (sparse)
  → PSQT weight rows for each active feature index (sparse)
```

The step size for each layer is governed by the Adam LR schedule.  Three separate LRs:
`TDLEAF_ADAM_LR0` (FC layers + FT biases), `TDLEAF_ADAM_FT_LR0` (FT weights), and
`TDLEAF_ADAM_PSQT_LR0` (PSQT).  FT weights use a higher LR than FC because sparse features
receive far fewer updates (~8 per 5000 games vs. every game for FC); PSQT at int32 scale
needs a different LR than FC at int8 scale.  See [Adam Optimizer](#adam-optimizer).

---

## Adam Optimizer

`nnue_apply_gradients()` uses AdamW with a fixed learning rate (constant after warmup).

### Algorithm

For each weight parameter w with accumulated gradient g and update count cnt:

```
t   ← t + 1                               (global session step counter)
m   ← β₁ m + (1−β₁) g                    (first moment)
v   ← β₂ v + (1−β₂) g²                   (second moment)
eff_t = cnt + 1                            (per-weight update count)
m̂   = (eff_t ≥ 20) ? m : m / (1 − β₁^eff_t)   (bc1 skipped when negligible)
v̂   = v / (1 − β₂^eff_t)                 (bc2 always applied)
Δw  = −LR0 × m̂ / (√v̂ + ε)
w   ← w + Δw − λ × LR0 × w              (AdamW weight decay, weights only)
cnt ← cnt + 1
```

Bias correction uses per-weight `eff_t = cnt + 1` rather than the global `t_adam`.
bc1 (β₁=0.9) is skipped at cnt≥20 because 0.9²⁰ ≈ 0.12, making bc1 ≈ 0.88 (close
to 1).  bc2 (β₂=0.999) is **always** applied: 0.999²⁰ ≈ 0.98, so bc2 = 0.02 at
cnt=20 — skipping would give ~7× oversized steps.  FT RMSProp retains global bc2
(from `t_adam`) because sparse features (~8 updates/5000g) need the growing global
correction.

### Per-Layer Configuration

| Layer | Update Rule | LR0 | Notes |
|-------|-------------|-----|-------|
| FC0/FC1/FC2 weights | Full Adam | `TDLEAF_ADAM_LR0 = 0.13` | Float shadow clamped to ±127 after each update |
| FC0/FC1/FC2 biases  | Full Adam | `TDLEAF_ADAM_LR0 = 0.13` | |
| FT weights | RMSProp (per-weight v, no m) | `TDLEAF_ADAM_FT_LR0 = 0.2` | Sparse; higher LR than FC to compensate for fewer updates |
| FT biases  | Full Adam | `TDLEAF_ADAM_LR0 = 0.13` | |
| PSQT       | Full Adam | `TDLEAF_ADAM_PSQT_LR0 = 1.6` | Separate LR0 required — see below |

### Why a Separate PSQT LR0?

Adam normalises gradient magnitude: the effective per-step size in weight-space is
approximately ±LR0 per update, independent of the raw gradient magnitude.  PSQT
weights are at int32 scale (std ≈ 36,000) while FC weights are at int8 scale (std ≈ 30)
— a ratio of ~1,000×.  Using the same LR0=0.2 for both caused PSQT to change negligibly
relative to its baseline scale.  `TDLEAF_ADAM_PSQT_LR0` is tuned separately for this reason.

### Why Float-Shadow Clamping for FC0/FC1?

FC0 and FC1 use int8 quantized inference weights clamped to ±127 on requantization.
Without a matching clamp on the float shadow, Adam can push `w_f32` arbitrarily beyond
±127 while the inference weight is stuck at the boundary.  These "zombie weights"
accumulate gradient updates with zero effect on the network.  After each weight update,
`w_f32 = clamp(w_f32, −127, 127)` keeps the float shadow aligned with the int8 inference
space.  Not applied to FC2 (output layer, no activation clamping required).

### Session-Local Moments

All m/v arrays and `t_adam` are zeroed at session start in both `nnue_init_fp32_weights()`
and `nnue_init_zero_weights()`.  Resetting both together keeps bias correction
`v̂ = v/(1−β₂ᵗ)` mathematically valid from game 1 of each session.

Moments are **not** persisted to `.tdleaf.bin` because:
- Persisting them would break the delta-merge mechanism
  used for concurrent multi-instance training.
- Session-local moments restart cleanly with full bias correction each session.
- The per-weight `cnt` arrays (which are persisted) track update history for
  per-weight bias correction and monitoring.

### Hyperparameters (`src/tdleaf.h`)

| Constant | Value | Notes |
|----------|-------|-------|
| `TDLEAF_ADAM_LR0` | 0.13 | Step size for FC layers + FT biases (float weight units) |
| `TDLEAF_ADAM_FT_LR0` | 0.2 | Step size for FT weights (sparse; need higher LR than dense FC) |
| `TDLEAF_ADAM_PSQT_LR0` | 1.6 | Step size for PSQT (int32 scale; ~1000× FC) |
| `TDLEAF_ADAM_BETA1` | 0.9 | First-moment decay (FC weights/biases, FT biases, PSQT) |
| `TDLEAF_ADAM_BETA2` | 0.999 | Second-moment decay (all layers) |
| `TDLEAF_ADAM_EPS` | 1e-8 | Numerical floor in denominator |
| `TDLEAF_ADAM_WARMUP` | 50 | Linear LR warmup: ramp from 0 to full LR over first N Adam steps (0 = disabled) |
| `TDLEAF_BATCH_SIZE` | 4 | Mini-batch: accumulate gradients across N games before each Adam step |
| `TDLEAF_WEIGHT_DECAY` | 1e-4 | AdamW decoupled weight decay coefficient (FC + FT weights only) |
| `TDLEAF_GRAD_CLIP_NORM` | 1.0 | Global gradient L2 norm clip threshold; 0 = disabled |

Set `TDLEAF_BATCH_SIZE = 1` to restore per-game Adam steps.
Set `TDLEAF_ADAM_WARMUP = 0` to disable warmup.
Set `TDLEAF_WEIGHT_DECAY = 0.0` to disable weight decay.
Set `TDLEAF_GRAD_CLIP_NORM = 0.0` to disable gradient clipping.

---

## Mini-Batch Gradient Accumulation

By default (`TDLEAF_BATCH_SIZE=4`), gradients are accumulated across 4 games before
a single Adam step is applied.  This gives the optimizer a more reliable gradient signal
per step, reducing single-game noise that otherwise causes Adam's first moment to chase
stochastic fluctuations.

### How it works

1. `tdleaf_update_after_game()` calls `tdleaf_accumulate_game()` on every game but only
   calls `nnue_apply_gradients()` + `nnue_requantize_fc()` + save when the batch counter
   reaches `TDLEAF_BATCH_SIZE`.
2. `tdleaf_replay()` always pushes the completed game into the ring buffer, but replay
   passes only run on batch boundaries (when the live batch was just applied).
3. `tdleaf_flush_batch()` applies any pending partial batch at session end (program exit
   or weight export), preventing gradient loss.

### Trade-offs

- **Pro:** each Adam step uses ~4× more gradient data, improving signal-to-noise ratio.
- **Pro:** file I/O reduced by ~4× (one write per batch instead of per game).
- **Con:** weight updates are delayed by up to `BATCH_SIZE-1` games (negligible in practice;
  the delay is <1 second at typical game durations).

Set `TDLEAF_BATCH_SIZE = 1` to restore the original per-game update behaviour.

---

## LR Warmup

A linear warmup ramps the learning rate from 0 to its full value over the first
`TDLEAF_ADAM_WARMUP` Adam steps (default 50).  The effective LR at step `t` is:

```
lr_effective = min(1.0, t / WARMUP) × lr_decay(LR0, cnt)
```

### Motivation

Adam's bias correction handles cold-start `m` and `v` mathematically, but in practice
the first few steps can produce disproportionately large effective step sizes because
`v` hasn't accumulated a reliable variance estimate.  For rarely-visited feature rows
that may not receive gradient for hundreds of games, the first update can overshoot.
Warmup smooths this transition at essentially zero implementation cost.

Set `TDLEAF_ADAM_WARMUP = 0` to disable warmup.

---

## Weight Persistence — `.tdleaf.bin` (version 4)

Saved at `{exec_path}nn-ad9b42354671.tdleaf.bin`.  Format:

```
[version(4) + 8 FC stacks: per-layer float32×128 weights/biases + uint32 counts]
[n_ft_rows(4 bytes)]
[per dirty row: fi(4) + ft_w[1024]×128 as float32[1024] + ft_cnt[1024] as uint32[1024]
                      + psqt_w[8]×128 as float32[8]    + psqt_cnt[8]  as uint32[8]]
[FT bias section (v4): ft_bias[1024]×128 as float32[1024] + ft_bias_cnt[1024] as uint32[1024]]
```

Values are stored at 128× resolution (divide by 128 on load) to preserve sub-integer
drift across sessions.  Update counts enable weighted averaging of concurrent training runs.

Version 3 files (FC + FT/PSQT, no FT bias) are still accepted on load; FT biases start
from the `.nnue` baseline and will be included on the next save.
Version 2 files (FC only) are also still accepted.  A notice is printed in both cases.

---

## Concurrent File Access

Multiple Leaf instances (e.g. several parallel self-play games) can share a single
`.tdleaf.bin` safely via POSIX file locking and delta-based merging.

### Design

**Problem:** If two instances both read the file, apply their gradient updates to their
in-memory weights, and then write back, the second write silently overwrites the first
instance's changes.

**Solution:** Each instance tracks only its own accumulated weight *deltas* since the last
file write (not the full weight values).  On each write:

1. Acquire `LOCK_EX` on a companion `.tdleaf.bin.lock` file.
2. Re-read the current `.tdleaf.bin` from disk.
3. Merge: `merged_value = file_value + our_delta` for every entry.
4. Update the in-memory float shadows to the merged values; zero the deltas.
5. Write the merged content to `.tdleaf.bin.tmp`.
6. `rename(.tmp, .tdleaf.bin)` — atomic on POSIX filesystems.
7. Release the lock (close the lock-file fd).

Reads use `LOCK_SH` (multiple simultaneous readers allowed; blocked only during a write).

### Implementation Details

**Lock file:** `.tdleaf.bin.lock` is a separate companion file so locking survives the
atomic `rename()` of the main file.  The lock is held only during the re-read/write cycle
(a few milliseconds), not across the entire game.

**Delta arrays** (in `nnue.cpp`):

| Array | Size | Contents |
|-------|------|----------|
| `delta_l0_w[8][1024×16]` | 512 KB | FC0 weight deltas |
| `delta_l0_b[8][16]` | 512 B | FC0 bias deltas |
| `delta_l1_w[8][32×32]` | 256 KB | FC1 weight deltas |
| `delta_l1_b[8][32]` | 1 KB | FC1 bias deltas |
| `delta_l2_w[8][32]` | 1 KB | FC2 weight deltas |
| `delta_l2_b[8]` | 32 B | FC2 bias deltas |
| `ft_delta_f32` (heap) | ~92 MB | FT weight deltas (all rows) |
| `psqt_delta_f32` (heap) | ~720 KB | PSQT weight deltas |
| `ft_bias_delta[1024]` | 4 KB | FT bias deltas (static) |

Deltas are zeroed after each successful write (either on first write or after a
re-read-merge write).  `nnue_load_fc_weights()` also zeros all deltas to establish a
clean baseline.

**Update-count merging:** counts use `max(file_count, our_count)` — an approximation
of the total cross-instance update count sufficient for monitoring; exact accumulation
is not required for correctness.

### Usage with match.py

When running parallel self-play via cutechess-cli with multiple concurrent TDLEAF
instances, add `--wait MS` to `match.py` to insert a pause between games.  This reduces
contention on the `.tdleaf.bin.lock` file and gives each instance time to complete its
write cycle before the next game starts:

```sh
python3 match.py Leaf_vtrain_a Leaf_vtrain_ro -n 200 -c 4 --wait 500
```

---

## Initialization

**Default (fine-tuning):** When no `<network>.tdleaf.bin` is found, the default network 
for that executable, `<network>.nnue`, is used as a starting point and training proceeds 
as gradient updates from that starting point.  If `<network>.tdleaf.bin` is present, corresponding 
to previous training, these values are loaded as updates to the default at startup. Any 
additional training will further update the `<network>.tdleaf.bin` file.

**Training from scratch:** Use `--init-nnue --write-nnue <file>` to create a randomly
initialised `.nnue` with no source file required:

```sh
perl comp.pl init_nnue NNUE=1 TDLEAF=1
./Leaf_vinit_nnue --init-nnue --write-nnue nn-fresh.nnue
```

This calls `nnue_alloc_arrays()` + `nnue_init_fp32_weights()` + `nnue_init_zero_weights()`:

| Component | Distribution | Notes |
|-----------|-------------|-------|
| FT weights (int16) | N(0, 44) | Zero mean; acc std ≈ √30 × 44 ≈ 241; ~40% CReLU active |
| FC0 weights (int8) | N(0, 4) | FC0 CReLU ≈ 3.8; keeps FC1→FC2 chain active |
| FC1 weights (int8) | N(0, 3) | Moderate — fan-in 30, low saturation risk |
| FC2 weights (int8) | N(0, 2) | Small — keeps initial positional output ≈ 0 cp |
| All biases | **0** (zero) | FT and FC biases zero-initialised |
| PSQT | Pure material (no PSQ bonuses) | Same value across all 8 buckets |

All int8 weight sampling uses **rejection sampling** (not clipping): values outside ±127 are
discarded and redrawn to avoid density spikes at the int8 boundary.

**Design philosophy: start quiet, let TDLeaf build structure from signal.**  Initial NNUE
positional output should be near zero so that classical material dominates early play
(reasonable game quality from game 1).  The network gradually grows its influence as
TDLeaf learns real patterns from self-play.  Zero means (He/Kaiming principle) are the
correct starting point — non-zero means from a trained network are endpoints, not priors.

**FT weights (σ=44):** ~30 features active per position, so accumulator std ≈ √30 × 44 ≈ 241.
CReLU divides by 64 (>>6 shift), so the accumulator needs values in [0, ~8128] for non-zero
output.  Acc std ≈ 241 gives ~40% non-zero CReLU activations with mean ~3 — rich, varied
input for FC0 learning from game 1.  Too-small σ (e.g. 5) kills >99% of activations,
causing FT bias drift and mode collapse.

**FC0 weights (σ=4):** each CReLU layer divides by 64 (>>6 shift), so FC0 raw output must
be large enough that FC0_CReLU = FC0_raw/64 gives useful FC1 inputs.  With σ=4 and ~400
active CReLU inputs of mean ~3: FC0 raw std ≈ 240, CReLU ≈ 3.8 — healthy FC1 input.
The passthrough (fwdOut) std ≈ 283 internal units ≈ 5 cp — still quiet.

**PSQT initialisation:** all 8 buckets receive identical pure material values from score.h
(P=5776, N=21776, B=23046, R=34425, Q=69144 internal units; scale = cp × 5776/100).
No piece-square bonuses are included — TDLeaf learns positional adjustments on top of
the material prior.  Own pieces contribute positively; opponent pieces negatively.

Biases are zero-initialised because random N(μ,σ) from an unrelated SF15.1 distribution
adds noise TDLeaf must overcome via its near-cancelling per-game gradient structure.
FT weights already break symmetry across dimensions, so zero FT biases yield varied
SqrCReLU activations from game 1.

Then build two symmetric training binaries pointing at the new file.  Both write to the
shared `.tdleaf.bin` via the `flock`+delta-merge mechanism; every game produces gradient
updates from both sides of the board:

```sh
perl comp.pl train_fresh_a NNUE=1 NNUE_NET=nn-fresh.nnue TDLEAF=1 OVERWRITE
perl comp.pl train_fresh_b NNUE=1 NNUE_NET=nn-fresh.nnue TDLEAF=1 OVERWRITE
```

The interactive `scripts/training_run.py` manager handles this build step automatically.
A read-only binary (`TDLEAF_READONLY=1`) is still useful for Elo testing against fixed
weight snapshots but is no longer part of the default training workflow.

---

## Hooks in Existing Code

| Location | Change |
|----------|--------|
| `src/define.h` | `#ifndef TDLEAF / #define TDLEAF 0 / #endif` |
| `src/chess.h` — `game_rec` | `TDGameRecord td_game` inside `#if TDLEAF` |
| `src/chess.h` — `tree_search` | `int id_scores[TD_ID_HIST]; int id_score_count;` (TD_ID_HIST=4) for ID history |
| `src/nnue.cpp` — `nnue_load()` | Calls `nnue_init_fp32_weights()` inside `#if TDLEAF` |
| `src/search.cpp` — search start | `id_score_count = 0;` reset at the start of each search |
| `src/search.cpp` — after each ID iteration | Appends current `g` to `id_scores[]` ring, increments `id_score_count` |
| `src/main.cpp` — after `ts.search()` | `tdleaf_record_ply()` with root acc + PV + `id_scores` + `id_score_count` |
| `src/main.cpp` — `game.over = 1` sites | `tdleaf_update_after_game()` then `tdleaf_replay()` |
| `src/main.cpp` — `new_game` / `setboard` | `td_game.n_plies = 0` |
| `src/Leaf.cc` | `#if TDLEAF #include "tdleaf.cpp" #endif` |
| `src/comp.pl` | `TDLEAF=1` flag → `-D TDLEAF=1` |

---

## Epoch-Based Replay

After `tdleaf_update_after_game()` applies the live gradient pass, `tdleaf_replay()`
runs `TDLEAF_REPLAY_K` (default 1) additional passes over the last `TDLEAF_REPLAY_BUF_N`
(default 8) completed games stored in a static ring buffer.

### How it works (Flavor A)

1. The completed `TDGameRecord` (accumulator snapshots, feature indices, and leaf
   positions) is pushed into the ring buffer, replacing the oldest entry when full.
2. For each replay pass, iterate over all buffered games oldest-first:
   a. `tdleaf_refresh_scores()` rebuilds each ply's accumulator from the stored leaf
      `position` using `nnue_init_accumulator()` against the **current FT weights**,
      re-enumerates active features, and re-evaluates `score_stm`.  This ensures
      FT gradients during replay are self-consistent with the current network.
   b. `tdleaf_accumulate_game()` computes TD errors and accumulates gradients exactly
      as in the live pass.
3. After all games in the pass are processed, `nnue_apply_gradients()` and
   `nnue_requantize_fc()` are called once, so the next pass's accumulator rebuild
   sees the updated weights.
4. Weights are saved to `.tdleaf.bin` after all K passes complete.

Score-change clipping and ID-stability weighting apply identically in replay passes
(the stored `id_score_variance` values are reused unchanged).

### Ablation results (K vs. Elo gain)

| K | Result |
|---|--------|
| 0 | Baseline — much weaker |
| 1 | **Current default — nearly as strong as K=2, more conservative** |
| 2 | Marginally better than K=1 in initial ablation; reduced to K=1 for stability |
| 3 | Slightly worse than K=2 |
| 6 | Large regression |

### Build flags

| Flag | Default | Effect |
|------|---------|--------|
| `TDLEAF_REPLAY_K` | 1 | Replay passes per game; 0 disables replay |
| `TDLEAF_REPLAY_BUF_N` | 8 | Ring buffer capacity (~4.5 MB × N static BSS) |

---

## Diagnostic Flags

| Flag | Effect |
|------|--------|
| `TDLEAF=1` | Enable all learning code |
| `TDLEAF_READONLY=1` | Load weights but skip gradient updates (inference only) |
| `TDLEAF_CHECK_SCORE=1` | Print direct vs propagated leaf score on every ply |

---

## Self-Play Driver

`scripts/training_run.py` manages the process of creating the necessary binaries
(if needed), specifies a baseline `.nnue` file, and sets up training matches.

`scripts/compare_nnue_learning.py` compares a `.tdleaf.bin`
file against the baseline `.nnue` and shows FC, FT, and PSQT weight statistics.

---

## Horizon Noise Mitigation

### Problem

TDLeaf uses consecutive leaf scores to form TD errors.  When the score changes
dramatically from one ply to the next (e.g., 300+ cp), it is often because the
*next* position falls into a tactical sequence that lies beyond the current search
horizon — a tactic the current position's evaluator cannot see.  Treating that
large score jump as a genuine evaluation signal distorts the gradient: the network
is penalised for correctly evaluating a position it cannot see past.

### Approach 1 — Score-change clipping (TDLEAF_SCORE_CLIP_CP)

When the white-POV score change between consecutive moves exceeds
`TDLEAF_SCORE_CLIP_CP` (default 200 cp), the `d[t+1] - d[t]` contribution to the
eligibility trace is scaled down *proportionally* so the effective change is capped
at 200 cp-equivalent:

```
delta_d  = d[t+1] - d[t]
delta_cp = |score_white[t+1] - score_white[t]|
if delta_cp > TDLEAF_SCORE_CLIP_CP:
    delta_d *= TDLEAF_SCORE_CLIP_CP / delta_cp
e[t] = delta_d + λ * e[t+1]
```

This preserves the *direction* of the update while reducing its magnitude when the
score swing is large.  Set `TDLEAF_SCORE_CLIP_CP` to a very large value (e.g., 1e6)
to disable this approach.

### Approach 2 — Iterative-deepening stability weighting (TDLEAF_ID_VAR_SIGMA2)

The last `TD_ID_HIST = 4` iterative-deepening scores are tracked in
`tree_search::id_scores[]`.  At each ply, their variance is stored in
`TDRecord::id_score_variance` (units: cp²).  During the update, the gradient scale
is multiplied by a soft weight:

```
id_weight = 1 / (1 + id_score_variance / TDLEAF_ID_VAR_SIGMA2)
grad_scale *= id_weight
```

Positions with stable ID scores (low variance) receive full weight; positions whose
score fluctuated across search depths are down-weighted.  `TDLEAF_ID_VAR_SIGMA2`
(default 10 000 cp²) is the reference variance — a position with variance equal to
`TDLEAF_ID_VAR_SIGMA2` receives half weight.  Set `TDLEAF_ID_VAR_SIGMA2` to a very
large value to disable this approach.

### Tuning guidance

| Hyperparameter | Default | Effect of increasing | Effect of decreasing |
|----------------|---------|---------------------|---------------------|
| `TDLEAF_SCORE_CLIP_CP` | 200 cp | Less clipping; more sensitive to large swings | More aggressive attenuation of large score changes |
| `TDLEAF_ID_VAR_SIGMA2` | 10 000 cp² | More tolerant of unstable ID scores | Stronger down-weighting of ID-unstable positions |

Both approaches are active simultaneously by default.  Use the ablation plan in
`docs/TODO.md` to isolate their individual contributions.  A good starting ablation:
run 500 games with each configuration and compare Elo gain per game vs. the baseline.
