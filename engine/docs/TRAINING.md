# Training — Online Learning, Offline Consolidation, and the Hybrid Loop

> For the experiments, abandoned approaches, and superseded hyperparameter values that
> led here, see `docs/history/TRAINING_HISTORY.md`.

## Overview

Leaf has two complementary NNUE training modes that together form the **hybrid loop**:

- **Online learning — TDLeaf(λ).** A temporal-difference reinforcement-learning
  algorithm adapted for minimax search (Baxter, Tridgell, and Weaver, 2000). While the
  engine plays games (self-play or otherwise), it uses the game result and the sequence
  of NNUE evaluations at PV leaf positions to form TD errors, then backpropagates those
  errors through the NNUE network to update weights after every game. See
  [Online Learning — TDLeaf(λ)](#online-learning--tdleafλ) below.
- **Offline consolidation — batch training.** Supervised training on quiet-position
  corpora harvested from root **and** leaf positions in games the engine has already played (`--batch-train`),
  extracting the full information content of those games with multi-epoch, shuffled,
  all-layer gradient descent — something online TD's single, in-order pass through each
  game cannot do. See
  [Offline Consolidation — Batch Training](#offline-consolidation--batch-training) below.
- **The hybrid loop** ties the two together: online self-play generates games and
  learns as it goes; offline consolidation re-extracts everything from those games with
  many epochs of shuffled gradient descent; the consolidated net re-enters online play
  to generate better games than the last cycle. `scripts/train.py` drives one full
  iteration of this loop end to end. See
  [The Hybrid Loop Workflow](#the-hybrid-loop-workflow--scriptstrainpy) below.

Both modes are built into any `NNUE=1 TDLEAF=1` binary:

```sh
perl comp.pl <version> NNUE=1 TDLEAF=1
```

All learning code is gated by `#if TDLEAF`; when `TDLEAF=0` (default) no overhead is added.

> **Protocol support — both xboard/CECP and UCI.**
> Under xboard/CECP the learning hooks run inside `make_move()` and the game
> result arrives via the protocol `result` command. Under UCI (the default for
> `match.py` / fastchess training runs) there is no protocol result command;
> `uci_finish_game()` derives the outcome via `tdleaf_self_adjudicate()` —
> terminal-position checks (mate / stalemate / 50-move / 3-fold / insufficient
> material) with a score-history fallback mirroring cutechess/fastchess
> adjudication defaults. Ambiguous outcomes (e.g. time forfeits) skip learning
> rather than guess.

---

## Quick Start

The easiest way to run one full hybrid-loop iteration against an existing net — online generation, offline
consolidation, and a gauntlet, in one command — is `scripts/train.py`:

```sh
cd learn/
python train.py --tag iter1 --games 400000 --depth 8 \
    --state <net>.tdleaf.bin --recompile --bt-K 220 --bt-threads 8 \
    --gauntlet-epochs --gauntlet Leaf_vclassic_eval
```

See [The Hybrid Loop Workflow](#the-hybrid-loop-workflow--scriptstrainpy) below, and
`SCRIPT_USE.md` for the full option table.

Manual online-training workflow, including initialization of a fresh net:

```sh
# 1. Build a training binary
perl comp.pl train NNUE=1 NNUE_NET=nn-fresh.nnue TDLEAF=1 OVERWRITE

# 2. Initialize a fresh network (default material values baked into PSQT).
#    Use --init-nnue-noprior for uniform 100 cp instead.  Delete any stale companion
#    .tdleaf.bin FIRST — --init-nnue over an existing one merge-saves, not resets.
./Leaf_vtrain --init-nnue --write-nnue nn-fresh.nnue

# 3. Run self-play matches (from run/ or learn/; UCI + fastchess is the default,
#    game outcomes reach the learner via UCI self-adjudication).  Fixed depth with
#    no early adjudication provides the best learning targets. 
python match.py Leaf_vtrain_a Leaf_vtrain_b -n 50000 -c 12 -tc inf --depth1 6 --depth2 6 --no-adjudication
```

The steps above will place the learned delta values in nn-fresh.tdleaf.bin which will be loaded alongside nn-fresh.nnue in future training rounds.   For a statistical and graphical analysis of the learned values use

```sh
# Analyze current state of the learned values
python compare_nnue_learning.py nn-fresh.nnue nn-fresh.tdleaf.bin

# To run another learning round that builds upon the same net, just repeat
# the above match.py command.  Training will pick up from the current
# state of nn-fresh.tdleaf.bin
python match.py Leaf_vtrain_a Leaf_vtrain_b -n 50000 -c 12 -tc inf --depth1 6 --depth2 6 --no-adjudication
```

Any online training match can optionally dump root and leaf node training corpa (.tsv files) to be used during offline training:

```sh
TDLEAF_DUMP_TSV=iter2 python match.py Leaf_vtrain_a Leaf_vtrain_b ...
```

writes two per-process files at game end:  `iter2.<pid>.root.tsv`  and `iter2.<pid>.leaf.tsv`.  See [Offline Consolidation — Batch Training](#offline-consolidation--batch-training) below for how to use training corpa for offline training.

---

## Online Learning — TDLeaf(λ)

### Material Representation — Pure-PSQT

**The bucketed PSQT is the only trainable material channel.** There is no separate
dense piece-value channel and no gauge-anchoring machinery in the current codebase —
both were fully removed (Phase B of the pure-PSQT mainstreaming effort; see
`docs/history/TRAINING_HISTORY.md` for why they existed and why they were deleted
rather than merely disabled). Relevant defaults (`src/define.h`):

- `NNUE_FIXED_PIECE_VALUES 1` — search keeps classical `value[]`
  (`P=100 N=380 B=400 R=600 Q=1200`). `nnue_extract_piece_values()` still computes the
  PSQT-implied values for the startup banner, but **report-only**; it does not overwrite
  `value[]`.

Seed a run with `--init-nnue-classical` (classical material + phase-interpolated
piece-square tables baked into PSQT) or `--init-nnue-noprior` (uniform 100 cp).

#### Why no gauge anchoring is needed

The old gauge machinery (pin + mean-centering + dw-centering + slot-mean recentering)
existed because PSQT and a separate dense `piece_val` channel **redundantly** encoded
material level — a gauge null direction that the multi-writer merge protocol could
amplify geometrically. With `piece_val` gone there is exactly **one** material
channel: no null direction, nothing for the merge to amplify. Absolute scale is
loss-anchored via `TDLEAF_K = 220 cp` under outcome-dominated λ-return targets.

**Validation (multi-writer merge smoke test, 2026-07-07, run before the removal was
finalized):** the pure-PSQT run was effectively single-lineage, so the one thing 500k
games never exercised was the concurrent merge path that historically drove gauge
blow-ups. 8000 games of two concurrent writers sharing one `.tdleaf.bin` (depth 6,
balanced self-play) showed the extracted pawn value pinned at 100 cp across every
merge cycle (sub-cp drift, below rounding) and FC bias means converging and
plateauing rather than accumulating. No accelerating drift → the merge path needs no
restored centering.

Measured *cosmetic* drift over long training: extracted pawn 100→107 cp per 500k online
games (+7 cp more per 4 offline epochs) — slow, and cosmetic because search piece values
are fixed by `NNUE_FIXED_PIECE_VALUES`. Note `TDLEAF_SCORE_CLIP_PAWNS` is in units of
`max(value[PAWN], 100 cp)`; with fixed classical values this stays 100 cp exactly, so the
clip does **not** stretch as extracted pawn drifts.

> **⚠️ Do NOT freeze PSQT.** PSQT freezing was tried and fails catastrophically,
> ~−200 Elo — do not re-attempt it. Full failure-mechanism writeup in
> `docs/history/TRAINING_HISTORY.md`.

> The retired dense-piece-value channel and its gauge-anchoring machinery — including
> why it was needed and how it worked — are documented in full in
> `docs/history/TRAINING_HISTORY.md`. Both were deleted from the codebase (not merely
> disabled by a flag) in Phase B of the pure-PSQT mainstreaming effort.

---

### Algorithm

For a game of T half-moves:

- `d_t` = sigmoid of the **NNUE static evaluation at the PV leaf position** at ply t,
  from White's perspective:
  `d_t = 1 / (1 + exp(-score_white_t / K))`, K = 220 cp.
- `z` = game result from White's perspective: 1.0 = White wins, 0.5 = draw, 0.0 = Black wins.

**Temporal difference errors (backward view):**

```
e_{T-1} = z - d_{T-1}
e_t     = clip(d_{t+1} - d_t) + λ * e_{t+1}     for t = T-2 … 0
```

where `clip(d_{t+1} - d_t)` applies proportional scaling when the white-POV score
change between consecutive moves exceeds `TDLEAF_SCORE_CLIP_PAWNS × max(value[PAWN], 100 cp)`
— see [Horizon Noise Mitigation](#horizon-noise-mitigation) below.

**Weight update (gradient ascent on prediction accuracy):**

```
Δw = α * Σ_t  e_t * ∇_w d_t
```

where `∇_w d_t = d_t * (1 - d_t) / K * ∇_w score_t`.

Defaults: `λ = 0.985` **per game-ply** (the trace applies `λ^dply` where `dply` is
the game-ply gap between consecutive records — 2 in the two-process harness, since
the engine records only its own moves; 1 under internal self-play, so one λ expresses
the same real-game horizon in both modes), `K = 220 cp`. 0.985 (down from an earlier
`√0.98 ≈ 0.98995`) was chosen from offline batch-training convergence testing — close
enough to the prior default that the same constant is now used everywhere, online and
offline alike. For the
calibration methodology and the reusable workflow to recalibrate either value in the
future, see [Recalibrating K/λ](#recalibrating-kλ-reproducing-the-analysis) below; for
how the current values were derived and revised over time, see
`docs/history/TRAINING_HISTORY.md`.
Gradient updates use Adam with per-weight LR decay; see [Adam Optimizer](#adam-optimizer) below.

**Key design choice:** `d_t` is computed from `nnue_evaluate()` (direct static eval of the
PV leaf), not from the search score propagated from the root. This ensures the sigmoid
value and the forward-pass gradient are computed from the same NNUE evaluation of the same
position, making the gradient self-consistent.

---

### Scope: All NNUE Weights Are Trained

| Layer | Parameters | Notes |
|-------|-----------|-------|
| FC0 weights/biases | 1,024×16 int8 + 16 int32, ×8 stacks | Quantized int8, float shadow |
| FC1 weights/biases | 32×32 int8 + 32 int32, ×8 stacks | Same |
| FC2 weights/bias   | 32 int8 + 1 int32, ×8 stacks | Same |
| FT biases          | 1,024 int16 | Dense update; static float shadow (4 KB) |
| FT weights         | 22,528×1,024 int16 | Sparse update; float shadow on heap (~92 MB) |
| PSQT weights       | 22,528×8 int32 | Sparse update; float shadow (~720 KB) |

FT weights and PSQT are updated sparsely: only the ~30–60 feature rows active at each
leaf position are touched. `ft_dirty[FT_INPUTS]` tracks which rows received gradient
during the game; only dirty rows are scanned in `nnue_apply_gradients`.

FT biases are updated densely every game (all 1,024 values): the gradient is the sum of
`g_acc[persp][d]` across both perspectives.

There used to be a fourth, densely-updated material channel (`piece_val[6]`, one value
per piece type) alongside PSQT, but it — and the gauge-anchoring machinery it needed —
was fully removed from the codebase; see
[Material Representation — Pure-PSQT](#material-representation--pure-psqt) above and
`docs/history/TRAINING_HISTORY.md` for why.

---

### Adam Optimizer

`nnue_apply_gradients()` uses AdamW with a fixed learning rate (constant after warmup).

#### Algorithm

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
to 1). bc2 (β₂=0.999) is **always** applied: 0.999²⁰ ≈ 0.98, so bc2 = 0.02 at
cnt=20 — skipping would give ~7× oversized steps. FT RMSProp retains global bc2
(from `t_adam`) because sparse features (~8 updates/5000g) need the growing global
correction.

#### Per-Layer Configuration

LRs follow the "0.001 × median(|w|)" heuristic measured on `nn-ad9b42354671`. Median
weight magnitudes (per section): FC0 weights ≈ 4, FC1 weights ≈ 9, FC2 weights ≈ 68,
FC biases ≈ 1500 (int32 scale), FT weights ≈ 16, FT biases ≈ 51, PSQT ≈ 13343.

| Layer | Update Rule | LR0 | Notes |
|-------|-------------|-----|-------|
| FC0/FC1 weights | Full Adam | `TDLEAF_ADAM_LR0 = 0.005` | Float shadow clamped to ±127 after each update |
| FC2 weights | Full Adam | `TDLEAF_ADAM_FC2_LR0 = 0.07` | Separate LR — 32→1 fan-in gives FC2 weights ~14× the leverage and ~14× the median magnitude of FC0/FC1 |
| FC0/FC1/FC2 biases  | Full Adam | `TDLEAF_ADAM_FC_BIAS_LR0 = 1.5` | Int32 scale; separate from int8-scale FC weights |
| FT weights | RMSProp (per-weight v, no m) | `TDLEAF_ADAM_FT_LR0 = 0.015` | Sparse update; per-session warmup damps first 100 steps |
| FT biases  | Full Adam | `TDLEAF_ADAM_FT_BIAS_LR0 = 0.02` | Hedged below the median-rule value to limit dying-ReLU risk from update-frequency asymmetry |
| PSQT       | Full Adam | `TDLEAF_ADAM_PSQT_LR0 = 13.0` | Int32 scale; the sole material channel |

#### Why a Separate PSQT LR0?

Adam normalises gradient magnitude: the effective per-step size in weight-space is
approximately ±LR0 per update, independent of the raw gradient magnitude. PSQT
weights are at int32 scale (median |w| ≈ 13343) while FC weights are at int8 scale
(median |w| ≈ 5 for FC0/FC1) — a ratio of ~2600×. Using the same LR0 for both caused
PSQT to change negligibly relative to its baseline scale. `TDLEAF_ADAM_PSQT_LR0 = 13.0`
keeps PSQT's per-step fractional change matched to the other sections.

#### Why a Separate FT Bias LR0?

FT biases are updated densely (~200 times per game), while FT weights are updated
sparsely (~8 per 5000 games for a typical feature row). Without a hedged LR, biases
race ahead of weights: they drift strongly negative before FT weights have learned
useful features, suppressing SqrCReLU activations and causing dying-ReLU.
`TDLEAF_ADAM_FT_BIAS_LR0 = 0.02` is hedged below the 0.001×median(|w|) value
(median |w| ≈ 51 would suggest 0.05) to limit that drift without freezing adaptation.

#### Adam step clipping

`TDLEAF_ADAM_STEP_CLIP = 30.0` bounds the unit-less Adam step
`|m_hat / sqrt(v_hat)|` (or `|g / sqrt(v_hat)|` for FT's RMSProp path) before the
LR multiply. Uniform across categories — Adam is scale-normalised by design, so a
single threshold catches the rare-feature pathology where a low running `v` makes a
normal gradient produce an oversized parameter change. Per-category max-step and
clip counts are logged when built with `-D TDLEAF_LOG_STEP_CLIPS=1`.

#### Per-session FT warmup

Beyond the global `TDLEAF_ADAM_WARMUP = 50` (keyed on the persisted `t_adam`, so
it fires only on the very first session), FT updates use an additional
per-session ramp `TDLEAF_FT_SESSION_WARMUP = 100`. This damps FT-weight updates
during the `v_ft_w` accumulation phase at every restart, regardless of whether `v`
was loaded from disk — protecting freshly-zeroed (cold) FT rows from oversized
first-batch steps.

#### Why Float-Shadow Clamping for FC0/FC1/FC2?

FC0, FC1, and FC2 all use int8 quantized inference weights clamped to ±127 on
requantization. Without a matching clamp on the float shadow, Adam can push
`w_f32` arbitrarily beyond ±127 while the inference weight is stuck at the boundary.
These "zombie weights" accumulate gradient updates with zero effect on the network.
After each weight update, `w_f32 = clamp(w_f32, −127, 127)` keeps the float shadow
aligned with the int8 inference space. The clamp is applied uniformly to FC0, FC1,
and FC2.

#### Persistent v and m

**Second-moment arrays (`v`) and `t_adam`** are persisted to `.tdleaf.bin` (v6+) so that
Adam's gradient scale knowledge survives across sessions. Without this, every restart
cold-starts with `v=0`, causing the first ~20 Adam steps to use poorly-scaled learning
rates until v accumulates a reliable variance estimate.

**First-moment arrays (`m`, momentum)** are persisted to `.tdleaf.bin` (v7+). Persisting
m eliminates a slow or negative Elo trend observed at the start of every training session:
without m, the optimizer has no directional bias and early gradient noise can push weights
the wrong way for thousands of games before a consistent signal accumulates. With m
restored, the optimizer continues in the learned gradient direction from the previous session.

**Multi-writer merge for v:** When multiple concurrent training instances save to the
same `.tdleaf.bin`, v arrays are merged per-element using `max(v_file, v_local)`. This
is conservative: a too-large v only slightly slows learning (larger denominator in
Adam's update), while a too-small v causes instability. The max-merge is safe because
v is always non-negative and represents gradient magnitude².

**Multi-writer merge for m:** m arrays are merged per-element using the element-wise
average `(m_file + m_local) / 2`. Workers seeing the same gradient direction reinforce
each other; conflicting directions reduce toward zero — appropriate, since uncertainty
about gradient direction should produce a smaller step rather than a random-direction step.

**FT weight v and m are NOT persisted** — v is ~92 MB (too large), and FT weight updates
are sparse enough (~3–50 per weight over 190k games) that v barely converges before process
restart. FT weights use RMSProp (no m array), so FT weight m does not exist.

**FT bc2 cold-start fix:** Because `v_ft_w` is zeroed at every startup, using the global
`t_adam` (persisted and large) for FT's bc2 would give `bc2≈1` while `v=0`, producing
`sv = sqrt(0/1+ε) = ε` and step sizes ~31× too large on the first batch — destroying
previously learned FT weights. The fix: FT uses `min(t_adam, t_ft_session)` for bc2,
where `t_ft_session` is a session-local counter starting at 0 on every process launch.
This gives standard Adam bias correction for the fresh v regardless of how large t_adam is,
keeping first-step magnitude at the intended ±LR0.

The per-weight `cnt` arrays (which are also persisted) track update history for
per-weight bias correction and monitoring.

#### Hyperparameters (`src/tdleaf.h`)

| Constant | Value | Notes |
|----------|-------|-------|
| `TDLEAF_ADAM_LR0` | 0.005 | FC0/FC1 weights (int8 scale; median \|w\| ≈ 5) |
| `TDLEAF_ADAM_FC2_LR0` | 0.07 | FC2 weights (int8 scale; median \|w\| ≈ 68 — final 32→1 layer) |
| `TDLEAF_ADAM_FC_BIAS_LR0` | 1.5 | FC0/FC1/FC2 biases (int32 scale; median ≈ 1500 across stacks) |
| `TDLEAF_ADAM_FT_LR0` | 0.015 | FT weights (sparse; int16 scale; median \|w\| ≈ 16) |
| `TDLEAF_ADAM_FT_BIAS_LR0` | 0.02 | FT biases (hedged below 0.001×median to limit dying-ReLU risk) |
| `TDLEAF_ADAM_PSQT_LR0` | 13.0 | PSQT (int32 scale; median ≈ 13343) |
| `TDLEAF_ADAM_BETA1` | 0.9 | First-moment decay (FC weights/biases, FT biases, PSQT) |
| `TDLEAF_ADAM_BETA2` | 0.999 | Second-moment decay (all layers) |
| `TDLEAF_ADAM_EPS` | 1e-8 | Numerical floor in denominator |
| `TDLEAF_ADAM_STEP_CLIP` | 30.0 | Bound on unit-less Adam step `\|m_hat/sqrt(v_hat)\|` before LR multiply; uniform across categories |
| `TDLEAF_ADAM_WARMUP` | 50 | Linear LR warmup over first N Adam steps; keyed on persisted `t_adam` (first session only) |
| `TDLEAF_FT_SESSION_WARMUP` | 100 | Per-session FT LR ramp over first N steps of each restart; keyed on `t_ft_session` (not persisted) |
| `TDLEAF_BATCH_SIZE` | 8 | Mini-batch: accumulate gradients across N games before each Adam step |
| `TDLEAF_WEIGHT_DECAY` | 1e-4 | AdamW decoupled weight decay coefficient (FC + FT weights only) |
| `TDLEAF_GRAD_CLIP_NORM` | 1.0 | Global gradient L2 norm clip threshold; 0 = disabled |
| `TDLEAF_REPLAY_LR_SCALE` | 0.3 | Multiplicative LR scale applied during replay-pass Adam steps |
| `TDLEAF_MIN_PLIES` | 8 | Skip games with fewer recorded TDLeaf plies than this |
| `TDLEAF_MIN_PLIES_REP` | 40 | Skip 3-fold repetition draws with fewer plies than this |
| `TDLEAF_SCORE_CLIP_PAWNS` | 1.0 | Clip threshold for inter-ply score-change attenuation: `score_clip_cp = SCORE_CLIP_PAWNS × max(value[PAWN], 100 cp)`. With PAWN fixed, the threshold is effectively constant at 100 cp. Set to a large value to disable |
| `TDLEAF_ID_VAR_SIGMA2` | 10000 cp² | Iterative-deepening stability weight: `id_weight = 1 / (1 + id_score_variance / SIGMA2)`. Larger values are more tolerant of ID score instability. Set to a large value to disable |

Runtime LR sweep via env vars `TDLEAF_LR_{FC,FC2,FC_BIAS,FT,FT_BIAS,PSQT}` (each a
multiplier on the corresponding `TDLEAF_ADAM_*_LR0`, default 1.0) and
`TDLEAF_FREEZE_PASSTHROUGH=1` (holds the FC0 passthrough row fixed).

Set `TDLEAF_BATCH_SIZE = 1` to restore per-game Adam steps.
Set `TDLEAF_ADAM_WARMUP = 0` to disable warmup.
Set `TDLEAF_WEIGHT_DECAY = 0.0` to disable weight decay.
Set `TDLEAF_GRAD_CLIP_NORM = 0.0` to disable gradient clipping.
Set `TDLEAF_ADAM_STEP_CLIP` to a very large value to effectively disable step clipping.

---

### Mini-Batch Gradient Accumulation

By default (`TDLEAF_BATCH_SIZE=8`), gradients are accumulated across 8 games before
a single Adam step is applied. This gives the optimizer a more reliable gradient signal
per step, reducing single-game noise that otherwise causes Adam's first moment to chase
stochastic fluctuations.

#### How it works

1. `tdleaf_update_after_game()` calls `tdleaf_accumulate_game()` on every game but only
   calls `nnue_apply_gradients()` + `nnue_requantize_fc()` + save when the batch counter
   reaches `TDLEAF_BATCH_SIZE`.
2. `tdleaf_replay()` always pushes the completed game into the ring buffer, but replay
   passes only run on batch boundaries (when the live batch was just applied).
3. `tdleaf_flush_batch()` applies any pending partial batch at session end (program exit
   or weight export), preventing gradient loss.

#### Trade-offs

- **Pro:** each Adam step uses ~8× more gradient data, improving signal-to-noise ratio.
- **Pro:** file I/O reduced by ~8× (one write per batch instead of per game).
- **Con:** weight updates are delayed by up to `BATCH_SIZE-1` games (negligible in practice;
  the delay is a few seconds at typical game durations).

Set `TDLEAF_BATCH_SIZE = 1` to restore the original per-game update behaviour.

---

### LR Warmup

A linear warmup ramps the learning rate from 0 to its full value over the first
`TDLEAF_ADAM_WARMUP` Adam steps (default 50). The effective LR at step `t` is:

```
lr_effective = min(1.0, t / WARMUP) × lr_decay(LR0, cnt)
```

#### Motivation

Adam's bias correction handles cold-start `m` and `v` mathematically, but in practice
the first few steps can produce disproportionately large effective step sizes because
`v` hasn't accumulated a reliable variance estimate. For rarely-visited feature rows
that may not receive gradient for hundreds of games, the first update can overshoot.
Warmup smooths this transition at essentially zero implementation cost.

Set `TDLEAF_ADAM_WARMUP = 0` to disable warmup.

---

### Weight Persistence — `.tdleaf.bin` (version 12)

Saved at `{exec_path}<network>.tdleaf.bin`. The writer always emits v12; the loader
accepts v2–v12, reading and discarding any dropped legacy sections to stay aligned
before upgrading the file to v12 on next save. Current (v12) format:

```
[magic(4) + version(12)]
[v10+: nnue_content_hash(4)  — FNV-1a over source .nnue FT weight bytes]
[8 FC stacks: per-layer float32×128 weights/biases + uint32 counts]
[n_ft_rows(4 bytes)]
[per dirty row: fi(4) + ft_w[1024]×128 as float32[1024] + ft_cnt[1024] as uint32[1024]
                      + psqt_w[8]×128 as float32[8]    + psqt_cnt[8]  as uint32[8]]
[FT bias section (v4+): ft_bias[1024]×128 as float32[1024] + ft_bias_cnt[1024] as uint32[1024]]
[Adam v section (v6+):
  t_adam as uint32
  8 FC stacks × { v_l0_b, v_l0_w, v_l1_b, v_l1_w, v_l2_b, v_l2_w } as raw float32
  v_ft_bias[1024] as float32
  n_psqt_v_rows as uint32
  per dirty row: fi(4) + v_psqt[8] as float32[8]
]
[Adam m section (v7+):
  8 FC stacks × { m_l0_b, m_l0_w, m_l1_b, m_l1_w, m_l2_b, m_l2_w } as raw float32
  m_ft_bias[1024] as float32
  n_psqt_m_rows as uint32
  per dirty row: fi(4) + m_psqt[8] as float32[8]
]
[Sparse FT v section (v8+):
  n_ft_v_rows as uint32
  per row with non-zero v: fi(4) + v_ft[1024] as float32[1024]
]
```

**v12 dropped four sections that earlier versions carried** — all belonged to the
retired dense-piece-value channel and its gauge-anchoring machinery (see
`docs/history/TRAINING_HISTORY.md`): the piece-value weight section (v5+), piece-value
Adam v (v6+) and m (v7+) inside their respective sections, and the v11 PSQT
init-slot-means block. Loading a pre-v12 file reads and discards these bytes rather
than skipping the version outright, so old files upgrade transparently on next save.

Weight values are stored at 128× resolution (divide by 128 on load) to preserve
sub-integer drift across sessions. Adam v and m arrays are stored as raw float32 (no scaling).
Update counts enable weighted averaging of concurrent training runs.

The Adam v section persists gradient scale across sessions so Adam doesn't cold-start
with v=0. The Adam m section persists gradient direction so the optimizer continues in
the learned direction from the previous session — eliminating the slow/negative Elo
trend at the start of each new training run. FT weight v/m (~92 MB each) are NOT
persisted — too large, and FT weights use RMSProp (no m array).

Versions 2–11 are accepted on load with appropriate defaults; a notice is printed for
any version upgrade.

#### Source-.nnue content hash (v10+)

The v10 header stores an FNV-1a fingerprint of the source `.nnue` FT weight bytes,
computed once at load/init time (see `nnue_update_content_hash()` in `nnue.cpp`).
On load, if the stored hash does not match the hash of the currently-loaded `.nnue`,
the file is refused — preventing accidental pairing of weight deltas with the wrong
baseline network. The same check guards the merge-read phase of every save, so a
worker running against a different `.nnue` cannot corrupt the file.

V5–V9 files have no hash and are accepted without a check; saving promotes them to
v10 carrying the current `.nnue`'s content hash. `--init-nnue` fingerprints the
freshly-initialised FT weights so the companion `.tdleaf.bin` is born consistent.
`scripts/train.py` also uses this hash as a pairing pre-flight before promoting a
`.tdleaf.bin` to the live training state — see
[The Hybrid Loop Workflow](#the-hybrid-loop-workflow--scriptstrainpy) below.

---

### Concurrent File Access

Multiple Leaf instances (e.g. several parallel self-play games) can share a single
`.tdleaf.bin` safely via POSIX file locking and delta-based merging.

The save path uses section-level buffered I/O with cached dirty-row bitmaps and 4 MB
stream buffers. Measured on depth-6 training matches, total learning overhead vs. a
no-learning control is ~+13% wall clock; at depth 8 it amortizes to ~10%. Saves that
find the lock busy are deferred with deltas retained.

#### Design

**Problem:** If two instances both read the file, apply their gradient updates to their
in-memory weights, and then write back, the second write silently overwrites the first
instance's changes.

**Solution:** Each instance tracks only its own accumulated weight *deltas* since the last
file write (not the full weight values). On each write:

1. Acquire `LOCK_EX` on a companion `.tdleaf.bin.lock` file.
2. Re-read the current `.tdleaf.bin` from disk.
3. Merge: `merged_value = file_value + our_delta` for every entry.
4. Update the in-memory float shadows to the merged values; zero the deltas.
5. Write the merged content to `.tdleaf.bin.tmp`.
6. `rename(.tmp, .tdleaf.bin)` — atomic on POSIX filesystems.
7. Release the lock (close the lock-file fd).

Reads use `LOCK_SH` (multiple simultaneous readers allowed; blocked only during a write).

#### Implementation Details

**Lock file:** `.tdleaf.bin.lock` is a separate companion file so locking survives the
atomic `rename()` of the main file. The lock is held only during the re-read/write cycle
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

**Delta count arrays** (parallel to weight deltas, track update counts since last sync):

| Array | Size | Contents |
|-------|------|----------|
| `delta_l0_w_cnt[8][1024×16]` | 512 KB | FC0 weight delta counts |
| `delta_l0_b_cnt[8][16]` | 512 B | FC0 bias delta counts |
| `delta_l1_w_cnt[8][32×32]` | 256 KB | FC1 weight delta counts |
| `delta_l1_b_cnt[8][32]` | 1 KB | FC1 bias delta counts |
| `delta_l2_w_cnt[8][32]` | 1 KB | FC2 weight delta counts |
| `delta_l2_b_cnt[8]` | 32 B | FC2 bias delta counts |
| `delta_ft_bias_cnt[1024]` | 4 KB | FT bias delta counts (static) |
| `delta_psqt_cnt` (heap) | ~720 KB | PSQT delta counts |

Deltas (both weight and count) are zeroed after each successful write (either on first
write or after a re-read-merge write). `nnue_load_fc_weights()` also zeros all deltas
to establish a clean baseline.

**Update-count merging:** counts use additive merge: `merged = file_count + delta_count`.
Each instance's delta count tracks only updates since the last file sync, so adding it
to the file's count correctly accumulates across concurrent instances and training cycles.
FT weight counts remain max-based (`max(file, ours)`) because per-weight delta counts
would require 92 MB; FT counts use global bias correction so exact counts are less
critical.

#### Usage with match.py

The recommended way to run training is via `scripts/training_run.py`, which handles
binary compilation, opponent rotation, checkpointing, .tdleaf.bin snapshots, and
startup backups automatically. For manual control:

When running parallel self-play via `match.py` (fastchess by default, `cutechess-cli`
via `--driver=cutechess`) with multiple concurrent TDLEAF instances, add `--wait MS`
to insert a pause between games. This reduces
contention on the `.tdleaf.bin.lock` file and gives each instance time to complete its
write cycle before the next game starts.

For symmetric self-play (both engines learning from the same `.tdleaf.bin`), use
`--no-repeat` to play each opening once (maximising position diversity) rather than the
default which replays each opening twice for color balance:

```sh
# Symmetric self-play — each opening once, maximum diversity
python3 match.py Leaf_vtrain_a Leaf_vtrain_b -n 200 -c 4 --wait 500 --no-repeat

# Asymmetric (learner vs. read-only reference) — default repeat for color balance
python3 match.py Leaf_vtrain_a Leaf_vtrain_ro -n 200 -c 4 --wait 500
```

---

### Initialization

**Default (fine-tuning):** When no `<network>.tdleaf.bin` is found, the default network
for that executable, `<network>.nnue`, is used as a starting point and training proceeds
as gradient updates from that starting point. If `<network>.tdleaf.bin` is present, corresponding
to previous training, these values are loaded as updates to the default at startup. Any
additional training will further update the `<network>.tdleaf.bin` file.

**Training from scratch:** Use `--init-nnue --write-nnue <file>` to create a randomly
initialised `.nnue` with no source file required:

```sh
perl comp.pl init_nnue NNUE=1 TDLEAF=1
./Leaf_vinit_nnue --init-nnue --write-nnue nn-fresh.nnue
```

Three priors are available — matching `train.py`'s `--init-nnue [material|classical|noprior]`
choices (engine flags `--init-nnue`, `--init-nnue-classical`, and `--init-nnue-noprior`
respectively; see [The Hybrid Loop Workflow](#the-hybrid-loop-workflow--scriptstrainpy)):
plain classical material values, classical material with phase-interpolated
piece-square tables baked in, or a uniform 100 cp PSQT with no material or positional
prior at all.

A variant `--init-nnue-noprior` initialises **all** piece PSQT slots at 100 cp (symmetric
own=+V, enemy=−V; P=N=B=R=Q=100 cp) instead of classical material values — materially
blind from move 1, with N/B/R/Q differentiating from that 100 cp baseline via PSQT
updates during training, while `value[PAWN]` stays anchored at 100 cp throughout
(preserves SEE / material-accounting semantics). The interactive
`scripts/training_run.py` offers all of these options when initialising a new network.

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

**Design philosophy: start quiet, let TDLeaf build structure from signal.** Initial NNUE
positional output should be near zero so that classical material dominates early play
(reasonable game quality from game 1). The network gradually grows its influence as
TDLeaf learns real patterns from self-play. Zero means (He/Kaiming principle) are the
correct starting point — non-zero means from a trained network are endpoints, not priors.

**FT weights (σ=44):** ~30 features active per position, so accumulator std ≈ √30 × 44 ≈ 241.
CReLU divides by 64 (>>6 shift), so the accumulator needs values in [0, ~8128] for non-zero
output. Acc std ≈ 241 gives ~40% non-zero CReLU activations with mean ~3 — rich, varied
input for FC0 learning from game 1. Too-small σ (e.g. 5) kills >99% of activations,
causing FT bias drift and mode collapse.

**FC0 weights (σ=4):** each CReLU layer divides by 64 (>>6 shift), so FC0 raw output must
be large enough that FC0_CReLU = FC0_raw/64 gives useful FC1 inputs. With σ=4 and ~400
active CReLU inputs of mean ~3: FC0 raw std ≈ 240, CReLU ≈ 3.8 — healthy FC1 input.
The passthrough (fwdOut) std ≈ 283 internal units ≈ 5 cp — still quiet.

**PSQT initialisation:** all 8 buckets receive identical pure material values from score.h
(P=5776, N=21776, B=23046, R=34425, Q=69144 internal units; scale = cp × 5776/100).
No piece-square bonuses are included — TDLeaf learns positional adjustments on top of
the material prior. Own pieces contribute positively; opponent pieces negatively.

Biases are zero-initialised because random N(μ,σ) from an unrelated SF15.1 distribution
adds noise TDLeaf must overcome via its near-cancelling per-game gradient structure.
FT weights already break symmetry across dimensions, so zero FT biases yield varied
SqrCReLU activations from game 1.

---

### Corpus Dumping (TDLEAF_DUMP_TSV)

Any TDLEAF build can emit offline-training corpora as a by-product of play — the
raw material for the offline consolidation mode documented in
[Offline Consolidation — Batch Training](#offline-consolidation--batch-training) below.
When the env var `TDLEAF_DUMP_TSV=<prefix>` is set, each engine process writes two
files at game end (append mode, per-process, format
`fen \t cp \t result \t ply \t depth \t gid \t endply`, cp/result white-POV;
`endply` = the game's final recorded ply, used by the batch trainer's
distance-decayed result weight):

| File | Position | `cp` label | `depth` column |
|------|----------|-----------|----------------|
| `<prefix>.<pid>.root.tsv` | played root of each recorded ply | root **search** score (search-amplified) | achieved ID depth |
| `<prefix>.<pid>.leaf.tsv` | PV leaf of each recorded ply | leaf **static** eval (self-distillation — the outcome label carries the signal) | 0 |

Quietness filters (both files): |static − search| ≤ `TDLEAF_DUMP_QUIET_CP`
(default 60 cp — unresolved tactics show up as static-vs-search disagreement) and
|cp| ≤ `TDLEAF_DUMP_MAX_CP` (default 1500). Only games that feed the TD update
are dumped, with the same outcome labels. The batch trainer gives `depth == 0`
records their own outcome-weight ceiling (`--bt-leaf-lambda`, default = the root
λ), so root and leaf corpora mix freely in one run.

When dumping is enabled, `tdleaf_record_ply` additionally snapshots the root
position and computes its static eval (one extra `nnue_evaluate` per recorded
ply; skipped entirely when the env var is unset).

**Generate-only play (`TDLEAF_FREEZE=1`, env):** freezes the weights at
runtime — games are recorded and dumped exactly as in a learning run, but
gradient accumulation, weight application, and all `.tdleaf.bin` writes are
skipped, so the corpus labels come from a fixed net.  This is the tool for
generate-only hybrid-loop iterations (see Parts 3–4 of
`Online_Learning_Investigation.md`).  Do **not** use the compile-time
`TDLEAF_READONLY=1` flag for this: it compiles out the record/update hooks
entirely, so a READONLY binary plays with frozen weights but **dumps no
corpus** (it exists for rating/inference binaries that load a `.tdleaf.bin`
pair).  Note the name: `TDLEAF_READONLY` is *only* a compile flag — exporting
it as an env var does nothing.

Frozen-pair determinism caveat: two identical frozen engines replay the exact
same game from each opening every time it comes up (including a color-swapped
repeat — the swap changes nothing when both sides are the same net), so the
unique-game count is capped at the opening-book size.  `train.py` guards
this automatically: generation always runs `--no-repeat`, it warns when
`--games` exceeds the book line count, and corpus assembly always drops
duplicate rows (which would otherwise straddle the trainer's by-game
train/val split and leak validation).  Still cap `--games` at the book size —
recycled openings are wasted generation compute even though dedup protects
the corpus.

---

### Outcome-Imbalance Drift — Guidance

Sustained training at a score far from 50% — e.g. vs. a fixed stronger opponent — can
collapse the net: the state-independent component of the TD outcome term is absorbed by
whatever channel can represent a constant (FC output biases first, then other
constant-capable channels), and positive feedback (pessimistic eval → worse play → more
losses) makes the drift monotone. Self-play is immune (wins/losses balance by
construction) and balanced play actively reverses accumulated drift. Diagnosed in
detail — including the incident that surfaced it and the mechanism analysis — in
`docs/history/TRAINING_HISTORY.md`.

**Guidance:**
- Keep the aggregate training score near 50% — mix stronger opponents with
  equal/weaker ones (opponent rotation in `training_run.py`); a ~75/25
  self-play/classic diet is stable without mirror segments.
- Fixed-opponent games harvest one learner trajectory per game vs. self-play's
  two — their value is diversity, not volume.
- Monitor the drift canaries between checkpoints with
  `scripts/diff_tdleaf_checkpoints.py`: per-stack fc2_bias, stack-0
  fc2_w[13]/[27], FC0 passthrough-row mean. Bias drift is visible within ~100k
  games, long before Elo shows it.
- A principled structural fix (outcome-baseline subtraction,
  `e'[t] = e[t] − EMA(engine-POV mean e)`, a no-op in balanced regimes) is
  designed but not implemented; adopt it if imbalanced-opponent training becomes
  a first-class mode.

---

### Horizon Noise Mitigation

#### Problem

TDLeaf uses consecutive leaf scores to form TD errors. When the score changes
dramatically from one ply to the next (e.g., 300+ cp), it is often because the
*next* position falls into a tactical sequence that lies beyond the current search
horizon — a tactic the current position's evaluator cannot see. Treating that
large score jump as a genuine evaluation signal distorts the gradient: the network
is penalised for correctly evaluating a position it cannot see past.

#### Approach 1 — Score-change clipping (TDLEAF_SCORE_CLIP_PAWNS)

The threshold is computed dynamically as
`score_clip_cp = TDLEAF_SCORE_CLIP_PAWNS × max(value[PAWN], 100 cp)`
(default `TDLEAF_SCORE_CLIP_PAWNS = 1.0`, i.e. 100 cp at the classical pawn value).
Under the current pure-PSQT default, `value[PAWN]` is fixed at 100 cp by
`NNUE_FIXED_PIECE_VALUES` (search keeps the classical value regardless of training), so
the threshold is effectively a constant `1.0 × 100 = 100 cp`. The `max(..., 100 cp)`
floor is retained as belt-and-braces against any future configuration that lets
`value[PAWN]` sag below the classical default; under `--init-nnue-noprior` the flat-100
cp PSQT init also keeps `value[PAWN]` anchored from move 1.
When the white-POV score change between consecutive moves exceeds `score_clip_cp`, the
`d[t+1] - d[t]` contribution to the eligibility trace is scaled down
*proportionally* so the effective change is capped at the threshold:

```
delta_d  = d[t+1] - d[t]
delta_cp = |score_white[t+1] - score_white[t]|
if delta_cp > score_clip_cp:
    delta_d *= score_clip_cp / delta_cp
e[t] = delta_d + λ * e[t+1]
```

This preserves the *direction* of the update while reducing its magnitude when the
score swing is large. Set `TDLEAF_SCORE_CLIP_PAWNS` to a very large value (e.g.,
1e4) to effectively disable this approach.

#### Approach 2 — Iterative-deepening stability weighting (TDLEAF_ID_VAR_SIGMA2)

The last `TD_ID_HIST = 4` iterative-deepening scores are tracked in
`tree_search::id_scores[]`. At each ply, their variance is stored in
`TDRecord::id_score_variance` (units: cp²). During the update, the gradient scale
is multiplied by a soft weight:

```
id_weight = 1 / (1 + id_score_variance / TDLEAF_ID_VAR_SIGMA2)
grad_scale *= id_weight
```

Positions with stable ID scores (low variance) receive full weight; positions whose
score fluctuated across search depths are down-weighted. `TDLEAF_ID_VAR_SIGMA2`
(default 10000 cp², equivalent to a 100 cp std-dev reference) is the reference
variance — a position with variance equal to `TDLEAF_ID_VAR_SIGMA2` receives half
weight. Set `TDLEAF_ID_VAR_SIGMA2` to a very large value to disable this approach.

#### Tuning guidance

| Hyperparameter | Default | Effect of increasing | Effect of decreasing |
|----------------|---------|---------------------|---------------------|
| `TDLEAF_SCORE_CLIP_PAWNS` | 1.0 (with 100 cp floor on `value[PAWN]`) | Less clipping; more sensitive to large swings | More aggressive attenuation of large score changes |
| `TDLEAF_ID_VAR_SIGMA2` | 10000 cp² (100 cp std-dev reference) | More tolerant of unstable ID scores | Stronger down-weighting of ID-unstable positions |

Both approaches are active simultaneously by default at the values above — no open
ablation is tracked for them currently. To isolate their individual contributions,
disable one at a time (set its constant to a very large value) and compare Elo over
~500 games against the baseline.

---

## Offline Consolidation — Batch Training

Offline consolidation is supervised training on quiet-position corpora harvested from
already-played games (self-play or PGN import), running many shuffled epochs of full
gradient descent over every layer — extracting more from existing games than online
TDLeaf's single, in-order pass can. It reuses almost all of the online machinery (same
forward pass, same per-position update math, same Adam optimizer and pure-PSQT material
representation) but drives it from a static corpus instead of live play.

### Corpus Format

Tab-separated text, one quiet position per row (produced by both corpus sources
below; `#` comment lines and a `fen`-prefixed header are skipped by consumers):

```
fen  cp  result  ply  depth  gid  [endply]
```

| Column | Meaning |
|---|---|
| `fen` | Shredder-FEN of the position (castling/ep unused by the trainer) |
| `cp` | eval label, **white POV**, centipawns |
| `result` | game outcome, **white POV**: `1` / `0.5` / `0` |
| `ply` | 1-based ply index within the game |
| `depth` | search depth of the eval label; **0 = no search label** (see below) |
| `gid` | stable game id — the trainer splits train/validation **by game** |
| `endply` | *(optional)* the game's true final ply — exact distance base for the result decay. When absent, the trainer falls back to the per-`gid` max ply seen in the corpus (short by the quiet-filtered game tail, a mild ~uniform over-weighting of the result). Both corpus producers write it since 2026-07-04. |

**Ply units:** both corpus producers write true **game plies** (every half-move)
in the `ply`/`endply` columns (game-ply λ^Δ era, since 2026-07-07) — the
in-engine dump's `ply` is the engine's own game-ply counter at record time, not
a per-record index, so it lines up exactly with PGN extraction's half-move
counting. The online trace uses the same convention: `dply` in
`tdleaf_update_after_game` is the true game-ply gap between consecutive records
(2 in the two-process harness, since each engine only records its own moves; 1
under internal self-play). `TDLEAF_LAMBDA` (0.985) is therefore a single
per-game-ply constant valid unchanged as `--bt-td-lambda` for either corpus
type — no conversion needed, and no reason to mix the two sources differently
in one decayed training run. (The K/λ calibration pipeline —
`extract_positions.py` / `analyze_calibration.py` — also fits λ directly per
*game ply* from PGNs, which is why its fitted band is directly comparable to
`TDLEAF_LAMBDA` without a conversion.)

**The depth-0 rule:** records with `depth == 0` (leaf-dump rows, whose `cp` is the
net's *own static eval* at dump time, acting as a magnitude anchor) get their own
outcome weight, `--bt-leaf-lambda` — **default: the same λ as roots** (the
recommended setting). Set `--bt-leaf-lambda 1.0` for the old outcome-only behaviour, or
any other value to sweep the leaf blend independently of the root blend. Records with
`depth > 0` carry a search-amplified label and always use `--bt-lambda`. This lets root
and leaf corpora mix freely in one training run.

### Building Corpora

#### 1. From PGNs — `scripts/extract_quiet_positions.py`

Replays existing games (python-chess, Chess960-aware, multiprocessed ~2,300
games/s) and emits root positions with their search-score labels from the move
comments. Quiet filters: in-check, played-move-is-capture/promotion/check,
missing/mate eval, |eval| cap, min-ply, fifty-move clock; duplicate control via
Zobrist hash with a per-position record cap (FRC openings repeat massively).
See `SCRIPT_USE.md` for the option table.

#### 2. During play — `TDLEAF_DUMP_TSV` (in-engine)

Any TDLEAF build dumps corpora as a by-product of online play — PGN retention
becomes optional:

```sh
TDLEAF_DUMP_TSV=iter2 python3 match.py Leaf_vtrain_a Leaf_vtrain_b ...
```

writes two per-process files at game end:

- `iter2.<pid>.root.tsv` — played root positions, **search-score labels**
  (white POV), `depth` = achieved ID depth. Root quietness test:
  |root static − root search| ≤ `TDLEAF_DUMP_QUIET_CP` (default 60 cp) — an
  operational filter (unresolved tactics show up as static-vs-search
  disagreement).
- `iter2.<pid>.leaf.tsv` — PV-leaf positions, static labels, `depth` 0. Leaves
  are distribution-matched to what the net actually evaluates in search; their
  training signal is the outcome label.

`TDLEAF_DUMP_MAX_CP` (default 1500) caps |eval| for both. Results are labelled
via the engine's normal game-outcome path (protocol result or UCI
self-adjudication), so only games that feed the TD update are dumped.

### The Batch Trainer — `--batch-train`

Built into any `NNUE=1 TDLEAF=1` binary (`src/nnue_batch_train.cpp`). Loads the
base `.nnue` + `.tdleaf.bin` next to the binary exactly like a normal training
session, trains on the given TSVs, writes per-epoch snapshots, and exits:

```sh
./Leaf_vbt --batch-train corpus_a.tsv,corpus_b.tsv --bt-out myrun \
           [--bt-epochs 3] [--bt-lambda 0.7] [--bt-leaf-lambda <λ>] \
           [--bt-td-lambda 0.985] [--bt-K 220] [--bt-lr 0.25] \
           [--bt-batch 512] [--bt-val 0.05] [--bt-seed 42] [--bt-max 0] \
           [--bt-threads 8] [--bt-clip-every 64] [--bt-loss-gamma 1.0]
```

> **Current defaults:** `--bt-K 220` cp with the default pure λ-return target —
> `--bt-lambda` and `--bt-leaf-lambda` default to `1.0` and stay dormant scale knobs;
> `--bt-td-lambda` (default `TDLEAF_LAMBDA` = 0.985) is the single knob of record for
> outcome-weight decay. When calibrating `--bt-td-lambda` on a **new** corpus, don't
> reuse a value blindly — check the trainer's printed mean decay and aim for a mean
> outcome mass ≈ 0.25–0.33 (see
> [Recalibrating K/λ](#recalibrating-kλ-reproducing-the-analysis) below).

Design principles:

- **Maximum reuse of the online machinery.** The per-position update goes through
  `nnue_init_accumulator` → `nnue_forward_fp32` → `nnue_accumulate_gradients` →
  gradient clipping / per-section Adam / requantize. The gradient-scale formula
  mirrors `tdleaf_accumulate_game`, so the online LR calibration carries over
  unchanged.
- **Target:** `p = w·result + (1−w)·sigmoid(cp/K)` with the distance-decayed
  result weight `w = λ_eff · td_λ^(N−ply)` — the outcome's credibility decays
  with distance from the game end, exactly as the terminal outcome's weight
  decays in the TD(λ) forward-view λ-return, and the freed weight returns to
  the eval bootstrap so the two weights always sum to 1 (targets stay
  calibrated at any decay). `λ_eff` is `--bt-lambda` for root rows and
  `--bt-leaf-lambda` for leaf rows; `N` is the game's final ply (`endply`
  column, or per-game corpus max as fallback); `--bt-td-lambda` defaults to
  `TDLEAF_LAMBDA` (0.985, `tdleaf.h`) so offline targets reconstruct the same
  λ-returns the online games were trained on, and `1.0` reproduces the flat
  blend `p = λ·result + (1−λ)·σ` bit-for-bit. Squared error in probability
  space (the same update form as the TD rule); all of λ/td_λ/K are
  training-time hyperparameters — never baked into the data. Because the decay
  compounds across many plies, a nominal ceiling λ delivers noticeably less than
  its full flat-blend outcome mass in practice — check the trainer's printed
  mean decay for a given corpus (see
  [Recalibrating K/λ](#recalibrating-kλ-reproducing-the-analysis)) rather than
  assuming the flat-blend equivalence holds unchanged under decay.
- **Loss shape** (`--bt-loss-gamma`, default 1.0): scales the per-position gradient by
  `(d·(1−d))^γ / K` — `γ=1.0` is standard MSE-on-probability (default), `γ=0.0` reduces
  to cross-entropy, `γ=0.5` is a focal-loss compromise between the two. A
  training-time hyperparameter, like λ/td_λ/K — never baked into the data.
- **Packed records** (occupancy bitboard + piece nibbles, ~40 B each): a 200M-
  position corpus fits in ~8 GB RAM.
- **Validation split by game** (hashed `gid`), reported per epoch — but rate
  snapshots by gauntlet, not by validation loss (validation MSE is a poor proxy for
  playing strength: it can oscillate while strength rises monotonically).
- Throughput ≈ 31k positions/s single-threaded (≈ 18 min/epoch per 34M
  positions); substantially higher with `--bt-threads 8` (see below).

#### Per-epoch snapshots and pairing rules

- `<prefix>_epN.nnue` — a fully self-contained snapshot (PSQT holds all trained
  material); use to compile rating binaries (plain `NNUE=1` build). Not a valid
  base for further training.
- `<prefix>_epN.tdleaf.bin` — pairs with the **original base `.nnue`** to resume
  or seed further training. The v10 content-hash check refuses a mispairing at
  load and save (loudly), so the two snapshot types cannot be confused silently.

The base `.nnue` never changes across iterations: every consolidated `.tdleaf.bin`
of every generation pairs with the original base.

### Threaded Training — `--bt-threads`

`--bt-threads N` parallelizes gradient computation **within each batch** in a
single process. A batch's positions are split contiguously across N worker
threads; every thread computes forward + backprop against the **same frozen
weights** into its own gradient buffer. The workers are then summed into the
global buffer in thread-index order (deterministic for a fixed thread count),
and a single Adam step + requantize runs on the merged gradient.

This is **synchronous data parallelism**: no optimizer step happens until every
thread's gradients for the batch are in, so it is *mathematically identical* to
single-threaded training — same weights seen by every gradient, same number and
sequence of Adam steps, same batch size, same LR calibration — up to
floating-point reduction order. `--bt-threads 1` reproduces the pre-threading
trainer **bit-for-bit** (verified by a git A/B: identical `.tdleaf.bin` and
`.nnue`). Measured **~2.85× on 8 performance cores** (30k → 86k pos/s on a 570k
corpus; a ~3–4 h gen-2 consolidation drops to ~1.3 h). Both the per-position
forward/backprop and the per-batch Adam step (FC stacks + FT rows) are
parallelized; a targeted requantize (only the rows the step touched) keeps the
serial tail cheap. The per-epoch log prints a phase-timing line (compute /
reduce / clear / serial-tail) to watch the remaining serial work.

The per-batch fixed costs (serial tail, gradient-buffer clear, the L2 clip-norm
scan) dominate wall clock at small batch sizes more than gradient compute itself.
Two knobs address this:

- **`--bt-clip-every N`** (default 64): the L2 clip-norm scan is serial and would
  otherwise touch every dirty FT/PSQT row each batch, but the clip essentially
  never fires on self-play corpora (batch norms run well under threshold in
  practice). The norm is computed on every Nth batch only; skipped batches still
  run the freeze-passthrough housekeeping. Safety: if a sampled norm ever exceeds
  half the threshold, per-batch scanning is restored for the rest of the run.
  `--bt-clip-every 1` recovers exhaustive per-batch scanning.
- **`--bt-batch`** (default 512): raising the batch size amortizes the fixed
  per-batch costs over more gradient compute, trading a small amount of final
  strength for throughput. **Keep `--bt-batch 512` for production consolidations
  (the promoted net); use `--bt-batch 2048` at the same `--bt-lr` for sweeps and
  probe arms** where turnaround matters more than the last few Elo. Do not scale
  `--bt-lr` up with batch size — Adam's normalized steps largely absorb the
  batch-size change already, and classical √/linear LR-scaling rules make
  larger-batch runs roll over early instead of converging further.

Multi-process sharded training (`--bt-sync`) was tried and removed in favor of
within-batch threading, which has no gradient staleness between synchronization
points; see `docs/history/TRAINING_HISTORY.md` for why sharding was abandoned.
The multi-process delta-merge save protocol itself remains in the engine — it's
what concurrent *online* self-play training still uses (see
[Concurrent File Access](#concurrent-file-access) above).

---

## Generation Modes — Internal Self-Play and the Actor/Learner Split

Three ways to generate training games, selectable per `train.py` run:

| Mode | Flag | Processes | Optimizer states | Notes |
|---|---|---|---|---|
| UCI pair (legacy) | *(default)* | 2 engines/game under fastchess | N, merged via multi-writer `.tdleaf.bin` | 2-ply TD chains; UCI self-adjudication skips ambiguous games — **undersamples draws in the corpus by ~40%** (21% dumped vs 35% played; see `learn/eqstudy_260717_work/RESULTS.md`) |
| Internal self-play | `--selfplay-gen` | N striped single engines | N, merged | Engine `--selfplay` driver: whole games in one process, both sides recorded (1-ply TD chains), exact in-engine results, ~3.5× per-core throughput |
| **Actor/learner** | `--actor-learner-gen` | N−1 frozen actors + **1 learner** | **1** | The settled Stage-1 architecture: single optimizer, sole state writer, no merge in the training hot path |

### The engine's internal self-play driver (`--selfplay`)

`Leaf_vX --selfplay --epd FILE --games N --depth D [--epd-offset K --epd-stride S]
[--epd-shuffle SEED] [--traj-out DIR] [--no-adjudication] [--max-ply P] [--tdleaf-out PATH]`

Plays whole games in-process: openings from plain 4-field EPD lines (FRC-ready),
`tdleaf_record_ply` after **every** search (records alternate root STM; the
`pow(λ, dply)` trace handles the 1-ply gap automatically), exact terminal
detection (mate/stalemate/50-move/3-rep/insufficient material/max-ply) plus
optional fastchess-style resign/draw adjudication.  Per-game reset mirrors
`ucinewgame` (full hash realloc — TT probes match on key only).  Deterministic:
a game is a pure function of (opening, depth, net) at 1 thread.  Learning env
is unchanged from harness play: `TDLEAF_FREEZE=1` + `TDLEAF_DUMP_TSV` = frozen
corpus generation; unfrozen = live online learning.

### The actor/learner split (`.tdg` trajectories + `--learn-stream`)

Actors run frozen with `--traj-out DIR`, writing one binary `.tdg` file per
learned game (write-then-rename; `src/selfplay_traj.h`): positions, search
outputs, and POV/gate metadata only — accumulators, features, and stack
indices are rebuilt by the learner (`tdleaf_rebuild_record`), which is **exact**
(integer accumulator rebuilds equal the incremental PV-walk snapshots; verify
any change with `TDLEAF_CHECK_ACC=1`).  The learner
(`--learn-stream DIR [--total-games N] [--refresh-scores] [--publish X.nnue]`)
consumes files oldest-first, validates magic/version/net-content-hash, and runs
the **exact online update** — with identical starting state and env it
reproduces a single-process online run's `.tdleaf.bin` byte-for-byte (the
Stage-1 bit-exactness gate; it caught the FRC castle accumulator bug).

`scripts/selfplay_run.py` drives the ensemble with epoch-style weight refresh:
the learner's state saves are atomic, and actors exit every `--games-per-actor`
games and respawn, reloading the latest state.

### Online-stability rules (hard-won; both cost a full iteration to learn)

1. **Play to natural termination.**  With learning in the loop, resign/draw
   adjudication forms a runaway spiral: evals inflate → games resign earlier →
   truncated outcome-dominated trajectories inflate evals further.  The
   `d8t-3al` collapse: 60%→97% resignations, ~27-ply games, entry net 0/400 vs
   every opponent — while **gradient norms stayed nominal throughout** and PSQT
   material stayed pinned.  Frozen generation tolerates adjudication (no loop).
2. **Keep TD targets on current weights.**  Epoch-refresh actors ship scores up
   to a refresh cycle stale (~9k games learner-side vs ~8 in the merge path),
   delaying the eval-scale feedback that keeps online TD self-correcting; the
   `d8t-3al2` run drifted from 37% to 12% draws within 40k games.  The learner
   must run `--refresh-scores` (re-evaluate leaf statics with current weights
   at consume time — the replay Flavor-A machinery).

**The health canary is the draw rate** (plus game length), not gradient
telemetry: healthy self-play holds a steady ~35–40% draws at d8; a sustained
slide toward decisiveness means the loop is inflating.  Both failures were
invisible to grad-norm/clip monitors.

`train.py --actor-learner-gen` bakes both rules in (actors get
`--no-adjudication`, the learner gets `--refresh-scores`).  Validated by the
`material_260708-d8t-3al3` iteration: draws steady 35–40% across 188k games,
final net +23±17 over its seed, foreign anchor +80 vs classic.

## The Hybrid Loop Workflow — `scripts/train.py`

One command per hybrid-loop iteration, run from `engine/learn/`:

```sh
cd engine/learn/
python3 train.py --tag iter3 --games 400000 --depth 8 \
    --state <consolidated>.tdleaf.bin --recompile \
    --bt-K 220 --bt-threads 8 \
    --gauntlet-epochs --gauntlet Leaf_vtdL10F10x6-ep4 Leaf_vclassic_eval
```

Chained iteration — `--continue <prev_tag>` reads `<prev_tag>_final.json` (the
sidecar written by the previous run) and defaults `--net` / `--state` /
`--gauntlet-anchors` from it, tracks `cumulative_games` across the whole chain,
and automatically adds `Leaf_v<prev_tag>-final` as a gauntlet opponent:

```sh
python3 train.py --tag iter4 --continue iter3 --games 1000000 --depth 8 \
    --bt-K 220 --bt-threads 8 --gauntlet-epochs
```

`--gauntlet-anchors <binary...>` is a **fixed opponent list** that carries forward
automatically across a `--continue` chain (e.g. `Leaf_vclassic_eval`); one-off
opponents for a single run still go in `--gauntlet` (combined with any anchors and,
under `--continue`, the auto-added previous-iteration net). See `SCRIPT_USE.md` for the
full option table, and the Obsidian `Hybrid_Loop_Runbook` note for the manual procedure
this script encodes.

### Phases

1. **Promote state** — copy `--state` (a `.tdleaf.bin`) over the live training
   state next to `--net`, after a content-hash pairing pre-flight (refuses to
   promote a `.tdleaf.bin` trained against a different base `.nnue`) and a backup
   of whatever state was live.
2. **Online generation** — self-play games at `--depth` (default 8), with
   `TDLEAF_DUMP_TSV` corpus dumping enabled; TDLeaf keeps learning throughout.
   Three modes (see [Generation Modes](#generation-modes--internal-self-play-and-the-actorlearner-split)):
   the legacy `match.py`/fastchess pair (default), `--selfplay-gen` (striped
   internal self-play, multi-writer merge), or `--actor-learner-gen` (frozen
   actors + one learner, single optimizer — the recommended mode).
3. **Post-generation checkpoint** — the post-online `.tdleaf.bin` is saved and
   a piece-value drift canary is run against it.
4. **Assemble corpus** — dump files (plus any `--corpus` files) are
   concatenated into one `corpus.tsv`, refusing to mix game-ply-axis and
   legacy record-index corpora in one run.
5. **Threaded consolidation** — `--batch-train` runs `--epochs` (default 6)
   epochs with `--bt-threads` (default 8) within-batch parallelism, writing a
   per-epoch snapshot. With `--gauntlet-epochs`, each epoch snapshot is rated
   (default 1000 games at 1+0.01) against the net **as it stood before offline
   training** as soon as that epoch finishes — the trainer is paused
   (SIGSTOP/SIGCONT) while each ladder match runs so training never contends
   with the games for cores.
6. **Best-epoch promotion** — with the ladder, the epoch with the best Elo is
   promoted (ties → later epoch); without it, the last epoch is used.
7. **Compile** — a TDLEAF-off rating binary `Leaf_v<tag>-final` is compiled from
   the promoted net.
8. **Gauntlet** — the promoted net plays `--gauntlet-games` (default 400) games
   at `--tc` (default `3+0.05`) against the resolved opponent list, and an Elo
   table is printed and logged.

Skippable phases cover the general workflows:
- `--skip-online` + `--corpus ...` — consolidate-only on existing corpora (also
  how LR/K/λ probe arms rerun on the same dumps without regenerating games;
  combined with `--force` it also resumes a crashed run from the consolidation
  phase, reusing the dump files already sitting in `<tag>_work/`).
- `--skip-train` — generate-only (games + corpora, no offline training).
- `--init-nnue [material|classical|noprior]` — initialise a fresh `--net` (and
  its companion `.tdleaf.bin`) before generating, turning the whole command
  into a start-to-finish run from a brand-new network. `material` (the bare
  flag's default) is classical-material-only PSQT; `classical` adds
  phase-interpolated piece-square tables; `noprior` is the uniform-100 cp,
  materially-blind init — see [Initialization](#initialization) above.

### Artifacts and the `<tag>_work/` archive

All artifacts are named by `--tag` and land in `learn/` and `<tag>_work/`:

| Artifact | Contents |
|---|---|
| `<tag>_final.nnue` | Consolidated, self-contained net — compile rating binaries from this |
| `<tag>_final.tdleaf.bin` | Seeds the next iteration — pairs with the **original base** `.nnue` |
| `<tag>_final.json` | Sidecar metadata: cumulative games, gauntlet anchors, epoch-ladder and final-gauntlet results — this is what `--continue` reads for the next iteration |
| `Leaf_v<tag>-final` | Compiled rating binary for the promoted net; always built (even with no gauntlet opponents, so it's ready as a future `--continue` anchor); never left resident in `run/` |
| `<netbase>.tdleaf.bin-pre<tag>` | Backup of the live state before promotion (only if `--state` was given; deleted on a successful run) |
| `<netbase>.tdleaf.bin-<tag>-online` | Post-generation online checkpoint (deleted on a successful run) |

`<tag>_work/` is a **permanent per-run archive** — it is never deleted on a successful
run. What it keeps and what gets pruned at end-of-run (unless `--keep-work` is passed,
which disables all pruning for that run):

- **Kept:** `corpus.tsv` (gzip'd), the online-generation PGN (gzip'd) and the
  final-gauntlet PGN(s), and `train/train.log` (with the epoch-ladder and gauntlet
  result tables appended after training finishes).
- **Pruned (single-use or regenerable):** the raw per-shard dump files (superseded by
  `corpus.tsv.gz`), non-winning epoch `.nnue` files (regenerable via `Leaf_vbt
  --write-nnue`), epoch-ladder PGNs (their Elo is already captured in the
  log/sidecar), and epoch rating binaries.

`--keep-epoch-states` additionally keeps **every** epoch's `.tdleaf.bin` in
`<tag>_work/train/` (default: only the promoted epoch's state survives; its
`.nnue` companion is always regenerable and is never kept either way). A failed
run's `<tag>_work/` is never pruned, so a postmortem always has the full raw
dumps and logs available.

### The `run/` invariant

No compiled binary — rating binaries, the trainer, anything — is ever executed with
`engine/run/` as its working directory. `run/` holds files like `main_bk.dat` that could
otherwise silently bias match outcomes if a training or rating binary picked them up.
Binaries are compiled in `run/` (a build-system requirement of `comp.pl`) but are then
copied or moved into `engine/learn/` or `<tag>_work/epoch_binaries/` before anything
executes them, and `match.py` is always given a resolved path so each engine's working
directory is derived from that path, not from `run/`.

### Current settled recipe

`--bt-K 220` with the default pure λ-return target is the current consolidation
recipe: `--bt-lambda` and `--bt-leaf-lambda` default to `1.0` and stay dormant scale
knobs, and `--bt-td-lambda` (default `TDLEAF_LAMBDA` = 0.985) is the single knob of
record for how fast the outcome term's weight decays away from the game end.
`--bt-threads 8` (the default) gives single-process, staleness-free consolidation
speed. When calibrating `--bt-td-lambda` on a **new** corpus, don't reuse a value
blindly — check the trainer's printed mean decay and pick a value that lands the mean
outcome weight mass around **0.25–0.33** (see
[Recalibrating K/λ](#recalibrating-kλ-reproducing-the-analysis) below).

### Practical guidance

- **Rate by gauntlet**, 300–400 games vs a fixed panel; never by validation-MSE
  *level*. The MSE trajectory *shape* is still useful live: smooth = healthy
  optimizer, oscillating = step-size trouble.
- Select the epoch by ladder (`--gauntlet-epochs`), not by assuming the last
  epoch — the best epoch has repeatedly landed mid-run rather than at the final
  epoch. A single 1000-game ladder point can swing ~±30 Elo — read the shape,
  replicate before trusting any single-point peak.
- Depth 8 generation is recommended (data-quality lever; the save-I/O overhead
  amortizes to ~10% at depth 8 vs ~13% at depth 6).
- Fresh-generation data first; add older corpora only as controlled arms.
- Check the drift canaries between checkpoints with `diff_tdleaf_checkpoints.py`
  (per-stack fc2_bias, stack-0 fc2_w[13]/[27], FC0 passthrough-row mean). Some
  movement is *productive* (material/positional learning) — judge by gauntlet,
  not by drift alone.
- Measure the promoted net vs. the fixed cross-family anchor **directly**: Elo
  chained through a family member reads optimistic relative to a direct
  measurement.
- Use `--continue <prev_tag>` and `--gauntlet-anchors` for chained iterations
  instead of hand-copying `--net` / `--state` / opponent flags between runs —
  the sidecar keeps the chain's cumulative game count and opponent panel
  consistent automatically.

---

## Reference

### File Structure

| File | Contents |
|------|----------|
| `src/tdleaf.h` | Hyperparameters, `TDRecord`, `TDGameRecord`, function declarations |
| `src/tdleaf.cpp` | PV walking, TD error computation, gradient backprop hooks |
| `src/nnue.cpp` | FP32 shadow arrays, forward pass, gradient accumulation, weight save/load |
| `src/nnue.h` | `NNUEActivations` struct, TDLeaf function declarations |
| `src/nnue_batch_train.cpp` | Offline batch trainer (`--batch-train`): supervised training on quiet-position TSV corpora, λ-blend targets, within-batch thread parallelism (`--bt-threads`) |

### Data Structures

#### `TDRecord` (per-ply snapshot, stored in `TDGameRecord`)

```cpp
struct TDRecord {
    int16_t acc [2][NNUE_HALF_DIMS];   // raw accumulator [perspective][dim]
    int32_t psqt[2][NNUE_PSQT_BKTS];  // PSQT sums [perspective][bucket]
    int     score_stm;                 // search score (cp, side-to-move POV)
    int     score_root_stm;            // root search score (engine POV, cp) — feeds
                                        //   UCI self-adjudication only, not gradients
    int     stack;                     // layer-stack index (piece_count-1)/4
    bool    wtm;                       // White to move at the leaf position
    bool    root_wtm;                  // White to move at the ROOT (recorded) position —
                                        //   alternates per-record under internal self-play
    int     game_ply;                  // 1-based game-ply of the root position; the gap
                                        //   between consecutive records (2 in the two-
                                        //   process harness, 1 under internal self-play)
                                        //   drives the λ^dply trace decay
    float   id_score_variance;         // variance of last TD_ID_HIST ID-depth scores (cp²)
    int     ft_idx[2][NNUE_MAX_FT_PER_PERSP]; // active FT feature indices
    int8_t  n_ft[2];                   // active feature count per perspective
    position pos;                      // leaf position for Flavor A replay
    // Corpus-dump fields (TDLEAF_DUMP_TSV — see "Corpus Dumping" below):
    position root_pos;                 // root snapshot (filled only when dumping)
    int      root_static;              // root STATIC eval, STM POV (root quietness test)
    int8_t   id_depth;                 // achieved ID depth (root-row depth column)
};
```

#### `TDGameRecord`

```cpp
struct TDGameRecord {
    TDRecord plies[MAX_GAME_PLY];   // one per half-move (max 400)
    int      n_plies;               // plies recorded so far
    int8_t   engine_color;          // root STM at every recorded ply (-1=unset, 0=black,
                                     //   1=white); maps "engine won/lost" to a white-POV
                                     //   result for UCI self-adjudication
};
```

One global instance in `game_rec`; `n_plies` reset to 0 at game start.

### PV Leaf Score

`tdleaf_record_ply` walks the PV from the root accumulator using `nnue_record_delta` /
`nnue_apply_delta`, then calls `nnue_evaluate(leaf_acc, leaf_wtm, pc)` to get the leaf
score. `leaf_wtm = root_wtm XOR (pv_len & 1)` — the side to move at the leaf flips once
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

### Gradient Flow

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

The step size for each layer is governed by the Adam LR schedule. Six separate LRs,
each sized to ~0.001 × median(|w|) of the corresponding weight section:
`TDLEAF_ADAM_LR0` (FC0/FC1 weights), `TDLEAF_ADAM_FC2_LR0` (FC2 weights — separate
because the 32→1 fan-in gives FC2 weights far higher leverage), `TDLEAF_ADAM_FC_BIAS_LR0`
(FC biases — int32 scale ~1500), `TDLEAF_ADAM_FT_LR0` (FT weights — int16 scale),
`TDLEAF_ADAM_FT_BIAS_LR0` (FT biases — hedged below the median-rule value to limit
dying-ReLU risk from update frequency asymmetry: biases ~200×/game vs FT weights ~8/5000g),
and `TDLEAF_ADAM_PSQT_LR0` (PSQT — int32 scale ~13000, the sole material channel).
See [Adam Optimizer](#adam-optimizer) above.

### Hooks in Existing Code

| Location | Change |
|----------|--------|
| `src/define.h` | `#ifndef TDLEAF / #define TDLEAF 0 / #endif` |
| `src/chess.h` — `game_rec` | `TDGameRecord td_game` inside `#if TDLEAF` |
| `src/chess.h` — `tree_search` | `int id_scores[TD_ID_HIST]; int id_score_count;` (TD_ID_HIST=4) for ID history |
| `src/nnue.cpp` — `nnue_load()` | Calls `nnue_init_fp32_weights()` inside `#if TDLEAF` |
| `src/search.cpp` — search start | `id_score_count = 0;` reset at the start of each search |
| `src/search.cpp` — after each ID iteration | Appends current `g` to `id_scores[]` ring, increments `id_score_count` |
| `src/main.cpp` — after NNUE/TDLeaf load | `nnue_extract_piece_values()` computes PSQT-derived material values for the startup banner (report-only under `NNUE_FIXED_PIECE_VALUES`, the default — does not overwrite `value[]`) |
| `src/main.cpp` — after `ts.search()` | `tdleaf_record_ply()` with root acc + PV + `id_scores` + `id_score_count` |
| `src/main.cpp` — `game.over = 1` sites | `tdleaf_update_after_game()` then `tdleaf_replay()` |
| `src/main.cpp` — `new_game` / `setboard` | `td_game.n_plies = 0` |
| `src/Leaf.cc` | `#if TDLEAF #include "tdleaf.cpp" #endif` |
| `src/comp.pl` | `TDLEAF=1` flag → `-D TDLEAF=1` |

### Diagnostic Flags

| Flag | Effect |
|------|--------|
| `TDLEAF=1` | Enable all learning code |
| `TDLEAF_READONLY=1` | Load weights but skip gradient updates (inference only) |
| `TDLEAF_CHECK_SCORE=1` | Print direct vs propagated leaf score on every ply |
| `TDLEAF_REPLAY_K` | Epoch-based replay passes per game after the live update (default `0` = disabled). Ablated and left off by default — see `docs/history/TRAINING_HISTORY.md` for why. |

### Recalibrating K/λ (Reproducing the analysis)

`K` and `λ` were originally calibrated empirically from self-play games using
maximum-likelihood sigmoid fitting (for `K`) and autocorrelation/decay fitting (for
`λ`) — see `docs/history/TRAINING_HISTORY.md` for the methodology and results that
produced the current `K = 220 cp` default and the original `λ ≈ 0.98995` value —
later revised to `λ = 0.985` from offline batch-training convergence testing (see
`docs/history/TRAINING_HISTORY.md`). The same pipeline is
reusable to recalibrate either value against a new corpus of self-play or reference
games:

```sh
# 1. Extract per-position parquet from the PGN directory (≈3–5 min for 200K games)
python3 scripts/extract_positions.py \
    --pgn-dir learn/pgn/nn-fresh-260410 \
    --out     learn/positions.parquet \
    --max-games 200000

# 2. Run calibration analysis and generate plots
python3 scripts/analyze_calibration.py \
    --input   learn/positions.parquet \
    --out-dir learn/calibration_plots \
    --stage 5 6
```

Output: `calibration_K.png` (NLL curve, reliability diagram, sigmoid comparison),
`lambda_decay.png` (4-panel autocorrelation and predictiveness plots), `summary.txt`.

For recalibrating `--bt-td-lambda` specifically (the offline result-decay knob) on a
new corpus, this parquet pipeline isn't required — the batch trainer itself prints its
mean decay per run; see [Current settled recipe](#current-settled-recipe) above for
the target mean-outcome-mass range.

---

## Self-Play Driver

`scripts/training_run.py` manages the process of creating the necessary binaries
(if needed), specifies a baseline `.nnue` file, and sets up online training matches
interactively — see [Quick Start](#quick-start) above.

`scripts/train.py` drives one full hybrid-loop iteration end to end: online generation
with corpus dumping → single-process, within-batch-threaded offline consolidation
(`--bt-threads`) → best-epoch promotion → gauntlet. It also supports `--continue
<prev_tag>` to chain a run's `--net` / `--state` / `--gauntlet-anchors` and cumulative
game count from a prior run's `<prev_tag>_final.json` sidecar. See
[The Hybrid Loop Workflow](#the-hybrid-loop-workflow--scriptstrainpy) above for the
full walkthrough.

`scripts/compare_nnue_learning.py` compares a `.tdleaf.bin`
file against the baseline `.nnue` and shows FC, FT, and PSQT weight statistics.

`scripts/diff_tdleaf_checkpoints.py` diffs two `.tdleaf.bin` checkpoints section
by section (piece values, FC/FT/PSQT movement) — the drift-canary monitor.
