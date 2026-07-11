# Auxiliary WDL (win/draw/loss) head

Status: **Phase 1a in progress** (branch `WDL-Head`).

## Motivation

The NNUE terminal layer (`FC2`, `32→1` per material bucket) emits a single scalar
evaluation. A scalar cannot distinguish a static `+0.5` (opposite-colour bishops,
dead-drawn) from a dynamic, winnable `+0.5`. A **win/draw/loss head** predicts a
3-way distribution `(p_w, p_d, p_l)` and unlocks:

1. **Draw awareness / calibration** — the model can say "ahead but drawish".
2. **Contempt / draw-avoidance** with an explicit `p_draw` lever (Phase 2).
3. **Better TD targets** — TDLeaf already funnels scores through a fixed-temperature
   sigmoid `σ(score/K=220)`. A WDL head learns a *position-dependent* score→win-prob
   mapping (sharper in endgames), i.e. a **learned temperature** (Phase 2 payoff).
4. **Anti-drift diagnostics** — a calibrated head is a natural monitor for the
   outcome-imbalance drift documented in `docs/TDLEAF.md`.

## Design: auxiliary head (not full WDL)

The scalar path is **left byte-for-byte untouched** — it keeps driving search, TT,
alpha-beta, `score.cpp`, everything. A parallel head reads the same per-bucket
`fc2_in[32]` activation the scalar `FC2` consumes, plus one material input, and
produces 3 logits → softmax.

```
                     ┌── FC2 (32→1)  ── positional ──┐
FT→FC0→FC1→ fc2_in[32]                                ├─ (psqt/2 + positional) → cp score  [UNCHANGED, drives search]
                     └── WDL (33→3) ── softmax ─(p_w,p_d,p_l)   [NEW, read-out only]
```

### Head input (33 = fc2_in[32] + material)

`fc2_in` is *positional only* — material lives in the separate PSQT channel. Draw
probability is dominated by material, so the head is fed one extra input: the
STM-POV centipawn eval `(psqt_diff/2 + positional)·100/5776`, scaled by
`WDL_MAT_SCALE` (1/16) to sit in the same magnitude band as the `[0,127]`
activations. This is what makes the head a *learned temperature*: it maps
(eval, positional features) → WDL.

### Gradient isolation (Phase 1a)

The head's backprop **stops at `fc2_in`** — it updates only `wdl_weights` /
`wdl_biases`, never the trunk. Consequences:

- The scalar net cannot be perturbed; a WDL build produces a **bit-identical**
  scalar net to the non-WDL build (verifiable by gauntlet).
- The pure-PSQT material anchor is untouched — none of the gauge/scale
  pathologies (`docs/TDLEAF.md`) can be reintroduced by the head.

Phase 2 may relax this (let head gradient flow into the trunk, and/or feed the
WDL value back as the scalar TD target).

## Training target: TD(λ) on the WDL distribution

Not outcome-only CE — we bootstrap the distribution so the head learns the same
horizon the scalar TD path does. Computed in **White POV**, then converted to
STM POV per ply:

```
P^W_t   = wtm_t ? softmax(logits_t) : flip_wl(softmax(logits_t))   (stop-grad for bootstrap)
πW_{T-1}= onehot_white(result)                                     (terminal)
πW_t    = (1-λ^dply)·P^W_{t+1} + λ^dply·πW_{t+1}                    (λ-return over distributions)
target_stm_t = wtm_t ? πW_t : flip_wl(πW_t)
d_logits_t   = softmax(logits_t) - target_stm_t                    (softmax+CE gradient, STM POV)
```

`λ^dply` reuses the existing game-ply trace decay (`TDLEAF_LAMBDA`, `dply` = game-ply
gap). `flip_wl` swaps the W and L components. `onehot_white(result)` = (1,0,0) /
(0,1,0) / (0,0,1) for White win / draw / Black win, using the existing 0.75/0.25
thresholds. The convex combination keeps `πW_t` a valid distribution.

Per-ply gradient is scaled by `TDLEAF_WDL_WEIGHT` and the existing `id_weight`
(ID-score-variance stability weight). Score-change clipping (`TDLEAF_SCORE_CLIP_PAWNS`)
is **not** applied to the WDL target in Phase 1a (softmax is already bounded) —
a Phase-2 refinement.

## Phase 1a scope decisions (deliberate simplifications)

| Concern | Phase 1a | Phase 2 upgrade |
|---|---|---|
| Adam moments (`m`,`v`) | **session-local** (zeroed each start), shared `t_wdl_session` counter for bias correction | persist like FC2 if warmup matters |
| `.tdleaf.bin` persistence | **absolute values** of `wdl_w/b` (v13 trailing section) | full delta-merge + counts |
| Multi-writer merge | **last-writer-wins** on the head (harmless: head doesn't affect play, only slows head convergence) | count-weighted merge like FC2 |
| Trunk gradient clip | head **excluded** from the shared L2 norm; bounded by `TDLEAF_ADAM_STEP_CLIP` only | own clip if needed |
| Inference read-out | **none** — head trained + persisted + telemetered only | `nnue_evaluate_wdl` + contempt (Phase 1b/2) |
| Forward passes | head computed via a second forward per ply during learning (opt-in build) | single-pass activation cache |

These keep the change surgical and the scalar path provably untouched.

## Format

- `.tdleaf.bin`: `TDLEAF_VERSION` bumps to **13** *only* in WDL builds. Non-WDL
  builds stay at v12 and **accept** v13 (they ignore the trailing WDL section).
  A WDL build reading a v12 file cold-starts the head.
- `.nnue`: unchanged in Phase 1a (head lives only in the `.tdleaf.bin` FP32
  shadow; no inference-side head yet).

## Build

```sh
perl src/comp.pl <ver> NNUE=1 NNUE_NET=learn/<net>.nnue TDLEAF=1 WDL_HEAD=1
```

`WDL_HEAD` defaults to 0; the mainline binary is unaffected.

## Validation (Phase 1a)

`WDL_DEBUG=1` prints the terminal-ply White-POV predicted distribution vs. the
actual game result per game. Learning is confirmed when predicted `p_w` tracks
realised win rate (reliability check on a held-out game set). Rate any downstream
use by gauntlet, never by val-loss — see memory `tdleaf-training-efficiency-260702`.

## Code touchpoints

- `src/define.h` — `WDL_HEAD` flag (default 0).
- `src/nnue.h` — `NNUE_WDL_OUT`/`NNUE_WDL_IN`, `NNUEActivations` head fields.
- `src/tdleaf.h` — `TDLEAF_ADAM_WDL_LR0`, `TDLEAF_WDL_WEIGHT`, `WDL_MAT_SCALE`.
- `src/nnue_training.cpp` — shadow weights, gradients, session Adam, forward head,
  `nnue_accumulate_wdl_gradients`, apply (both paths), init (both sites),
  gradbuf clear/merge, v13 save/load.
- `src/tdleaf.cpp` — TD-on-WDL target recursion + head gradient in
  `tdleaf_accumulate_game`, `WDL_DEBUG` telemetry.

Scalar hot path (`src/nnue.cpp` `nnue_evaluate`), `src/score.cpp`, and search are
**not** touched in Phase 1a.
