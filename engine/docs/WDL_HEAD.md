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

## Scope by phase

| Concern | Phase 1a | Phase 2 (done) | Later |
|---|---|---|---|
| Multi-writer merge | last-writer-wins | **additive delta-merge** (`shadow = file + delta`), like FC2 — 12 concurrent writers combine | — |
| `.tdleaf.bin` persistence | absolute values (v13 trailing section) | **delta-merged** on save (v13 bytes unchanged; only save *behavior* changes) | — |
| Adam moments (`m`,`v`) | session-local, `t_wdl_session` bias-correction counter | **still session-local** (correct within one continuous run; each writer's Adam is independent, deltas sum) | persist for cross-restart warmup |
| Head → trunk gradient | isolated (stops at `fc2_in`) | **still isolated** by default | **Phase 4: `WDL_TRUNK_GRAD`** (opt-in) lets it co-train the trunk (below) |
| Inference read-out | none | none | **Phase 1b: `nnue_evaluate_wdl` done** (below); contempt next |
| Trunk gradient clip | head excluded (bounded by `TDLEAF_ADAM_STEP_CLIP`) | unchanged | own clip if needed |
| Forward passes | second forward per learning-ply | unchanged | single-pass activation cache (perf) |

**Phase 2 rationale:** under concurrency N, last-writer-wins discards ~(N−1)/N of each save cycle's head updates. The additive delta-merge (each writer tracks `Σ applied dw` since last sync and adds it onto the re-read file value) makes all N writers' updates combine — the prerequisite for a concurrent-12 training run. Adam stays session-local because within one run every process runs the whole time, so per-writer moments are correct and only the *weights* need merging. The head remains a pure read-out (no trunk perturbation), so the scalar net is still bit-identical.

## Phase 4 — WDL as an auxiliary objective on the trunk (`WDL_TRUNK_GRAD`, opt-in)

By default the head is a read-out: its gradient stops at `fc2_in`. With
`-D WDL_TRUNK_GRAD=1` (requires `WDL_HEAD=1`), the head's gradient w.r.t. `fc2_in`
is also backpropped into the **shared trunk** (FC1/FC0/FT + biases) via
`nnue_backprop_wdl_trunk`, so the WDL objective co-trains those weights as an
auxiliary task. Deliberately excluded:
- **FC2** (scalar output layer) — WDL has its own head weights.
- **the FC0 passthrough** (`fc0_raw[15]`) — a direct score channel `fc2_in`
  doesn't depend on.
- **PSQT / the material scale** — the `wdl_mat` input is treated as
  **stop-gradient**, so the WDL loss cannot perturb the pure-PSQT anchor.

The scalar backprop function is left untouched; the helper is a faithful copy of
its `fc2_in → FT` sub-path minus those three channels.

**Verified (deterministic offline A/B on the 100k-game `material_wdl` net +
571k-position corpus, 2 epochs, single-thread):**
- `WDL_TRUNK_GRAD=0` → scalar `val MSE` is **byte-identical** to a non-WDL build
  (0.004079 → 0.003564 → 0.003432), while the head trains (Brier 0.488→0.433).
  The isolation invariant holds under real training.
- `WDL_TRUNK_GRAD=1` → scalar net **diverges** (mechanism works), but at full
  weight (`TDLEAF_WDL_WEIGHT=1`) it **hurt both** metrics over 2 epochs: scalar
  `val MSE` 0.003432 → **0.004154** and WDL Brier 0.433 → **0.454**.

**Read:** the co-training mechanism is correct, but naive equal-weight
multi-tasking degrades here (classic un-balanced multi-task behaviour).

**Tuning knob (`TDLEAF_WDL_TRUNK_WEIGHT`, default 0.1).** Scales *only* the
gradient into the trunk — the head's own weights always learn at full
`TDLEAF_WDL_WEIGHT`. **Runtime-overridable** via env `TDLEAF_WDL_TRUNK_WEIGHT`
so it can be swept without recompiling (0 = head stays a pure read-out that step;
1.0 = the degrading full-strength A/B above). Sweep e.g. 0.05 / 0.1 / 0.25 and
rate by **gauntlet** — a higher offline training loss doesn't necessarily mean
weaker play.

**hybrid_loop:** `--wdl-trunk-grad` (implies `--wdl-head`) compiles it in;
`--wdl-trunk-weight W` sets the env for the launched binaries. Off by default.

## Phase 3a — offline WDL training in the batch trainer (done)

Phase 1a/2 trained the head only in the **online** self-play path. Phase 3a adds
the WDL loss to the **offline** batch trainer (`nnue_batch_train.cpp`), so a
`--wdl-head` hybrid_loop run trains the head over the offline consolidation
epochs too (on the full quiet-position corpus).

Offline positions are independent (no per-position TD bootstrap), so the target
is the **game-outcome one-hot** (`result2`, STM POV), weighted by the same
result-decay `w` the scalar target applies to its outcome term — the
Lc0/KataGo-style approach where the head learns draw probability from the
*frequency* of draws across similar positions. Gradient
`d_logits = softmax − onehot`, scaled by `w`, into the same per-thread gradbuf →
`merge_dense` → apply → delta-merge as the online path. Head-only (trunk
untouched). A `WDL_Brier` column is added to the batch-train `val` line for
monitoring.

Verified: `WDL_Brier` descends monotonically in the realistic regime (15k
positions, 6 epochs: 0.788 → 0.608). A pathological tiny-corpus × many-epoch run
(2.4k positions, 40 epochs) overfits (val Brier climbs) as expected for a
768-param head with too little data — not a concern at hybrid_loop scale.

> **Latent note:** the head's material input is `score_cp/16` (unbounded); under
> very aggressive trunk sharpening it can grow faster than the slow head (LR
> 0.001) adapts. A future refinement is a bounded input (e.g. `sigmoid(cp/K)`).
> It did not bite in the realistic regime.

**Not covered by Phase 3a:** the head still lives in the `.tdleaf.bin`, not the
promoted `.nnue` — carrying it into a promoted net needs `.nnue` embedding
(separate step). And the gauntlet still rates the scalar net, not the head.

## Phase 1b — inference read-out (done)

`nnue_evaluate_wdl(acc, psqt, wtm, piece_count, out[3])` returns the head's
STM-POV `(p_win, p_draw, p_loss)` for a position, reusing `nnue_forward_fp32`
(so the distribution is identical to what the head is trained against — correct
by construction). It is **not on any hot path** — call at the root or for
diagnostics only; `nnue_evaluate` (the int8 leaf path) is untouched.

Observable via the interactive CLI `wdl` command (prints the current position's
distribution alongside `score`). Verified STM-consistent: the same material
(side-with-queen to move) yields the identical STM-POV distribution for either
colour, matching the scalar score's POV flip.

**Contempt (next):** with `p_draw` available at the root, a draw-avoidance /
draw-seeking bias can be applied. This changes play, so it must be gated
off-by-default and gauntlet-validated, and wants a well-calibrated head (from a
long run) first — deliberately **not** bundled with the read-out.

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
