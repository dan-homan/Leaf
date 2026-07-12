# Training System — Historical Record

This document is a chronological/thematic record of experiments, abandoned
approaches, and superseded hyperparameter values that shaped Leaf's current
training system.  It is kept for context — to explain *why* the system works
the way it does, and to make sure abandoned approaches (dense piece values,
PSQT freezing, sharded offline training, low-K/low-λ regimes, CE loss) are
not silently re-attempted.  For how training actually works **today**, see
`../TRAINING.md`.

---

## Material Representation — Dense Piece Values & the Gauge Machinery (retired)

*Current material representation: see `../TRAINING.md` (pure-PSQT — the bucketed
PSQT is the sole trainable material channel).*

The following is retained essentially verbatim from `TDLEAF.md`, where it was
already self-labeled as historical/inactive under the pure-PSQT default.  It
is the definitive record of the "gauge machinery" — pawn pin, gradient
mean-centering, post-Adam dw-centering, and persisted slot-mean recentering —
and why it existed before pure-PSQT removed the need for it.

### Dense Piece Values

> **HISTORICAL / INACTIVE under the pure-PSQT default.**  The dense `piece_val`
> channel and its gauge machinery are gated off by `TDLEAF_PURE_PSQT = true`.
> This section documents how the second material channel worked and why it
> needed anchoring; it is kept for context and was removed in Phase B of
> `docs/MAINSTREAM_PLAN.md`.

#### Motivation

PSQT weights in the feature transformer are updated sparsely — only the ~30–60
active feature rows are touched per position.  Any given piece-type/king-bucket
combination appears infrequently, so material-scale corrections via PSQT
converge very slowly (~8 updates per 5000 games for a typical feature row).

Dense piece values provide a fast-converging material correction channel.  The
`piece_val[6]` array (one float per piece type PAWN..KING = 6 floats) is
initialized to zero; all per-bucket and per-square variation of material value
is encoded in PSQT.  Every position contributes gradient for every piece type
present on the board, giving ~200 gradient updates per game — orders of
magnitude more than sparse PSQT.

#### Evaluation

`nnue_dense_piece_val(pos, stm, pc)` computes the piece value correction in
centipawns:

```
piece_count_diff[pt] = count(stm, pt) - count(opp, pt)    for pt in PAWN..QUEEN
pv_raw = Σ_pt  piece_count_diff[pt] × piece_val[pt]
pv_score = (pv_raw / 2) × 100 / 5776                       (same scale as PSQT scoring)
```

The result is added to `nnue_evaluate()` in `score.cpp`.

#### Startup Piece Value Extraction

At startup, after the `.nnue` and `.tdleaf.bin` are loaded,
`nnue_extract_piece_values()` reads the PSQT weights to compute an average
material value per piece type and overwrites `value[1..5]` in `score.h` with
those values in centipawns.  In TDLEAF builds, `piece_val[pt] × 0.5` is added
to the PSQT average before conversion (since both encode the same material
correction in different channels).

This ensures that search heuristics (SEE, MVV-LVA, LMR, futility pruning) use
the NNUE-derived material values rather than the hard-coded classical
defaults.

#### NNUE Export (Baking piece_val into PSQT)

When `nnue_write_nnue()` exports a checkpoint `.nnue` file, it bakes the
current `piece_val[pt]` correction into the PSQT weights so the exported file
is self-contained (no `.tdleaf.bin` required at runtime):

```
baked[fi][b] = psqt_weights_f32[fi][b] ± piece_val[ps_slot] × 0.5
```

where `+` applies to own-piece features (`fi % 704 % 128 < 64`) and `−` to
opponent features.  The ÷2 mirrors the `/2` in the scoring formula to preserve
the centipawn scale.  After baking, `piece_val` and `nnue_evaluate()` together
produce the same score as `piece_val=0` would with the baked PSQT.

> **Warning:** loading a baked `.nnue` alongside a `.tdleaf.bin` that still
> contains a non-zero `piece_val` double-counts the correction.  Baked exports
> are intended for read-only opponent/validator binaries; training always
> continues against the original `.nnue` + `.tdleaf.bin`.

#### Gradient

The gradient follows the same path as PSQT but is dense: for each position in
the TD update, every piece type present contributes.  `piece_count_diff[6]` is
stored in `NNUEActivations` alongside the accumulator snapshot.  During
backprop, the gradient for `piece_val[pt]` is:

```
g_pv[pt] += grad_scale × piece_count_diff[pt]
```

#### Optimizer

Full Adam with `TDLEAF_ADAM_PV_LR0 = 13.0` (matches `TDLEAF_ADAM_PSQT_LR0`).
No weight decay is applied.  Dense piece value gradients are included in the
global L2 norm for gradient clipping.  `piece_val_f32[pt]` is clamped ≥ 0 after
each Adam step (Adam moments continue to advance through the clamp so it
self-lifts when the gradient turns constructive) — negative piece values
invert material evaluation and create an unrecoverable death spiral.

#### Evaluation Anchoring (Gauge Fix)

> **HISTORICAL / INACTIVE under the pure-PSQT default.**  Every mechanism in
> this subsection (pawn pin, gradient mean-centering, post-Adam dw centering,
> persisted slot-mean re-centering) is gated off when `TDLEAF_PURE_PSQT =
> true`.  With a single material channel there is no gauge null direction to
> fix.  Retained for context.

Both PSQT and `piece_val` can encode material-scale corrections, and
`TDLEAF_K = 220 cp` is the only term in the loss that fixes the absolute score
scale — so without intervention the training problem is non-identifiable: the
optimizer can shift material mass arbitrarily between the two channels, and
the absolute magnitude of either channel is unbounded.  Empirically,
`piece_val[PAWN]` climbed without ceiling and PSQT slot-means drifted by tens
of cp per session.  Under the multi-writer merge protocol any drift compounds
geometrically across save cycles, eventually producing catastrophic blow-ups
(observed: `piece_val[Bishop]` +515 cp in 571 batches).

The fix is a four-layer gauge anchor that completes the parameter
identification while leaving the legitimate degrees of freedom free to learn.

**1. PAWN piece_val pin (`TDLEAF_PIN_PAWN_VALUE`, src/tdleaf.h).**
`piece_val[PAWN]` is frozen at its init value (0 for `--init-nnue` and
`--init-nnue-noprior`; both put 100 cp of pawn material in PSQT).  Its
accumulated gradient is discarded each batch; m/v moments are left untouched.
This fixes the unit of absolute material scale — N/B/R/Q `piece_val` still
adapt freely and end up expressed in pawn-equivalent units.

**2. Per-batch gradient mean-centering (`nnue_mean_center_psqt_gradients`,
called before the L2 clip).**  For each of the 11 HalfKAv2_hm piece-type slots
(`fi % 704 / 64`: own/opp × {pawn, knight, bishop, rook, queen} + king), the
**per-slot aggregate** gradient summed across all NNUE_PSQT_BKTS buckets and
all dirty features is forced to zero by subtracting the per-cell mean
uniformly.  Per-bucket distribution within the slot is left untouched — only
the absolute material level is constrained.  This is what lets the bucketed
PSQT learn phase-dependent piece values (e.g. pawn worth more in deep
endgame), the entire reason HalfKAv2_hm has 8 PSQT buckets.

**3. Per-batch post-Adam dw centering (`nnue_apply_gradients` Pass 2).**
Mean-centering the gradient alone is not sufficient: Adam's per-weight
`1/√v_hat` normalisation makes the applied step `dw = LR × m_hat / √v_hat` not
zero-sum within a slot, because sparse features (low v) take proportionally
larger steps than dense ones.  Pass 2 sums the applied dw per slot (aggregate
across buckets), then adds the per-cell mean back to every dirty (fi, b) cell
— restoring `Σ_(slot fi dirty, b) Δpw[b] = 0` exactly.  m and v are left
untouched.  Without this step, PSQT pawn-slot mean drifted +13% over 50k games
even with `piece_val[PAWN]` hard-pinned.

**4. Persisted slot-mean targets + load-time and post-merge re-centering
(`nnue_recenter_psqt_slot_means`).**  Per-(slot, bucket) init slot-means are
snapshotted at `--init-nnue` time and persisted in `.tdleaf.bin` v11 as an
11×8 float array.  At every load (after merging in the .tdleaf.bin dirty rows)
AND at every save (after the multi-writer merge block, before the write), the
per-slot aggregate PSQT mean is computed and snapped to the persisted target
by uniformly shifting every feature in the slot.  The correction is applied to
`psqt_weights_f32` only — NOT to `psqt_delta_f32`; treating a gauge
restoration as a fake "delta" would cause the merge to re-apply it on top of
the file value (which already includes other workers' corrections), giving
geometric error growth.  The post-merge re-center is essential under
concurrent training: the merge protocol replaces in-memory shadow with
`file_value + pd`, which discards the load-time anchor, and across 24 parallel
workers the drift compounds without it.

**Why per-slot aggregate (not per-(slot, bucket))**: per-(slot, bucket)
locking would anchor PSQT[bishop_slot][bucket] separately at each of the 8
buckets, forcing the average bishop value across buckets to be constant.
Combined with a single global `piece_val[B]` (no bucket dependence), this
leaves no representational room for phase-dependent piece values.  Aggregate
centering anchors only the slot total, leaving the per-bucket distribution
free.

**Pre-v11 .tdleaf.bin compatibility:** files without persisted slot-means use
the loaded state as the target (locks in any existing drift but prevents
further accumulation).  Re-init via `--init-nnue` is the clean reset.

Mean-centering is active only when `piece_val_active` is true (i.e., when
dense piece values are in use).  Without `piece_val`, PSQT is the only
material correction channel and should not be mean-centered.

#### Persistence

Stored in `.tdleaf.bin` v9 format as 6 float32 values (at 128× resolution)
plus 6 uint32 update counts, appended after the FT bias section.  V4–V8 files
are accepted on load; V5–V8 files contain the old `piece_val[6][8]` (48
floats) which are averaged per piece type on load.  V4 files have no piece
values; they start from zero.

---

## PSQT Freezing — Tried and Abandoned

*Current guidance: see `../TRAINING.md` (a one-line "do not freeze PSQT"
warning); this section is the full failure-mechanism writeup.*

### ⚠️ Do NOT freeze PSQT — the phase-1/1b failure mechanism

Freezing PSQT was tested and **fails catastrophically (~−200 Elo)**.  Do not
re-attempt it.

- **Phase 1** (freeze PSQT + `piece_val`, all material channels pinned to the
  classical prior) and **Phase 1b** (freeze PSQT only, `piece_val` trainable
  with the pawn pin) both collapsed to roughly −200 Elo vs the full-gauge
  reference.
- **Mechanism:** the outcome-scaling pressure that TDLeaf applies is
  **phase-dependent** (it differs per material bucket).  The bucketed PSQT is
  the only **linear** absorber for it.  `piece_val` cannot substitute — it is
  bucket-**independent** (one value per piece type, no phase dependence),
  which phase-1b proved directly.  With PSQT frozen, the pressure instead
  routes through the int8-clipped FC stacks and blows out per-bucket output
  scale (a won KQP-vs-K endgame evaluated +4033 cp vs the reference +1286 cp;
  gauntlet draw rates collapsed).
- The takeaway that makes pure-PSQT work is the same one that makes freezing
  fail: the bucketed PSQT must stay **free** so it can absorb phase-dependent
  outcome scaling.  Pure-PSQT keeps it free and simply removes the
  *redundant* second channel.

---

## Mainstreaming Pure-PSQT — Phases A–C (completed)

*Current state: see `../TRAINING.md`; the implementation roadmap below (from
`MAINSTREAM_PLAN.md`) covers Phases A–C, all completed.  Phases D/E remain
live/pending work and stay in the trimmed `MAINSTREAM_PLAN.md`.*

### 0. Experimental context (why these changes)

All results below from branch `frozen-psqt`, 500k-game depth-6 training runs
with matched batching (single fastchess invocations of 50k/50k/100k/300k
games, c=12, tc=inf, no adjudication, bookless binaries in `learn/`), rated by
gauntlet at 3+0.05 in a combined bayeselo frame anchored by
`Leaf_vclassic_eval` / `Leaf_vmaterial_eval` and a shared 0g checkpoint.

| Experiment | Config | Result @500k |
|---|---|---|
| Reference (`psqt-prior-*`) | full gauge machinery: mean-centering, dw centering, slot-mean recentering, pawn pin, trainable piece_val | +78 (frame) |
| Phase 1 (freeze PSQT + piece_val) | all material channels frozen at classical prior | **FAILED** ~−200 vs ref |
| Phase 1b (freeze PSQT only) | piece_val trainable w/ pawn pin | **FAILED** ~−200 vs ref |
| Pure-PSQT (`pure-*`) | one trainable channel: PSQT free, NO centering/pin/piece_val, search values fixed classical | +47 (frame), −33±26 vs ref |
| **Pure-PSQT + offline (`pure-bt-ep4`)** | above + 4 epochs batch-train on own 80M-position corpus | **+149 (frame): +71 over ref, +102 over own online endpoint** |

**Failure mechanism (phases 1/1b), do not re-attempt freezing:**
outcome-scaling pressure is phase-dependent (per material bucket). The
bucketed PSQT is the only *linear* absorber for it; `piece_val` cannot
substitute (it is bucket-independent — 1b proved this directly). With PSQT
frozen, the pressure routes through the int8-clipped FC stacks and blows out
per-bucket output scale (won KQP-vs-K endgame evaluated +4033cp vs reference
+1286cp; collapsed gauntlet draw rates).

**Why pure-PSQT is safe without the gauge machinery:** the machinery existed
because PSQT and `piece_val` redundantly encode material level → a gauge null
direction the multi-writer merge can amplify geometrically. Delete
`piece_val` as a training channel and there is ONE material channel: no null
direction, nothing to amplify. Absolute scale is loss-anchored via
`TDLEAF_K = 220` under outcome-dominated λ-return targets. Measured drift:
extracted pawn 100→107cp per 500k online games, +7cp more per 4 offline
epochs — slow, and cosmetic because search piece values are fixed (see
`NNUE_FIXED_PIECE_VALUES`).

**Rating discipline:** rate by gauntlet, never by val-MSE level. Val-MSE is a
gate for "not diverging," not a fitness function.

### Phase A — Merge pure-PSQT to main (defaults on, no deletions yet)

#### A1. Merge `frozen-psqt` → `main`

Defaults after merge (already the state at `da9e57a`):

- `src/tdleaf.h`: `TDLEAF_PURE_PSQT = true`, `TDLEAF_FREEZE_PSQT = false`
  (`TDLEAF_PIN_PAWN_VALUE` is moot in pure mode)
- `src/define.h`: `NNUE_FIXED_PIECE_VALUES 1` — search keeps classical
  `value[] = {0,100,377,399,596,1197,10000}`; `nnue_extract_piece_values()` in
  `src/nnue.cpp` computes implied values for the banner ("report only; search
  uses classical") but does not assign `value[]`.
- Gates already in `src/nnue_training.cpp` (verify they survive the merge):
  - piece_val gradient accumulation and piece_val Adam block:
    `if (piece_val_active && !TDLEAF_PURE_PSQT)`
  - Pass 2 dw centering: `if (!TDLEAF_FREEZE_PSQT && !TDLEAF_PURE_PSQT && ...)`
  - `nnue_mean_center_psqt_gradients()` and `nnue_recenter_psqt_slot_means()`:
    early-return on `TDLEAF_PURE_PSQT`

#### A2. Multi-writer merge smoke test — the ONE thing 500k didn't exercise

The pure run was effectively single-lineage. The historical reason for
centering was merge-amplified gauge drift. Pure-PSQT removes the null
direction *in theory*; test it with the actual merge code:

1. Two TDLEAF training processes, same `.tdleaf.bin`, concurrent self-play,
   ~5–10k games.
2. Diff checkpoints with `scripts/diff_tdleaf_checkpoints.py`; watch extracted
   piece values (`P` especially) and FC bias means across a few save/merge
   cycles.
3. **Gate:** no accelerating drift across merge cycles (pawn stays within a
   few cp of its pre-test value; FC bias means stable). If it drifts, stop and
   investigate before Phase B — the merge path may need count-weighting fixes
   rather than restored centering.

#### A3. Docs

Update `docs/TDLEAF.md` and `engine/CLAUDE.md`: pure-PSQT is the recipe; move
gauge-anchoring to a "historical / why not" section; record the phase-1/1b
failure mechanism prominently so freezing is never re-attempted. Log in
`docs/change_log.txt`.

### Phase B — Delete the gauge machinery, format v12

Pure deletion, after A2 passes.

#### B1. Delete from `src/nnue_training.cpp` / `src/tdleaf.cpp` / `src/tdleaf.h`

- `nnue_mean_center_psqt_gradients()` (and its call site)
- post-Adam dw centering in `nnue_apply_gradients` Pass 2
- `nnue_recenter_psqt_slot_means()` (load-path and save-path call sites)
- pawn pin (`TDLEAF_PIN_PAWN_VALUE`)
- the entire `piece_val` training channel: gradient accumulation, Adam
  moments, the ≥0 clamp, LR `TDLEAF_LR_PV` env override
- flags `TDLEAF_FREEZE_PSQT` / `TDLEAF_FREEZE_MATERIAL` / `TDLEAF_PURE_PSQT`
  (pure is now the only mode; keep the *name* out of the code, keep the
  *explanation* in TDLEAF.md)

Keep: `--init-nnue-classical` and `--init-nnue-noprior` unchanged (both bake
material into PSQT — exactly what one-channel wants). Keep
`NNUE_FIXED_PIECE_VALUES` as a define defaulting to 1.

#### B2. `.tdleaf.bin` format v12

- Drop: piece_val weights + their Adam moments; the 88 persisted PSQT init
  slot-means (v11 field).
- Loader: accept v11 (ignore dropped fields) and v12; writer emits v12 only.
- Update `scripts/merge_tdleaf.py` (class `TDLeafFile`),
  `scripts/compare_nnue_learning.py`, `scripts/diff_tdleaf_checkpoints.py` for
  the new layout.
- **Gotcha that will bite:** `--init-nnue` over an existing `.tdleaf.bin`
  MERGES with it (save path is merge-save). Any fresh-init workflow must `rm`
  the old file first. Preserve this warning in docs; better, make
  `--init-nnue` refuse to run if the companion `.tdleaf.bin` exists.

#### B3. Drift canary (replaces the pin — monitor, don't constrain)

- `scripts/train.py`: log extracted piece values (the report-only banner
  values) at every checkpoint; warn if pawn leaves [85, 130] cp.
- The only *functional* coupling to drift is `TDLEAF_SCORE_CLIP_PAWNS` (units
  of max(value[PAWN], 100cp) — with fixed classical values this stays 100cp
  exactly, so the clip does NOT stretch with drift. Good; note it in
  TDLEAF.md).
- Do NOT add a scale regularizer preemptively. If the canary ever fires at
  the 2.4M-game horizon, the fix is a soft pull of extracted-pawn toward 100
  with tiny weight (~10 lines).

#### B4. Gate

A fresh `--init-nnue-classical` + 50k-game depth-6 run on the v12 build must
track the pure-PSQT 50k checkpoint (`Leaf_vpure-5e4g`, still in `learn/`)
within noise in a 1000-game 1+0.01 match.

### Phase C — Ply semantics: per-record STM + game-ply λ^Δ (harness mode)

These are the mode-agnostic wins from the internal-self-play plan, landed and
validated *in the existing two-process harness mode* where behavior can be
regression-tested bit-for-bit. Original design in
`single-process-selfplay-tdleaf-plan.md`; the additions below are critique
fixes.

#### C1. Per-record STM/sign (do this first — biggest correctness surface)

Current code hard-assumes a single-color trajectory:

- `rec.engine_color` is per-GAME (`src/tdleaf.cpp:53` at da9e57a — re-verify
  line numbers)
- propagated-score sign flip at `tdleaf.cpp:91` derives from it
- TSV dump uses `root_wtm = rec.engine_color` (`tdleaf.cpp:322`)

Changes:

1. Add per-record STM to `TDRecord`; outcome term z applied from each record's
   own STM POV.
2. **Centralize ALL POV/sign logic in ONE helper.** A half-inverted-targets
   bug here is silent.
3. **Edit surface beyond the original plan** — two mechanisms operate on
   *successive records* and are POV-sensitive once STM can alternate:
   - `TDLEAF_SCORE_CLIP` (compares consecutive scores — must negate across a
     POV flip, or every quiet position looks like a ±2×eval swing and the
     clip fires constantly, silently neutering learning)
   - ID-stability variance weighting (`TDLEAF_ID_VAR_SIGMA2`, operates on
     `id_scores[]` histories — same negation requirement)
   Route both through the central helper.
4. Rework the `tdleaf.cpp:91` sanity check and the `TDLEAF_DUMP_TSV` dump for
   per-ply color.

**Gate C1 (invariance tests, write them before the refactor):**
- (a) In harness mode (all records same STM) the refactor is a no-op: fixed
  game → **bit-identical gradients** vs pre-change build.
- (b) Sign-symmetry: same fixed game recorded from white POV vs black POV →
  identical updates.
- (c) Mirror test: color-flipped game → exactly mirrored gradients.

#### C2. Game-ply accounting + λ^Δ

1. Store true game-ply in each `TDRecord`.
2. Trace update (`tdleaf.cpp:192` at da9e57a): `e[t] = delta_d + lambda*e[t+1]`
   → `e[t] = delta_d + pow(lambda, dply)*e[t+1]`, `dply` = game-ply gap to the
   next record. Harness mode has dply=2; internal self-play will have dply=1;
   ONE λ expresses the same real-game horizon in both.
3. **λ default retune — not in the original plan, mandatory:** today λ=0.98
   decays per own-move step = per 2 game-plies. Under λ^Δ, preserving current
   harness behavior requires `TDLEAF_LAMBDA = sqrt(0.98) ≈ 0.98995`. Leaving
   0.98 would silently change the harness horizon to λ_eff=0.9604.
4. TSV dump: ply column becomes game-ply.
5. **Corpus version marker — not in the original plan, mandatory:** the
   offline trainer (`--batch-train` in `src/nnue_batch_train.cpp`) computes
   its own λ-return over TSV records and must switch to the game-ply axis
   (λ^Δ over the ply column) in the SAME commit. Add a version header/column
   to the TSV format; `--batch-train` must refuse pre-change corpora (e.g.
   `pure500k.*.tsv`) or handle them with the old axis explicitly. Note
   `--bt-lambda 1.0` (the settled gen-3 recipe) is exponent-invariant, but the
   td_λ=0.98 component and any future λ<1 sweep are not.

**Gate C2:** harness mode with λ = √0.98 → bit-identical gradients to the old
code on a fixed game (up to fp rounding in `pow`; if not exactly identical,
assert max relative deviation < 1e-6).

---

## Depth Curriculum — Superseded Strategy (pre-hybrid-loop)

*Superseded by the hybrid loop; current recipe: see `../TRAINING.md` and the
"Hybrid Loop Adoption Log" section below.  This section is explicitly flagged
in the source (`TODO.md`) as superseded.*

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
corrects its weights for the deeper signal, then stalls at the new
equilibrium.

**Key insight — Adam v is NOT the cause of plateaus:**
After ~1.4M games, FC max v = 4.5e-6 → effective FC LR = 0.01/sqrt(4.5e-6) ≈
4.7. Adam v is negligibly small; the FC plateau is caused by small TD errors
(gradients are tiny when the engine is near its depth-equilibrium), not by
accumulated v damping updates. Optimizer reset would give a brief burst but
not sustained improvement.

**Recommended strategy going forward (at the time):**
Depth curriculum: spend ~10-15k games per depth step, then advance.
e.g. d8 → d10 → d12 → d14. Each transition gives one calibration kick.
Staying at any depth beyond ~10-15k games past the initial burst yields
diminishing returns. The depth itself is the primary lever, not game volume.

**Future experiments considered at the time (superseded by the hybrid loop):**
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

---

## Offline Consolidation — Sweep History (gen-1 through gen-3+)

*Current recipe: see `../TRAINING.md` § Offline Consolidation.*

### Motivation and Results

Online TDLeaf is single-pass and within-game correlated: each game is seen
once, in order, and per-game weight movement decays as Adam's second moment
accumulates. Analysis of the 260628 training run showed the depth-6 self-play
Elo curve saturating ~150–200 Elo below the classical hand-crafted eval, with
data *quality* (not volume, LR, or capacity) as the binding constraint.

Offline consolidation converts the question from "can single-pass TD extract
more?" to "is the information in the data?" Measured results (2026-07-02, all
matches 3+0.05, FRC openings):

| Experiment | Corpus | vs its online endpoint | vs classic_eval |
|---|---|---|---|
| Pilot (4 epochs, single process) | 41.5M positions (d8 + rotation) | +143 ± 22 | −90 ± 18 |
| Pure self-play (6 epochs, 8-way sharded) | 183M positions (260628 d6+d8 only) | **+139 ± 22** | **−87 ± 18** |
| *online endpoint baseline (2.4e6g)* | — | 0 | −214 |

Both runs recovered **+125–130 Elo of genuine cross-family strength** from
games the engine had already played — roughly 60% of the then-remaining gap
to classic_eval — in ~1.5–2 hours of training and zero new games. Notably the
pure self-play corpus alone matched the mixed corpus: the binding factor was
extraction depth, not data diversity. A further finding, replicated in both
runs: **validation MSE is a poor proxy for playing strength** (it oscillated
while strength rose monotonically) — rate snapshots by gauntlet, never by
validation loss.

Two data-scaling caveats established empirically:
- Elo gain is strongly sublinear in position count: the 183M-position run
  matched the 41.5M pilot because the added d6-era positions (from much
  weaker net generations) carried little information. A generation of
  self-play data supports roughly +130–145 over its online endpoint,
  extractable from its best ~40M positions. The route to more is *better
  games* (the hybrid loop), not more epochs over old ones.
- Multi-process sharding does **not** multiply effective LR (total Adam step
  mass is conserved); it adds gradient staleness between syncs, which was
  benign for the large generation-1 backlog signal but **destroyed the
  subtler generation-2 signal**. Sharding was therefore removed in favour of
  within-batch threading (`--bt-threads`), which is staleness-free — see the
  "Threaded Batch Trainer — Tuning History" section below.

### Generation 2 (iteration 2, 2026-07-03/04)

The second loop iteration — 400k d8 online games from the consolidated net
(`iter2-online` = +27 over the gen-1 consolidation, −79.5 vs classic measured
directly), then re-consolidation on the 57M-position in-play dump corpus —
initially regressed under the gen-1 settings, and a systematic arm series
resolved why. Final matrix (all arms trained from the iter2-online state):

| Arm | Sharding | K | λ | Leaf rows | vs gen-1 net | vs classic_eval | Q piece_val drift |
|---|---|---|---|---|---|---|---|
| all 7 initial arms | 8-way | 165–220 | 0.3–0.7 | various | −87 … | −123 … −279 | up to +339 cp |
| iter2s | **none** | 220 | 0.7 | blend | +39 | −90 | +254 cp |
| iter2ks | none | 165 | 0.7 | blend | +12 | −119 | +25 cp |
| iter2ks2 | none | 165 | 0.7 | outcome-only | +9 | −136 | +47 cp |
| **iter2s2** | **none** | **220** | **0.3** | **blend** | **+55 ± 18** | **−64 ± 18** | +89 cp |

Lessons, in order of importance:

1. **Single-process training at the frontier.** Every 8-way sharded arm
   regressed; the identical unsharded control succeeded. Sync-merge staleness
   that the large gen-1 backlog signal absorbed destroys the subtle gen-2
   signal. Live tell: the validation-MSE *trajectory shape* — smooth when
   healthy, oscillating under staleness (the absolute level still doesn't map
   to Elo).
2. **λ, not K, is the knob for outcome-driven piece-value inflation.**
   Consolidation sharpens evals, so gen-2 labels fit a smaller K (165 vs 220);
   training with too-large K makes the outcome term inflate eval magnitudes,
   which lands in `piece_val` (the only free material channel). But refitting
   K to 165 removed the overshoot *and* the productive material correction
   hiding inside it (iter2s's minors moved toward classical values), costing
   30–55 Elo. Cutting λ 0.7 → 0.3 instead tames the drift ~3× while keeping
   the correction.
3. **Leaf rows need the blend anchor**: outcome-only leaves cost ~46 Elo vs
   leaves trained on the λ-blend with their dump-time static as a magnitude
   anchor. (Blended leaves are now the trainer default; the run-time knob is
   `--bt-leaf-lambda`, with 1.0 recovering outcome-only.)

**Settled gen-2+ recipe:** `--bt-K 220 --bt-lambda 0.3` (leaf rows follow λ by
default; single process, ~4 h single-threaded / under an hour with
`--bt-threads 8` on a 57M corpus). Consolidation remains gauntlet-positive per
generation: iter2s2 is +55 over the gen-1 net and +28 over its own online
endpoint, cross-family. Epoch count matters at gen-2+: iter2s2 peaked at
**epoch 4** of 6 in the in-family ladder (gen-1 was still improving at 6) —
select the epoch by a fast ladder (`train.py --gauntlet-epochs`: 1000 games at
1+0.01 per epoch snapshot, ±19, minutes each) rather than assuming the last
epoch. Direct classic anchor: ep4 −62 ± 30 vs ep6 −64 ± 18 — statistically
identical cross-family (the +24 in-family edge is inside the error bars at 400
games), so the ladder pick costs nothing and may gain. (Superseded as the
iteration-3 seed by the decayed λ-return net below, which edged it on the
direct anchor.)

### The λ sweep and the distance-decayed result weight (2026-07-04)

A single-epoch λ × leaf-λ ladder sweep (1000 games at 1+0.01 each, vs a shared
anchor) showed that **the corpus-mean outcome weight is the knob, not the
root/leaf split**: arms with the same mean `0.43·λ_root + 0.57·λ_leaf` (the
corpus is 43% roots / 57% leaves) were statistically identical, with a plateau
at mean weight ≈ 0.2–0.3 and falloff on both sides. A flat λ can only set that
mean; TD(λ) says the *per-position* weight should depend on distance from the
game end. That motivated the decayed target (`--bt-td-lambda`, default
`TDLEAF_LAMBDA`): near-terminal positions trust the result, early positions
lean on the eval bootstrap. Under decay the nominal ceilings roughly double
(mean decay 0.502 on the iter2 corpus), so the plateau maps to diagonal
ceilings λ ≈ 0.4–0.65 — swept before iteration 3 (`learn/sweep_td.sh`:
diagonal λ = leaf-λ ∈ {0.3, 0.5, 0.7, 1.0} plus two crossed arms that test
whether the mean-is-the-knob result still holds under decay).

### Sweep results: the pure λ-return is the settled gen-3+ recipe (2026-07-05)

The decay sweep's winner was the **pure λ-return end**: λ = leaf-λ = 1.0, all
moderation supplied by the td_λ = 0.98 distance decay. Decay *shape* beats the
flat mean — the decayed arm at corpus-mean weight 0.50 won where the flat
family was already declining at 0.44 — and the crossed arms were again inert
(root/leaf split doesn't matter under decay either). A 3000-game head-to-head
vs the λ = 0.5 diagonal arm was a tie (+3 ± 10), so 1.0 wins on simplicity. A
6-epoch confirmation run (`tdL10F10x6`, per-epoch ladder) showed **no
multi-epoch rollover** and **self-limiting piece-value drift**: per-epoch Q
increments +57/+49/+34/+25/+19/+15 cp — geometric convergence (~×0.73/epoch,
asymptote ≈ +240 cp) toward a new material equilibrium, not iter2's
compounding spiral. The ladder peaked at **epoch 4** again (second time, after
iter2s2), and the direct classic anchor on ep4 measured **−58.6 ± 20** (1000
games at 3+0.05) — the best of any net, vs iter2s2-ep4's −62 ± 30.
**`tdL10F10x6_p0_ep4.tdleaf.bin` seeds iteration 3.**

Measurement note, learned the hard way: a single 1000-game ladder point can
swing ~±30 Elo (the *bit-identical* ep1 net measured +29 at 1+0.1 and −5 at
1+0.01 vs the same opponent).

### td_λ calibration sweep (2026-07-10): 0.985 on the material corpus — the knob is corpus-mean outcome mass ≈ 0.2–0.33

Four td_λ points on the 137.7M-position material-line corpus (game-ply axis,
exact endply; batch 2048, lr 0.25, per-epoch ladders vs a shared pretrain
anchor, 1000 games/point):

| td_λ | mean outcome mass | ep1…ep6 ladder | read |
|---|---|---|---|
| 0.975 | 0.196 | +68 +74 +91 +78 +99 +81 | oscillating plateau ~+85–99 |
| **0.985** | **0.330** | +65 +67 +77 +87 +90 **+95** | **monotone, still rising at ep6** |
| 0.98995 (default) | 0.451 | +38 +72 +76 +82 +77 +78 | plateau ~+78 |
| 0.995 | 0.651 | +42 +40 +43 +56 | clearly worst |

Head-to-head between the two best snapshots (td0985-ep6 vs td0975-ep5, 2000
games): 49.55% — a tie. **0.985 is the pick** on curve shape (smooth monotone
vs oscillating) and proximity to the default. Its ep6 net is the strongest of
the material line (+95 vs pretrain, vs the b512 reference's +86 peak) and was
still rising — more epochs may extract more.

The deeper reading: mapped to corpus-mean outcome mass, the sweep reproduces
the flat-λ sweep's plateau (mass ≈ 0.19–0.33, falloff above) under the decayed
target. The default td_λ (0.98995 → mass 0.451 on this corpus) sits past the
plateau edge and cost ~10–15 Elo here.

---

## Hybrid Loop Adoption Log (TODO.md, 2026-07-02 → 2026-07-10)

*Current state: the hybrid loop is now the primary training strategy — see
`../TRAINING.md`.  This section preserves the dated narrative and resolved
open-items history from `TODO.md`; still-open backlog items remain in the
live `TODO.md`, not here.*

### UPDATE 2026-07-02 — hybrid loop supersedes pure-online strategies

The offline-consolidation work (see `OFFLINE_TRAINING.md`) changed the
training picture substantially and superseded parts of the depth-curriculum
plan (see "Depth Curriculum — Superseded Strategy" above):

- **Confirmed:** the depth-equilibrium finding still holds for *online*
  learning — the d6 self-play Elo curve on run 260628 extrapolates to an
  asymptote ~150–200 Elo below classic_eval; data quality is the binding
  constraint.
- **New result:** offline multi-epoch consolidation of the SAME games
  recovers +125–140 Elo past the online endpoint (~60% of the then-remaining
  gap to classic_eval) in ~2 h. The hybrid loop (generate → consolidate →
  regenerate with the stronger net) became the primary strategy, with depth
  increases remaining the quality lever *within* the generation phase (d8
  recommended).

**UPDATE 2026-07-04 — iteration 2 complete; gen-2+ recipe settled.** The
generation-2 consolidation experiment series (see "Generation 2" above)
resolved the initial iteration-2 regression and settled the recipe:
`--shards 1 --bt-K 220 --bt-lambda 0.3 --bt-leaf-blend`. Best net is now
`iter2s2` (+55 vs the gen-1 consolidation, −64 vs classic_eval; gap to classic
down to ~64 Elo). Findings folded into the open items below: sharded sync
staleness destroys the subtle gen-2 signal (single-process required for now);
λ — not K — is the knob for outcome-driven piece-value inflation; leaf rows
need the blend anchor (`--bt-leaf-blend`, committed 2026-07-04).

**UPDATE 2026-07-05 — pure λ-return settled as the gen-3+ recipe.** The
distance-decayed result weight (`w = λ_eff·td_λ^(N−ply)`, committed 5ce7714)
sweep found the λ = leaf-λ = 1.0 end best: decay shape beats any flat mean,
piece-value drift self-limits (geometric convergence, no multi-epoch
rollover), and `tdL10F10x6-ep4` posted the best direct classic anchor yet
(**−58.6 ± 20**; gap ~59 Elo). `--bt-lambda` now defaults to 1.0 (trainer +
train.py); `--bt-td-lambda` (= `TDLEAF_LAMBDA` 0.98) is the single knob of
record. The λ-fine-tuning item below was superseded by td_λ calibration.

### Resolved items (folded in from the Open items checklist)

These items from the `TODO.md` "Open items" checklist under the 2026-07-02
UPDATE were resolved, with their inline rationale preserved here.  (The
still-open `[ ]` items from that same checklist remain live backlog in
`TODO.md`.)

- ~~td_λ calibration on a larger corpus~~ — resolved 2026-07-10 on the
  137.7M material-line corpus: **td_λ 0.985** (game-ply) beats the 0.98995
  default by ~10–15 Elo (+95 still-rising vs +78 plateau vs the shared
  pretrain anchor); 0.975 ties it (noisier); 0.995 clearly worst. The
  transferable knob is corpus-mean outcome mass ≈ 0.25–0.33 (matches the old
  flat-λ plateau) — pick td_λ per corpus from the trainer's printed mean
  decay. See "Offline Consolidation — Sweep History" above.
- ~~`--bt-sync` frontier fix~~ — resolved 2026-07-08 by *removing* sharding:
  replaced with within-batch thread parallelism (`--bt-threads`, single
  process, staleness-free — mathematically identical to 1 thread up to float
  summation order; measured ~2.85× on 8 cores). See "Threaded Batch Trainer —
  Tuning History" below and `docs/BT_PARALLEL_PLAN.md`.
- ~~λ fine-tuning around 0.3; per-source λ (roots vs leaves)~~ — resolved
  2026-07-05: root/leaf split is inert (confirmed twice); flat λ superseded
  by the pure λ-return with td_λ decay (see UPDATE above).
- ~~Dirty-row-only requantize~~ — resolved by the `--bt-threads` work
  (`nnue_requantize_fc_applied`, targeted rows only). Further serial-tail
  trims landed 2026-07-09: zero-on-merge worker clearing + sampled clip scan
  (`--bt-clip-every`, default 64) — +16% at batch 512, bit-identical; see
  "Threaded Batch Trainer — Tuning History" below. Batch-size sweep resolved
  2026-07-10: batch 2048 at *unchanged* lr 0.25 plateaus ~7 Elo below batch
  512 at 1.8× speed (LR scaling rules hurt — Adam absorbs the batch change;
  scaled arms roll over). Recipe: 512 for production consolidations, 2048 for
  sweep/probe arms.

---

## Cross-Entropy Loss Experiment (focal-γ) — Tried, Not Adopted

*Current loss function: see `../TRAINING.md` (offline trainer uses MSE in
probability space, γ=1, by default).  This experiment is self-labeled "DONE,
NOT ADOPTED" in the source.*

### ~~Cross-entropy loss for offline batch-train (focal-γ variant)~~ — DONE, NOT ADOPTED (2026-07-10)

**Result: γ<1 loses on the near-equal quiet corpus.** Implemented
`--bt-loss-gamma` (`sig_grad = (d(1−d))^γ/K`; γ=1 MSE bit-identical, γ=0 CE,
γ=0.5 between) + an NLL(blend) val metric. Epoch ladders (batch 2048, td_λ
0.985, 1000g/pt vs shared pretrain anchor, lr paired ÷2 at γ=0.5 / ÷4 at γ=0):

| γ | ladder (ep1…6) | best |
|---|----------------|------|
| 1.0 (MSE) | +65 +78 +94 +73 +91 +91 | +94 |
| 0.5 | +54 +70 +60 +64 +84 +71 | +84 |
| 0.0 (CE) | +49 +54 +51 +48 (ep1–4, then aborted) | +54 |

The ordering is cleanly **monotone in γ**: γ=1 (+94) > γ=0.5 (+84) > γ=0
(~+50, flat) — each step of tail-emphasis (lower γ) makes it progressively
worse. γ=0 (CE) is the worst by ~40 Elo and never climbs, so its remaining
epochs were aborted. This is exactly the "null-or-negative on balanced quiet
data" outcome predicted below: targets and `d` cluster near 0.5 where
`d(1−d)≈0.25` is maximal and MSE≈CE, so CE's confidently-wrong-tail advantage
has almost no mass to act on — and removing the `d(1−d)` damping just
amplifies the noisy near-0.5 bulk. Flag kept (default γ=1, fully inert) in
case a future fatter-tailed corpus (e.g. a blunder-heavy line) warrants a
retest; **not adopted for the mainstream recipe.** Original analysis retained
below.

**Idea:** Leaf currently minimizes squared error *in probability space*
everywhere, not cross-entropy. Both learning paths map the white-POV eval
through a sigmoid `d = σ(score/K)` and descend `(target − d)²`:
- Online TDLeaf (`tdleaf.cpp:194,204`): `sig_grad = d*(1−d)/K`,
  `grad_scale = e[t] · sig_grad · …`, where `e[t]` is the backward, λ-traced
  sum of consecutive-leaf sigmoid deltas (`delta_d = d[t+1] − d[t]`,
  `tdleaf.cpp:182`).
- Offline `--batch-train` (`nnue_batch_train.cpp:457-464`): `e = target − d`;
  `se += e·e`; `sig_grad = d*(1−d)/K`; `grad_scale = e · sig_grad · …`. Header
  comment (line 30) literally says the loss is `(p_target − d)²`. `target` is
  the λ-blend soft label (`bt_target`).

For a **fixed (soft) target**, MSE and cross-entropy differ by *exactly one
factor* — the sigmoid Jacobian `d(1−d)` — because of the standard sigmoid+CE
cancellation:

| loss          | ∂L/∂score                 |
|---------------|---------------------------|
| MSE (current) | −(target − d) · d(1−d)/K   |
| Cross-entropy | −(target − d) / K          |

So in the **offline** trainer, "switch to CE" ≈ drop `d(1−d)` from
`sig_grad`, keeping `e = target − d` (a soft label → soft-label cross-entropy).
~One line.

**Why CE could help:** MSE's `d(1−d)` factor →0 at the confident tails
(`d→0/1`), so a position the net rates winning (`d=0.98`) that was actually
lost (`target=0`) gets a near-zero gradient — the blunder/horizon corrections
you most want are throttled hardest. CE keeps full `(target − d)` strength
there (this is why nnue-pytorch / Stockfish train NNUE with a CE-style loss).
Better calibration, often faster.

**Do it OFFLINE ONLY.** The offline path has genuine fixed targets, so CE is
a clean, correct drop-in and A/B-able via the normal gauntlet without
touching the online gauge machinery. Leave online TDLeaf on MSE — there the
`d(1−d)` is **not** a discretionary MSE-damping term but the chain-rule
Jacobian `∂d/∂score`, because `e[t]` is built in *d-space* (a traced sum of
`delta_d`). Dropping it online yields a gradient inconsistent with its own
objective, not "CE"; a real logit-space CE would require redefining the
eligibility trace over logits — a different algorithm, out of scope.

**No numerical blowup to fear.** With the sigmoid+CE cancellation the CE
gradient is `(d − target)/K`, bounded by `1/K` — at most ~4× MSE's peak of
`0.25/K`. Only the *reported* NLL metric can overflow (`log(d)` as `d→0/1`);
clamp `d` to `[0.05,0.95]` (or `+ε` inside the log) for the metric only. **Do
not clamp to "protect" training** — any clamp tight enough to tame magnitude
just re-introduces MSE-style tail damping and gives back the whole point of
switching.

**Implementation — focal-γ knob (preferred over binary MSE-vs-CE):**
- `sig_grad = powf(d*(1−d), γ) / K`. `γ=1` = current MSE, `γ=0` = CE, `γ=0.5`
  between.
- Behind a `--bt-loss-gamma <γ>` flag (default 1.0 = no behavior change).
- Gives the tail-emphasis/stability tradeoff as a *curve* over a small grid
  (γ ∈ {0, 0.5, 1}) for the same gauntlet cost, instead of a coin flip.

**Two things that must accompany the switch (else the A/B is confounded):**
1. **Global LR rescale, ~÷4 — not a seven-LR re-tune.** `d(1−d)` is a *per-
   position scalar* that multiplies `grad_scale`, scaling **every section's
   gradient uniformly** (FC0/FC1/FC2/FT/PSQT). It never changes the
   *relative* magnitude between sections, so the inter-section LR ratios are
   preserved; only a global `lr_scale ÷ ~4` is needed (most data sits near
   `d=0.5` where `1/(d(1−d)) ≈ 4`).
2. ~~**Bump `TDLEAF_GRAD_CLIP_NORM` ~×4.**~~ **Empirically moot (measured
   2026-07-09):** clip telemetry over the 137.7M-position material-line run
   showed batch grad norms of 0.053–0.082 against the 1.0 threshold — 12–19×
   headroom, zero fires in ~1M batches — so even CE's ~4× larger raw
   gradients (~0.2–0.33) never reach the clip. No clip change needed; only
   the LR ÷4 pairing above. (Caveat kept for the record: on a corpus with
   much larger norms the original reasoning applies — the clip runs on the
   raw, pre-LR gradient, so lowering `lr_scale` would not relieve it. Note
   the clip scan is now *sampled* by default (`--bt-clip-every 64`) with an
   automatic fall-back to per-batch scanning if a sampled norm exceeds half
   the threshold.)

**Expectation management:** the payoff is likely *modest*. Offline corpora
are quiet positions from near-equal self-play, where most targets and `d` sit
near 0.5 — exactly where `d(1−d) ≈ 0.25` is maximal and MSE ≈ CE. The
confidently-wrong tail cases CE helps with are rare in balanced data (the
same distribution that makes outcome-imbalance drift benign — see
"Outcome-Imbalance Drift — Diagnosis" below). A small or null result is a
legitimate outcome. Watch the piece_val / PSQT scale spectrum when testing,
since CE's tail emphasis can interact with material scale.

**Plan (as originally proposed):** add `--bt-loss-gamma` + paired
`GRAD_CLIP_NORM` handling → sweep γ ∈ {1, 0.5, 0} with `lr_scale ÷ 4` at γ<1
→ gauntlet each net vs the current-MSE consolidation → keep γ only if it
clearly wins. Report NLL (with metric-only clamp) alongside
MSE(blend)/MSE(outcome) in `val_loss()`.

---

## Hyperparameter Calibration History (K, λ)

*Current values: `TDLEAF_K = 220 cp`, `TDLEAF_LAMBDA = √0.98 ≈ 0.98995`; see
`engine/CLAUDE.md` and `../TRAINING.md`.*

### Sigmoid temperature K

**Result (stages 5–6, 10M positions):**

| | K (cp) | NLL/position | Brier score |
|---|---|---|---|
| Optimal (MLE) | **240** | 0.54037 | 0.12604 |
| Previous value | 290 | 0.54296 | 0.12707 |

The optimal K of ~239 cp was rounded to **240 cp**. The earlier value of 290
cp was fitted from a different training stage; as the network improved its
evaluations became sharper, narrowing the effective sigmoid. The reliability
diagram confirms K=240 produces well-calibrated win probabilities across the
full score range.

**Operational history:**
- *Pre-2026-05-02:* K=400 (raised from earlier calibrated value to fight
  piece-value drift)
- *2026-05-02:* lowered to **K=200**, after 200k+ training games at K=400
  showed piece values converging ~2× too high — the larger K underweighted
  material differences in the gradient signal.
- *2026-05-23:* lowered further to **K=150**, after classical-PSQT-prior
  calibration runs (see `Learning_Rate_Experiment` in the Obsidian vault)
  showed residual piece-value drift at K=200 and a moderately wider effective
  gradient was needed for FC/FT to track PSQT shape changes. K=150 is below
  the MLE optimum (~240 cp from raw sigmoid fit) but produces a balanced
  piece-value spectrum and faster FC/FT response. Probability calibration
  remains acceptable across the score range used in TDLeaf updates (most
  positions are within ±400 cp of zero where the difference between K=150 and
  K=240 sigmoids is small).
- *Current:* **K=220**, recalibrated 2026-05-25 via MLE over 58M positions
  from the classical-eval side of `match_nn-fresh-260514-1.39e6g_9.5e5g.pgn`
  (1.015M games). Optimum 217.71 cp, rounded to 220. As the network's
  PSQT/piece_val converged closer to the classical prior, the per-position
  score distribution sharpened again and the MLE optimum drifted upward from
  the earlier sub-150 cp regime.

### Eligibility trace decay λ

**Result (stages 5–6):**

| Method | λ decisive | λ draw |
|---|---|---|
| Even-lag autocorr | 0.985 | 0.989 |
| d_t-vs-result decay | 0.970 | 0.986 |
| Previous (separate) | 0.800 | 0.500 |

Key findings:
- Both methods agree: λ ≈ 0.97–0.99. The previous values (0.8 / 0.5) were
  substantially below the empirical decay rate — the engine's depth-6
  evaluations are far more temporally stable than those values assumed.
- **Decisive and draw games have essentially the same temporal correlation
  structure.** The two-lambda scheme offered no empirical benefit, so it was
  collapsed to a single `TDLEAF_LAMBDA = 0.98`.
- The ply-alternation oscillation in the raw autocorrelation (period-2 ripple
  caused by white/black mover-optimism bias) is a real data feature, not an
  artifact. Even-lag analysis removes it cleanly; the odd-lag trough at lag 1
  suggests adjacent-ply pairs are weakly anticorrelated within a game.

Rounded to **λ = 0.98** (splitting the difference between the two methods).
(This value was later reinterpreted as a per-game-ply decay,
`TDLEAF_LAMBDA = √0.98 ≈ 0.98995`, in Phase C of the mainstreaming plan — see
"Mainstreaming Pure-PSQT" above.)

---

## Epoch-Based Replay — Ablation

*Current state: replay is disabled by default (`TDLEAF_REPLAY_K = 0`); see
`../TRAINING.md`.*

After `tdleaf_update_after_game()` applies the live gradient pass,
`tdleaf_replay()` runs `TDLEAF_REPLAY_K` (default 0, disabled) additional
passes over the last `TDLEAF_REPLAY_BUF_N` (default 8) completed games
stored in a static ring buffer.

### How it works (Flavor A)

1. The completed `TDGameRecord` (accumulator snapshots, feature indices, and
   leaf positions) is pushed into the ring buffer, replacing the oldest
   entry when full.
2. For each replay pass, iterate over all buffered games oldest-first:
   a. `tdleaf_refresh_scores()` rebuilds each ply's accumulator from the
      stored leaf `position` using `nnue_init_accumulator()` against the
      **current FT weights**, re-enumerates active features, and
      re-evaluates `score_stm`. This ensures FT gradients during replay are
      self-consistent with the current network.
   b. `tdleaf_accumulate_game()` computes TD errors and accumulates
      gradients exactly as in the live pass.
3. After all games in the pass are processed, `nnue_apply_gradients()` and
   `nnue_requantize_fc()` are called once, so the next pass's accumulator
   rebuild sees the updated weights.
4. Weights are saved to `.tdleaf.bin` after all K passes complete.

Score-change clipping and ID-stability weighting apply identically in
replay passes (the stored `id_score_variance` values are reused unchanged).

### Ablation results (K vs. Elo gain)

| K | Result |
|---|--------|
| 0 | **Current default — replay disabled; no long-term benefit observed beyond ~5000 games** |
| 1 | ~+20 Elo over first 5000 games but benefit fades; FC-only replay required |
| 2 | Marginally better than K=1 in initial ablation |
| 3 | Slightly worse than K=2 |
| 6 | Large regression |

Replay with FT/PSQT gradients causes eval divergence (FT feedback loop); only
FC-only replay (`fc_only=true` in `nnue_accumulate_gradients`) is viable.
Even FC-only replay showed no long-term benefit after the first ~5000 games,
so replay is disabled by default.

### Build flags

| Flag | Default | Effect |
|------|---------|--------|
| `TDLEAF_REPLAY_K` | 0 | Replay passes per game; 0 disables replay |
| `TDLEAF_REPLAY_BUF_N` | 8 | Ring buffer capacity (~4.5 MB × N static BSS) |

---

## Outcome-Imbalance Drift — Diagnosis

*Current guidance/monitoring: see `../TRAINING.md` and
`scripts/diff_tdleaf_checkpoints.py`.*

**Failure mode (diagnosed 2026-07-02):** sustained training at a score far
from 50% — e.g. vs. a fixed stronger opponent — collapses the net. In run
260628, 1M games vs. classic_eval took the learner from 40% to 4.5% score and
cost ~570 pool Elo.

**Mechanism:** the terminal TD term `result − d` is net-negative for a
persistently losing learner (net-positive for a persistently winning one).
Its state-independent component cannot be expressed by minimax-relevant
features and is absorbed by whatever channel can represent a constant: the FC
output biases first (measured: mean fc2_bias drift −860 raw over the 1M
games; endgame stack −504 → −2277 ≈ −39 cp STM-POV), and — if the biases are
frozen — specific FC2 weights on constant-ish activations, the FC0
passthrough row, and piece_val deflation. Pinning channels is whack-a-mole;
the source is the imbalance. Positive feedback (pessimistic eval → worse play
→ more losses) makes the drift monotone. The evaluation-gauge anchors (PAWN
pin, PSQT slot-means) do not cover this direction.

**Why self-play is immune:** wins and losses balance by construction, so the
DC term is zero-mean. Better: **balanced play actively reverses** accumulated
drift — the calibration equilibrium (mean prediction must match mean result)
pulls the bias back. Verified by a 200k-game rotation experiment (20k-game
segments alternating classic_eval / frozen mirror): vs-classic Elo stable
across all five classic segments, and previously accumulated drift recovered.

---

## Threaded Batch Trainer — Tuning History

*Current defaults/usage: see `../TRAINING.md` (`--bt-threads`, `--bt-batch`,
`--bt-clip-every`).*

**Serial-tail trims (2026-07-09, +16% at batch 512, bit-identical).** Phase
timing on the 137.7M-position material-line run showed the per-batch fixed
costs — not gradient compute — dominating wall clock (serial tail 36–58%,
reduce+clear another ~22%). Two changes:

- **Zero-on-merge:** `nnue_gradbuf_merge_ft_rows` re-zeroes each worker's
  FT/PSQT row (and its dirty flag) right after summing it into `g_grad`,
  while the cache lines are hot. The separate clear phase — previously a
  memory-bandwidth-bound O(dirty-rows) memset pass — shrinks to the dense FC
  grads plus the dirty-list cursor (measured 6.7s → 0.5s per epoch on an
  8M-position slice).
- **Sampled clip scan (`--bt-clip-every`, default 64):** the L2 clip-norm
  scan is serial and touches every dirty FT/PSQT row each batch, yet the clip
  *never fires* on self-play corpora — measured batch norms 0.053–0.082
  against the 1.0 threshold (12–19× headroom, zero fires in ~1M batches). The
  norm is now computed on every Nth batch only; skipped batches still run the
  freeze-passthrough housekeeping. Safety: if a sampled norm ever exceeds
  half the threshold, per-batch scanning is restored for the rest of the run.
  `--bt-clip-every 1` recovers the old behaviour exactly.

Both changes verified bit-identical to the previous trainer at 8 threads and
at 1 thread (same corpus/seed, identical `.nnue` + `.tdleaf.bin` MD5s).
Combined throughput at batch 512: 82k → 95k pos/s on 8 cores.

**Batch-size sweep (2026-07-09/10): batch 2048 trades ~7 Elo for 1.8×
speed; do NOT raise the LR.** Three arms at `--bt-batch 2048` on the
137.7M-position material-line corpus, identical starting state to the
batch-512 reference run, 4–6 epochs, each epoch laddered vs the same pretrain
anchor (1000 games at 1+0.01, ±11):

| arm | ep1 | ep2 | ep3 | ep4 | ep5 | ep6 | plateau |
|---|---|---|---|---|---|---|---|
| b512, lr 0.25 (reference) | +50 | +59 | **+86** | +84 | — | — | ~+85 |
| b2048, lr 0.25 | +38 | +72 | +76 | **+82** | +77 | +78 | ~+78 |
| b2048, lr 0.5 (√4 scaling) | +69 | +68 | +75 | +59 | — | — | rolls over |
| b2048, lr 1.0 (linear scaling) | +74 | +59 | +76 | +47 | — | — | rolls over |

Findings: (1) **LR scaling rules don't apply** — Adam's normalized steps
largely absorb the batch-size change, so the classical √/linear batch-LR
scaling just runs hot: both scaled arms peak early then roll over, losing
20–30 Elo by epoch 4-6 equivalents. (2) At unchanged LR, batch 2048 converges
~1 epoch later to a plateau ~7 Elo below batch 512's (~+78 vs ~+85, each
point ±11 — borderline individually, consistent across three plateau points
each). (3) Throughput at 2048 was 103–152k pos/s in production vs 66–83k at
512 (~1.8×). Val-MSE note: the lr 1.0 arm's val MSE fell monotonically while
its strength collapsed after ep3 — the strongest rate-by-gauntlet-never-by-
val-loss datapoint yet.

**Why sharding was removed (historical).** The earlier `--bt-sync` scheme ran
N independent optimizer processes on diverging weight copies, delta-merging
every ~256 batches. That gradient *staleness* was benign for the large
generation-1 backlog signal but **destroyed the subtler generation-2 signal**
— every 8-way sharded arm regressed while the identical single-process run
gained (the live tell was oscillating per-epoch validation MSE).
Within-batch threading has no staleness anywhere — nobody steps the weights
while another thread computes against an old copy — so it delivers the
sharded throughput with exact single-process semantics. The multi-process
delta-merge save protocol itself stays in the engine; concurrent *online*
self-play training still uses it.

---

## Superseded Adam Hyperparameters

> **⚠️ HISTORICAL VALUES — DO NOT USE.**  This section documents an OLD Adam
> hyperparameter regime from 190k-game weight-distribution analysis
> (`TODO.md`, pre-dating the current tuning) and is superseded and
> contradicted by current values.  The current Adam LR values live in
> `engine/CLAUDE.md`'s TDLeaf section and `../TRAINING.md`: FC0/FC1 weights
> **0.005**, FC2 weights **0.07**, FC biases **1.5**, FT weights **0.015**,
> FT bias **0.02**, PSQT **13.0**.  The values below — FC LR0 0.1/0.13, FT
> LR0 1.0, FT bias LR0 0.01, PSQT LR0 1.6 — do NOT match current reality; they
> are preserved only as a historical record of an earlier calibration.

### Adam hyperparameter tuning

The Adam optimizer uses five separate LRs tuned from 190k-game weight
distribution analysis. Key monitoring points:

- **FC LR0 (0.1):** FC1 saturation at 0.5% after 190k games; stacks 5,6 at
  1.2–1.7%. Reduced from 0.13 to extend runway before saturation becomes
  problematic.
- **FT LR0 (1.0):** FT weights barely changed (std 44.006 vs 44.0 init). With
  only 3–50 updates per weight, FT learning is very slow; high LR compensates.
- **FT bias LR0 (0.01):** Separate LR prevents dying-ReLU from update
  frequency asymmetry.
- **PSQT LR0 (1.6):** PSQT barely moves (std change -44 from 35642 over 190k
  games). Correct behavior — dense piece_val handles material corrections.

---

## Resolved / Implemented Changelog (from TODO.md)

*This is the full dated changelog of implemented/resolved training-system
changes carried over from `TODO.md`, preserved wholesale for historical
reference.*

### ~~Persistent Adam v~~ ✓ Implemented (2026-04-02)
Adam second-moment (v) arrays and t_adam persisted to .tdleaf.bin v6.
Multi-writer merge uses max(v_file, v_local) per element. FT weight v (~92
MB) excluded. Momentum (m) not persisted — recovers in ~10 steps. ~564 KB
file size increase.

### ~~Separate FT bias LR~~ ✓ Implemented (2026-04-01)
`TDLEAF_ADAM_FT_BIAS_LR0 = 0.01` (10× slower than FC) prevents dying-ReLU
from update frequency asymmetry. FT biases update ~200×/game vs FT weights
~8/5000g.

### ~~--init-nnue-noprior~~ ✓ Implemented (2026-04-01)
All piece PSQT values initialised at 100cp (uniform) instead of classical
material. Forces material value learning from scratch. training_run.py
offers the choice.

### ~~FC-only replay~~ ✓ Implemented (2026-03-29)
Replay with FT/PSQT gradients causes eval divergence; `fc_only=true`
suppresses FT/PSQT/FT-bias/piece_val gradients during replay. +20 Elo over
5000 games. Subsequently disabled (TDLEAF_REPLAY_K=0) as benefit faded after
first 5000 games.

### ~~Init-nnue redesign~~ ✓ Implemented (2026-03-23)
Weight initialization redesigned for TDLeaf training (decoupled from SF15.1
statistics). FT weights N(0,5), FC weights N(0,{1,3,2}), all means zero,
PSQT pure material (no piece-square bonuses). Separate
`TDLEAF_ADAM_FT_LR0=1.0` for sparse FT weights.

### ~~Flavor A replay~~ ✓ Implemented (2026-03-21)
Replay now rebuilds accumulators from stored leaf positions using current FT
weights, ensuring FT gradients during replay are self-consistent with the
current network. `TDRecord` stores the leaf `position` (~300 bytes/ply, ~6%
size increase).

### ~~Per-weight bias correction~~ ✓ Implemented (2026-03-21)
FC and PSQT Adam steps use per-weight bias correction (`eff_t = cnt + 1`)
instead of global `t_adam`. bc1 skipped at cnt≥20 (negligible); bc2 always
applied. FT RMSProp retains global bc2 (sparse features need growing global
correction).

### ~~Per-weight LR decay removed~~ ✓ Removed (2026-03-22)
Per-weight LR decay (`TDLEAF_ADAM_C`, `TDLEAF_ADAM_LR_FLOOR`) removed. AdamW
weight decay now handles regularization; LR0 tuned directly to the right
value (0.01 FC, 1.6 PSQT) instead of starting high and decaying. `--set-cnt`
and `_prompt_init_cnt` also removed as they existed only to prime the LR
decay schedule.

### ~~AdamW decoupled weight decay~~ ✓ Implemented (2026-03-21)
`TDLEAF_WEIGHT_DECAY=1e-4` applied to FC weights and FT weights after each
Adam step. Skipped for biases (no benefit) and PSQT (would fight classical
prior).

### ~~Gradient clipping by global norm~~ ✓ Implemented (2026-03-21)
`TDLEAF_GRAD_CLIP_NORM=1.0` clips the global L2 gradient norm before each
Adam step. Applied in `tdleaf_update_after_game`, `tdleaf_replay`, and
`tdleaf_flush_batch`. Set to 0 to disable.

### ~~Asymmetric lambda~~ ✓ Implemented (2026-03-21)
`TDLEAF_LAMBDA_DECISIVE=0.8` for wins/losses, `TDLEAF_LAMBDA_DRAW=0.5` for
draws. Decisive games get longer eligibility traces; draws use shorter
traces to reduce balanced-position noise. Set both to the same value for
symmetric behaviour.

### ~~Mini-batch gradient accumulation~~ ✓ Implemented (2026-03-19)
Gradients accumulated across `TDLEAF_BATCH_SIZE=16` games before each Adam
step. Reduces single-game gradient noise and file I/O. `tdleaf_flush_batch()`
applies any pending partial batch at session end. Set `TDLEAF_BATCH_SIZE=1`
to restore per-game updates.

### ~~Per-weight FT second moment~~ ✓ Implemented (2026-03-19)
FT weights upgraded from per-row RMSProp v (~88 KB) to per-weight v (~92 MB,
OS lazy-paged). Each of the 1024 dimensions within a feature row now has its
own variance estimate, allowing the optimizer to adapt step sizes
per-dimension rather than using a coarse per-row average.

### ~~LR warmup~~ ✓ Implemented (2026-03-19)
Linear warmup over first `TDLEAF_ADAM_WARMUP=50` Adam steps. Prevents
early-training instability from cold-start v estimates. Set
`TDLEAF_ADAM_WARMUP=0` to disable.

### ~~Adam optimizer~~ ✓ Implemented (2026-03-15), LR decay removed (2026-03-22)
Adam optimizer with fixed LR (constant after warmup). Per-weight LR decay
was removed in favour of direct LR tuning + AdamW weight decay. FC:
`TDLEAF_ADAM_LR0=0.01`; FT: `TDLEAF_ADAM_FT_LR0=1.0`; PSQT:
`TDLEAF_ADAM_PSQT_LR0=1.6`. FC0/FC1 float shadows clamped to ±127 to prevent
zombie weights. See `../TRAINING.md` for the current Adam optimizer reference.

### ~~Epoch-based replay~~ ✓ Implemented (2026-03-11)
Flavor B is live with `TDLEAF_REPLAY_K=1` (default) and
`TDLEAF_REPLAY_BUF_N=8`. Ablation: K=1 is the current conservative default
(K=2 marginal gain; K=6 large regression).

### ~~Bias initialisation~~ ✓ Implemented (2026-03-11)
FC biases (FC0/FC1/FC2) and FT biases are zero-initialised in `--init-nnue`
mode. Random N(μ,σ) from SF15.1 was removed — it added noise TDLeaf must
first cancel.

### ~~Linux compilation~~ ✓ Resolved (2026-03-11)
AVX2 x86-64 SIMD paths added to `nnue.cpp` for SqrCReLU, FC0, and FC1.
Fallback chain: NEON (ARM) → AVX2 (x86-64 with `-mavx2`) → scalar. Default
build uses `-march=x86-64-v3`; use `NATIVE=1` for CPU-native tuning.
</content>
