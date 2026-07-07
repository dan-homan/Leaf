# Mainstreaming Plan: Pure-PSQT + Ply Semantics + Internal Self-Play

**Date:** 2026-07-07
**Status:** approved direction; implementation not started
**Baseline:** branch `frozen-psqt` at commit `da9e57a` (experimental state, validated); `main` at `fa20daa`
**Companion docs:** `docs/TDLEAF.md`, `docs/OFFLINE_TRAINING.md`,
`~/.claude/projects/-Users-homand-Leaf/memory/single-process-selfplay-tdleaf-plan.md` (original
internal-self-play design), `~/.claude/projects/-Users-homand-Leaf/memory/frozen-psqt-experiment.md`
(full experimental record)

This document is the implementation roadmap. It is self-contained: experimental context first,
then five phases in strict order, each with its validation gate. **Do not reorder the phases** —
each is the test platform for the next.

---

## 0. Experimental context (why these changes)

All results below from branch `frozen-psqt`, 500k-game depth-6 training runs with matched batching
(single fastchess invocations of 50k/50k/100k/300k games, c=12, tc=inf, no adjudication, bookless
binaries in `learn/`), rated by gauntlet at 3+0.05 in a combined bayeselo frame anchored by
`Leaf_vclassic_eval` / `Leaf_vmaterial_eval` and a shared 0g checkpoint.

| Experiment | Config | Result @500k |
|---|---|---|
| Reference (`psqt-prior-*`) | full gauge machinery: mean-centering, dw centering, slot-mean recentering, pawn pin, trainable piece_val | +78 (frame) |
| Phase 1 (freeze PSQT + piece_val) | all material channels frozen at classical prior | **FAILED** ~−200 vs ref |
| Phase 1b (freeze PSQT only) | piece_val trainable w/ pawn pin | **FAILED** ~−200 vs ref |
| Pure-PSQT (`pure-*`) | one trainable channel: PSQT free, NO centering/pin/piece_val, search values fixed classical | +47 (frame), −33±26 vs ref |
| **Pure-PSQT + offline (`pure-bt-ep4`)** | above + 4 epochs batch-train on own 80M-position corpus | **+149 (frame): +71 over ref, +102 over own online endpoint** |

**Failure mechanism (phases 1/1b), do not re-attempt freezing:** outcome-scaling pressure is
phase-dependent (per material bucket). The bucketed PSQT is the only *linear* absorber for it;
`piece_val` cannot substitute (it is bucket-independent — 1b proved this directly). With PSQT
frozen, the pressure routes through the int8-clipped FC stacks and blows out per-bucket output
scale (won KQP-vs-K endgame evaluated +4033cp vs reference +1286cp; collapsed gauntlet draw rates).

**Why pure-PSQT is safe without the gauge machinery:** the machinery existed because PSQT and
`piece_val` redundantly encode material level → a gauge null direction the multi-writer merge can
amplify geometrically. Delete `piece_val` as a training channel and there is ONE material channel:
no null direction, nothing to amplify. Absolute scale is loss-anchored via `TDLEAF_K = 220` under
outcome-dominated λ-return targets. Measured drift: extracted pawn 100→107cp per 500k online
games, +7cp more per 4 offline epochs — slow, and cosmetic because search piece values are fixed
(see `NNUE_FIXED_PIECE_VALUES` below).

**Rating discipline:** rate by gauntlet, never by val-MSE level. Val-MSE is a gate for "not
diverging," not a fitness function.

---

## Phase A — Merge pure-PSQT to main (defaults on, no deletions yet)

### A1. Merge `frozen-psqt` → `main`

Defaults after merge (already the state at `da9e57a`):

- `src/tdleaf.h`: `TDLEAF_PURE_PSQT = true`, `TDLEAF_FREEZE_PSQT = false`
  (`TDLEAF_PIN_PAWN_VALUE` is moot in pure mode)
- `src/define.h`: `NNUE_FIXED_PIECE_VALUES 1` — search keeps classical
  `value[] = {0,100,377,399,596,1197,10000}`; `nnue_extract_piece_values()` in `src/nnue.cpp`
  computes implied values for the banner ("report only; search uses classical") but does not
  assign `value[]`.
- Gates already in `src/nnue_training.cpp` (verify they survive the merge):
  - piece_val gradient accumulation and piece_val Adam block: `if (piece_val_active && !TDLEAF_PURE_PSQT)`
  - Pass 2 dw centering: `if (!TDLEAF_FREEZE_PSQT && !TDLEAF_PURE_PSQT && ...)`
  - `nnue_mean_center_psqt_gradients()` and `nnue_recenter_psqt_slot_means()`: early-return on
    `TDLEAF_PURE_PSQT`

### A2. Multi-writer merge smoke test — the ONE thing 500k didn't exercise

The pure run was effectively single-lineage. The historical reason for centering was
merge-amplified gauge drift. Pure-PSQT removes the null direction *in theory*; test it with the
actual merge code:

1. Two TDLEAF training processes, same `.tdleaf.bin`, concurrent self-play, ~5–10k games.
2. Diff checkpoints with `scripts/diff_tdleaf_checkpoints.py`; watch extracted piece values
   (`P` especially) and FC bias means across a few save/merge cycles.
3. **Gate:** no accelerating drift across merge cycles (pawn stays within a few cp of its
   pre-test value; FC bias means stable). If it drifts, stop and investigate before Phase B —
   the merge path may need count-weighting fixes rather than restored centering.

### A3. Docs

Update `docs/TDLEAF.md` and `engine/CLAUDE.md`: pure-PSQT is the recipe; move gauge-anchoring to a
"historical / why not" section; record the phase-1/1b failure mechanism prominently so freezing is
never re-attempted. Log in `docs/change_log.txt`.

---

## Phase B — Delete the gauge machinery, format v12

Pure deletion, after A2 passes.

### B1. Delete from `src/nnue_training.cpp` / `src/tdleaf.cpp` / `src/tdleaf.h`

- `nnue_mean_center_psqt_gradients()` (and its call site)
- post-Adam dw centering in `nnue_apply_gradients` Pass 2
- `nnue_recenter_psqt_slot_means()` (load-path and save-path call sites)
- pawn pin (`TDLEAF_PIN_PAWN_VALUE`)
- the entire `piece_val` training channel: gradient accumulation, Adam moments, the ≥0 clamp,
  LR `TDLEAF_LR_PV` env override
- flags `TDLEAF_FREEZE_PSQT` / `TDLEAF_FREEZE_MATERIAL` / `TDLEAF_PURE_PSQT` (pure is now the
  only mode; keep the *name* out of the code, keep the *explanation* in TDLEAF.md)

Keep: `--init-nnue-classical` and `--init-nnue-noprior` unchanged (both bake material into PSQT —
exactly what one-channel wants). Keep `NNUE_FIXED_PIECE_VALUES` as a define defaulting to 1.

### B2. `.tdleaf.bin` format v12

- Drop: piece_val weights + their Adam moments; the 88 persisted PSQT init slot-means (v11 field).
- Loader: accept v11 (ignore dropped fields) and v12; writer emits v12 only.
- Update `scripts/merge_tdleaf.py` (class `TDLeafFile`), `scripts/compare_nnue_learning.py`,
  `scripts/diff_tdleaf_checkpoints.py` for the new layout.
- **Gotcha that will bite:** `--init-nnue` over an existing `.tdleaf.bin` MERGES with it (save
  path is merge-save). Any fresh-init workflow must `rm` the old file first. Preserve this
  warning in docs; better, make `--init-nnue` refuse to run if the companion `.tdleaf.bin`
  exists.

### B3. Drift canary (replaces the pin — monitor, don't constrain)

- `scripts/hybrid_loop.py`: log extracted piece values (the report-only banner values) at every
  checkpoint; warn if pawn leaves [85, 130] cp.
- The only *functional* coupling to drift is `TDLEAF_SCORE_CLIP_PAWNS` (units of
  max(value[PAWN], 100cp) — with fixed classical values this stays 100cp exactly, so the clip
  does NOT stretch with drift. Good; note it in TDLEAF.md).
- Do NOT add a scale regularizer preemptively. If the canary ever fires at the 2.4M-game
  horizon, the fix is a soft pull of extracted-pawn toward 100 with tiny weight (~10 lines).

### B4. Gate

A fresh `--init-nnue-classical` + 50k-game depth-6 run on the v12 build must track the pure-PSQT
50k checkpoint (`Leaf_vpure-5e4g`, still in `learn/`) within noise in a 1000-game 1+0.01 match.

---

## Phase C — Ply semantics: per-record STM + game-ply λ^Δ (harness mode)

These are the mode-agnostic wins from the internal-self-play plan, landed and validated *in the
existing two-process harness mode* where behavior can be regression-tested bit-for-bit. Original
design in `single-process-selfplay-tdleaf-plan.md`; the additions below are critique fixes.

### C1. Per-record STM/sign (do this first — biggest correctness surface)

Current code hard-assumes a single-color trajectory:

- `rec.engine_color` is per-GAME (`src/tdleaf.cpp:53` at da9e57a — re-verify line numbers)
- propagated-score sign flip at `tdleaf.cpp:91` derives from it
- TSV dump uses `root_wtm = rec.engine_color` (`tdleaf.cpp:322`)

Changes:

1. Add per-record STM to `TDRecord`; outcome term z applied from each record's own STM POV.
2. **Centralize ALL POV/sign logic in ONE helper.** A half-inverted-targets bug here is silent.
3. **Edit surface beyond the original plan** — two mechanisms operate on *successive records*
   and are POV-sensitive once STM can alternate:
   - `TDLEAF_SCORE_CLIP` (compares consecutive scores — must negate across a POV flip, or every
     quiet position looks like a ±2×eval swing and the clip fires constantly, silently neutering
     learning)
   - ID-stability variance weighting (`TDLEAF_ID_VAR_SIGMA2`, operates on `id_scores[]`
     histories — same negation requirement)
   Route both through the central helper.
4. Rework the `tdleaf.cpp:91` sanity check and the `TDLEAF_DUMP_TSV` dump for per-ply color.

**Gate C1 (invariance tests, write them before the refactor):**
- (a) In harness mode (all records same STM) the refactor is a no-op: fixed game →
  **bit-identical gradients** vs pre-change build.
- (b) Sign-symmetry: same fixed game recorded from white POV vs black POV → identical updates.
- (c) Mirror test: color-flipped game → exactly mirrored gradients.

### C2. Game-ply accounting + λ^Δ

1. Store true game-ply in each `TDRecord`.
2. Trace update (`tdleaf.cpp:192` at da9e57a): `e[t] = delta_d + lambda*e[t+1]` →
   `e[t] = delta_d + pow(lambda, dply)*e[t+1]`, `dply` = game-ply gap to the next record.
   Harness mode has dply=2; internal self-play will have dply=1; ONE λ expresses the same
   real-game horizon in both.
3. **λ default retune — not in the original plan, mandatory:** today λ=0.98 decays per
   own-move step = per 2 game-plies. Under λ^Δ, preserving current harness behavior requires
   `TDLEAF_LAMBDA = sqrt(0.98) ≈ 0.98995`. Leaving 0.98 would silently change the harness
   horizon to λ_eff=0.9604.
4. TSV dump: ply column becomes game-ply.
5. **Corpus version marker — not in the original plan, mandatory:** the offline trainer
   (`--batch-train` in `src/nnue_batch_train.cpp`) computes its own λ-return over TSV records
   and must switch to the game-ply axis (λ^Δ over the ply column) in the SAME commit. Add a
   version header/column to the TSV format; `--batch-train` must refuse pre-change corpora
   (e.g. `pure500k.*.tsv`) or handle them with the old axis explicitly. Note `--bt-lambda 1.0`
   (the settled gen-3 recipe) is exponent-invariant, but the td_λ=0.98 component and any future
   λ<1 sweep are not.

**Gate C2:** harness mode with λ = √0.98 → bit-identical gradients to the old code on a fixed
game (up to fp rounding in `pow`; if not exactly identical, assert max relative deviation < 1e-6).

---

## Phase D — Internal self-play (single board)

One process plays whole games against itself, both sides recorded, learning at game end.
Original design: `single-process-selfplay-tdleaf-plan.md`. Benefits: 1-ply TD bootstrapping,
true full-game traces, ½ the processes, own the result (clean mate/repetition/50-move detection,
no `tdleaf_self_adjudicate` fallback), maximal outcome symmetry (strongest anti-drift — see
`tdleaf-fixed-opponent-bias-collapse` memory), **and it eliminates the FT session-warmup trap**
(`TDLEAF_FT_SESSION_WARMUP=100` batches paid once per run instead of per fastchess invocation —
this is what forced the games-per-invocation ≥50k rule and cost a scrapped 500k run).

### D1. The game loop

New mode (suggest `--selfplay <N games>`): pick opening from EPD (FRC+book,
`learn/training_openings.epd`), search every ply at fixed depth (reproducibility; depth 6
matches the experimental rig), `tdleaf_record_ply` with per-record STM (Phase C machinery),
detect result in-process, `tdleaf_update_after_game`. Reuse: `tdleaf_record_ply`
(`src/uci.cpp:481`, `src/main.cpp:664` at da9e57a), `tdleaf_update_after_game`, opening EPDs,
and (for N>1 processes) the multi-writer merge.

Games run to completion (mate/stalemate/threefold/50-move/max-moves) — the no-adjudication
training rule is now enforced by construction; keep it that way.

**Capacity check:** both-sides recording = 2× records per game. Verify `TDGameRecord` sizing
against `MAX_GAME_PLY` for completion-rule games; expect 2× TSV size (the 500k corpus was
already 80M positions). Batch-of-8-games now holds 2× records — Adam's step bounding absorbs
most of it, but eyeball first-run step-clip telemetry (`TDLEAF_LOG_STEP_CLIPS=1`) rather than
assuming.

### D2. THE decisive experiment — run this before building the TT salt

The mode's headline benefit (1-ply bootstrapping) and its main risk are the same quantity: the
1-ply TD error between my eval at ply t and the opponent's at t+1 is exactly what collapses
toward zero if both searches share a TT and a net (negamax-consistent pair). So, **first**, run
internal mode WITHOUT any decorrelation and instrument:

- |TD error| distribution per ply
- bootstrap-vs-outcome contribution split in the applied gradients

Compare vs the two-process baseline (same net, same openings, same depth).

- Distributions match → skip the salt entirely (simpler); proceed to D4.
- TD errors collapsed → implement D3, confirm recovery, then D4.
- Salt doesn't recover → the mode keeps only its efficiency/symmetry benefits; decide then
  whether that justifies it.

### D3. TT salt (only if D2 says so)

XOR a fixed per-side salt into the TT key ONLY, at the 4 call sites
(`put_hash`/`get_hash`/`get_move`/`put_move`): pass `pos.hcode ^ side_salt`, salt chosen by
root STM, constant through the subtree. Fixed constant, not per-game random. Cost: ~2× TT
working set (fine for training).

**LANDMINE (from the original plan, preserved verbatim in spirit):** do NOT salt `pos.hcode`
itself. The same value goes into `ts->plist[]` for repetition detection (`search.cpp:1195`,
`:2092` at da9e57a) — salting it silently breaks threefold/fifty-move legality. Repetition,
plist, and the pawn/NNUE/score eval caches stay on the TRUE hcode (eval caches are
position-deterministic; sharing them is correct, not a leak).

If residual coupling remains after the salt: next lever is per-side history/killer/countermove
tables — but TT is ~90% of it; don't build until measured.

### D4. Rate it — the frozen-psqt rig is the reusable harness

500k games depth 6 from the same `--init-nnue-classical` seed, checkpoints at 50k/1e5/2e5/5e5,
baked nets, gauntlets at 3+0.05 vs the existing `learn/` anchor population
(`Leaf_vclassic_eval`, `Leaf_vpsqt-prior-*`, `Leaf_vpure-*`, `Leaf_vpure-bt-ep4`), combined
bayeselo. This frame resolved both +98 and −33±26 cleanly; trust it. Then run the offline
consolidation pass (4 epochs, `--bt-lambda 1.0`, on the new game-ply-axis corpus) and gauntlet
the epochs — screening 1000g @ 1+0.01 vs the online endpoint, best epoch → 500g @ 3+0.05 finals.

**All experiment binaries live in `learn/` (bookless). `run/` contains `main_bk.dat` — a binary
in `run/` gets book moves and corrupts the comparison.** match.py auto-detects FRC openings.

---

## Phase E — Multi-board: N games, one process, one Adam stream (contingent)

The natural phase 2 of D, worth its own audit because it deletes the last distributed-training
complexity: the multi-writer `.tdleaf.bin` merge protocol.

**What it buys:**
- ONE Adam stream: no count-weighted averaging, consistent `t_adam`, no merge-amplification
  pathway even in principle (pure-PSQT removed the null direction; this removes the amplifier).
- Memory: today each of ~12 processes holds its own FP32 shadow weights + Adam state (~the
  263MB `.tdleaf.bin`, in RAM) + TT. One shared copy frees several GB → one large shared TT.
- Batch composition for free: games finish asynchronously across boards, so the gradient
  accumulator naturally interleaves across games (fixes the adjacent-ply-correlation concern
  structurally). Game-end updates are rare vs search; a mutex on accumulate/apply is ~free.

**The feasibility question — audit BEFORE committing (bounded, do first):** Leaf is
one-game-per-process with lazy-SMP threads assisting a single search. Multi-board inverts that:
N single-threaded searches over N independent game states. Lazy SMP already makes search
thread-safe against a shared TT (the hard part). Audit what `tdleaf_record_ply` and the game
loop touch that is NOT per-`tree_search`: `game` (`game_rec`), accumulator stacks, TDLeaf
per-game recording buffers, `engine_cfg`-adjacent globals (see `src/engine_globals.h`). If the
answer is "a handful of globals" → mechanical `struct GameContext` refactor. If game state is
tentacled everywhere → stay with N single-board processes + merge (Phase D standing alone loses
little).

**Known subtlety:** cross-*game* TT sharing correlates games with each other (shared opening
analysis), partially undoing batch diversity. Correctness is unaffected. If the D2 metric shows
it matters, the same salt mechanism extends per-board — but measure first, same discipline as D3.

---

## Deferred (do NOT bundle with any phase above)

- **WDL head** (FC2 32→3 + translation to cp for search): strongest phase-2 *architecture* idea;
  motivation strengthened by phase-1's blowup (outcome term stretching an unbounded cp head is
  the root distortion). Orthogonal to everything here; needs its own branch and its own 500k rig
  run. Do not share a validation run with the ply/process work.
- Scale regularizer for PSQT drift: only if the B3 canary fires.

## Sequencing summary

```
A  merge pure-PSQT, defaults on          gate: merge smoke test (A2)
B  delete gauge machinery, v12           gate: 50k tracks Leaf_vpure-5e4g (B4)
C  per-record STM + λ^Δ, harness mode    gate: bit-identical-gradient tests (C1/C2)
D  internal self-play, single board      gate: TD-error metric (D2), then 500k rig (D4)
E  multi-board                           gate: globals audit, then rig again
```

Line numbers cited are as of `da9e57a` — re-verify before editing; the unity build means LSP
per-file diagnostics are unreliable (expected false positives).
