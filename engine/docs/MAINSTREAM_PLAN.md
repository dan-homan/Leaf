# Mainstreaming Plan: Internal Self-Play

**Date:** 2026-07-07 (Phases A–C); this file trimmed to the still-pending phases
**Status:** Phases A–C (pure-PSQT + ply semantics) are **done** — see
`docs/history/TRAINING_HISTORY.md` for the full experimental record, phase-by-phase
implementation notes, and gates. Phases D–E below are **approved direction, not
started**.
**Baseline:** branch `frozen-psqt` at commit `da9e57a` (experimental state, validated); `main` at `fa20daa`
**Companion docs:** `docs/TRAINING.md`,
`~/.claude/projects/-Users-homand-Leaf/memory/single-process-selfplay-tdleaf-plan.md` (original
internal-self-play design), `~/.claude/projects/-Users-homand-Leaf/memory/frozen-psqt-experiment.md`
(full experimental record)

This document is the implementation roadmap for internal self-play (Phases D and E).
It assumes the pure-PSQT material representation and the per-record-STM/game-ply
ply semantics from Phases A–C (`docs/history/TRAINING_HISTORY.md`) are already in
place — Phase D's `tdleaf_record_ply` reuse and gradient path depend on both.

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

**All training/rating binaries execute only from `learn/` or `<tag>_work/epoch_binaries/`,
never `run/`.** `run/` contains `main_bk.dat` — a binary executing there gets book moves and
corrupts the comparison; `run/` is only ever a transient compile-output location (see
`docs/TRAINING.md`, "The `run/` invariant"). `match.py` auto-detects FRC openings.

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
  the root distortion — see `docs/history/TRAINING_HISTORY.md`). Orthogonal to everything here;
  needs its own branch and its own 500k rig run. Do not share a validation run with the ply/process work.
- Scale regularizer for PSQT drift: only if a drift canary fires (see `docs/TRAINING.md`).

## Sequencing summary

```
A  merge pure-PSQT, defaults on          DONE — see docs/history/TRAINING_HISTORY.md
B  delete gauge machinery, v12           DONE — see docs/history/TRAINING_HISTORY.md
C  per-record STM + λ^Δ, harness mode    DONE — see docs/history/TRAINING_HISTORY.md
D  internal self-play, single board      gate: TD-error metric (D2), then 500k rig (D4)
E  multi-board                           gate: globals audit, then rig again
```

Line numbers cited are as of `da9e57a` — re-verify before editing; the unity build means LSP
per-file diagnostics are unreliable (expected false positives).
