# Leaf TODO

Planned investigations, improvements, and open questions. Resolved items and
experiment write-ups have moved to `docs/history/TRAINING_HISTORY.md` — this
file tracks only what's still open.

---

## TDLeaf(λ) Training

### Internal self-play — Phases D & E (DELIVERED)

**Full spec:** `docs/MAINSTREAM_PLAN.md` (Phases D and E).  Original design:
`~/.claude/projects/-Users-homand-Leaf/memory/single-process-selfplay-tdleaf-plan.md`.

Phases A–C are done (pure-PSQT + format v12, gauge machinery deleted, per-record
STM + game-ply λ^Δ), and **Phases D and E have since landed:**

- **Phase D — internal self-play (single board)** shipped as the engine's
  `--selfplay` driver (`src/selfplay.cpp`): one process plays whole games against
  itself, records every ply (per-record STM, `dply=1`), owns the result (clean
  terminal detection), and learns at game end.
- **Phase E — single Adam stream** shipped as the **actor/learner split**
  (`--traj-out` / `--learn-stream`, driven by `scripts/selfplay_run.py`): N−1
  frozen actors emit `.tdg` trajectories, ONE learner owns the optimizer.  This is
  now the default `train.py` generation mode.  Fully retiring the in-engine
  multi-writer `.tdleaf.bin` merge protocol (only the legacy `--selfplay-gen`
  path still needs it) is tracked as Phase 3 in `docs/SIMPLIFICATION_PLAN.md`.

### Open items

- [ ] Iteration 3+: long d8 online generation from a consolidated net (needs
      `--recompile` so the dump binaries emit the exact `endply` column);
      gauntlet vs a fixed opponent panel, plus a direct promoted-net-vs-classic
      anchor match (family-chained Elo reads ~20–30 optimistic).
- [ ] Online `TDLEAF_K` runtime override: compile-time 220 in `tdleaf.h`, but
      consolidated nets' evals sometimes fit a lower K — likely explains mild
      online piece-scale drift. Add an env override and test online generation
      at the refit K.
- [ ] Outcome-baseline subtraction (`e' = e − EMA(engine-POV mean e)`) —
      designed, unimplemented; needed only if imbalanced-opponent training
      (score far from 50%) becomes a first-class mode.  See
      "Outcome-Imbalance Drift" in `TRAINING.md`.
- [ ] Bayeselo pool rating (not head-to-heads) once a consolidated net gets
      close to classic_eval.

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

### Tactical signal under NNUE — follow-ups to qchecks restoration

The 2026-05-04 fix restored `pos.qchecks[]` (king-tropism) under NNUE,
which recovered the check-extension and qsearch-with-checks paths.
Several follow-up investigations are worth running now that the search
has a working tactical signal again.

**Tune the qchecks threshold for NNUE.**  The classical-eval gating
(`gstage < KING_SAFETY_STAGE` where `KING_SAFETY_STAGE = 10`) was
calibrated for the classical eval's score distribution.  NNUE may
benefit from extending the gate further into the endgame (e.g. 12 or
14) — NNUE handles complex king-and-rook endgames where classical eval
was weaker, and check-extensions there might help.  Self-play match
with KING_SAFETY_STAGE=12 vs 10 to measure.

**Singular-extension margin (SMARGIN) tuning.**  The 25-cp default was
calibrated against classical scores.  NNUE scores have a different
noise distribution (more depth-to-depth variance in the opening, less
in the middlegame).  Test SMARGIN={25, 40, 60} in self-play; the
correct value with NNUE may be ~40 cp.  See also the per-position
analysis in the 2026-05-04 diagnostic — startpos fired sing_ext 22×
more than EXchess, suggesting SMARGIN=25 is too tight in symmetric
opening positions.

**Option B from the original qchecks investigation: NNUE-native
proxy.**  The current fix runs a small classical-style piece loop on
every cache miss.  An alternative is to derive qchecks from cheap
board-state info that doesn't reference the classical eval at all,
e.g. `popcount(enemy_attackers_within_2_squares_of_king)`.
Behaviorally similar but removes the residual coupling to the
classical eval code path.  Worth A/B testing if the current fix's
~5% node-count overhead is meaningful.

**Re-tune extension trigger thresholds.**  The check-extension branch
also gates on `moves.mv[mi].score > 1000000`.  This score threshold
was set against the classical move-ordering scoring scheme.  With S1
(per-ply killers) and the new qchecks signal, the move scores hitting
this branch may have shifted.  Worth profiling which moves trigger
extensions and verifying the threshold still selects high-quality
checking moves.

**Consider NNUE-derived king-safety bonus alongside qchecks.**  Now
that we're running a small classical-style loop on every NNUE eval
miss, it's cheap to extract more information.  The NNUE accumulator's
PSQT terms encode king safety information; adding a small
king-safety adjustment from NNUE psqt_diff (in addition to the main
NNUE score) could in principle replace the classical king-safety
score that's still missing.  Speculative; worth investigating only
after the simpler tuning above is exhausted.

### Pawn hash under NNUE
The classical eval stores pawn structure scores in a pawn hash table.  The NNUE eval
bypasses classical eval entirely, so `pawn hash hits` is always 0 in NNUE mode and the
pawn hash memory (≈19 MB) is wasted.  Disabling or shrinking it at build time when
`NNUE=1` would recover that memory (no effect on playing strength).

### Multi-thread accumulator correctness

**Open correctness question, not just a nice-to-have.** The SMP search allocates
one `ts_thread_data` per thread, each with its own `search_node n[MAXD+1]` stack
including per-node accumulators.  Each thread's root accumulator is
independently initialised.  Thread interactions have not been tested under
NNUE; correctness is expected but **unverified with `THREADS > 1`**.

---

### Win-only .tdleaf.bin writes (`TDLEAF_WIN_ONLY_WRITE`)

Compile-time flag that suppresses writing `.tdleaf.bin` after draws and losses.
Gradients still applied to in-memory weights and other-process deltas still merged from disk on
all games; only the disk write is gated on `td_result >= 1.0`.  Requires refactoring
`nnue_save_fc_weights` to split its read+merge phase from its write phase.
See memory for full implementation plan.

---

# Internal Self-Play

> **DELIVERED (2026-07).**  Phases D and E below shipped: Phase D as the engine's
> `--selfplay` driver (`src/selfplay.cpp`) and Phase E as the actor/learner split
> (`--traj-out`/`--learn-stream` via `scripts/selfplay_run.py`), now the default
> `train.py` generation mode.  The roadmap below is retained as the historical
> implementation spec.  Remaining follow-on (retire the in-engine multi-writer
> merge) is Phase 3 in `docs/SIMPLIFICATION_PLAN.md`.

**Date:** 2026-07-07 (Phases A–C); this file trimmed to the still-pending phases
**Status:** Phases A–C (pure-PSQT + ply semantics) are **done** — see
`docs/history/TRAINING_HISTORY.md` for the full experimental record, phase-by-phase
implementation notes, and gates. Phases D–E shipped (see banner above).
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

------

## Known Issues

(none currently open)
