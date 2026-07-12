# Leaf TODO

Planned investigations, improvements, and open questions. Resolved items and
experiment write-ups have moved to `docs/history/TRAINING_HISTORY.md` — this
file tracks only what's still open.

---

## TDLeaf(λ) Training

### Internal self-play — Phases D & E (deferred; tackle together)

**Full spec:** `docs/MAINSTREAM_PLAN.md` (Phases D and E).  Original design:
`~/.claude/projects/-Users-homand-Leaf/memory/single-process-selfplay-tdleaf-plan.md`.

Phases A–C of that roadmap are **done** (pure-PSQT mainstreamed + format v12; the
gauge machinery deleted; per-record STM + game-ply λ^Δ landed and gated in the
two-process harness — see `docs/history/TRAINING_HISTORY.md` for how that work
went).  The remaining phases are the payoff — the first code that actually
exercises the alternating-STM / `dply=1` paths C1/C2 built:

- **Phase D — internal self-play (single board).**  One process plays whole games
  against itself, records *every* ply (per-record STM, `dply=1`), and learns at
  game end.  Benefits: 1-ply TD bootstrapping, true full-game traces, ½ the
  processes, own the result (clean mate/rep/50-move detection, no
  `tdleaf_self_adjudicate`), maximal outcome symmetry, and it retires the
  FT-session-warmup-per-invocation trap.  **Decisive gate D2 first:** run internal
  mode WITHOUT decorrelation and measure the |TD error| distribution +
  bootstrap-vs-outcome split vs the two-process baseline — that decides whether
  the TT salt (D3) is needed at all before spending the 500k rig (D4).

- **Phase E — multi-board (contingent).**  N games in one process, one Adam
  stream, retiring the multi-writer `.tdleaf.bin` merge protocol entirely.  Gated
  on a bounded globals audit first (`GameContext` refactor feasibility).

Reuse: `tdleaf_record_ply` (now takes per-record STM + `game_ply`),
`tdleaf_update_after_game`, opening EPDs, and (for N processes) the merge
protocol.  See MAINSTREAM_PLAN.md for the exact gates and sequencing.

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

## Known Issues

(none currently open)
