# Leaf TODO

Planned investigations, improvements, and open questions. Resolved items and
experiment write-ups have moved to `docs/history/TRAINING_HISTORY.md` — this
file tracks only what's still open.

> **Training/learning items: the rationale lives in
> `docs/Learning_Investigation.md`.**  That document holds the graded evidence
> ledger, the regime boundaries that decide whether an old result still holds, the
> closed lines with what would reopen each, and the ranked open questions (§6).
> This file is the checklist; where the two disagree about status, fix this one.

---

## TDLeaf(λ) Training

### Internal self-play — Phases D & E (DELIVERED)

**Full spec:** the "Internal Self-Play" roadmap later in this file (the former
`docs/MAINSTREAM_PLAN.md`, folded in here 2026-07-14).  Original design:
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
  now the sole `train.py` generation mode (the multi-writer `--selfplay-gen` and
  `--uci-pair-gen` paths were removed).  The follow-on cleanup that unblocked —
  replacing the in-engine multi-writer `.tdleaf.bin` merge with a plain atomic
  write — **also landed** (Phase 3, `docs/history/SIMPLIFICATION_PLAN.md`), validated
  byte-exact.  Nothing in this section is outstanding.

### Offline-phase plateau — ranked experiment plan (2026-09-02)

Evidence and measurements: `docs/history/Offline_Learning_Investigation.md` Part 1.  The
offline phase is data-starved on both of its channels at once — the eval bootstrap
is gated shut by `TDLEAF_DUMP_QUIET_CP = 60` (45% of root positions discarded,
ranked by informativeness), and the outcome channel gets one 500k-game corpus per
pass and overfits by epoch 2.  Arms below are ordered by expected value per unit
of compute.  All of them are judged by **foreign-anchor gauntlet**
(`Leaf_vclassic_eval`), never by validation MSE level; `--bt-diag` on a fixed
held-out corpus is the cheap between-arm proxy.

- [x] **A1 — Multi-corpus consolidation (offline only, no new games). DONE, +40 Elo.**
      Row-matched arms (95,028,415 rows each, 2 epochs, same seed/LR/steps/wall
      clock) differing only in game diversity: 500k games vs 2.5M.  Result
      **+35.8 ± 16.7 / +37.4 ± 16.6 vs the foreign anchor** (ep1/ep2), **+45.4 ± 11.1
      head-to-head**, and **+50.0 ± 11.1 over the net actually promoted** for that
      iteration — for zero extra compute and zero new games.  Validation MSE ordered
      the arms *backwards* and had no power to see it.  Full write-up:
      `docs/history/Offline_Learning_Investigation.md` Part 2.

- [x] **A1a — Fold A1 into `train.py`. DONE.**  `--corpus-window N` (default 4)
      walks the `--continue` chain, pulls each ancestor's archived corpus, and
      splits a fixed row budget (`--corpus-rows`, default = this run's own dump
      size) evenly across the sources — so epoch cost is unchanged and only the
      game diversity rises.  `--corpus-window-max-stale` filters on **generator**
      Elo; every source's generator Elo is logged and recorded in the sidecar
      (`corpus_window`, `corpus_rows`, `corpus_games`).  `--corpus-window 0`
      restores pre-A1 behaviour.  The loader `gid` fix landed with it.

- [ ] **A1b — Drop the stale corpus and re-run A1.**  `a1m` unknowingly included
      `3.5e6g`, whose labels come from a **+34.2** generator while the other four
      come from +103…+113 (the setup rationale used each iteration's own Elo, not
      its generator's — corrected in Part 2.1).  Re-run with `4e6g`+`4.5e6g`+`5e6g`
      +`5.5e6g` only, row-matched again.  Predicts a further gain; bounds how much
      label staleness costs.  Cheap to build: the archived union assigns gids in
      file order, 500k games each, so `3.5e6g` is exactly `gid <= 500000` — filter
      it out of `union_d8d10.tsv` and re-run the Bresenham row-match in
      `learn/offline_a1/build_a1m.sh`.

- [ ] **A1c — Epoch count and ladder opponent.**  Epoch 1 beat epoch 2 in *both* A1
      arms (+112.9/+105.3 and +148.7/+142.7), while the production family-opponent
      epoch ladder picked epoch 2 (48.3 vs 56.4) — the family ladder inverted the
      foreign anchor's ordering, not merely compressed it.  Re-examine whether the
      epoch ladder should use a foreign anchor, and whether 1 epoch is the right
      default on a diverse corpus.
- [x] **Loader: compact `gid` at load. DONE.**  `bt_load_file` sized `gid_N` by raw
      gid *value*, costing **7.2 GB** of padding on `4e6g` (max gid 3.6e9) and
      overflowing `uint32_t` across ~4 files.  Now maps each distinct raw gid to a
      sequential id (per-file map reset preserves the old cross-file namespacing):
      gid table 1.4 GB → **1.0 MB** on a 500k-game corpus, verified
      behaviour-identical (`--bt-diag` output bit-for-bit unchanged).

- [ ] **G1 — Generation questions raised by Part 3 (revisit before the next
      generation run).**  Root-only training won by +36 Elo, which puts three
      generation-side decisions on the table: (a) stop dumping leaf rows at all
      (they are 54% of dump I/O and disk for rows we no longer train on) — but
      settle Part 3.5's blend confound first, since it is the one thing that
      could rehabilitate them; (b) keep *more* root rows per game, since the
      root arm already consumed 92% of available root rows and scaling further
      needs more games, not more sampling; (c) **retain the generation PGN by
      default — DONE (2026-09-15).**  Actors write the games they play
      (`--selfplay --pgn-out`, fastchess-shaped `{score/depth time}` comments);
      `train.py` turns it on by default and archives
      `<tag>_work/<tag>_gen.pgn.gz`.  Measured at +0.23% of generation wall
      clock (inside the run-to-run spread; bound it at 0.5%) and ~304 MB
      gzip'd per 300k games.  See `TRAINING.md` "Game records".

- [x] **A2 — Widen the corpus quiet gate. DONE — the gate is correct; widening
      does not help.**  Enabled by the `gate` column (dumps re-cut offline via
      `--bt-quiet-cp`).  Row-matched arms over the same 100k games (7,829,099 rows
      each) at gate 60 / 120 / 200 / none: 60–200 flat, **none is 27.9 ± 11.3 Elo
      worse** on the pooled foreign anchor (2000 games/arm, replicated).  The
      discarded tail carries the label information (`ΔMSE_out` +52% in the top
      bin) but it is **not learnable** — those labels are good because search
      resolved a tactic.  Part 1's "the gate throws away the signal" reading is
      retracted; see `history/Offline_Learning_Investigation.md` Part 4.4.
      Scale caveat: 7.8M rows / ~15k steps had power for 28 Elo, not for 5–10, so
      a 60-vs-120-vs-200 difference of that size is not excluded — and a gate
      *tighter* than 60 was never tested (the diagnostic's negative `ΔMSE_out`
      below 40 cp hints the optimum may sit under 60).

- [ ] **A3 — Direct tactical exclusion instead of the eval-gap proxy.**  Now
      *lower* priority than when filed: A2 showed the eval-gap gate is not merely
      a proxy that happens to work, it is doing real damage-prevention, and
      widening it buys nothing.  A direct test (root in check, best move is a
      capture/promotion/check) might still dominate it — admitting mis-evaluated
      quiet positions while still excluding tactics is a strictly better filter
      *if* the two populations separate.  The A2 machinery makes this cheap to
      evaluate offline once the dump carries the tactical flags; that column is
      the only new work.

- [ ] **A4 — Drop or down-weight leaf rows.**  Leaf rows are 54% of every corpus
      and every offline pass, and buy `ΔMSE_out = +0.41%` against root rows'
      +2.08%.  `--bt-rows root` at 2× epochs costs the same wall clock.  Check
      first whether leaf rows are still earning their keep as distribution
      matching / magnitude anchor for the outcome term.
- [ ] **A5 — Epoch/LR schedule on the enlarged corpus.**  `--bt-epochs 2
      --bt-lr 0.25` flat has never been swept in this regime, and epoch 2 is
      currently harmful (validation `MSE(outcome)` rises).  On a 3–5× larger
      corpus (A1) the right answer probably moves; sweep epochs 1–4 with an LR
      decay across epochs.
- [ ] **A6 — Draw rate as a generation constraint.**  Depth 10 pushed self-play
      draws from 41% to 52%, cutting outcome information ~25% and costing the
      iteration 13 Elo.  Before raising depth again, treat draw rate as a hard
      health gate (35–40%) and buy decisiveness elsewhere — more unbalanced
      opening lines, or a wider book — rather than with depth.

### PV-leaf approximation quality — open line (2026-09-16)

Branch `tdleaf-pv-telemetry`.  The PV recorded in the triangular array is an
APPROXIMATION of the minimax line: the leaf position, its accumulator and its
static eval are all provably correct (TDLEAF_CHECK_ACC = 0 mismatches; an
off-by-one probe matches the leaf 43% against one-ply-back 2%), but the root
search score is not that leaf's static eval in ~45-55% of records.  Six causes
ruled out by experiment: aspiration clamping, score hash, leaf quiescence,
off-by-one, accumulator rebuild, fail-hard clamping.

The mis-approximation is SYMMETRIC — leaf higher 23.3% / lower 21.5%, mean
+1.13 cp against sd 65 — so it is variance, not bias.  That matters because
Sigma (gradient-noise covariance) is the one lever 7.14/7.15 left standing, and
7.15 showed it pays (batch 8->32 = +64.0 Elo, 5.0 sigma).

Approximation quality vs net maturity, m260720 chain, 120 games/point at d8/4000n:

  state     n      EXACT   <=10cp  <=25cp  >50cp    sd    bias
  1e5g    16,977   38.46%  82.48%  93.54%  3.42%   67.7  +1.16
  5e5g    14,267   49.29%  77.99%  89.36%  5.43%   72.4  +1.10
  1e6g    16,713   49.45%  80.31%  90.59%  4.53%   63.0  -0.22
  2.5e6g  14,738   55.46%  78.04%  88.01%  6.38%   64.8  +1.10
  7e6g    15,961   55.13%  79.86%  88.89%  6.27%   65.4  +1.13

The exact-match SPIKE grows with maturity (38% -> 55%, saturating ~2.5M) but the
error SCALE does not (sd flat at 63-72, <=10cp band flat at 78-82%).  So a fixed
cp threshold is defensible -- the gate does not need a maturity schedule.

- [ ] **Rules for using / replacing / skipping a leaf** (D. Homan's framing).
      Well-approximated leaves carry gradients related to the root score and are
      fine; poorly-approximated ones cannot.  Candidate gate:
      `|leaf_static - propagated_root| <= T`, one constant.  From the bands,
      T=25 keeps ~89%, T=10 keeps ~80%, and the >50 cp tail is 3-6%.
      Judge by foreign-anchor gauntlet, not by label statistics -- four prior
      interventions were correct at the label level and bought no Elo.

- [ ] **⚠️ OFFLINE RETARGETING IS AFFECTED — do not lose this.**
      `--bt-rescore` (SCRIPT_USE.md; Online 7.7) retargets a stored ROOT label to
      its paired PV LEAF re-evaluated on current weights.  That mode assumes the
      stored leaf is the position the root score came from.  It is not, in
      ~45-55% of records, and the error is +/- 65 cp of symmetric noise.
      7.7 already measured retargeting as null (-2.5 +- 21.0) WITHOUT knowing
      this; the null now has a candidate explanation rather than being a
      mystery.  Decide whether the mode is dropped or re-envisioned -- e.g.
      gated on the same |leaf - root| tolerance, so it only retargets records
      where the pairing is actually tight.

      **REVISIT REQUIRED after the tdleaf-pv-telemetry changes (2026-09-16).**
      Two things moved underneath this mode and neither is reflected in it yet:

      1. Leaf rows are now dumped only when |leaf_static - propagated root
         search| <= TDLEAF_LEAF_MATCH_CP (10 cp), the same test the online trace
         uses.  So on corpora generated from that branch onward, the stored
         (root, leaf) pairs are ALREADY tight by construction -- which is the
         gating this item proposed, applied at generation time instead.
         `--bt-rescore` on a NEW corpus is therefore a different experiment from
         `--bt-rescore` on the archives, where pairs are 60 cp-gated and ~45-55%
         mismatched.  Do not pool results across the two.
      2. PV_LAST_RESOLVED changes WHICH leaf is stored: an unresolved iteration
         now yields the last resolved iteration's PV rather than a 2-ply
         fail-high stub.  Archived corpora contain the stubs; new ones do not.

      So the honest status is that 7.7's null was measured on a corpus whose
      pairings were far worse than what the loop now produces, and the mode has
      never been tested on tight pairs.  Either re-run it on a post-change corpus
      or retire it explicitly -- but it should not simply be left in the tree
      carrying a null that no longer describes the data it would run on.

### Post-restart learning queue (2026-09-15)

Ranked in `docs/Learning_Investigation.md` §6, which carries the rationale, the
arm design and the pre-committed reading for each.  Checklist form only here.
⚠️ = previously ruled out under a regime or criterion that has since changed.

**Everything the investigations established was measured on a net at 2.2–7M
cumulative games.**  R7 is the first young chain since those records begin, so
cheap questions are worth re-asking on it rather than assuming the mature-net
answer carries.

- [ ] **1. Batch 50 on a full leg — damage reduction vs leg *yield*.**  7.15 rates
      handoff damage only, and 6.16's batch-16 arm cut damage 4× while making the
      loop worse.  Pass `--gauntlet-tdleaf` on **every** leg; dropping it after
      `3e6` is how the online drift went unseen for five legs.
- [ ] **2. ⚠️ Does the handoff happen on a YOUNG net — absent, or masked?**  The
      damage is only *apparent* after ~1M games of learning; the chain read
      +18/+7/−2/−1 at 100k–1M before turning negative at 2.2M.  Either it does not
      occur on an unsaturated net, or it occurs every time and is netted out by
      concurrent learning gains.  Decides whether "minimise handoffs" is a
      whole-chain rule or a late-life one.  **Arm:** 5k-game damage protocol plus a
      within-leg checkpoint ladder at 100k / 500k / 1M / 2M of the R7 chain, each
      rated against its own starting net.  ~50 min per state, on states the chain
      produces anyway.  The R7 chain is the first opportunity since the records
      begin.
- [ ] **3. Attack Σ directly — shuffle records across a pool of games in the
      learner**, rather than buying decorrelation only through larger B.  The only
      arm that would confirm Σ positively rather than by elimination.
- [ ] **4. ⚠️ Re-read alpha and rbar against the 5k-game damage protocol.**  Both
      were rejected on leg total, judged partly through "online Δ" — now known to
      be a handoff cost — and both predate the eps fix.  ~50 min each.
- [ ] **5. ⚠️ Depth, on the new chain.**  d10 was rejected on one leg confounded
      by a 52% draw rate, predating the corpus-window/root-row wins and the eps
      fix.  Gate on draw rate first (see A6); consider fixed-nodes generation.
- [ ] **6. ⚠️ A quiet gate *tighter* than 60 cp** — never tested, and `--bt-diag`
      reads negative ΔMSE_out below 40.  One `--bt-quiet-cp` sweep on disk data.
- [ ] **7. Restore the deferred offline wins** — `--corpus-window` (+36 anchor /
      +45 paired) and `--bt-rows root` (+35.6 paired) are deferred for a clean
      restart, not retired.  A1b below bounds what label staleness costs.
- [ ] **8. Actor refresh cadence** — blocked on a corpus-diversity observable.
      `cos(online, offline)` is the candidate; **measure its repeat-run noise
      floor first**, which was never done.
- [ ] **9. Leaf rows: settle the blend confound** (`--bt-leaf-lambda`) before
      deciding whether to keep generating them.  See A4 and Offline 3.5.
- [ ] **10. Log per-section bias RMS per leg.**  Nearly free, and unlike the draw
      rate it is not blind to uniform decay.  Monotone growth was still rising at
      7M games on the old chain and is unexplained.
- [ ] **11. The root-vs-mix confound** — root-only at 86M rows, matching the root
      count inside `mix`, discriminates "leaf rows are harmful" from "mix was
      root-starved".  The density question from the same arms is unresolved.

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

### ~~Online learning-rate scale — the last untested magnitude knob~~ — CLOSED

Run and closed.  `--lr-scale` (7.9.5) removed the need for the parked 6.17
constants-edit recipe, and the arms settled it: damage scales as η exactly
(displacement ∝ √η, Elo ∝ displacement², 7.13.1), but `lr25` and `lr100` produced
final nets that were **identical head-to-head at −0.7 ± 8.3** — η cuts signal and
noise together.  Displacement magnitude is not the axis (four independent
demonstrations, `Learning_Investigation.md` §1 D).  The online LR dropped 4× under
the 2026-09-15 restart as a by-product of unifying the two phases' LR sets, not as
an experiment.

**What replaced it as the live magnitude-adjacent line is Σ** — see
`Learning_Investigation.md` §6 items 1–3.

### Search parameter tuning
The search's pruning parameters (null-move margins, futility thresholds, aspiration
windows, LMR reduction tables) were tuned for the classical eval.  The NNUE eval has a
different score distribution and may benefit from re-tuning these constants.  CLOP or
a self-play tournament with systematic variation would be the appropriate approach.

### Under-promotion move ordering (gauntlet needed)

`moves.cpp` `add_move` scores a queen promotion at 20,000,000 and derives the
under-promotions from it by subtraction:

```
    queen   20,000,000
    rook    20,000,000 - 9,000,050 = 10,999,950
    bishop  20,000,000 - 9,000,060 = 10,999,940
    knight  20,000,000 - 9,000,070 = 10,999,930
```

Captures top out around `10,000,000 + 1000*PTYPE + pawn_bonus` ~= 10,006,000, the
counter-move is 8,000,000 and the killers are 6,000,000 / 4,000,000 — so **all
three under-promotions currently sort ahead of every capture, killer and
counter-move.**  The 50/60/70 offsets deliberately encode the R > B > N
preference, and the placement is deliberate too (confirmed 2026-08-29), so this
is NOT a bug — but it has never been measured.

Open question: is ordering R/B/N promotions above all captures actually worth it?
Knight under-promotions are occasionally decisive (fork/check), but rook and
bishop under-promotions essentially never are outside of stalemate avoidance.
An arm that moves the under-promotion band below the capture band (subtract
~10,000,000 instead of ~9,000,050) is a one-line change and a clean gauntlet
target.  Judge with a foreign anchor, not a family match.

---

## Release (1.0)

- [ ] **Refresh the training numbers in `README.md` before tagging 1.0.** The
      "Offline Training & the Hybrid Loop" section currently cites the `m260720`
      chain at 2.5M games (+17 ± 11 vs `Leaf_vclassic_eval`), which is the last
      iteration archived in this repo. The chain is still running and is well past
      that; take the final figure from the completed chain's sidecar rather than
      from this snapshot.

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
Gradients still applied to in-memory weights; only the disk write is gated on
`td_result >= 1.0`.

**Note (2026-08):** this item was written for the multi-writer era, where the save
path re-read the file and merged other processes' deltas, so skipping a write also
skipped an *import*.  That machinery is gone — `nnue_save_fc_weights` is now a plain
atomic write of in-memory state, so the proposed refactor is moot and the flag would
be a one-line gate.  Whether selective checkpointing is desirable at all under the
single-writer learner is unexamined; the learner's saves are already infrequent.

---

# Internal Self-Play

> **DELIVERED (2026-07).**  Phases D and E below shipped: Phase D as the engine's
> `--selfplay` driver (`src/selfplay.cpp`) and Phase E as the actor/learner split
> (`--traj-out`/`--learn-stream` via `scripts/selfplay_run.py`), now the default
> `train.py` generation mode.  The roadmap below is retained as the historical
> implementation spec.  Remaining follow-on (retire the in-engine multi-writer
> merge) is Phase 3 in `docs/history/SIMPLIFICATION_PLAN.md`.

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
D  internal self-play, single board      DONE — shipped as --selfplay (src/selfplay.cpp)
E  multi-board                           DONE (as actor/learner split, not multi-board)
```

Line numbers cited are as of `da9e57a` — re-verify before editing; the unity build means LSP
per-file diagnostics are unreliable (expected false positives).

------

## Known Issues

(none currently open)
