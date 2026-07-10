# Leaf TODO

Planned investigations, improvements, and open questions.

---

## TDLeaf(λ) Training

### Internal self-play — Phases D & E (deferred; tackle together)

**Full spec:** `docs/MAINSTREAM_PLAN.md` (Phases D and E).  Original design:
`~/.claude/projects/-Users-homand-Leaf/memory/single-process-selfplay-tdleaf-plan.md`.

Phases A–C of that roadmap are **done** (pure-PSQT mainstreamed + format v12; the
gauge machinery deleted; per-record STM + game-ply λ^Δ landed and gated in the
two-process harness).  The remaining phases are the payoff — the first code that
actually exercises the alternating-STM / `dply=1` paths C1/C2 built:

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

### UPDATE 2026-07-02 — hybrid loop supersedes pure-online strategies

The offline-consolidation work (see `OFFLINE_TRAINING.md`) changed the training
picture substantially and supersedes parts of the depth-curriculum plan below:

- **Confirmed:** the depth-equilibrium finding below still holds for *online*
  learning — the d6 self-play Elo curve on run 260628 extrapolates to an
  asymptote ~150–200 Elo below classic_eval; data quality is the binding
  constraint.
- **New result:** offline multi-epoch consolidation of the SAME games recovers
  +125–140 Elo past the online endpoint (~60% of the then-remaining gap to
  classic_eval) in ~2 h.  The hybrid loop (generate → consolidate → regenerate
  with the stronger net) is now the primary strategy; depth increases remain the
  quality lever *within* the generation phase (d8 recommended).

**UPDATE 2026-07-04 — iteration 2 complete; gen-2+ recipe settled.**  The
generation-2 consolidation experiment series (see "Generation 2" in
`OFFLINE_TRAINING.md`) resolved the initial iteration-2 regression and settled
the recipe: `--shards 1 --bt-K 220 --bt-lambda 0.3 --bt-leaf-blend`.  Best net
is now `iter2s2` (+55 vs the gen-1 consolidation, −64 vs classic_eval; gap to
classic down to ~64 Elo).  Findings folded into the open items below: sharded
sync staleness destroys the subtle gen-2 signal (single-process required for
now); λ — not K — is the knob for outcome-driven piece-value inflation; leaf
rows need the blend anchor (`--bt-leaf-blend`, committed 2026-07-04).

**UPDATE 2026-07-05 — pure λ-return settled as the gen-3+ recipe.**  The
distance-decayed result weight (`w = λ_eff·td_λ^(N−ply)`, committed 5ce7714)
sweep found the λ = leaf-λ = 1.0 end best: decay shape beats any flat mean,
piece-value drift self-limits (geometric convergence, no multi-epoch rollover),
and `tdL10F10x6-ep4` posted the best direct classic anchor yet (**−58.6 ± 20**;
gap ~59 Elo).  `--bt-lambda` now defaults to 1.0 (trainer + hybrid_loop);
`--bt-td-lambda` (= `TDLEAF_LAMBDA` 0.98) is the single knob of record.  The
λ-fine-tuning item below is superseded by td_λ calibration.

Open items:
- [ ] Iteration 3: long d8 online generation (~400k–1M games) from
      `tdL10F10x6_p0_ep4.tdleaf.bin` via `hybrid_loop.py` (needs `--recompile`
      so the dump binaries emit the exact `endply` column); gauntlet vs
      {tdL10F10x6-ep4, classic_eval}, plus a direct promoted-net-vs-classic
      anchor match (family-chained Elo reads ~20–30 optimistic).
- [x] ~~td_λ calibration on a larger corpus~~ — resolved 2026-07-10 on the
      137.7M material-line corpus: **td_λ 0.985** (game-ply) beats the 0.98995
      default by ~10–15 Elo (+95 still-rising vs +78 plateau vs the shared
      pretrain anchor); 0.975 ties it (noisier); 0.995 clearly worst.  The
      transferable knob is corpus-mean outcome mass ≈ 0.25–0.33 (matches the
      old flat-λ plateau) — pick td_λ per corpus from the trainer's printed
      mean decay.  See `OFFLINE_TRAINING.md` "td_λ calibration sweep".
- [x] ~~`--bt-sync` frontier fix~~ — resolved 2026-07-08 by *removing* sharding:
      replaced with within-batch thread parallelism (`--bt-threads`, single
      process, staleness-free — mathematically identical to 1 thread up to float
      summation order; measured ~2.85× on 8 cores).  See `OFFLINE_TRAINING.md`
      "Threaded training" and `docs/BT_PARALLEL_PLAN.md`.
- [ ] Online `TDLEAF_K` runtime override: compile-time 220 in `tdleaf.h`, but the
      consolidated nets' evals fit K≈165 — likely explains the mild online
      piece-val drift (+15–34 cp/400k games).  Add an env override and test
      online generation at the refit K.
- [x] ~~λ fine-tuning around 0.3; per-source λ (roots vs leaves)~~ — resolved
      2026-07-05: root/leaf split is inert (confirmed twice); flat λ superseded
      by the pure λ-return with td_λ decay (see UPDATE above).
- [ ] Outcome-baseline subtraction (`e' = e − EMA(engine-POV mean e)`) — designed,
      unimplemented; needed only if imbalanced-opponent training (score far from
      50%) becomes a first-class mode.  See "Outcome-Imbalance Drift" in TDLEAF.md.
- [ ] Bayeselo pool rating (not head-to-heads) once a consolidated net approaches
      classic_eval (+598) — iter2s2 at ~−64 is getting close.
- [x] ~~Dirty-row-only requantize~~ — resolved by the `--bt-threads` work
      (`nnue_requantize_fc_applied`, targeted rows only).  Further serial-tail
      trims landed 2026-07-09: zero-on-merge worker clearing + sampled clip
      scan (`--bt-clip-every`, default 64) — +16% at batch 512, bit-identical;
      see `OFFLINE_TRAINING.md` "Serial-tail trims".  Batch-size sweep
      resolved 2026-07-10: batch 2048 at *unchanged* lr 0.25 plateaus ~7 Elo
      below batch 512 at 1.8× speed (LR scaling rules hurt — Adam absorbs the
      batch change; scaled arms roll over).  Recipe: 512 for production
      consolidations, 2048 for sweep/probe arms.

### Training depth curriculum (2026-04-11; see UPDATE above)

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
corrects its weights for the deeper signal, then stalls at the new equilibrium.

**Key insight — Adam v is NOT the cause of plateaus:**
After ~1.4M games, FC max v = 4.5e-6 → effective FC LR = 0.01/sqrt(4.5e-6) ≈ 4.7.
Adam v is negligibly small; the FC plateau is caused by small TD errors
(gradients are tiny when the engine is near its depth-equilibrium), not by
accumulated v damping updates. Optimizer reset would give a brief burst but
not sustained improvement.

**Recommended strategy going forward:**
Depth curriculum: spend ~10-15k games per depth step, then advance.
e.g. d8 → d10 → d12 → d14. Each transition gives one calibration kick.
Staying at any depth beyond ~10-15k games past the initial burst yields
diminishing returns. The depth itself is the primary lever, not game volume.

**Future experiments to try:**
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

### Cross-entropy loss for offline batch-train (focal-γ variant)

**Idea:** Leaf currently minimizes squared error *in probability space* everywhere,
not cross-entropy.  Both learning paths map the white-POV eval through a sigmoid
`d = σ(score/K)` and descend `(target − d)²`:
- Online TDLeaf (`tdleaf.cpp:194,204`): `sig_grad = d*(1−d)/K`,
  `grad_scale = e[t] · sig_grad · …`, where `e[t]` is the backward, λ-traced sum of
  consecutive-leaf sigmoid deltas (`delta_d = d[t+1] − d[t]`, `tdleaf.cpp:182`).
- Offline `--batch-train` (`nnue_batch_train.cpp:457-464`): `e = target − d`;
  `se += e·e`; `sig_grad = d*(1−d)/K`; `grad_scale = e · sig_grad · …`.  Header
  comment (line 30) literally says the loss is `(p_target − d)²`.  `target` is the
  λ-blend soft label (`bt_target`).

For a **fixed (soft) target**, MSE and cross-entropy differ by *exactly one factor* —
the sigmoid Jacobian `d(1−d)` — because of the standard sigmoid+CE cancellation:

| loss          | ∂L/∂score                 |
|---------------|---------------------------|
| MSE (current) | −(target − d) · d(1−d)/K   |
| Cross-entropy | −(target − d) / K          |

So in the **offline** trainer, "switch to CE" ≈ drop `d(1−d)` from `sig_grad`, keeping
`e = target − d` (a soft label → soft-label cross-entropy).  ~One line.

**Why CE could help:** MSE's `d(1−d)` factor →0 at the confident tails (`d→0/1`), so a
position the net rates winning (`d=0.98`) that was actually lost (`target=0`) gets a
near-zero gradient — the blunder/horizon corrections you most want are throttled
hardest.  CE keeps full `(target − d)` strength there (this is why nnue-pytorch /
Stockfish train NNUE with a CE-style loss).  Better calibration, often faster.

**Do it OFFLINE ONLY.**  The offline path has genuine fixed targets, so CE is a clean,
correct drop-in and A/B-able via the normal gauntlet without touching the online gauge
machinery.  Leave online TDLeaf on MSE — there the `d(1−d)` is **not** a discretionary
MSE-damping term but the chain-rule Jacobian `∂d/∂score`, because `e[t]` is built in
*d-space* (a traced sum of `delta_d`).  Dropping it online yields a gradient
inconsistent with its own objective, not "CE"; a real logit-space CE would require
redefining the eligibility trace over logits — a different algorithm, out of scope.

**No numerical blowup to fear.**  With the sigmoid+CE cancellation the CE gradient is
`(d − target)/K`, bounded by `1/K` — at most ~4× MSE's peak of `0.25/K`.  Only the
*reported* NLL metric can overflow (`log(d)` as `d→0/1`); clamp `d` to `[0.05,0.95]`
(or `+ε` inside the log) for the metric only.  **Do not clamp to "protect" training**
— any clamp tight enough to tame magnitude just re-introduces MSE-style tail damping
and gives back the whole point of switching.

**Implementation — focal-γ knob (preferred over binary MSE-vs-CE):**
- `sig_grad = powf(d*(1−d), γ) / K`.  `γ=1` = current MSE, `γ=0` = CE, `γ=0.5` between.
- Behind a `--bt-loss-gamma <γ>` flag (default 1.0 = no behavior change).
- Gives the tail-emphasis/stability tradeoff as a *curve* over a small grid
  (γ ∈ {0, 0.5, 1}) for the same gauntlet cost, instead of a coin flip.

**Two things that must accompany the switch (else the A/B is confounded):**
1. **Global LR rescale, ~÷4 — not a seven-LR re-tune.**  `d(1−d)` is a *per-position
   scalar* that multiplies `grad_scale`, scaling **every section's gradient uniformly**
   (FC0/FC1/FC2/FT/PSQT).  It never changes the *relative* magnitude between sections,
   so the inter-section LR ratios are preserved; only a global `lr_scale ÷ ~4` is
   needed (most data sits near `d=0.5` where `1/(d(1−d)) ≈ 4`).
2. ~~**Bump `TDLEAF_GRAD_CLIP_NORM` ~×4.**~~  **Empirically moot (measured
   2026-07-09):** clip telemetry over the 137.7M-position material-line run
   showed batch grad norms of 0.053–0.082 against the 1.0 threshold — 12–19×
   headroom, zero fires in ~1M batches — so even CE's ~4× larger raw gradients
   (~0.2–0.33) never reach the clip.  No clip change needed; only the LR ÷4
   pairing above.  (Caveat kept for the record: on a corpus with much larger
   norms the original reasoning applies — the clip runs on the raw, pre-LR
   gradient, so lowering `lr_scale` would not relieve it.  Note the clip scan
   is now *sampled* by default (`--bt-clip-every 64`) with an automatic
   fall-back to per-batch scanning if a sampled norm exceeds half the
   threshold.)

**Expectation management:** the payoff is likely *modest*.  Offline corpora are quiet
positions from near-equal self-play, where most targets and `d` sit near 0.5 — exactly
where `d(1−d) ≈ 0.25` is maximal and MSE ≈ CE.  The confidently-wrong tail cases CE
helps with are rare in balanced data (the same distribution that makes
outcome-imbalance drift benign — [[tdleaf-fixed-opponent-bias-collapse]]).  A small or
null result is a legitimate outcome.  Watch the piece_val / PSQT scale spectrum when
testing, since CE's tail emphasis can interact with material scale
([[frozen-psqt-experiment]]).

**Plan:** add `--bt-loss-gamma` + paired `GRAD_CLIP_NORM` handling → sweep
γ ∈ {1, 0.5, 0} with `lr_scale ÷ 4` at γ<1 → gauntlet each net vs the current-MSE
consolidation → keep γ only if it clearly wins.  Report NLL (with metric-only clamp)
alongside MSE(blend)/MSE(outcome) in `val_loss()`.

### Adam hyperparameter tuning

The Adam optimizer uses five separate LRs tuned from 190k-game weight distribution
analysis.  Key monitoring points:

- **FC LR0 (0.1):** FC1 saturation at 0.5% after 190k games; stacks 5,6 at 1.2–1.7%.
  Reduced from 0.13 to extend runway before saturation becomes problematic.
- **FT LR0 (1.0):** FT weights barely changed (std 44.006 vs 44.0 init).  With only
  3–50 updates per weight, FT learning is very slow; high LR compensates.
- **FT bias LR0 (0.01):** Separate LR prevents dying-ReLU from update frequency asymmetry.
- **PSQT LR0 (1.6):** PSQT barely moves (std change -44 from 35642 over 190k games).
  Correct behavior — dense piece_val handles material corrections.

### Horizon noise mitigation — ablation testing plan

Two mechanisms reduce the influence of tactics-beyond-horizon on TD errors:
score-change clipping (`TDLEAF_SCORE_CLIP_CP`) and ID-stability weighting
(`TDLEAF_ID_VAR_SIGMA2`).  Their individual contributions should be isolated.

**Recommended ablation (500 games per arm, same starting network):**

| Arm | TDLEAF_SCORE_CLIP_CP | TDLEAF_ID_VAR_SIGMA2 | Description |
|-----|---------------------|---------------------|-------------|
| A (baseline) | 1e6 (disabled) | 1e6 (disabled) | Original algorithm |
| B (clip only) | 200 cp | 1e6 (disabled) | Approach 1 only |
| C (ID weight only) | 1e6 (disabled) | 10 000 cp² | Approach 2 only |
| D (combined) | 200 cp | 10 000 cp² | Both active (current default) |

**Metric:** Elo gain per game vs. the starting network (use `bayeselo_ratings.py` on
a 100-game test match against the starting network after each 500-game training run).

**Override at build time:**

```sh
# Arm A — no mitigation
perl comp.pl train_arm_a NNUE=1 NNUE_NET=nn-start.nnue TDLEAF=1 \
  -D TDLEAF_SCORE_CLIP_CP=1000000.0f -D TDLEAF_ID_VAR_SIGMA2=1000000.0f

# Arm B — clip only
perl comp.pl train_arm_b NNUE=1 NNUE_NET=nn-start.nnue TDLEAF=1 \
  -D TDLEAF_SCORE_CLIP_CP=200.0f -D TDLEAF_ID_VAR_SIGMA2=1000000.0f
```

(Similarly for arms C and D.)

**After the ablation:** if one approach dominates, drop the other to reduce complexity.

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
The SMP search allocates one `ts_thread_data` per thread, each with its own
`search_node n[MAXD+1]` stack including per-node accumulators.  Each thread's root
accumulator is independently initialised.  Thread interactions have not been tested
under NNUE; correctness is expected but unverified with `THREADS > 1`.

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

---

## Resolved / Implemented

### ~~Persistent Adam v~~ ✓ Implemented (2026-04-02)
Adam second-moment (v) arrays and t_adam persisted to .tdleaf.bin v6.  Multi-writer
merge uses max(v_file, v_local) per element.  FT weight v (~92 MB) excluded.  Momentum
(m) not persisted — recovers in ~10 steps.  ~564 KB file size increase.

### ~~Separate FT bias LR~~ ✓ Implemented (2026-04-01)
`TDLEAF_ADAM_FT_BIAS_LR0 = 0.01` (10× slower than FC) prevents dying-ReLU from update
frequency asymmetry.  FT biases update ~200×/game vs FT weights ~8/5000g.

### ~~--init-nnue-noprior~~ ✓ Implemented (2026-04-01)
All piece PSQT values initialised at 100cp (uniform) instead of classical material.
Forces material value learning from scratch.  training_run.py offers the choice.

### ~~FC-only replay~~ ✓ Implemented (2026-03-29)
Replay with FT/PSQT gradients causes eval divergence; `fc_only=true` suppresses
FT/PSQT/FT-bias/piece_val gradients during replay.  +20 Elo over 5000 games.
Subsequently disabled (TDLEAF_REPLAY_K=0) as benefit faded after first 5000 games.

### ~~Init-nnue redesign~~ ✓ Implemented (2026-03-23)
Weight initialization redesigned for TDLeaf training (decoupled from SF15.1 statistics).
FT weights N(0,5), FC weights N(0,{1,3,2}), all means zero, PSQT pure material (no
piece-square bonuses).  Separate `TDLEAF_ADAM_FT_LR0=1.0` for sparse FT weights.

### ~~Flavor A replay~~ ✓ Implemented (2026-03-21)
Replay now rebuilds accumulators from stored leaf positions using current FT weights,
ensuring FT gradients during replay are self-consistent with the current network.
`TDRecord` stores the leaf `position` (~300 bytes/ply, ~6% size increase).

### ~~Per-weight bias correction~~ ✓ Implemented (2026-03-21)
FC and PSQT Adam steps use per-weight bias correction (`eff_t = cnt + 1`) instead of
global `t_adam`.  bc1 skipped at cnt≥20 (negligible); bc2 always applied.  FT RMSProp
retains global bc2 (sparse features need growing global correction).

### ~~Per-weight LR decay removed~~ ✓ Removed (2026-03-22)
Per-weight LR decay (`TDLEAF_ADAM_C`, `TDLEAF_ADAM_LR_FLOOR`) removed.  AdamW weight
decay now handles regularization; LR0 tuned directly to the right value (0.01 FC, 1.6
PSQT) instead of starting high and decaying.  `--set-cnt` and `_prompt_init_cnt` also
removed as they existed only to prime the LR decay schedule.

### ~~AdamW decoupled weight decay~~ ✓ Implemented (2026-03-21)
`TDLEAF_WEIGHT_DECAY=1e-4` applied to FC weights and FT weights after each Adam step.
Skipped for biases (no benefit) and PSQT (would fight classical prior).

### ~~Gradient clipping by global norm~~ ✓ Implemented (2026-03-21)
`TDLEAF_GRAD_CLIP_NORM=1.0` clips the global L2 gradient norm before each Adam step.
Applied in `tdleaf_update_after_game`, `tdleaf_replay`, and `tdleaf_flush_batch`.
Set to 0 to disable.

### ~~Asymmetric lambda~~ ✓ Implemented (2026-03-21)
`TDLEAF_LAMBDA_DECISIVE=0.8` for wins/losses, `TDLEAF_LAMBDA_DRAW=0.5` for draws.
Decisive games get longer eligibility traces; draws use shorter traces to reduce
balanced-position noise.  Set both to the same value for symmetric behaviour.

### ~~Mini-batch gradient accumulation~~ ✓ Implemented (2026-03-19)
Gradients accumulated across `TDLEAF_BATCH_SIZE=16` games before each Adam step.
Reduces single-game gradient noise and file I/O.  `tdleaf_flush_batch()`
applies any pending partial batch at session end.  Set `TDLEAF_BATCH_SIZE=1` to restore
per-game updates.

### ~~Per-weight FT second moment~~ ✓ Implemented (2026-03-19)
FT weights upgraded from per-row RMSProp v (~88 KB) to per-weight v (~92 MB, OS lazy-paged).
Each of the 1024 dimensions within a feature row now has its own variance estimate,
allowing the optimizer to adapt step sizes per-dimension rather than using a coarse
per-row average.

### ~~LR warmup~~ ✓ Implemented (2026-03-19)
Linear warmup over first `TDLEAF_ADAM_WARMUP=50` Adam steps.  Prevents early-training
instability from cold-start v estimates.  Set `TDLEAF_ADAM_WARMUP=0` to disable.

### ~~Adam optimizer~~ ✓ Implemented (2026-03-15), LR decay removed (2026-03-22)
Adam optimizer with fixed LR (constant after warmup).  Per-weight LR decay was removed
in favour of direct LR tuning + AdamW weight decay.
FC: `TDLEAF_ADAM_LR0=0.01`; FT: `TDLEAF_ADAM_FT_LR0=1.0`; PSQT: `TDLEAF_ADAM_PSQT_LR0=1.6`.
FC0/FC1 float shadows clamped to ±127 to prevent zombie weights.  See `docs/TDLEAF.md`.

### ~~Epoch-based replay~~ ✓ Implemented (2026-03-11)
Flavor B is live with `TDLEAF_REPLAY_K=1` (default) and `TDLEAF_REPLAY_BUF_N=8`.
Ablation: K=1 is the current conservative default (K=2 marginal gain; K=6 large regression).

### ~~Bias initialisation~~ ✓ Implemented (2026-03-11)
FC biases (FC0/FC1/FC2) and FT biases are zero-initialised in `--init-nnue` mode.
Random N(μ,σ) from SF15.1 was removed — it added noise TDLeaf must first cancel.

### ~~Linux compilation~~ ✓ Resolved (2026-03-11)
AVX2 x86-64 SIMD paths added to `nnue.cpp` for SqrCReLU, FC0, and FC1.
Fallback chain: NEON (ARM) → AVX2 (x86-64 with `-mavx2`) → scalar.
Default build uses `-march=x86-64-v3`; use `NATIVE=1` for CPU-native tuning.
