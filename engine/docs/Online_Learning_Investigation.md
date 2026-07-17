# Online Learning Investigation — material_260708 hybrid-loop chain

Date: 2026-07-14 (updated same day with the learning-target redesign and the
first A/B results — see "Part 2" below)
Analysis: Claude Code session, investigating stalling/regressing online (TDLeaf)
phases in the `material_260708` hybrid-loop training chain, followed by the
design, implementation, and first validation of replacement learning targets
(branch `tdleaf-score-trace`, commit 898ff44).

---

## Initial question (Daniel Homan)

> I have been running a series of online (tdleaf) and offline training sessions
> for Leaf using the train.py script. The tags for this session are
> material_260708-1e5g etc... with the total of number of games indicated in
> the tag. While progress overall has been excellent, I am troubled by the
> trend in the online "tdleaf" versions which often seem to hold steady or
> regress with the offline learning making up the difference in the later
> rounds. In early rounds the tdleaf portion is very efficient at learning. I
> want to analyze what is going on. A couple of possibilities present
> themselves to me, but there may be others. First, it might simply be that
> there is less signal in the later games and so learning is unpredictable and
> noisy. Second, there may be some rare, but large, regression style 'event'
> that corrupts the .tdleaf.bin file and those are simply more likely in long
> (1 million game) sessions. Third, it might be that we have a non-optimal
> formula for back propagating game results through the tdleaf game tree, and
> using something closer to the offline training design (I have an idea in
> mind) might dramatically improve learning when the signal is not as strong.
> Here are the bayeselo learning results... "tdleaf" branches are before the
> offline learning phase and "final" are after offline learning on the corpus.

```
(base) homand@Omen-Laptop:~/Leaf/engine/learn$ python ../scripts/bayeselo_ratings.py *tdleaf*pgn */*final*pgn

Bayesian Elo ratings — 30 PGN files combined
12000 games loaded, 16 players rated

Rank  Name                                 Elo     ±  Games   Score   Oppo  Draws
---------------------------------------------------------------------------------
   1  Leaf_vmaterial_260708-5e6g-final    +290    15   1600     63%   +198    29%
   2  Leaf_vmaterial_260708-4e6g-final    +242    13   2000     55%   +206    30%
   3  Leaf_vclassic_eval                  +199    13   2400     59%   +128    17%
   4  Leaf_vmaterial_260708-4e6g-tdleaf   +187    20    800     46%   +214    39%
   5  Leaf_vmaterial_260708-3e6g-final    +186    12   2400     50%   +186    32%
   6  Leaf_vmaterial_260708-5e6g-tdleaf   +167    22    800     37%   +266    18%
   7  Leaf_vmaterial_260708-3e6g-tdleaf   +102    20    800     44%   +140    37%
   8  Leaf_vmaterial_260708-2e6g-final     +94    13   2000     49%   +104    30%
   9  Leaf_vmaterial_260708-1e6g-final     +22    14   2400     60%   -108    20%
  10  Leaf_vmaterial_260708-2e6g-tdleaf    +10    20    800     43%    +58    30%
  11  Leaf_vmaterial_260708-1e6g-tdleaf    -49    21    800     46%    -23    27%
  12  Leaf_vmaterial_260708-5e5g-final     -69    15   2000     59%   -180    20%
  13  Leaf_vmaterial_260708-5e5g-tdleaf   -151    21    800     49%   -141    31%
  14  Leaf_vmaterial_260708-1e5g-final    -213    15   2000     50%   -243    20%
  15  Leaf_vmaterial_260708-1e5g-tdleaf   -331    25    800     61%   -450    14%
  16  Leaf_vmaterial_eval                 -686    30   1600      5%   -148     4%
```

---

## Findings

### TL;DR

The data rules out corruption events (hypothesis 2), mostly rules out the
simple "fewer decisive games" version of hypothesis 1, and strongly supports a
refined combination of 1 and 3: **the online phase is still taking full-sized
Adam steps but the remaining learnable signal has shrunk below its noise
floor, so late online phases are a random walk** — expected gain ≈ 0, high
variance. Offline consolidation keeps working because its design (λ-return
targets, quiet-position filtering, shuffled mini-batches, epoch selection with
a validation ladder) extracts signal *from the exact same games* that online
wastes.

### The numbers, cleaned up

Direct 400-game head-to-heads (better than the pooled BayesElo, which mixes
opponents):

| Online phase | tdleaf_N vs final_{N−1} | Offline gain (final_N vs tdleaf_N) |
|---|---|---|
| →1e5 | +355 (vs material_eval) | +135 |
| →5e5 | +61 | +80 |
| →1e6 | +36 | +90 |
| →2e6 | −22 | +78 |
| →3e6 | 0 | +80 |
| →4e6 | +21 | +82 |
| →5e6 | **−71** | +116 |

Online: monotone decay to a zero-mean, ±~40–70 Elo random variable. Offline:
+78–116 every single iteration, no decay. And since the offline baseline is
the *post-online* checkpoint, the online regression directly drags down each
iteration's final — fixing it would compound.

### Evidence against hypothesis 2 (corruption events)

Every `<tag>_work/train/` kept its post-online live state
(`nn-material_260708.tdleaf.bin`), so each online phase could be diffed at the
weight level with `diff_tdleaf_checkpoints.py`:

- The regressing 5e6 phase and the healthy 4e6 phase have **nearly identical
  movement signatures** — fc0_bias med|dw| 104 vs 124, fc1_bias 52 vs 60,
  psqt 146 vs 152, similar max|dw|, no saturated or blown-out section
  anywhere.
- Game character within the 1M-game 5e6 self-play PGN is flat: per-50k-window
  draw rate steady at ~26–27%, average length steady at ~140 plies, no
  mid-run discontinuity that a corruption event would leave.
- The offline logs show zero gradient-clip fires across all 6 epochs (norms
  ~0.04 vs threshold 1.0).

The −71 at 5e6 is drift of the *usual* magnitude that happened to point in a
bad direction — a bad draw from the noise ball, not an event.

### Evidence on hypothesis 1 (signal)

The naive version fails: decisive-game rate is stable at ~73% across all 1M
games of every late run, and early runs actually drew *more* (33–35% at 1e5).
Outcome signal per game is undiminished.

The refined version holds, and here's the key comparison for the 5e6
iteration:

- **Online phase** (1M games, ~250k batch updates of 8 games): fc0_bias
  med|dw| = 104, PSQT med|dw| = 146 → **−71 Elo**.
- **Offline phase** (picked epoch 1, ~250k batches of 512 over the same
  games' corpus): fc0_bias med|dw| = 69, PSQT med|dw| = 31 → **+72 Elo**.

Online moves the weights *more* than the productive offline pass and gains
nothing. Back at 5e5, that same movement magnitude was buying +61 Elo on
weights half the size. Fixed Adam LRs (tuned when med|w| and residual error
were much larger) now produce a stationary noise distribution whose radius
exceeds the remaining signal. This is the classic constant-LR plateau; the
−22/0/+21/−71 sequence is exactly its sampling behavior.

Two secondary observations worth watching:

1. **FC bias magnitudes creep monotonically** across the chain (fc0_bias
   median 117 → 623, fc1_bias 170 → 421, fc2_bias 64 → 385), and online
   phases show a systematic *negative* mean bias drift that offline partially
   reverses. This is slow constant-component absorption — nowhere near the
   collapse signature, but it's a canary to keep on the dashboard.
2. **Offline is starting to overfit too**: `picked_epoch` went 4, 3, 2, 3, 4,
   2, **1**, and at 5e6 epochs 2–6 are all worse than epoch 1. The
   bt_lr=0.25 pass gets everything useful in one epoch now.

### On hypothesis 3

Directly supported by the strongest fact in the data: the corpus offline
trains on is dumped from the very games the online phase played, and offline
reliably finds +80 Elo there that online missed. The differences are the
candidate list for what the online formula lacks: λ-return targets with
per-ply distance decay rather than the online eligibility trace,
quiet-position filtering (online updates on every recorded ply, including
noisy tactical ones), decorrelated shuffled batches of 512 vs sequential
within-game batches of 8, and a validation-gated stopping rule. So moving the
online update toward the offline design is well-motivated by this data.

### Recommendations, in order of cost

1. **Cheapest, no code:** decay the online LRs across the chain via the
   existing `TDLEAF_LR_*` env vars — e.g. have train.py scale them
   ∝ 1/√(cumulative_games) or just 0.25× for iterations past 1e6. If the
   noise-ball theory is right, the online endpoint stops regressing almost
   immediately.
2. **Control experiment:** re-run one 1M-game iteration generate-only with a
   `TDLEAF_READONLY=1` pair (net frozen, corpus still dumped), then
   consolidate. If that final matches or beats the normal final, online
   learning late in the chain is contributing nothing but downside as
   currently configured.
3. **Hypothesis-3 redesign** (online update moved toward the offline design),
   validated against the ladder — with the post-online forensic states proven
   to survive in each `_work/train/`, any online-formula change can be A/B'd
   at the weight level, not just by Elo.

Also drop offline `--epochs` to ~2 for late iterations — the ladder says
everything past epoch 1–2 is wasted compute at this point.

---

## Methodology notes (for reproducing this analysis)

- Per-iteration configs and epoch ladders: `learn/material_260708-*_final.json`.
- Direct pairwise Elo: scored each
  `learn/match_Leaf_vmaterial_260708-*-tdleaf_vs_*.pgn` from the tdleaf
  side's perspective.
- Draw-rate / game-length windows: parsed `[Result]` / `[PlyCount]` headers
  from `material_260708-{1e5g,2e6g,5e6g}_work/match_*_d6.pgn.gz` in 50k-game
  windows.
- Weight-level phase diffs: `diff_tdleaf_checkpoints.py <seed state> <post
  state>`, where each online phase is
  `material_260708-<prev>_final.tdleaf.bin` →
  `material_260708-<tag>_work/train/nn-material_260708.tdleaf.bin`, and each
  offline phase is that post-online state →
  `material_260708-<tag>_final.tdleaf.bin`.
- `t_adam` in the live file merges by max across writer processes
  (`nnue_training.cpp` load path), so the small per-run increment (~7.4k) is
  per-process, not the total update count (~250k batches per 1M games).

---

# Part 2 — Learning-target redesign (2026-07-14)

Follow-up to the findings above: two replacement online learning targets were
designed, implemented behind `TDLEAF_TARGET` (env, default = legacy trace,
byte-for-byte unchanged), and the first — "blend" — was validated with a
1e5-game A/B chain (`material_260714`).  Code: branch `tdleaf-score-trace`,
commit 898ff44 (`src/tdleaf.h`, `src/tdleaf.cpp`).

## The blend target (`TDLEAF_TARGET=blend`)

Replaces the λ-decayed eligibility trace with a local per-record error
(sigmoid space, White POV, matching the offline trainer's target form):

    e_t = w·(result − d_t) + (1−w)·(d_{t+1} − d_t),   w = λ^(N − game_ply_t)

- `N` = last recorded root game-ply (same result-decay reference as the TSV
  dump), so the final record's `w = λ^0 = 1` reproduces the legacy
  outcome-only `e[T−1]` with no special case.
- Quiet gate: records where the white-POV score moved more than
  `TDLEAF_QUIET_CP` (default 60 cp, env-overridable) between consecutive
  searches contribute no gradient.  The gate is deliberately the DIRECT
  consecutive-score test, not the dump's static-vs-search position test: the
  opponent moves between records, so position-quietness at t cannot certify
  the transition (D. Homan's point, adopted).
- The legacy score-change clip is subsumed (dead in this mode); the
  ID-variance weight (`TDLEAF_ID_VAR_SIGMA2`) still applies in ALL modes —
  it scales `grad_scale` in step 3 of `tdleaf_accumulate_game`, outside the
  target computation.  Removing it in blend/hybrid is a separate A/B.
- Rationale: this is the offline `bt_target` run online (the next search's
  score standing in for the frozen cp label), attacking the exact variance
  mechanism identified in Part 1 — under the legacy trace, every distant
  one-step swing δ_j leaks into e_t with weight λ^(j−t).
- Telemetry: batch-apply stderr line reports cumulative `quiet-accept %`.
  Note fastchess swallows engine stderr; capture via a wrapper script
  (`exec engine "$@" 2>>log`) when telemetry is needed.

## Blend A/B at the 1e5 mark (`material_260714` vs `material_260708`)

Same recipe as the original chain's first iteration (100k games, depth 6,
same offline consolidation), blend target online.  Pool + direct results:

| | old (trace) | new (blend) |
|---|---|---|
| tdleaf (post-online) | +88 | −36  (direct match: −124 vs old-tdleaf) |
| final (post-offline) | +210 | +45 |
| offline gain (epoch ladder) | +161 | +80 |
| offline baseline val MSE(blend) | 0.0518 → 0.0169 (ep1) | **0.0101** → 0.0091 (ep1) |
| quiet-gate accept (from PGN evals) | (87% counterfactual) | 81–85% |

Interpretation (the val-MSE row is the decisive evidence):

1. **The gate is not starving the learner** — 81–85% of transitions pass.
   Sample count is not why blend learns slower.
2. **Blend online converges to the offline objective** — its post-online net
   arrives already at the offline trainer's fixed point (baseline MSE 0.0101
   vs the trace net's 0.0518), which is why offline gains only +80 after it.
   "Offline gains less because online already did that work" is confirmed in
   objective space.
3. **But at this stage that objective is not the binding constraint.**  The
   badly-calibrated trace net plays 124 Elo stronger.  In the old loop,
   online (trace) and offline (blend-form) were COMPLEMENTARY objectives;
   blend online makes them redundant, and the pipeline total collapses to
   what one blend pass extracts.
4. **The mechanism the trace has and blend lacks** is replicated backward
   credit for loud real events: a queen falling at ply k enters every earlier
   record's error with weight λ^(k−t) — dozens of coherent gradient
   contributions per decisive event.  Early in training those swings are
   overwhelmingly real (material actually fell) and this is the fast teacher;
   late in training they are mostly search blunders, and the same channel is
   what randomly walked the net at 5e6 (Part 1).  Blend forfeits the channel
   twice: the gate rejects the loud record, and the event survives only inside
   the ±1 outcome diluted by w ≈ 0.3.  (Per-record error MAGNITUDE is not the
   mechanism — Adam/RMSProp renormalize scale via v.)
5. "Slower, not worse" is plausible (right objective, less information per
   game) but unproven; the risk case is that outcome-only material learning is
   Texel-slow.  Distinguishing tests: continue the 260714 chain (does the gap
   narrow?), and the still-decisive late-regime A/B from
   `material_260708-5e6g_final` (does blend hold where trace lost −71?).

Curriculum implication: trace early (its loud-event channel is real signal
when the net is ignorant), blend-form late (when that channel is the noise
source).  The env-var switch makes per-run target selection free.

## The hybrid target (`TDLEAF_TARGET=hybrid`)

Designed to restore the early-regime channel inside the blend structure
(D. Homan's proposal):

    e_t     = w·(result − d_t) + (1−w)·trace_t,   w = λ^(N − game_ply_t)
    trace_t = (d_{t+1} − d_t) + λ_trace·trace_{t+1}      (trace_{T−1} = 0)

- `d_t + trace_t` telescopes to `(1−λ_trace)·Σ_k λ_trace^k·d_{t+1+k}` — a
  NORMALIZED geometric average of the next ~1/(1−λ_trace) records' evals, so
  targets stay calibrated automatically.  `λ_trace` (default 0.7, env
  `TDLEAF_TRACE_LAMBDA`) is fully decoupled from `TDLEAF_LAMBDA`, which
  shapes only the outcome weight w.  At 0.7 the trace horizon is ~3 records
  (~6 game-plies in the harness) — local event credit with strong damping,
  vs the legacy trace's ~65-record horizon.  `λ_trace = 0` reproduces blend
  exactly (with the widened gate below in place of blend's cp-only gate).
- **Trace gate — predicted OR quiet**: the trace flows through record t if
  the opponent played the engine's PREDICTED reply (search t's pv[1]) OR the
  transition was quiet (`|Δcp| ≤ TDLEAF_QUIET_CP`, same 60 cp default and env
  override as blend).  Only transitions that are both loud AND uncalculated
  break it — a swing the engine foresaw is calculated signal, a quiet
  transition is harmless regardless of prediction, and only genuine surprises
  (blunders, unforeseen tactics) sever the credit chain.  Prediction is
  verified by position hash, nearly free: the PV walk in `tdleaf_record_ply`
  snapshots `cur.hcode` after pv[0] (`key_own`) and after pv[0]+pv[1]
  (`key_reply`) into the TDRecord; at update time the transition is predicted
  iff the next record's `root_key` equals `key_reply` (dply 2, harness) or
  `key_own` (dply 1, internal self-play — trivially true, so the prediction
  half only bites in the harness).  PVs shorter than 2 plies count as
  unpredicted.  (First implementation gated on prediction alone — ~42% flow;
  the widened gate was adopted the same day.)
- **Break semantics: a gated-out transition breaks the trace** (`trace_t = 0`,
  propagating upstream through the recursion) **but the record still trains
  on its outcome term.**  The gate throttles only the eval-difference
  channel, and no sample is ever discarded — addressing both the sample-loss
  and lost-loud-event concerns in one stroke.  No score clip in this mode:
  predicted swings are calculated, not accidental.
- Telemetry: batch-apply line reports cumulative `trace-gate % pass` plus
  `predicted %` — the latter is effectively a free policy-stability meter.

Smoke (depth-5 self-play, stale default net): trace-gate ~75% pass, of which
predicted 40–43%; expect prediction higher at depth 6 with a mature net,
rising as the net stabilizes.

## Hybrid A/B at the 1e5 mark (`material_260714h2`, 2026-07-15)

Same 1e5 recipe, `TDLEAF_TARGET=hybrid` with the widened gate.  Combined pool
(different anchor set from the Part-1 pool, so compare within this table):

| | trace (260708) | blend (260714) | hybrid (260714h2) |
|---|---|---|---|
| tdleaf (post-online) | +61 | −58 | **+19** |
| final (post-offline) | +185 | +25 | **+91** |

The hybrid recovered roughly two-thirds of the online-phase gap to the legacy
trace (−119 → −42) and half the final gap, while post-online val MSE stayed at
the offline fixed point (the blend property that matters).  Reading: the
prediction-gated short trace restores most of the early-regime loud-event
channel without reopening the long-horizon noise path.

## Online root learning (`TDLEAF_ROOT=1`, 2026-07-15)

The remaining channel offline had and online lacked: the corpus ROOT rows'
search-amplified labels.  Now mirrored online — a second gradient per record
at the root position (blend/hybrid modes only; legacy warns and disables):

    e_root_t = w·(result − d_root_t) + (1−w)·(d_t − d_root_t)

- `d_root_t` = root's own static eval (sigmoid space, from `root_static`
  captured at record time); `d_t` = the record's search score (the existing
  leaf label); same `w = λ^(N − game_ply_t)` as the leaf error.  The
  `(1−w)(d_t − d_root_t)` term is search-amplified self-distillation — pull
  the static eval toward what depth-6 search concluded from this exact
  position.  No trace on the root error: the search itself is the lookahead.
- Gate: `|root_static − score_root_stm| ≤ TDLEAF_QUIET_CP` — a WITHIN-search
  test (no opponent move intervenes), matching the TSV dump's root gate; the
  transition-quietness argument does not apply here.
- Plumbing: `tdleaf_record_ply` snapshots the root accumulator/PSQT/features/
  stack into the TDRecord (`root_acc` was already a parameter); the update
  pass runs `nnue_forward_fp32` + `nnue_accumulate_gradients` on the root
  exactly as on the leaf, signed by `root_wtm`, scaled by
  `id_weight × TDLEAF_ROOT_WEIGHT` (env, default 1.0).  Root gradients apply
  even on records whose leaf error was gated out.  Nothing new persists —
  `.tdleaf.bin` v12 unchanged.
- Memory: TDRecord +4.4 KB (~+4.4 MB on the live game record).  Side fix
  found during sizing: `tdleaf_replay` copied every finished game into the
  8-slot ring buffer even with replay disabled (`TDLEAF_REPLAY_K=0`), paging
  in ~40 MB of dead BSS per process — now an early return when
  `tdleaf_replay_k <= 0`.
- Telemetry: `root-accept %` line at each batch apply.
- Smoke (depth-5, stale default net): hybrid+root and blend+root run with
  root-accept ~54–56%; legacy+root refuses cleanly; hybrid-without-root
  unchanged.  Expect higher accept on mature nets at depth 6.

Prediction: if online root+leaf learning fully replicates the offline
objective, the offline epoch ladder should collapse toward zero on late
iterations — at which point consolidation can be shortened or skipped and the
loop economics change.

## Status / next steps

- Committed on `tdleaf-score-trace`: targets (898ff44), doc (43f50d1),
  widened hybrid gate (2efc810), online root learning (this commit).
- Next: 1e5 sanity with `TDLEAF_TARGET=hybrid TDLEAF_ROOT=1` — watch the
  tdleaf-phase Elo close toward the trace's mark and the epoch ladder shrink.
- Then the design-target test: 1M games from `material_260708-5e6g_final`,
  best candidate vs legacy — the question is holding/gaining where the trace
  lost −71.
- Open A/Bs queued behind the target choice: drop `id_weight` in
  blend/hybrid; `TDLEAF_ROOT_WEIGHT` sweep; offline `--epochs 2` for late
  iterations; online LR decay across the chain (complementary to any target
  change).

*(The design-target test ran 2026-07-15/16 — results and a substantially
revised picture in Part 3 below.  In particular, Part 1's "offline reliably
extracts +80 that online missed" conclusion is corrected there.)*

---

# Part 3 — The design-target test, the seed-consolidation control, and what the offline gain actually is (2026-07-16)

Both new targets ran the design-target test from Part 2: 1M-game online
continuations from `material_260708-5e6g_final` (the chain's best net, "the
seed" below), followed by the standard offline consolidation.  Analysis
sequence (Claude Code session, 2026-07-16): the online losses replicated
across targets → per-bucket weight forensics → a direct label-quality test of
the endgame-staleness hypothesis (D. Homan) → a seed-consolidation control
that overturned Part 1's interpretation of the offline gain.

## 3.1 The design-target test failed: the online loss is target-independent

| Online run (1M games, d6, from the 5e6g seed) | tdleaf vs seed (500g) | offline ladder gain (picked ep) | final vs seed (400g) |
|---|---|---|---|
| legacy trace (Part 1, →5e6) | −71 | +116 | — |
| `TDLEAF_TARGET=hybrid` + `TDLEAF_ROOT=1` (`material_260708-6e6g`) | −50 | +88 (ep4 of 83/88/73/88/59/84) | +18 |
| `TDLEAF_TARGET=blend`, no root (`material_260708b-6e6g`) | −95 | +100 (ep4 of 89/80/79/100) | −8 |

Three independent 1M-game runs from the identical seed, three completely
different error formulas (65-record eligibility trace / local one-step blend /
prediction-gated short trace with root distillation), all losing 50–95 Elo
online.  Under Part 1's zero-mean-random-walk model this is a ~1% event.  The
online loss is **systematic** and lives in the shared update machinery, not
the target math.  Part 1's framing needed one correction to see why:

**A fixed-LR stochastic update process started from a validated optimum has
strictly negative expected Elo even with a perfectly unbiased gradient.**  The
seed is not a random point — it is the epoch-ladder-selected best of an
offline consolidation, i.e. a local Elo optimum (plus ~+10–15 of max-of-N
selection luck; the ladders now pick among epochs whose val MSE is *identical*
to 4 decimal places, so the pick is pure noise selection).  Displacing it by
any radius loses Elo roughly quadratically in the radius, regardless of
direction.  The online phase at the current fixed Adam LRs guarantees a
displacement of roughly constant radius (the stationary noise ball).  Early in
the chain the real signal dwarfed that cost (+355 at 1e5); by 5e6 the signal
is ~0 and the cost is unchanged, so *every* continuation loses — "systematic
in expectation, random in direction."  −50/−71/−95 are draws of the
displacement magnitude, not three discoveries of the same bad direction.

Shared-machinery amplifiers of the displacement radius, all absent offline
(candidate list for later mitigation, none yet individually confirmed):

1. **Correlated batch-8 same-game updates** — all of one game's records land
   in one batch; late-game records carry the same-sign outcome error onto the
   same rows dozens of times per batch (see 3.4).
2. **~13–16 concurrent writers with stale-baseline delta merging** (t_adam
   +9.4k per process ≈ 75k games/process over 1M games) — ~W× the effective
   single-writer LR near a fixed point; the same staleness physics that killed
   the sharded offline trainer.
3. **Phase-boundary Adam-v mismatch** — the online phase inherits `v` from
   the offline trainer's batch-512 gradients, ~8× smaller in noise scale than
   online's batch-8 gradients, so every process opens the run with
   step-clip-sized updates until `v` re-adapts.  (`t_ft_session` guards FT
   against exactly this; the FC/PSQT Adam state has no equivalent.)
   Checkable with a `TDLEAF_LOG_STEP_CLIPS=1` build.

Ruled out for these runs: label corruption from UCI self-adjudication
(generation runs `--no-adjudication`, so results come from natural
terminations where the terminal-position checks are reliable); gate
starvation (accept rates healthy, and the very differently-gated targets
landed the same); the Part-1 fc1-bias-drift canary (online shifted fc1_bias
coherently +61 mean, but offline pushed it *further the same direction* +38 —
whatever offline repairs, it is not the bias creep).

## 3.2 Bucket forensics: where the online damage lives

`scripts/bucket_phase_analysis.py` breaks each phase diff down by
HalfKAv2_hm material bucket (0 = 1–4 pieces = deep endgame … 7 = 29–32 =
opening), using the persisted per-weight update counts to separate exposure
from per-update violence, and computes per bucket how much of the online
displacement the offline phase *reverses* (projection; negative = repair,
positive = confirm-and-extend).  Replicated findings across both runs:

- **Online per-update PSQT movement is ~2× more violent in the deep-endgame
  bucket**: med |dw|/√updates 12.7 (bucket 0) falling to ~6.8 (buckets 5–7)
  for hybrid; 10.6 → 5.6 for blend.  Offline moves the same rows ~3× more
  gently (~2.0–2.5 everywhere).
- **Offline actively reverses online's low-bucket FC-stack movement**:
  fc0_bias projection in bucket 1 = **−0.80 (hybrid) and −0.83 (blend)**;
  the bucket-0 fc2 output bias moved −182/−112 online and was pushed back
  +365/+225 offline.  In buckets 4–7 the projections are *positive*
  (+0.35…+0.75) — offline confirms online's opening/middlegame direction.
- Online's large endgame PSQT displacement is nearly **orthogonal** to
  offline's movement (cos ≈ −0.1): neither confirmed nor repaired, it
  persists into the final net as unvetted noise (hybrid's online phase moved
  one PSQT row by 17,739 raw — 77% of the median |weight| — vs max ~3,900 in
  every other phase).

So by the offline objective's lights, online's low-material learning was
counterproductive while its opening/middlegame learning was directionally
right — consistent with an endgame-specific pathology.

## 3.3 Endgame-staleness hypothesis: tested at the label level and rejected

Hypothesis (D. Homan): depth-6 PV-leaf targets are "stale" — the leaf is 6+
plies off-game, horizon effects steer the game elsewhere, so game-derived
corrections are misdirected — worst in endgames where depth 6 is very short;
motivated by earlier pure-TDLeaf generations where d6 plateaued and switching
to d8 gave a clear bump.  Proposed fix: game-phase-dependent depth limits.

Direct test (`scripts/label_quality_by_bucket.py`, 1.2M root rows sampled
from the hybrid run's corpus): per bucket, how well do the depth-6 search
scores actually predict game outcomes?

| bucket | MSE(outcome), K=220 | \|cp\|≥150 converts to win | advantaged side loses | mean plies to game end |
|---|---|---|---|---|
| 0 (deep endgame) | **0.011** | **91.8%** | **0.0%** | 33 |
| 1 | 0.029 | 82.9% | 0.4% | 70 |
| 2 | 0.049 | 84.4% | 2.2% | 80 |
| 3 | 0.078 | 83.9% | 4.7% | 87 |
| 4 | 0.115 | 80.7% | 8.2% | 97 |
| 5 | 0.151 | 77.2% | 11.9% | 107 |
| 6 | 0.185 | 72.3% | 17.0% | 118 |
| 7 (opening) | **0.207** | **66.5%** | **22.8%** | 132 |

Depth-6 **endgame labels are the cleanest in the corpus by an order of
magnitude** — a ≥150 cp endgame advantage converts 92% of the time and
essentially never loses.  The stalest labels are in the *opening*, where the
outcome is 132 plies of play away.  Leaf-row MSE tracks root-row MSE within
~5% in every bucket — if stale leaf positions were absorbing off-trajectory
corrections, their static evals would be less outcome-consistent than the
root search scores, and they are not.  The endgame-staleness mechanism, as
stated, is contradicted.  (The d6→d8 historical bump has a different
explanation — see 3.6.)

## 3.4 Reinterpretation: endgame *correlation*, not endgame *staleness*

The mechanism that fits both 3.2 and 3.3: endgame records are the tail of
every game — dozens of near-identical positions hitting the *same* PSQT rows
and the *same* FC stack — and by then the outcome weight `w = λ^(N−ply)` has
gone to ~1, so every late record in a game carries the **same-sign** error on
the **same parameters**, all inside one batch (batch = 8 games).  Sharp labels
do not help when the update is a coherent 30-hit hammer on one bucket's
weights: Adam's `m` spikes, steps run near the clip, and the phase overshoots
along per-game directions.  Offline's global shuffle is precisely the
antidote — each endgame position's gradient is averaged against 511 unrelated
positions — which is why offline both moves those weights ~3× more gently
*and* reverses online's displacement there.  The violence and the reversal
are overshoot signatures, not wrong-label signatures.

Phase-dependent depth limits are therefore not the indicated fix (they buy
better labels where labels are already cleanest).  Mechanism-targeted A/Bs,
in order of cost: `TDLEAF_BATCH_SIZE` 8 → 64 (dilutes within-game coherence,
free); per-bucket down-weighting or record subsampling for buckets ≤ 2
online; online LR decay (blunt-instrument fix for the overall displacement).
Note: if depth reallocation is ever wanted, switching training games from
fixed-depth to fixed-nodes gives phase-adaptive depth for free (narrow
endgame trees search deeper at constant cost).

## 3.5 The seed-consolidation control: the offline "+80–116" was repair, not signal

Part 1 left two readings of the reliable offline gain open.  World A: it is
mostly *repair* of the online displacement.  World B: it is fresh-signal
extraction that would accrue from any starting point — in which case online
learning is pure downside and could simply be skipped.  The discriminating
experiment turned out to need no new games at all: batch-train the
**undisplaced seed** on the hybrid run's existing corpus — the *identical
data* whose consolidation "gained +88" from the displaced start — and ladder
each epoch against the seed itself:

```sh
gzip -kdc material_260708-6e6g_work/corpus.tsv.gz > corpus_6e6g.tsv
python3 train.py --tag seedctl-260716 --skip-online \
    --net nn-material_260708.nnue \
    --state material_260708-5e6g_final.tdleaf.bin \
    --corpus corpus_6e6g.tsv \
    --epochs 2 --bt-K 220 --bt-threads 8 \
    --gauntlet-epochs --no-final-gauntlet
```

Result (1000 games/epoch vs the seed):

| | W/L/D | Elo vs seed |
|---|---|---|
| epoch 1 | 298/434/268 | **−48 ± 11** |
| epoch 2 | 361/380/259 | −7 ± 11 |

Consolidating the seed on a fresh 1M-game corpus does not gain +88 — it
**loses 48 Elo at epoch 1** and claws back to −7 at epoch 2.  World A is
confirmed decisively, and then some: the offline gains throughout the late
chain were repair of online self-damage measured against a damaged baseline;
the fresh-signal content of 1M new depth-6 games for this net is **zero
within noise**.  The loop at this maturity is a treadmill — online damages,
offline repairs, the ladder picks a lucky epoch — netting +18/−8 per
iteration.

## 3.6 Why epoch 1 *loses*: the corpus distills its generator

The −48 is not mere diffusion.  Three measurements lock together:

1. **The corpus objective prefers the displaced generator over the stronger
   seed.**  Baseline val MSE(blend) on the identical corpus: displaced
   post-online net 0.00901, seed 0.00972.  The −50 Elo net fits the data
   better than the +0 net — because it *made* the data: the cp labels are its
   search scores.
2. **Consolidating the seed drags it toward the generator.**
   `scripts/distill_alignment.py` projects the seed's ep1 movement (B) onto
   the generator's online displacement (A = post-online − seed):
   **cos(A,B) = +0.67…+0.80 in every major section** (fc0_bias +0.67,
   fc1_bias +0.72, ft_w +0.69, PSQT +0.75 overall and +0.62…+0.80 per
   bucket), with ep1 replicating ~40–50% of the generator's displacement
   vector.  Epoch 2's partial Elo recovery came mostly from the fc2 output
   biases snapping back (cos(A, ep2−ep1) = −0.72 there) while the bulk kept
   drifting generator-ward.
3. The Elo landed accordingly: seed dragged ~halfway toward a −50 net → −48.

Caveat: part of the +0.7 alignment could be "any trainer on this data
distribution moves in correlated directions" rather than pure label
distillation; the readonly-generation control below de-confounds it (labels
from the seed itself → consolidation movement should align with nothing and
Elo should not drop).

**The unified picture.**  The engine of the whole hybrid loop has always been
the bootstrap **E ← search_d6(E)**: a depth-6 search of the current eval is a
better evaluator than the eval itself, so distilling search scores (plus
outcome anchoring) improves the net — while search_d6(E) is meaningfully
better than E.  The flat epoch ladders, epoch-1-does-everything, and now
seedctl ≈ 0 all say the static eval has converged to depth-6 search on quiet
positions: **the d6 bootstrap is saturated.**  Past that point the only
content left in a corpus's score labels is the generator's own noise and
displacement, so the online phase displaces the generator, the corpus
faithfully records the displaced net's evaluations, and offline consolidation
propagates that displacement to whatever net is trained on it.  The loop
cannot climb above its generator any more.

This also puts the historical d6-plateau → d8-bump observation in its correct
frame: not phase-dependent leaf staleness (3.3 — endgame labels are the
cleanest), but **global bootstrap saturation** — deeper search makes
search(E) > E again, restoring headroom by construction.

## 3.7 Recommendations

1. **Stop online weight updates at this maturity** — generate with a frozen
   pair: `TDLEAF_FREEZE=1` (runtime env var, added 2026-07-16 for exactly
   this experiment; records + dumps the corpus but skips all gradient
   updates and `.tdleaf.bin` writes).  NOT the compile-time
   `TDLEAF_READONLY=1` flag, which compiles out the record/update hooks and
   therefore dumps no corpus — and which silently does nothing when exported
   as an env var (discovered the hard way: the first attempt at this run
   exported it and the pair kept learning).  This matters more than Part 1's
   framing suggested: displacement doesn't just cost the online phase its
   Elo, it *poisons the labels* for the offline phase and any future
   consolidation.  A frozen-generated corpus is labeled by the seed itself;
   consolidating on it is the clean, unconfounded test of whether *any* d6
   signal remains (predicted: ~0).
2. **Next iteration at depth 8.**  ~250–400k games at d8 costs about the same
   as 1M at d6 and each label carries genuinely new information.  Both the
   saturation theory and the historical d6→d8 bump predict this is where the
   next real gain lives.
3. If online learning is ever re-enabled in the late regime, attack the
   displacement machinery (3.1/3.4), verified by mid-phase Elo checkpoints of
   the live state (the damage-timing curve discriminates the boundary
   transient from noise-ball diffusion) and a `TDLEAF_LOG_STEP_CLIPS=1`
   build; targets themselves are exonerated.
4. Optional replicate: seedctl on the blend run's corpus
   (`material_260708b-6e6g_work/corpus.tsv.gz`, generator displaced −95
   rather than −50) — distillation predicts a *worse* ep1 than −48.
5. Measurement hygiene: with val MSE identical across epochs, the
   ladder-max pick carries ~+10–15 Elo of pure selection inflation, baked
   into every seed and erased by every continuation — drop to `--epochs 2`
   (done here) and treat small final-vs-seed deltas accordingly.

*(Recommendation 1 ran the same day and closed the question — with two
detours that were themselves informative (a frozen-pair duplication landmine
and a book-diversity hypothesis, both resolved).  Recommendation 4's
blend-corpus replicate was mooted by the direct control.  See Part 4.)*

## Methodology notes (Part 3)

- Head-to-heads: `learn/match_Leaf_vmaterial_260708{,b}-6e6g-tdleaf_vs_*.pgn`
  (500 games each), scored with `bayeselo_ratings.py`; sidecars
  `learn/material_260708{,b}-6e6g_final.json` for ladders and final
  gauntlets.
- Per-bucket phase forensics: `scripts/bucket_phase_analysis.py <seed>
  <post_online> <final>`, where post-online is
  `<tag>_work/train/nn-material_260708.tdleaf.bin`.
- Label quality: `gzip -cd <work>/corpus.tsv.gz | awk 'NR % 40 == 0' |
  python3 scripts/label_quality_by_bucket.py` (root rows = depth > 0;
  result column is White-POV {0, 0.5, 1}, cp is White-POV).
- Distillation alignment: `scripts/distill_alignment.py` (paths hardcoded to
  this experiment's four states; seedctl epoch states survive in
  `learn/seedctl-260716_work/train/`).
- Baseline/epoch val MSEs: `<tag>_work/train/train.log` of each run.

---

# Part 4 — The frozen-generation control, the duplication landmine, and the book-diversity test (2026-07-16)

Part 3's recommendation 1 (generate with frozen weights, consolidate, measure
the true fresh-signal content of d6 data) ran the same day.  It took three
attempts to get a clean number — each failure was itself informative — and the
day ended with the d6 loop formally closed, a second hypothesis (opening-book
diversity, D. Homan) tested and retired, and the loop tooling hardened for
the d8 iteration.

## 4.1 TDLEAF_FREEZE — and the env-var trap that motivated it

The first frozen run was launched with `export TDLEAF_READONLY=1`, which does
nothing: `TDLEAF_READONLY` is a compile-time flag, and the env vars that do
work at runtime (`TDLEAF_TARGET`, `TDLEAF_ROOT`) made the pair a normal
learning run.  Discovered ~25 minutes in via the live `.tdleaf.bin`'s
advancing Adam counters; run killed, work dir deleted (its dumps would have
been globbed into any `--force` rerun's corpus).

A compiled READONLY binary would not have worked either: the
`#if !TDLEAF_READONLY` guards compile out the record/update hooks entirely,
so a READONLY pair plays frozen but **dumps no corpus**.  The fix is
`TDLEAF_FREEZE=1` (runtime env var, commit cc74ebd): records and dumps
exactly as a learning binary, but skips gradient accumulation, weight
application, and every `.tdleaf.bin` write path (gate after the dump call in
`tdleaf_update_after_game`; with nothing accumulated, batch apply / save /
exit flush are all naturally no-ops).  Smoke-verified: startup notice, zero
batch applies, byte-identical `.tdleaf.bin` md5, leaf+root TSVs dumped.

## 4.2 The frozen run crashed — and the crash was a duplication artifact

`material_260708r-6e6g`: 1M frozen games at d6 from the seed, standard
consolidation, ladder vs the seed (= pretrain, since the state never moves):

| epoch | W/L/D | Elo vs seed | val MSE(blend) |
|---|---|---|---|
| (baseline) | | | 0.007830 |
| 1 | 327/394/279 | −23 ± 11 | 0.006273 |
| 2 | 282/452/266 | −60 ± 11 | 0.005642 |
| 3 | 256/466/278 | −74 ± 11 | 0.005239 |
| 4 | 253/505/242 | −89 ± 11 | 0.004938 |

Monotone Elo collapse while val MSE *fell* 37% — the only run in the chain
where val ever moved after epoch 1.  Diagnosis (game-signature analysis,
FEN+PlyCount+Result per game):

- The 1M-game PGN contains **exactly 188,571 distinct games — precisely the
  line count of `training_openings.epd`** (mean 5.3 plays per opening,
  max 6).  Two identical deterministic engines at fixed depth replay the
  same game from an opening every time it comes up, **including the
  color-swapped `-repeat` game** (same net on both sides ⇒ the swap changes
  nothing).  The learning-pair 6e6g run, same book and game count: 990,870
  distinct — the online weight drift Part 3 indicted was also the only
  source of game diversity.
- The duplicates carry different gids, so they land on **both sides of the
  trainer's by-game train/val split**.  The falling "val" MSE was the
  trainer memorizing 188k unique games at ~5.3 effective epochs per nominal
  epoch (~21 by ep4), graded by a leaked validation set.
- Baseline val MSE 0.00783 was the lowest ever seen in the chain — the seed
  nearly predicts its own labels; the corpus was clean, just 5.3× smaller
  than nominal and ground in 4 epochs deep.

## 4.3 The dedup control: the d6 loop is closed

Dropping duplicate rows (identical in every field except gid) cut the corpus
134,048,352 → 25,144,224 rows (5.33×, matching the game-level count) and the
seed was re-consolidated on it (`seedctl-dedup`):

| epoch | Elo vs seed | val MSE(blend) |
|---|---|---|
| (baseline) | | 0.007845 |
| 1 | **−3 ± 11** | 0.007638 |
| 2 | −10 ± 11 | 0.007571 |

Flat.  A fresh 1M-game d6 corpus, labeled by the seed itself, deduplicated,
adds **nothing** — the unconfounded closure of the d6 loop (and retroactive
confirmation that 4.2's crash was pure duplication overfitting; same data
minus duplicates is simply zero).  The frozen run's final gauntlet agreed:
−21 ± 27 vs the seed (it promoted ep1 ≈ −23).  `material_260708-5e6g_final`
remains the chain's best state; the r-run and both 6e6g finals should never
seed a `--continue` chain.

Taken with Part 3: fresh d6 data contains zero extractable signal for this
net whether labels are clean (this control) or generator-drifted (seedctl,
−48), and the only thing that ever made late-chain consolidation look
productive (+80–116) was repairing online damage.

## 4.4 The book-diversity hypothesis: tested and retired

Hypothesis (D. Homan): `training_openings.epd` is only 188,571 lines — by
6.5M chain games each stem has been played ~34 times, and even with games
diverging, the repeated opening themes might bound the learnable manifold;
perhaps *this*, not depth, is the 5e6 plateau.  Genuinely open at that
point: every plateau measurement was equally consistent with "d6 exhausted"
and "book exhausted at d6" (both predict fresh same-book games add 0), and —
a confound worth recording — **every Elo number in the chain is measured on
the training book** (`train.py` ladders and gauntlets default to
`--openings training_openings.epd`), so book overfit would also inflate the
measurements themselves.

Two measurements answered it:

1. **Literal position repetition in the learning corpus is low.**
   Hash-partition sampling (keep all copies of a 2% FEN subset, so duplicate
   counts are unbiased) over the 6e6g corpus, copies per distinct position
   by ply band: 1.28 (plies 0–12), 1.10 (13–24), 1.11 (25–40), 1.12
   (41–80), 1.31 (81+).  The only fat tail is deep endgames (one position
   2,585×) — inherent to few-piece chess, not the book.  Any book effect
   would have to act through theme-level generalization, not memorized
   positions.

2. **Out-of-book strength is identical to in-book strength.**  A disjoint
   holdout book was generated with the same recipe and a different RNG seed
   (`make_training_epd.py --total 200000 --frc-fraction 0.2
   --random-suffix 2 --quiet-only --seed 2607`), and the seed rated vs
   `Leaf_vclassic_eval` at 1+0.01, 1000 games per condition:

   | condition | Elo |
   |---|---|
   | training book (run 1) | −22 ± 20 |
   | training book (run 2, independent replicate) | −29 ± 20 |
   | holdout book (never seen) | −29 ± 18 |

   In-book pooled ≈ −26 ± 14 vs out-of-book −29 ± 18: **no book overfit at
   all** — the net plays openings it has never seen at exactly its
   trained-book strength.  (Classical eval knows neither book, so the
   *difference* isolates the NNUE's book dependence.)  Two side lessons:
   the scary-looking −29 was the **time control**, not the book — at 1+0.01
   classic_eval's nps advantage is worth ~25–30 Elo relative to the 3+0.05
   gauntlets where these nets measure +0/+6 vs classic; keep
   cross-eval-type comparisons at 3+0.05 (NNUE-vs-NNUE ladders at 1+0.01
   are fine, speed is symmetric).  And the diversity hypothesis is retired
   in its strong form: a wider book from the same generator recipe samples
   the theme space the net has already mastered, so a fresh-book d6 corpus
   (Part 4's "test A") would almost certainly read ~0 and is not worth
   generation budget.

## 4.5 Where this leaves the loop: depth is the last lever standing

By elimination — targets exonerated (Part 3), online learning retired
(Part 3/4.3), book diversity retired (4.4) — the binding constraint is the
**depth-6 label ceiling**, exactly matching the historical pure-TDLeaf
observation that d6 plateaus broke at d8.  The loop redesign:

- **New loop shape: freeze-generate → consolidate → new seed.**  No online
  learning phase at all; the generator can never displace, so the corpus
  labels are always the seed's own.
- **Frozen economics:** anything beyond one game per opening is duplicated
  compute, so the natural d8 iteration is ~188k games ≈ 750k d6-equivalents
  at the ×4 depth cost — *cheaper* than the old 1M-game d6 iteration.
- **Tooling (commit 3140cb2):** `train.py --no-repeat` (generation-phase
  passthrough to `match.py`; ladders/gauntlets keep paired openings) and
  `--dedup-corpus` (phase-4 row dedup on every field except gid,
  **auto-enabled whenever `TDLEAF_FREEZE` is set** so the guard cannot be
  forgotten; ~no-op on learning corpora at 1.01×).
- The d8 iteration in flight as of this writing:

```sh
export TDLEAF_FREEZE=1
python3 train.py --tag material_260708-d8-1 --net nn-material_260708.nnue \
    --state material_260708-5e6g_final.tdleaf.bin \
    --games 188000 --depth 8 --concurrency 12 --recompile --no-repeat \
    --bt-threads 12 --epochs 2 --gauntlet-epochs \
    --gauntlet Leaf_vclassic_eval Leaf_vmaterial_260708-5e6g-final
```

The epoch ladder rates directly against the untouched seed, so ep1 *is* the
measurement: clearly positive ⇒ d8 labels reopen the bootstrap and
freeze-generate/consolidate at increasing depth becomes the recipe; ~0 ⇒ the
label ceiling is not depth-limited at this net capacity, and the
investigation turns to architecture.

## Methodology notes (Part 4)

- Game-signature duplication: `awk` extraction of `[FEN]`+`[PlyCount]`+
  `[Result]` per game from the generation PGNs, `sort | uniq -c` (frozen
  run: 188,571 distinct / 1M; learning run: 990,870 / 1M).
- Position repetition: hash-partition sampling (`md5(fen)[0] % 50 == 0`)
  over the 6e6g corpus — keeps every copy of the sampled positions, so
  per-position copy counts are exact.
- Dedup control: gid-excluded row dedup of the r-run corpus (134.0M →
  25.1M rows), then `train.py --skip-online --corpus ... --state <seed>`
  with the ladder vs the incoming seed; artifacts in
  `learn/seedctl-dedup_work/`.
- Book test: `learn/holdout_openings.epd` (seed 2607), match PGNs
  `learn/inbook_test.pgn` / `learn/oob_test.pgn`, plus the second in-book
  replicate.
- Freeze smoke test and gate placement: commit cc74ebd; duplication tooling:
  commit 3140cb2.
