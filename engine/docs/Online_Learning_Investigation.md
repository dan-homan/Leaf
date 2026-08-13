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

## 4.6 The d8 iteration: the bootstrap reopened — and the family ladder badly understates it (2026-07-17)

`material_260708-d8-1` completed: 188,000 frozen d8 games (26.0M corpus rows
after auto-dedup — same rows-per-game as the deduped d6 corpus, confirming
zero surviving duplication), consolidated in 2 epochs:

| measurement | result |
|---|---|
| epoch ladder vs seed (1+0.01, 1000g) | ep1 **+10 ± 11**, ep2 **+15 ± 11** (picked ep2) |
| final vs seed (3+0.05, 400g) | +13 ± 17 |
| final vs `Leaf_vclassic_eval` (3+0.05, 400g) | **+57 ± 18** |
| seed vs `Leaf_vclassic_eval` (3+0.05, 400g, direct control) | −8 ± 18 |

Three findings:

1. **The d8 bootstrap is open.**  The first positive consolidation of the
   entire late chain (every d6 control read ≤ 0: −3 clean, −48 drifted,
   −23…−89 duplicated).  E ← search_d8(E) has genuine headroom, exactly as
   the historical d6-plateau → d8-bump observation predicted.
2. **The iteration was data-limited, not signal-limited.**  Epoch 2 beat
   epoch 1 (+15 vs +10) and val MSE was still falling at ep2
   (0.006344 → 0.006106 → 0.006051) — the opposite of the late-d6
   epoch-1-does-everything pattern.  26M rows is a fifth of what the d6
   iterations trained on; the 188k-game cap is the book size, not a law.
   Next iteration: bigger fresh-seed book (the generator recipe provides
   unlimited disjoint lines), `--continue` from the d8-1 final, `--epochs 3`.
3. **Within-family matches severely compress real improvements.**  The
   triangle: d8-final is +13 over the seed head-to-head, but +57 vs classic
   while the seed measures −8 vs classic (direct control) — i.e. **~+65 of
   style-robust gain showing as +13 within the family**.  Family games (46%
   draws here) are decided inside blind spots both nets inherited from the
   same lineage; a tactically foreign opponent probes what actually changed.
   Consequences for measurement hygiene: the epoch ladder (NNUE-vs-seed)
   remains fine for epoch *selection* (relative ordering within a run), but
   iterations must be judged by a foreign-anchor gauntlet —
   `Leaf_vclassic_eval` should stay in `--gauntlet-anchors` for the whole
   chain, ideally joined by a second anchor of a different style to avoid
   overfitting the metric to classic specifically.

Status: the freeze-generate → consolidate loop at d8 is the recipe of
record; the chain continues from `material_260708-d8-1_final` with an
expanded book.

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
- d8 iteration (4.6): sidecar `learn/material_260708-d8-1_final.json`
  (ladder + gauntlets), trainer log
  `learn/material_260708-d8-1_work/train/train.log` (corpus size, val MSEs);
  seed-vs-classic direct control: `learn/seed_vs_classic_3s.pgn`.

# Part 5 — Internal self-play, the equivalence study, and the actor/learner split (2026-07-17/18)

The two-process harness was replaced from underneath the loop.  Branch:
`internal-selfplay`; the staged design (game loop → actor/learner → optional
in-process threads) and the anti-goal — never again N optimizer states merged
after the fact (the offline trainer's sharding pathology) — are recorded in
the plan sidecars and `docs/TRAINING.md` "Generation Modes".

## 5.1 Stage 0: the engine plays itself in one process

`--selfplay` (src/selfplay.cpp): whole games in-process, both sides recorded
every ply (dply=1 — the `pow(λ, dply)` trace from the score-trace branch
absorbs the change with no retune), exact in-engine results.  Deterministic at
fixed depth; FRC openings work because `setboard` was already
X-FEN/Shredder-FEN complete.

## 5.2 The equivalence study: no TD-error collapse, and a corpus bias found

5k frozen d8 games per arm vs the fastchess pair on the live net
(`learn/eqstudy_260717_work/RESULTS.md`):

- Same-side (gap-2) |Δd| ratio internal/baseline = **0.931** — the
  negamax-consistency concern is dead; the gated TT-salt mechanism was
  **rejected permanently** (do not build it).
- **The pair path undersamples draws ~40%** (corpus 21.0% vs 35.1% played):
  UCI self-adjudication skips ambiguous games, which are disproportionately
  draws.  Every prior corpus was decisive-skewed; internal self-play matches
  the true mix (33.2% vs ~33%).
- +29% rows/game, ~3.5× per-core throughput, identical lengths/quiet-accept/
  |Δcp| distributions.  `train.py --selfplay-gen` landed with this study.

First production iteration (`material_260708-d8t-2sp`, online d8 trace-target,
1-ply chains, draw-complete corpus): final **+34±17 over its d8t-1 seed**,
foreign anchor 46→+77 vs classic; the online phase lost only −24 vs seed
(d8t-1's had lost −56) — less systematic online damage, as predicted.

## 5.3 Stage 1: one optimizer, and a bit-exactness gate that paid for itself

Actors (frozen, `--traj-out`) emit binary `.tdg` trajectories; ONE learner
(`--learn-stream`, sole `.tdleaf.bin` writer) rebuilds records exactly and
runs the online update.  The acceptance gate — learner must reproduce a
single-process online run **byte-for-byte** — initially failed, and the trace
led to a real engine bug present in every FRC game ever played: on castles
whose destination held the castling side's own rook (or the king itself,
from==to), `nnue_record_delta` subtracted a phantom enemy piece from the
opponent-perspective accumulator, corrupting search evals until the opponent's
next king move and online-TDLeaf gradients throughout (offline corpora were
clean — FEN rebuilds).  Fixed; gate now passes; permanent diagnostics
`TDLEAF_CHECK_ACC=1` / `TDLEAF_TRACE_UPDATE=<file>`.

## 5.4 Two online-stability landmines, one iteration each

- **d8t-3al (collapse #1 — adjudication):** the driver left resign/draw
  adjudication on.  Feedback spiral: evals inflate → earlier resignations →
  short outcome-dominated trajectories → more inflation.  60%→97%
  resignations, ~27-ply games, entry net 0/400 vs everything — with nominal
  gradient norms and pinned PSQT material throughout.  Online self-play must
  play to natural termination.
- **d8t-3al2 (collapse #2 — score staleness):** adjudication off, still
  drifted 37%→12% draws by 40k games.  Epoch-refresh actors ship scores up to
  a refresh cycle stale (~9k games learner-side vs ~8 in the merge path),
  delaying the eval-scale negative feedback.  Fix: learner `--refresh-scores`
  (Flavor-A re-evaluation of leaf statics with current weights at consume
  time).  Aborted at the 75-minute canary.
- **The canary of record is the draw rate** (healthy ≈ 35–40% at d8, held
  rock-steady by every good run), not gradient telemetry — both collapses
  never tripped a grad monitor.

## 5.5 Stage 1 validated in production

`material_260708-d8t-3al3` (8 actors + learner w/ refresh-scores, castle-fix
binaries, 188k d8 games in 4h26m): draws steady 35–40% end-to-end; final
**+23±17 over its d8t-2sp seed** (LOS 97%), anchors classic +80 / 5e6g +70.
The single-optimizer path is equal-or-better to the 9-writer merge and is now
`train.py --actor-learner-gen` (stability defaults baked in; sidecar records
`gen_mode`).  Chain head: `material_260708-d8t-3al3_final`.

Caveat for the next comparison: d8t-3al3 is the first iteration carrying the
castle fix, so its delta vs d8t-2sp mixes fix + architecture + iteration.

## Methodology notes (Part 5)

- Equivalence study: `learn/eqstudy_260717_work/` (RESULTS.md, analyze_eq.py,
  both arms' TSV dumps).
- Bit-exactness matrix: online±emission and learner×2 all byte-compared;
  per-record gradient traces in exact hex (`TDLEAF_TRACE_UPDATE`) localized
  the castle bug to two records with identical Δaccsum.
- Collapse forensics: `learn/material_260708-d8t-3al_work/` (kept),
  actor logs' per-generation W/D/L + termination lines are the draw-rate
  canary source; healthy reference trajectory in
  `material_260708-d8t-2sp_work/selfplay_*.log` (37% flat).
- Sidecars: `material_260708-d8t-2sp_final.json`,
  `material_260708-d8t-3al3_final.json`.

---

# Part 6 — The m260720 chain: freeze-generation is closed at d8 too, and the online phase is what carries the iteration (2026-08-13)

The `m260720` chain ran 3.0M games from a fresh seed through eight
hybrid-loop iterations — six at d6, then d8 from the 2.2e6 mark — with the
actor/learner split throughout (`gen_mode: actor-learner` in every sidecar).
The final 500k games were generated with `TDLEAF_FREEZE=1`, following Part
4.5's recipe of record.  That frozen iteration is the control Part 4 never
ran at d8 with a mature chain behind it, and it inverts the recipe.

## 6.1 Foreign-anchor decomposition of the chain

Every iteration's sidecar carries a 1000-game gauntlet of both the
post-online (`tdleaf`) and post-offline (`final`) net against
`Leaf_vclassic_eval` (±11).  Because it is the same foreign opponent every
time, differencing that column separates the two phases without the
family-compression distortion of Part 4.6:

| iter | games | depth | tdleaf vs classic | final vs classic | online Δ | offline Δ | iter total | final vs prev final (family) | foreign/family |
|---|---|---|---|---|---|---|---|---|
| 1e5 | 100k | 6 | −404 | −349 | — | +55 | — | — | — |
| 2e5 | 100k | 6 | −330 | −294 | +18 | +36 | +55 | +87 | 0.63 |
| 5e5 | 300k | 6 | −287 | −199 | +7 | +89 | +95 | +113 | 0.85 |
| 1e6 | 500k | 6 | −201 | −134 | −2 | +67 | +65 | +91 | 0.71 |
| 2e6 | 1M | 6 | −135 | −61 | −1 | +74 | +73 | +116 | 0.63 |
| 2.2e6 | 200k | 8 | −88 | −36 | −27 | +52 | +24 | +37 | 0.65 |
| 2.5e6 | 300k | 8 | −68 | +17 | −32 | +85 | **+53** | +38 | 1.39 |
| 3e6 | 500k | 8 **frozen** | — | +24 | — | +7 | **+7** | +29 | **0.25** |

(The early rows sit at −300…−400 where the Elo scale is stretched and
differences are less reliable; the comparison that matters — 2.5e6 vs 3e6 —
is in the near-even zone where the anchor is trustworthy.)

Two readings jump out:

1. **The frozen iteration used 1.7× the games of the one before it and
   returned +7 against +53.**
2. **Its family/foreign ratio inverts.**  Every learning iteration converts
   63–139% of its within-family gain into gain against a foreign opponent;
   the frozen iteration converts 25%.  Part 4.6 established that family
   matches *understate* real improvement (~+65 style-robust showing as +13);
   the frozen iteration is the first in either chain where family
   *overstates* it.  Its +29 over the seed is largely family-local.

## 6.2 The objective-space evidence, which needs no Elo at all

The trainer's own validation curve settles it independently of any match
result:

| iteration | baseline val MSE | epoch 1 | epoch 2 |
|---|---|---|---|
| 2e6 (learning, d6) | 0.009186 | 0.008359 | 0.008356 |
| 2.2e6 (learning, d8) | 0.006946 | 0.006423 | 0.006491 |
| 2.5e6 (learning, d8) | 0.006829 | 0.006347 | 0.006363 |
| **3e6 (frozen, d8)** | **0.005928** | **0.005954** | **0.005998** |

The frozen consolidation made validation MSE **rise at every epoch**.  Epoch
1's running train MSE opens at 0.005989 — *above* the baseline — and closes
at 0.005903.  The seed already sat at the optimum of a corpus it had labeled
itself; there was no descent direction to find, and both epochs are noise
around a fixed point.  This is Part 4.3's `seedctl-dedup` result (flat, −3
and −10 Elo, val 0.007845 → 0.007638 → 0.007571) reproducing at d8 with a
mature chain: **freeze-generate → consolidate is closed at depth 8, exactly
as it closed at depth 6.**  Part 4.6's d8-1 iteration was positive because
it was the *first* d8 corpus for a d6-saturated net; once the chain has run
d8 iterations, a frozen d8 corpus carries nothing new.

Corpus size was not the limiter: 69.7M rows (139.5 rows/game after the
auto-dedup that `TDLEAF_FREEZE` enables — a ~25% reduction from the 186–189
rows/game the d8 learning runs produced) against 55.8M for the 2.5e6
iteration that gained +53.  More data, less information.  Generation health
was normal in both (draw rate 34.7% frozen, 36.0% learning, both inside the
35–40% canary band; learner batch applies 0 frozen vs 300k learning,
confirming the freeze took).

## 6.3 What this does to the Part-3 picture

Part 3.5 concluded from the `seedctl` control that the reliable offline gain
was **repair of online self-damage measured against a damaged baseline**, and
Part 3.7/4.5 drew the consequence: retire online learning, freeze the
generator.  The m260720 chain says that consequence does not survive contact
with a healthy online phase.

The correction is not that Part 3 measured wrong — `seedctl` is sound, and
the online phase here still reads −27/−32 in its own right.  It is that
**the online phase's product is not its Elo, it is the corpus.**  A frozen
generator labels positions with the seed's own evaluations, so consolidating
that corpus is a no-op by construction (6.2 measures exactly this).  A
learning generator moves off the seed's fixed point during generation, so
its corpus contains labels the seed does not already reproduce — and offline
consolidation converts those into a properly-fit, style-robust net worth +52
to +85 against a foreign anchor.  The online phase pays ~30 Elo of
displacement to buy that; the loop nets +24 to +53.  Frozen, the loop nets
+7.

Part 3's seedctl was run in the 13-writer merge regime, where online
displacement was large enough to dominate whatever signal the drift carried.
Under the actor/learner split with `--refresh-scores` and natural
termination, the displacement is smaller and the balance flips.  The two
results are consistent; the regime changed underneath the conclusion.

**Recipe of record, revised: keep online learning on during generation.**
`TDLEAF_FREEZE=1` remains the right tool for isolating label quality in a
control, and remains mandatory-with-dedup if ever used for production
generation, but it should not be the default generation mode.

Caveat worth stating plainly: this rests on one frozen iteration (n=1) for
the Elo half.  The val-MSE half (6.2) is not a noisy measurement and is what
the conclusion mainly leans on.

## 6.4 The online displacement is still endgame-concentrated — measured in the current regime

With online learning reinstated, its −27/−32 cost becomes the thing worth
fixing.  `bucket_phase_analysis.py` on the 2.5e6 phase
(`m260720-2.2e6g_final` → `m260720-2.5e6g_work/train/m260720.tdleaf.bin` →
`m260720-2.5e6g_final`), PSQT per-update violence `med|dw|/√updates`:

| bucket | online | offline | online exposure share | offline share | PSQT proj | cos |
|---|---|---|---|---|---|---|
| 0 (1–4 pieces) | **14.85** | 3.35 | 4.2% | 1.8% | −0.03 | −0.07 |
| 1 | 11.04 | 3.30 | 18.3% | 15.4% | −0.07 | −0.10 |
| 2 | 11.38 | 3.35 | 18.7% | 17.9% | +0.01 | +0.02 |
| 3 | 10.10 | 2.95 | 15.9% | 15.6% | −0.04 | −0.06 |
| 5 | 8.30 | 2.31 | 11.6% | 13.2% | −0.19 | −0.31 |
| 7 (opening) | **7.94** | 2.15 | 7.6% | 9.9% | −0.18 | −0.32 |

- Bucket 0 is **1.87× bucket 7** — the *identical* ratio Part 3.2 measured on
  the old 13-writer harness (12.7 → 6.8).  The actor/learner split fixed the
  multi-writer problem and did nothing to this one.
- Offline is 3–4× gentler per update in every bucket.
- PSQT projections in buckets 0–2 are ≈ 0 with cos ≈ −0.1: offline neither
  confirms nor repairs online's endgame PSQT movement.  It persists into the
  final net as unvetted noise, replicating Part 3.2 exactly.
- The concentrated-parameter version shows in the FC stacks: the bucket-0
  fc2 output bias moved **−183** online and offline pushed back only **+87**
  (bucket 1: −73 → +32).  In Part 3 the same parameter moved −182/−112
  online with a +365/+225 offline repair — the drift is unchanged and the
  repair is now weaker.

## 6.5 Mechanism: coherence, not magnitude

`|dw|/√n` is the displacement a *random walk* of n independent updates would
produce.  That this ratio grows 1.87× toward the deep endgame means the
updates there are **correlated**, not merely more numerous — the implied
effective coherence in bucket 0 is ~3.5× that of bucket 7.  Gradient
magnitude is not the mechanism: Adam's `v` renormalizes scale by design (the
same point Part 2 made about per-record error magnitude).

Two within-game sources, both amplified by internal self-play's dply=1
recording of every ply:

1. **Outcome-term saturation in the game tail.**  `e[t]` carries
   `λ^(T−1−t)·(result − d[T−1])`, which → 1 as t → T.  Every late record in a
   game therefore holds the same sign and nearly the same magnitude.
2. **Near-identical positions.**  Those records share an FC stack, a PSQT
   bucket, and most FT feature indices.  Corpus measurement over 4000 games
   of the 2.5e6 leaf rows: 44.5% of rows sit in buckets ≤2 at a median 31
   rows/game, and *literal* within-game position repeats are 13.8% of
   bucket-0 rows (max 13 copies of one position), 8.0% in bucket 1, 0.9% in
   bucket 7.

`TDLEAF_BATCH_SIZE` is 8 **games**, so a bucket's effective sample size per
Adam step is the game count regardless of how many correlated records each
game contributes — while the step size is normalized per weight like
everything else.  Endgame weights random-walk further per step by
construction.  This is Part 3.4's reinterpretation, now measured rather than
inferred, and it survives every architectural change since.

Suspected amplifier, not yet measured: `id_weight = 1/(1 + var/σ²)`
upweights records with stable iterative-deepening scores, and fixed-depth
endgame searches are the most stable in the game — so the ID-stability
weighting should be systematically boosting exactly the over-coherent
records.  Checkable from a `TDLEAF_TRACE_UPDATE` dump (the `var=` field
against `stack=`).

## 6.6 The fix: TDLEAF_STACK_NORM_ALPHA (implemented 2026-08-13)

Each record's gradient is divided by `pow(n_stack, alpha)`, where `n_stack` is
how many records its game contributes to that record's FC/PSQT material
bucket — a pre-pass in `tdleaf_accumulate_game`, applied to `grad_scale`
alongside the existing `id_weight`.  At `alpha = 1` a game teaches each
material bucket exactly one lesson no matter how many plies it spent there;
`alpha = 0.5` is the sqrt-n compromise; `alpha = 0` is the previous
behaviour.  Compile-time default `TDLEAF_STACK_NORM_ALPHA = 0.0`, env-
overridable for the sweep (and added to `tdleaf_check_env()`'s allowlist —
the binary refuses to start otherwise).

It subsumes the literal-repetition problem for free, and it does not touch
the target math, which Part 3.1 exonerated across three completely different
error formulas.

Validation before any training run:

- **alpha = 0 is byte-for-byte the old code.**  A 24-game strict `--selfplay`
  run produces an md5-identical `.tdleaf.bin` to a binary built from the
  pre-change sources.  The divisor is *skipped* at alpha = 0 rather than
  computed as `pow(n, 0)`, which is what makes this exact; the
  actor/learner bit-exactness gate (5.3) depends on it.
- **The arithmetic is exactly 1/n_stack.**  A `TDLEAF_TRACE_UPDATE` diff of
  alpha 0 vs 1 over one game: observed `gs` ratio per stack matches `1/n`
  to a max relative error of 8e-8 (stacks of n = 6, 10, 14, 19), while
  `max|e_alpha1 - e_alpha0| = 0.0` — the targets are bit-identical, only the
  per-record weighting moved.

## 6.7 The pre-test, and two traps in setting it up

`scripts/alpha_pretest.sh` runs one short actor/learner generation per alpha
from an identical seed and diffs seed → post-online per material bucket.  No
gauntlet: a flattened `on/upd` profile is what earns a full iteration.

Two failure modes were hit while building it, both worth recording because
each produces a run that *looks* completely healthy:

1. **The `.tdleaf.bin` content-hash trap.**  A state file carries a hash of
   the `.nnue` it was trained against and refuses to load against any other —
   then the engine plays on from freshly-derived shadow weights, so the run
   completes normally, the logs look right, and the experiment measures
   nothing.  The tell is in the state file, not the log: update counts
   *reset* (a toy run showed `psqt_cnt` total 16,608 against the seed's
   12.85e9) and `bucket_phase_analysis.py` reports **negative** `upd_on`.
   The cause: the chain's states are all anchored to the BASE net
   `m260720.nnue`, not to the baked `m260720-*_final.nnue` exports.  The
   script now pairs base net + base-name state (seeded by copying the chain
   head onto it) and hard-fails on any "Refusing to load" line.
2. **The measurement has a resolution floor in game count.**  A 400-game d6
   smoke showed the profile *inverted* (bucket 0 = 1.90, bucket 7 = 7.27), and
   a full 20k-game d8 control arm still read b0/b7 = 0.80 — bucket 0 the
   *quietest* bucket, against 1.87 in the 300k reference.  The initial reading
   was that `|dw|/sqrt(n)` grows as sqrt(n) for coherent updates, implying the
   test needed ~110k games.  **That reasoning is wrong**, as the chain's own
   completed iterations show when profiled the same way:

   | depth | games | b0/b7 |
   |---|---|---|
   | d6 | 100k | 1.32 |
   | d6 | 300k | 1.42 |
   | d6 | 500k | 1.45 |
   | d6 | 1M | 1.34 |
   | d8 | 200k | 2.12 |
   | d8 | 300k | 1.87 |
   | d8 | 20k (control arm) | 0.80 |

   The ratio does **not** scale with n: four independent d6 iterations spanning
   10x in game count all land in 1.32-1.45.  What actually happens is a
   resolution floor — below a threshold between 20k and 100k games the deep
   endgame accumulates too few updates (0.7M at 20k vs 9.5M at 300k) to
   resolve, and above it the value is flat.  **>= 100k games/arm; the number
   read off an arm is comparable to any other arm at any n above the floor.**

3. **The signature is depth-dependent, and d6 is the better test bed.**
   d8 shows it more strongly (1.87-2.12) but d6 shows it with a far tighter
   baseline (1.32-1.45 across four runs, sigma ~0.06) at ~5x the throughput.
   That the effect is *stronger at greater depth* is independent support for
   the `id_weight` amplifier hypothesis in 6.5: deeper search makes endgame ID
   scores relatively more stable, so the stability weight upweights exactly
   the over-coherent records more at d8.  Pre-test at d6 for power per hour;
   validate the winning alpha at d8, the production depth, with a foreign
   anchor.

Reading the result:

- **PASS** — `on/upd` flattens (bucket0:bucket7 ratio drops toward 1)
  *and* aggregate displacement stays the same order of magnitude as
  alpha = 0.
- **FAIL-1** — profile flattens but aggregate displacement collapses: alpha
  is acting as a blanket LR cut, not a decorrelator.  Re-run the winning
  alpha with the section LRs scaled up to match alpha = 0's displacement
  before believing anything.
- **FAIL-2** — profile unchanged: within-game per-stack coherence is not the
  mechanism; test the `id_weight` amplifier from 6.5 instead.

In every arm the draw rate must hold 35–40% (the 5.4 canary) — a quiet run
flatters every other number on the page.

## Methodology notes (Part 6)

- Foreign-anchor decomposition: `learn/m260720-*_final.json`,
  `tdleaf_gauntlet` / `final_gauntlet` entries for `Leaf_vclassic_eval`
  (1000 games each).  Family column is the same sidecars' `final_gauntlet`
  entry against the previous iteration's final.
- Val MSE and corpus sizes: `learn/m260720-*_work/train/train.log`
  (`batch-train: ... positions loaded`, `val MSE(blend)` lines).
- Freeze confirmation: `TDLEAF_FREEZE=1` startup notice in
  `m260720-3e6g_work/traj/actor_*.log`; 0 learner batch-apply lines vs 300k
  for 2.5e6.  Draw-rate canary from the actors' running `+W =D -L` lines.
- Bucket forensics: `python3 scripts/bucket_phase_analysis.py
  m260720-2.2e6g_final.tdleaf.bin
  m260720-2.5e6g_work/train/m260720.tdleaf.bin
  m260720-2.5e6g_final.tdleaf.bin`.
- Endgame concentration / repetition: leaf rows (`$5==0`) of
  `m260720-2.5e6g_work/corpus.tsv.gz`, first 4000 gids, bucket =
  `(piece_count−1)/4` from the FEN board field.
- Alpha knob: `src/tdleaf.h` (`TDLEAF_STACK_NORM_ALPHA`,
  `tdleaf_stack_norm_alpha()`), `src/tdleaf.cpp` (pre-pass in
  `tdleaf_accumulate_game`, allowlist entry, config banner).
- Byte-exactness gate: baseline binary built from `git show HEAD:` copies of
  `tdleaf.{cpp,h}` (do NOT use `git stash` for this — the repo carries
  unrelated stash entries), both run over 24 `--selfplay` games at d6 from the
  same seed, `.tdleaf.bin` md5 compared.
- Pre-test driver: `scripts/alpha_pretest.sh` (arms under
  `learn/alphapre_work/a<alpha>/`; each arm bookless by design — `run/` carries
  `main_bk.dat` and a book would collapse the opening diversity the
  measurement depends on).

## 6.8 The pre-test result: alpha rejected, and the pathology is gone from the chain head (2026-08-13)

Three arms, 200k games each at d6, all from an identical copy of
`m260720-3e6g_final`, draw rate 27.4-28.2% throughout (the d6 band):

| arm | per-bucket violence b0..b7 | b0/b7 | spread | PSQT med\|dw\| | ft_w med\|dw\| |
|---|---|---|---|---|---|
| alpha=0 | 6.85 8.78 8.97 7.87 7.20 7.42 7.50 7.28 | **0.94** | 1.31 | 42.95 | 0.16 |
| alpha=0.5 | 2.51 2.59 3.64 4.06 4.46 4.76 4.74 4.66 | 0.54 | 1.90 | 18.06 | 0.04 |
| alpha=1 | 1.65 1.13 1.72 2.30 2.78 2.99 2.97 3.07 | 0.54 | **2.70** | 11.84 | 0.02 |

**The control never reproduced the pathology.**  b0/b7 = 0.94 against
1.32-1.45 in every historical d6 iteration — bucket 0 is the *quietest*
bucket here, not the loudest.  Exposure rules out a sampling artefact: 31.7
updates/game in bucket 0 versus 33.1 for the closest historical run, and the
whole per-bucket exposure vector matches within ~5%.  With no dynamic range
to flatten, the treatment arms cannot be read against the intended criterion.

**Alpha also fails on its own terms** (FAIL-1 of the pre-committed reading).
It does not flatten the profile, it *inverts* it — spread 1.31 -> 1.90 ->
2.70 — while cutting PSQT displacement 3.6x and FT weights 8x but FC biases
only 1.3x.  That is a differential LR cut, not a decorrelator.  Obvious in
hindsight: dividing by `n_stack` bites hardest exactly where records-per-game
is highest, and it overshoots straight past uniform.

### The control anomaly tracks the SEED, not depth or game count

| seed (Elo vs classic) | run | b0 violence | b0/b7 |
|---|---|---|---|
| 1e5_final (-349) … 1e6_final (-134) | four d6 iterations | 10.4 - 11.4 | 1.32 - 1.45 |
| 2e6_final (-61), 2.2e6_final (-36) | two d8 iterations | 14.9 - 16.2 | 1.87 - 2.12 |
| **3e6_final (+24)** | pre-test d6 **and** the 20k d8 calibration | **6.85 / 6.54** | 0.94 / 0.80 |

Both arms seeded from the current chain head land at ~6.5-7.2 regardless of
depth.  This also reframes 6.7's calibration: the 20k arm's low b0 was read
as "under-resolved", but it was probably measuring this same real effect.

One confound separated the pre-test from the historical iterations —
`--games-per-actor 500` where train.py's production default is 1000, i.e.
actors refreshed twice as often, which reduces score staleness and therefore
TD-error size.  Re-run at production cadence, everything else identical:

| control | b0..b7 | b0/b7 | PSQT med\|dw\| |
|---|---|---|---|
| cadence 500 | 6.85 8.78 8.97 7.87 7.20 7.42 7.50 7.28 | 0.94 | 42.95 |
| cadence 1000 | 7.16 8.67 8.93 7.84 7.33 7.47 7.31 7.33 | **0.98** | 44.09 |

Indistinguishable.  Cadence is not the confound, and with it matched the
harness is methodologically identical to a historical d6 iteration except for
the seed.

### Conclusions

1. **`TDLEAF_STACK_NORM_ALPHA` is shelved at its default 0.0** (a verified
   byte-exact no-op).  The knob and its pre-test harness stay in the tree —
   the measurement is reusable and the mechanism may return at other seeds or
   architectures — but nothing should ship it as a non-zero default on this
   evidence.
2. **The endgame over-coherence has resolved as the net matured.**  6.4's
   measurement was real when taken (2.5e6 iteration, seed at -36 vs classic);
   it is absent at +24.  The plausible mechanism is self-limiting: a
   well-calibrated endgame eval makes `result - d[T-1]` small, so the
   saturated same-sign tail gradient shrinks on its own.  The pathology was a
   symptom of an under-trained endgame, not a structural defect of the update
   rule.
3. **The -27/-32 Elo online displacement at the chain head (6.1) therefore
   needs a different explanation.**  Endgame coherence is ruled out at this
   maturity; targets were ruled out in Part 3.1; the multi-writer merge was
   removed in Part 5.  The remaining candidates from 3.1 are the
   phase-boundary Adam-v mismatch (checkable with `TDLEAF_LOG_STEP_CLIPS=1`)
   and plain fixed-LR noise-ball diffusion around an offline-selected optimum
   — the latter now the leading hypothesis, and it predicts that online LR
   decay across the chain is the indicated fix rather than any reweighting.

### What remains untested

The seed explanation rests on one seed.  The clean confirmation is to run this
same harness from an older state (e.g. `m260720-1e6g_final`) at d6/200k: if
b0/b7 returns to ~1.4, seed maturity is confirmed as the variable; if it does
not, something else about this harness differs from the production iterations
and the historical comparison is unsafe.  ~1h.

## Methodology notes (6.8)

- Arms: `learn/alphapre_work/a{0,0.5,1}/`, driver log `learn/alphapre.log`.
- Production-cadence control: `learn/alphactl_gpa1000_work/a0/`, log
  `learn/alphactl_gpa1000.log` (`GAMES_PER_ACTOR=1000 OUT=... ALPHAS=0`).
- 20k d8 calibration: `learn/alphapre_calib_20k_d8_work/`.
- Draw rates aggregated over all 11 actor logs per arm, not the first four
  the script prints.
