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

> **Where the code lives.**  Every knob Part 6 tests was rejected, so none of it
> is on `main` — `main` carries the update rule unchanged, which is the baseline
> all of these arms were measured against.  The experimental branches are kept
> for reproduction only:
>
> | knob | branch | verdict |
> |---|---|---|
> | `TDLEAF_STACK_NORM_ALPHA` | `tdleaf-stack-norm-alpha` | rejected, 6.10 |
> | `TDLEAF_FEATURE_DEDUP`, `TDLEAF_FEATURE_RBAR`, `TDLEAF_RBAR_LR_COMP`, `TDLEAF_REP_HIST` | `tdleaf-feature-dedup` | rejected, 6.13 |
>
> All are byte-exact no-ops at their defaults, so the branches differ from
> `main` in behaviour only when a knob is explicitly enabled.  This document is
> the research record and is carried on `main` regardless of which branch the
> code sat on.

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

## 6.9 RETRACTION of 6.8's seed hypothesis — the harness, not the seed (2026-08-13)

6.8 concluded that the endgame over-coherence had "resolved as the net
matured", inferring it from two runs seeded at the chain head both reading
b0/b7 ~ 0.94-0.98 against 1.32-2.12 in six production iterations.  The
confirmation test named in 6.8's "what remains untested" was run and
**falsifies that conclusion.**

Same harness, same depth, same game count, same cadence, alpha=0, seeded from
`m260720-1e6g_final` (-134 vs classic) — the exact state the historical 2e6g
iteration used, which produced b0 = 10.37 and b0/b7 = 1.34:

| run | seed (Elo vs classic) | b0..b7 | b0/b7 |
|---|---|---|---|
| production 2e6g (1M games) | 1e6_final (-134) | 10.37 9.51 8.46 8.81 8.56 8.38 8.02 7.75 | **1.34** |
| this harness (200k games) | 1e6_final (-134) | 7.70 8.94 9.01 8.63 7.91 7.90 7.58 7.74 | **0.99** |

Predicted 1.34, measured 0.99.  Draw rate 27.1%, in band.  All three harness
runs — two seeds differing by 158 Elo — cluster at 0.94/0.98/0.99, while all
six production iterations sit at 1.32-2.12.  **The variable is the harness,
not the seed, not depth, not game count, not cadence.**

Excluded so far, each checked rather than assumed:

- **Binary / source**: production `Leaf_vtrain_hl_a` (Aug 12) and
  `Leaf_valphapre` are built from the same `engine/src/` — no source commit
  between 2026-07-20 and this work — with identical flags
  (`NNUE=1 NNUE_NET=m260720.nnue TDLEAF=1`).  The only delta is the alpha knob,
  proven a byte-exact no-op at 0.
- **Invocation**: matches train.py's `selfplay_run.py` call in every argument
  (`--actors --depth --games-per-actor --total-games --traj-dir
  --delete-consumed --refresh-scores`), `--tdleaf-out` equivalent by naming.
- **The TSV dump env** (`TDLEAF_DUMP_TSV`/`_QUIET_CP`, which production sets
  and this harness does not): excluded by code — `tdleaf_record_ply` gates root
  capture on `tdleaf_dump_wanted() || tdleaf_capture_root`, and
  `selfplay.cpp:360` sets `tdleaf_capture_root` whenever `--traj-out` is
  active, which is always in the actor/learner path.  The dump only makes the
  learner write corpus TSVs; it does not touch the gradient path.
- **Refresh cadence**: tested directly, 500 vs 1000 gives 0.94 vs 0.98.
- **Game count**: the production d6 series is flat from 100k to 1M
  (1.32/1.42/1.45/1.34), so 200k cannot explain 0.99.
- **Opening book**: `training_openings.epd` is dated 2026-07-18, unchanged
  across every m260720 iteration and every harness run.

Top remaining candidate: **the opening-shuffle `--seed`**.  train.py derives it
per iteration (`zlib.crc32(tag)`); this harness hardcoded `20260813` for all
three runs — which would also explain why three runs from very different weight
states cluster so tightly.  Testable in ~1h by re-running one arm with a
different `--seed`.

### What this invalidates, and what survives

- **6.8 conclusion 2 is RETRACTED.**  There is no evidence that the endgame
  over-coherence resolves with net maturity.  6.4's measurement stands as a
  property of production iterations; its absence here is a harness artefact of
  unknown origin.
- **6.8 conclusion 1 is weakened but not void.**  The control genuinely lacked
  dynamic range, so the alpha arms could not be read against the intended
  criterion — that much holds.  But the *reason* is now unknown rather than
  "the pathology is gone", so this is not evidence that alpha is unnecessary.
  What survives independently is the internal comparison: alpha inverts the
  bucket profile (spread 1.31 -> 1.90 -> 2.70) and acts as a differential LR
  cut (PSQT -3.6x, FT -8x, FC biases -1.3x).  That is measured within one
  harness against its own control and does not depend on the production
  comparison.  Alpha stays shelved on that ground alone.
- **6.8 conclusion 3 is withdrawn.**  Endgame coherence is NOT ruled out as
  the mechanism behind the -27/-32 online displacement; it is simply untested,
  because the instrument does not currently reproduce the phenomenon.

**The actionable finding is a tooling one:** `alpha_pretest.sh` cannot validate
or reject any online-update change until the ~1.4-vs-~0.97 gap against
production is explained.  Fix the instrument before running more arms.

## 6.10 The production alpha A/B: the knob works, the theory does not — and online displacement is net-productive (2026-08-14)

D. Homan's call: skip further harness repair and run alpha where the signature
demonstrably exists — a full production iteration, same seed, same everything,
against the already-completed alpha=0 iteration.

```sh
env TDLEAF_STACK_NORM_ALPHA=1.0 python3 train.py \
    --tag m260720-2.5e6g-a1 --continue m260720-2.2e6g \
    --games 300000 --depth 8 --concurrency 12 --recompile \
    --gauntlet-anchors Leaf_vclassic_eval --gauntlet-epochs --gauntlet-tdleaf --gauntlet
```

This is the exact command that produced `m260720-2.5e6g` with `alpha` unset,
modulo the tag and the env prefix.  Both iterations: 300k games at d8 from
`m260720-2.2e6g_final`, ~55.4M/55.8M corpus rows, draw rate 35.3% (in the d8
band), identical baseline val MSE 0.006829.  A cleaner A/B than anything the
pre-test harness could have produced.

### The knob works, decisively, at the weight level

| bucket | alpha=0 `on/upd` | alpha=1 `on/upd` | upd_on (M), a0 / a1 |
|---|---|---|---|
| 0 | 14.85 | **1.82** | 9.5 / 9.5 |
| 1 | 11.04 | 1.36 | 41.7 / 42.1 |
| 2 | 11.38 | 2.04 | 42.6 / 42.2 |
| 3 | 10.10 | 2.69 | 36.2 / 35.8 |
| 4 | 9.06 | 3.21 | 29.8 / 29.3 |
| 7 | 7.94 | 3.04 | 17.3 / 17.3 |
| **b0/b7** | **1.87** | **0.60** | |

Exposure matches bucket-for-bucket, so this isolates the update rule.  Alpha=1
suppressed deep-endgame displacement 8x and inverted the profile — exactly what
the harness predicted (0.60 in production against 0.54 there).  **The harness's
*relative* prediction about alpha was correct** even though its absolute control
level was not (6.9); that discrepancy is now moot for decision-making, since
production is the instrument of record.

### It bought nothing in Elo

| measurement | alpha=0 | alpha=1 |
|---|---|---|
| tdleaf vs classic | −67.9 | **−53** |
| tdleaf vs 2.2e6-final | −20.9 | **−49** |
| final vs classic | +16.7 | **+3** |
| final vs 2.2e6-final | +38.0 | **+48** |
| epoch ladder | ep1 +39.1, ep2 +63.2 | ep1 +52.5, ep2 +64.7 |
| val MSE | 0.006829 → 0.006347 → 0.006363 | 0.006829 → 0.006310 → 0.006340 |

Decomposed against the foreign anchor (seed = −36.3 vs classic):

| | online Δ | offline Δ | iteration total |
|---|---|---|---|
| alpha=0 | −31.6 | +84.6 | **+53.0** |
| alpha=1 | **−16.7** | **+56.0** | **+39.3** |

Alpha halved the online damage and the offline gain fell by twice as much.

**Statistical honesty:** at 1000 games each (±11), a difference of two
measurements carries ±16.  The four comparisons run 0.6σ–1.8σ, so *none* is
significant, and the two anchors contradict each other on the post-online net —
alpha=1 is 15 Elo better vs classic but 28 Elo worse vs the seed.  No
conclusion should be drawn from any single cell of that table.  What follows
leans on the pattern across interventions, not on these deltas.

### The finding: online displacement is net-productive at the margin

Three interventions on the same loop, ordered by how much online weight
movement they permit:

| online displacement | iteration total (foreign anchor) |
|---|---|
| frozen — zero (6.1, 3e6 iteration) | +7 |
| alpha=1 — ~8x suppressed | +39 |
| alpha=0 — full | +53 |

Monotone.  Suppressing online movement degraded the iteration in proportion to
how much was suppressed.  This promotes 6.3's thesis from observation to
intervention: **the online phase's displacement is not damage to be minimised,
it is the mechanism that moves the generator off its fixed point so the corpus
carries labels the seed does not already reproduce.**  The −27/−32 Elo the
online phase costs is the price of that signal, not a defect.

(Caveats: three points, each ±16, and the frozen run was 500k games at a later
chain position.  The monotonicity across three mechanically very different
interventions is what carries the argument, not any pair.)

This retroactively unifies the whole investigation.  The frozen iteration
failed because zero displacement means zero corpus signal.  Alpha failed the
same way in milder form.  And the premise this line started from — that
endgame over-learning was damaging the online phase, so fixing it would
compound — is **wrong**: the endgame over-coherence is real at the weight level
(6.4, replicated across two chains and confirmed controllable here), but it is
not costing Elo.

### Status of the alpha line: CLOSED

- `TDLEAF_STACK_NORM_ALPHA` stays at its default **0.0**, a verified byte-exact
  no-op.  Do not ship a non-zero default.  The knob remains in the tree as the
  reproduction handle for this experiment; `scripts/alpha_pretest.sh` was
  deleted (2026-08-14) under the repo's retire-and-remove convention — it never
  reproduced production's bucket profile (6.9) and production iterations are
  the instrument of record.  Its runs survive in `learn/alphapre_work/` and
  `learn/alphactl_*_work/`.
- **Do not run alpha=0.5.**  It interpolates between two arms that already
  differ by less than the measurement error.
- If alpha=1 vs alpha=0 is ever worth settling properly, the cheap route is a
  direct head-to-head of the existing finals (`m260720-2.5e6g_final` vs
  `m260720-2.5e6g-a1_final`) at 4000–6000 games, which needs no generation and
  sidesteps the anchor contradiction.  The point estimate and the mechanism
  both favour alpha=0, so this is for the record, not for a decision.

### What is now settled, and what remains open

Settled: targets (3.1), the multi-writer merge (Part 5), book diversity (4.4),
freeze-generation (6.1/6.2), and now endgame-coherence reweighting (here) are
all excluded as levers on the online phase.  Reducing online displacement is
not the direction.

Open:

1. **The harness gap (6.9)** — `alpha_pretest.sh` measures b0/b7 ≈ 0.94–1.04
   where production measures 1.32–2.12, unexplained after excluding binary,
   flags, dump env, cadence, weight seed, game count, book, and shuffle seed.
   No longer blocking (production is the instrument), but the harness must not
   be trusted for absolute levels until it is explained.
2. **Whether the online phase can be made *more* productive rather than less.**
   The dose-response curve points the opposite way from every intervention
   tried so far: if displacement is the product, the question is whether more
   or better-directed displacement helps — the reverse of the LR-decay
   hypothesis carried since 3.1, which should now be regarded as doubtful for
   the same reason alpha failed.

## Methodology notes (6.10)

- Runs: `learn/m260720-2.5e6g_final.json` (alpha=0) and
  `learn/m260720-2.5e6g-a1_final.json` (alpha=1); trainer logs and post-online
  states under each `<tag>_work/train/`.
- Alpha confirmed in-run via the learner banner
  (`grep stack_norm_alpha <tag>_work/traj/learner.log` → `stack_norm_alpha=1`).
- Bucket profiles: `bucket_phase_analysis.py <2.2e6_final> <work/train state>
  <final>` for each arm.
- Draw rate aggregated over all 11 actor logs (35.3%, n=6400).

## 6.11 The repeated-measures line: per-feature vote normalisation (2026-08-14)

After 6.10 closed the alpha line, D. Homan reframed the target: alpha had never
addressed the original concern.  It divided by `n_stack` — every record the game
contributed to a material bucket — whether or not anything actually recurred.
The concern was *repeated measures*: within one game it is hard to tell whether
a strong repeated signal is real evidence or the same observation counted many
times.  This section records the mechanical answer to how gradients are applied,
the statistical analysis that follows from it, the implementation, and a
calibration that refuted the premise the design was first pitched on.

### 6.11.1 How gradients are actually applied

Two things happen at two different rates, and conflating them was the source of
several wrong intuitions in this investigation.

- **Accumulation is per ply.**  `tdleaf_accumulate_game` loops over every record
  and calls `nnue_accumulate_gradients`, which for PSQT does
  `grad_psqt_w[fi*PSQT_BKTS + s] += g_psqt_diff * psqt_sign` for every active
  feature of both perspectives.  Nothing is applied.
- **The Adam step is per 8 GAMES.**  `tdleaf_update_after_game` counts games and
  only at `TDLEAF_BATCH_SIZE` calls `nnue_apply_gradients`, which takes one step
  per `(fi, bucket)` with non-zero accumulated gradient, then zeroes.

So a feature present for 40 plies contributes 40 additive terms into one
gradient cell and a single step is taken on their sum, pooled with the other 7
games.  **`psqt_weights_cnt` increments once per batch-apply, not per ply** — so
every "updates" figure in this document (6.4 included) counts batches.  The
median bucket-0 PSQT cell had m = 597 against ~125,000 batches in the
production 1M-game run: touched in ~0.5% of them.

### 6.11.2 Why offline batch=512 differs from online batch=8

D. Homan's framing, confirmed in code and arithmetic:

- The offline trainer does a **full global shuffle** of the training index each
  epoch (`std::shuffle(train_idx...)`), then takes contiguous 512-slices.  With
  55.8M rows over ~300k games (186 rows/game), the expected number of same-game
  *pairs* in a batch is **0.43**, and any given row has a **0.17%** chance of
  sharing its game with another row in its batch.  An offline batch is
  effectively 512 independent games.
- The online batch is **not smaller — it is larger**.  Mean game length is 149
  plies (median 139, n = 200k games), and internal self-play records every ply,
  so 8 games ≈ **1192 records** against 512.  Sample size was never the issue;
  effective independent sample size is.

### 6.11.3 The Adam analysis: the mean is absorbed, the variance is not

Write a cell's batch gradient as `r * g0`.  For sparse PSQT/FT rows `v` is
updated **only when the gradient is non-zero**, so it is an EMA over the batches
in which the feature actually appears, and bias correction uses the per-weight
count — making `v_hat` an approximately unbiased estimate of `E[g^2]` even at
small counts.  Rare features therefore *are* normalised.  What Adam removes is
the **mean** of `r`; what survives is its **variance**:

    step      = r / sqrt(E[r^2]) = r / (rbar * sqrt(1 + CV^2))
    mean step = 1 / sqrt(1 + CV^2)
    step CV   = CV                     (CV = sd(r)/mean(r) for that cell)

**CV(r) is exactly the Adam step-size CV.**  A worked case: a cell whose r is 1
at 99% and 40 at 1% has `sqrt(E[r^2]) = 4.12`, so the rare repeat-heavy game
takes a step of 9.7 while every normal step is suppressed to 0.24.  The scheme
is bad in both directions at once — outliers dominate *and* desensitise the
well-behaved majority.

This also rules out a tempting half-measure: excluding the within-game
repetition from `v` alone.  That restores normal steps to 1.0 but sends the
outlier to 40 (clipped at 30) with no normalisation at all.  Only removing `r`
from numerator and denominator *together* — averaging within the game — fixes
both.

### 6.11.4 A correction to the premise: features are king-relative

The design was pitched on the assumption that a "stationary pawn" repeatedly
hammers one weight.  The architecture largely prevents that:

    halfkav2_feature(persp, ksq, psq, ptype, pside)
        = KingBuckets[ksq_f] + ps + psq_o

`KingBuckets` maps 64 king squares onto 32 buckets — exactly two squares per
bucket, horizontal mirrors of each other, and those two carry *opposite*
orientation (`orient = ((ksq_f & 7) < 4) ? 7 : 0`).  **Any king move re-indexes
every piece in that perspective**: either the bucket changes, or for the
mirror-partner square the bucket is preserved while every `psq_o` flips.  And it
is per perspective, so a white pawn re-indexes when *either* king steps.

So in an endgame with active kings a static pawn's signal is spread across a
fresh row on every king move — the same mechanism that spreads a *moving*
pawn's.  What does land repeatedly on identical cells is a literally repeated
position (shuffling, 3-fold run-ups), because the whole position recurs, kings
included; and since the bucket is material count, which a shuffle does not
change, repeats hit exactly the same `(fi, bucket)` pairs.

### 6.11.5 Implementation

Two modes, both applying **only** to the per-feature sections (FT weights,
PSQT).  FC weights/biases and FT biases are dense — every record touches them,
so there is no per-feature repetition to remove.

- `TDLEAF_FEATURE_DEDUP` = beta — exponent form, weight `r^-beta`.
  beta = 0 disables and is BYTE-EXACT (the divisor is skipped, not computed as
  `r^0`); beta = 1 is full averaging.
- `TDLEAF_FEATURE_RBAR` = k — scale-neutral form, weight `rbar_shrunk / r` with
      rbar_shrunk = (n * rbar_cell + k * rbar_prior) / (n + k)
  and `rbar_prior` the mean r of the cell's material bucket (a global mean for
  FT rows, which have no bucket).  Mutually exclusive with the exponent —
  setting both is a hard error.
- `TDLEAF_REP_HIST=1` — a pure observer (byte-exact against baseline, verified)
  reporting the marginal r distribution, the across-game CV(r), and coverage.

`nnue_accumulate_gradients` consumes precomputed per-cell weight arrays, so all
mode logic stays out of the hot loop behind one null check.  Three passes share
a single `walk_features` lambda (count, histogram, teardown) so they cannot
drift apart; the across-game statistics are collected once per (cell, game) by a
first-visit trick in the teardown walk.

### 6.11.6 Calibration — and the refutation of "surgical"

The design was pitched, by Claude, as touching "roughly a twentieth of the
gradient mass", reasoning from the exact-position-repeat rate (13.8% of bucket-0
rows, 5.0% overall).  **That was wrong**, and the histogram says so:

| r | 1 | 2 | 3 | 4-7 | 8-15 | 16-31 | 32+ |
|---|---|---|---|---|---|---|---|
| % PSQT mass, d8 500g | 7.65 | 5.99 | 5.15 | 19.84 | 29.89 | 24.37 | 7.12 |
| % PSQT mass, d6 1200g | 6.34 | 5.27 | 4.74 | 18.65 | 31.80 | 26.37 | 6.83 |

92.4% of PSQT mass sits at r >= 2 (d8, 2.15M contributions); 93.7% at d6 over
5.03M contributions; 94% in the first 24-game probe.  Mass-weighted mean
r = 15.9.  FT rows are more extreme still (95.5-96.5% at r >= 2).

The reason is that **r counts feature PERSISTENCE, not position repetition**.  A
pawn on e4 with a static king gives r = 40 across a quiet stretch in which no
position ever recurs.  The exact-repeat rate measured a different thing.

Consequently the exponent form is not surgical at all.  Implied gradient-mass
reduction: **1.75x at beta = 0.25, 2.86x at 0.5, 5.94x at beta = 1** — i.e.
beta = 1 is a *larger* intervention than the rejected alpha = 1 (3.6x
displacement), so 6.10's dose-response applies with full force.

### 6.11.7 CV(r): the mechanism is real

| CV band | <0.1 | 0.1-0.25 | 0.25-0.5 | 0.5-1.0 | 1.0-2.0 | >2.0 |
|---|---|---|---|---|---|---|
| % contributions, d8 500g | 0.8 | 0.3 | 3.1 | 60.4 | 34.3 | 1.1 |
| % contributions, d6 1200g | 0.5 | 0.1 | 2.0 | 65.3 | 31.2 | 1.0 |

**Weighted mean CV = 0.95 (d8) / 0.94 (d6)**, replicated across depth on
independent samples.  96% of contributions sit on cells with CV >= 0.5.  Since
CV(r) is the step-size CV, a typical PSQT cell's Adam step swings by ~±95%
purely from how long the feature happened to persist that game, and Adam is
currently attenuating mean steps to `1/sqrt(1+CV^2) = 0.73` to pay for it.

Two reasons this is conservative: the estimator uses the `/n` population form,
biased low (by 29% at n = 2), and CV ≈ 1 is what an exponential distribution of
persistence times gives, which is what one would expect physically.

### 6.11.8 Why the scale-neutral form, and why shrinkage

The exponent fixes the variance but also cuts the mean ~5.9x.  Adam is
scale-invariant in steady state, so a uniform cut *ought* to cost nothing — it
costs something because `v` adapts over ~1000 touches (beta2 = 0.999) while a
median PSQT cell receives only ~450 touches in a 300k-game iteration.  **`v`
never catches up within a run, so a mass cut acts as a straight LR cut for the
whole iteration.**  That is the most likely explanation of alpha = 1's 3.6x
displacement drop, and 6.10 showed such cuts lose Elo.

`rbar/r` avoids this: the current game's r cancels exactly, so r's variance is
removed immediately and completely *however poor rbar is*.  What remains is
rbar's drift between batches, of size ~CV/sqrt(n), which decays as the run
proceeds — so k damps the early estimate rather than capping quality.

An earlier min-games threshold was replaced by shrinkage.  The threshold value
(4) was arbitrary, and the measurement that appeared to support it (min = 1 and
min = 4 both giving 0.93x) could not discriminate them: over 1200 games nearly
every cell accumulates far more than 4 appearances, so both spent the run in the
same regime.  Worse, a hard cutoff gives cells below it *no* normalisation —
full CV = 0.95 noise — precisely for the rare features the argument is about.
Coverage measured after 1200 d6 games:

    n=0: 3.4%   n=1-3: 7.5%   n=4-15: 17.0%   n=16-63: 27.2%   n=64+: 44.9%

72% of contributions come from cells with >= 16 prior games (where a threshold
is irrelevant), and ~11% from cells with n < 4 (where a threshold would have
switched the mechanism off).  Shrinkage covers that 11% with the bucket prior
instead of skipping it.

### 6.11.9 Validation

1200 games at d6 from an identical seed, PSQT displacement over touched cells:

| mode | mean\|dw\| | vs baseline |
|---|---|---|
| off | 37.26 | 1.00x |
| `TDLEAF_FEATURE_DEDUP=1` | 15.32 | **0.41x** |
| `TDLEAF_FEATURE_RBAR=8` | 38.87 | **1.04x** |

The exponent cuts displacement 2.4x — the disguised LR cut.  The scale-neutral
form leaves it at 1.04x, marginally above baseline, in the direction the
mean-restoration argument predicts (0.73 -> 1.0) though well short of the
predicted +37%.

Gates, all re-verified after the weight-array refactor:

- default byte-identical to a binary built from pre-change sources;
- `TDLEAF_REP_HIST=1` alone byte-identical — the diagnostic is a pure observer;
- exponent mode still produces its *pre-refactor* md5, proving the refactor
  behaviour-preserving;
- rbar produces a distinct state; both modes together hard-error.

### 6.11.10 Open items

1. **The +37% did not materialise** (1.04x, not 1.37x).  No confirmed
   explanation.  The likeliest is that `v` is inherited from the seed's long
   history at the pre-normalisation scale, so the restored mean cannot show
   until `v` re-adapts — which predicts the ratio drifts up over a real
   300k-game run.  That is a prediction, not a measurement.
2. **rbar makes the update history-dependent**, so the actor/learner
   bit-exactness gate of 5.3 will NOT hold: a learner restarted mid-run rebuilds
   its rbar history from scratch and diverges from an uninterrupted run.
   Tolerable for an A/B, but it must be decided before this becomes a default,
   and closing it means persisting the `rv_*` tables into `.tdleaf.bin` (a
   format bump).
3. **No Elo measurement yet.**  Everything above is mechanism and weight-level.
   The natural production arm is `TDLEAF_FEATURE_RBAR=8` against unset with
   `--continue m260720-2.2e6g`, directly comparable to the alpha pair's
   +53 / +39.  Note 6.10's measurement-power caveat: 1000-game gauntlets carry
   ±16 on a difference, so only large effects will read.

## Methodology notes (6.11)

- Gradient path: `nnue_training.cpp` `nnue_accumulate_gradients` (per-record
  accumulation, per-feature loop) and `nnue_apply_gradients` (per-batch step,
  `pcnt[b]++`); batch trigger in `tdleaf.cpp` `tdleaf_update_after_game`.
- Feature indexing: `nnue.cpp` `halfkav2_feature` and the `KingBuckets` table.
- Offline batching: `nnue_batch_train.cpp` epoch-level `std::shuffle` of
  `train_idx`; collision arithmetic from corpus rows/games in
  `m260720-2.5e6g_work/train/train.log`.
- Game length: `accumulated N-ply game` lines of
  `m260720-2.5e6g_work/traj/learner.log` (200k games).
- Calibration runs: single-process `--selfplay` with `TDLEAF_REP_HIST=1` — 500
  games at d8 and 1200 at d6, both seeded from `m260720-3e6g_final`.  Read the
  LAST cumulative report in the log, not the first (each batch apply prints one).
- Displacement comparison: `psqt_w` deltas over cells with a positive count
  delta, four arms from an identical seed copy at 1200 games each.
- Byte-exactness gate: baseline binary from `git show <merge-commit>:` copies of
  `tdleaf.{cpp,h}` and `nnue_training.cpp`, 24 `--selfplay` games at d6,
  `.tdleaf.bin` md5 compared (baseline `86c89ad4d951f0e6f2e3201bc9a33b0c`).

## 6.12 The rbar production arm: the mechanism works, and that is why it lost (2026-08-14)

The arm proposed in 6.11.10 was run: `TDLEAF_FEATURE_RBAR=8`, 300k games at d8
from `m260720-2.2e6g_final`, otherwise identical to the alpha pair.

```sh
env TDLEAF_FEATURE_RBAR=8 python3 train.py \
    --tag m260720-2.5e6g-rbar8 --continue m260720-2.2e6g \
    --games 300000 --depth 8 --concurrency 12 --recompile \
    --gauntlet-anchors Leaf_vclassic_eval --gauntlet-epochs --gauntlet-tdleaf --gauntlet
```

### 6.12.1 The prediction of 6.11.10 item 1 was confirmed

The 1200-game d6 pre-test measured only 1.04x displacement against a predicted
1.37x, and 6.11.10 guessed the shortfall was the `v` transient — `v` inherited
from the seed's long history at the pre-normalisation scale — which predicted
the ratio would drift up over a real 300k-game run.  It did:

| bkt | alpha=0 `on/upd` | rbar8 `on/upd` | ratio |
|---|---|---|---|
| 0 | 14.85 | 16.37 | 1.10 |
| 1 | 11.04 | 14.29 | 1.29 |
| 2 | 11.38 | 13.77 | 1.21 |
| 3 | 10.10 | 15.81 | 1.57 |
| 4 | 9.06 | 13.88 | 1.53 |
| 5 | 8.30 | 11.34 | 1.37 |
| 6 | 8.14 | 14.91 | 1.83 |
| 7 | 7.94 | 16.68 | 2.10 |
| **b0/b7** | **1.87** | **0.98** | |

Exposure-weighted: **1.47x**, against the sqrt(1+CV^2) = 1.38 prediction.

Crucially, the gradients themselves did not change.  From the learner clip
telemetry (final cumulative block, N = 37 500 batches):

| arm | mean grad L2 norm | clip fires |
|---|---|---|
| alpha=0 | 0.147 | 0 |
| alpha=1 | **0.009** | 0 |
| rbar8 | **0.153** | 0 |

rbar8's gradient mass is within 4% of baseline — exactly the scale-neutrality
the mode was designed for — while the *applied step* grew 1.47x.  That is the
mechanism working as specified: removing r's variance shrinks Adam's `v` by
sqrt(1+CV^2) and leaves `m` alone, so the same gradient buys a larger step.
(Contrast alpha=1, which cut gradient mass 16x to achieve a 3.6x displacement
cut — Adam absorbing most of a blanket scale cut, as 6.11.8 argued.)

### 6.12.2 It cost 76 Elo

Seed = −36.3 vs `Leaf_vclassic_eval`.

| arm | rel. displacement | b0/b7 | online Δ | offline Δ | **iteration total** |
|---|---|---|---|---|---|
| frozen (6.1) | 0 | — | 0 | +7 | +7 |
| alpha=1 | 0.26x | 0.60 | −16.7 | +56.0 | +39 |
| alpha=0 | 1.00x | 1.87 | −31.6 | +84.6 | **+53** |
| rbar8 | **1.47x** | 0.98 | **−190.7** | +167.8 | **−23** |

A 1.47x step increase produced 6x the online damage.  The offline phase clawed
back +168 (epoch ladder +181 / +200 against its own pretrain — a repair ladder,
not a quality one) and still finished 76 Elo behind the baseline arm.

The corpus degraded with the actors.  The *same* seed net, evaluated on each
run's own held-out slice before any offline training:

| arm | seed val MSE | outcome MSE | draw rate |
|---|---|---|---|
| alpha=0 | 0.006829 | 0.0886 | 34.6% |
| alpha=1 | 0.006829 | 0.0883 | 34.5% |
| rbar8 | **0.007559** | 0.0905 | 33.6% (35.0 -> 33.2 over the run) |

Weaker actors, more decisive games, a noisier corpus.  6.10's framing — the
online phase's product is the corpus — cuts both ways.

### 6.12.3 What this does to the dose-response curve

6.10 read three interventions as monotone in displacement and left open
"whether the online phase can be made *more* productive rather than less."  The
fourth point breaks the monotonicity: the curve is an inverted U with its peak
at or just below the current default.  Going 3.8x *below* baseline costs 14
Elo; going 1.47x *above* costs 76.  That asymmetry — slow decay below the
optimum, a cliff above it — is the classic learning-rate response, which
suggests **the online LR is already at the edge of its stable range.**

Two things are worth recording as closed or damaged:

- rbar8 delivered the flat bucket profile the alpha=0.5 proposal was aiming for
  (b0/b7 = 0.98) and produced the worst iteration in the series.  Third
  independent result against flattening as a lever.
- 6.10's open item 2 is answered in the negative *for this direction*: more
  displacement of this kind is not better.

### 6.12.4 The confound, and why the LR is the only place to fix it

The run cannot distinguish two explanations:

1. **magnitude** — a 1.47x step is simply past the stability edge, and any
   intervention producing it would lose the same way;
2. **the reweighting itself** — the within-game repetition count `r` carries
   real signal, and averaging it away discards evidence.

They are confounded *by construction*, because the step rise is not an
implementation choice: removing the within-game variance necessarily raises the
Adam step by sqrt(1+CV^2), and **no rescaling of the weight can undo it, since
Adam absorbs constant scale factors per weight.**  The learning rate is the one
factor that survives the `m/sqrt(v)` normalisation, so it is the only available
compensation.

This also settles the status of `TDLEAF_FEATURE_DEDUP=1` as a "next thing to
try": it is not a separate hypothesis.  Per weight, dedup's gradient stream is
rbar's divided by the per-feature constant `rbar_i`, and Adam is scale-invariant
per weight, so **in steady state the two produce identical steps.**  They differ
only in the transient — dedup's new scale is ~1/16 of the `v` inherited from
2.2M games, so it would start near 0.4x and climb toward the same 1.47x.  Dedup
is rbar at an accidental, drifting, per-feature-varying lower LR: it might score
better by luck and would explain nothing.

### 6.12.5 Arm A — rbar at matched displacement

Build the actor/learner binary with the FT-weight and PSQT LRs scaled by
1/1.47 = **0.68**:

```sh
env TDLEAF_FEATURE_RBAR=8 python3 train.py \
    --tag m260720-2.5e6g-rbar8lr --continue m260720-2.2e6g \
    --games 300000 --depth 8 --concurrency 12 --recompile --online-lr-comp 0.68 \
    --gauntlet-anchors Leaf_vclassic_eval --gauntlet-epochs --gauntlet-tdleaf --gauntlet
```

0.68 is the MEASURED ratio, not the theoretical 1/sqrt(1+CV^2) = 0.725 implied
by CV(r) = 0.95 — `v` had probably still not fully re-adapted at 300k games.  A
single scalar can only match the exposure-weighted mean; per-bucket ratios ran
1.10x (b0) to 2.10x (b7), so the compensated arm should land slightly under
baseline in b0 and over in b7.  Check with `bucket_phase_analysis.py` before
reading the Elo.

Reading the result:

- **near +53** — variance removal is Elo-neutral at matched step, and the
  repeated-measures line closes exactly as the alpha line did;
- **well above +53** — the line lives, and the exact per-weight form is worth
  building: feed Adam's `v` the UNWEIGHTED gradient magnitude while stepping
  with the denoised numerator, which is self-calibrating per weight instead of
  using one hand-fitted global scalar (cost: a second accumulator stream on the
  FT hot path);
- **still well below** — the reweighting itself is harmful independent of step
  size, and `r` is carrying signal rather than noise.

6.10's measurement-power caveat applies: 1000-game gauntlets carry ±16 on a
difference, so only the third outcome would read unambiguously.

### 6.12.6 Implementation

`TDLEAF_RBAR_LR_COMP` is a **compile-time** flag, not an env var, and that is a
deliberate consequence of the byte-exactness gate.  A runtime multiplier must be
read inside `nnue_apply_gradients`, and doing so — even behind an `if` that is
never taken when compensation is off — changed what the optimiser did with the
surrounding FP code under `-O3 -ffast-math -flto`: 1164 of 263 MB of
`.tdleaf.bin` bytes differed by ±1 after 80 d6 games.  A pure 1-ulp scatter, but
a gate failure.  A probe build carrying every other edit with only the multiply
neutralised reproduced the baseline md5 exactly, isolating the multiply as the
sole cause.  As a macro expanding to nothing, the default build's token stream is
identical to the pre-change sources and the gate holds by construction.

Because it is compile-time it would also apply to `--batch-train`, where the
gradients are *not* reweighted and the factor would be a bare LR cut.  Two hard
guards close that: `tdleaf_check_env()` refuses to start a compensated binary
with `TDLEAF_FEATURE_RBAR` unset, and `nnue_batch_train()` refuses outright.  In
the normal `train.py` flow neither can fire — `--online-lr-comp` goes to the
`train_hl_a` actor/learner binary and `bt` is compiled separately without it.

Gates, 80 `--selfplay` games at d6 from a shared `m260720-2.2e6g_final` copy:

| build | env | md5 | |
|---|---|---|---|
| pre-change (`7b94d9a`) | — | `bdbabfae…` | |
| Arm A sources | — | `bdbabfae…` | **identical** |
| pre-change | `RBAR=8` | `705c675b…` | |
| Arm A sources | `RBAR=8` | `705c675b…` | **identical** — rbar itself untouched |
| `TDLEAF_RBAR_LR_COMP=0.68f` | `RBAR=8` | `e6c95980…` | distinct |

Mean |ΔPSQT| over touched cells across those runs: off 9.526, rbar 9.841
(1.03x — the same `v` transient as the 1200-game d6 calibration), compensated
6.431, i.e. **0.65x of uncompensated** against the intended 0.68.

## Methodology notes (6.12)

- Runs: `learn/m260720-2.5e6g{,-a1,-rbar8}_final.json`; per-run trainer logs and
  post-online states under each `<tag>_work/train/`.
- Bucket profiles: `bucket_phase_analysis.py <2.2e6_final> <work/train state>
  <final>` per arm.
- Gradient norms: last `TDLeaf clip stats` block of each
  `<tag>_work/traj/learner.log` (cumulative, N = 37 500 batches).
- Draw rate: `result=` fields of the learner log in 20k-game blocks (300k games
  per arm, so ±0.3% per block — the 35.0 -> 33.2 drift is real).
- Corpus quality: first `val MSE` line of each `<tag>_work/train/train.log` —
  the seed net scored on that run's own held-out 5% before any offline step.
- Byte-exactness gates: 80 `--selfplay` games at d6, `--tdleaf-out` redirecting
  the write so all arms share one unmodified input state; baseline binary built
  from a `git worktree` at `7b94d9a`.  Baseline md5s above.

## 6.13 Arm A result: the reweighting itself is the cost — line closed (2026-08-15)

```sh
env TDLEAF_FEATURE_RBAR=8 python3 train.py \
    --tag m260720-2.5e6g-rbar8lr --continue m260720-2.2e6g \
    --games 300000 --depth 8 --concurrency 12 --recompile --online-lr-comp 0.68 \
    --gauntlet-anchors Leaf_vclassic_eval --gauntlet-epochs --gauntlet-tdleaf --gauntlet
```

### 6.13.1 The compensation did its job

Exposure-weighted PSQT displacement against the alpha=0 baseline:

| arm | rel. displacement | per-bucket b0 -> b7 |
|---|---|---|
| rbar8 | 1.468x | 1.10 1.29 1.21 1.57 1.53 1.37 1.83 2.10 |
| rbar8 + comp 0.68 | **1.039x** | 0.76 0.91 0.88 1.03 1.06 0.97 1.37 1.60 |

Ratio of the two arms: 0.708 against the 0.68 applied — the LR landed where it
was aimed.  The per-bucket shape is exactly what 6.12.5 predicted from a single
scalar against ratios spanning 1.10x-2.10x: under baseline in the deep endgame,
over it in the opening.  Total displacement is matched to within 4%.

### 6.13.2 And it bought 11 of the 76 Elo

Seed = −36.3 vs `Leaf_vclassic_eval`.

| arm | rel. displacement | b0/b7 | online Δ | offline Δ | **iteration total** |
|---|---|---|---|---|---|
| frozen (6.1) | 0 | — | 0 | +7 | +7 |
| alpha=1 | 0.26x | 0.60 | −16.7 | +56.0 | +39 |
| alpha=0 (default) | 1.00x | 1.87 | −31.6 | +84.6 | **+53** |
| **rbar8 + comp** | **1.04x** | 0.89 | **−142.7** | +131.0 | **−12** |
| rbar8 | 1.47x | 0.98 | −190.7 | +167.8 | −23 |

**At matched displacement the reweighting is still 65 Elo behind baseline.**
Removing the 1.47x step increase recovered 11 Elo of the 76-Elo gap; the other
~85% is the reweighting itself.  Corpus quality moved with it — seed val MSE on
each run's own held-out slice: 0.006829 (baseline), 0.007188 (compensated),
0.007559 (uncompensated).  Draw rate 33.6%, drifting 34.6 -> 33.0, same as the
uncompensated arm.  Gradient L2 norm 0.153, zero clip fires, identical to rbar8
as expected (the LR change does not touch gradients).

This is outcome 3 of the three listed in 6.12.5.

### 6.13.3 What was actually wrong with the premise

`r` is not a repeated measurement.  6.11.6 already established that it counts
feature **persistence**, not position repetition — a pawn on e4 with a static
king yields r = 40 with no position ever recurring, which is why the
mass-weighted mean r is 15.9 and 92% of PSQT gradient mass sits at r >= 2.  That
was recorded at the time as "the intervention is not surgical."  The right
reading, visible now, is stronger: **duration is evidence.**  A feature true for
60 plies of a won game is better support for that outcome than one true for 3,
and `r` is the natural encoding of it.  Averaging each game to one vote per
feature tells the learner those two are equally informative, and that is a loss
of real signal, not of noise.

The variance argument (6.11.7: CV(r) = 0.95 IS the Adam step-size CV) was
correct as arithmetic and irrelevant as a diagnosis.  The step-size variation it
identified is not a defect to be removed — it is the mechanism by which
persistence reaches the weights.

### 6.13.4 Retraction: there is no evidence of an online LR ceiling

6.12.3 read the four-point curve as an inverted U and inferred that "the online
LR is already at the edge of its stable range," from the single observation that
rbar8's 1.47x cost 76 Elo where alpha's 3.8x cut cost 14.  **That inference is
withdrawn.**  It rested on a point now known to be dominated by the reweighting
rather than by step size: at 1.04x the same reweighting still loses 65.

Removing the confounded point, the genuine magnitude evidence is unchanged from
6.10 and still monotone increasing — frozen +7, alpha=1 (0.26x) +39, alpha=0
(1.00x) +53 — and the only within-family magnitude pair runs the *other* way
(rbar 1.47x = −23 vs 1.04x = −12).  The synthesis that fits both: displacement
magnitude is not the axis; **direction quality is.**  Well-directed displacement
pays for itself and more of it is better (the alpha family); misdirected
displacement costs, and more of it costs more (the rbar family).  6.10's open
item 2 therefore reverts to open — "can the online phase be made more
productive" has not been answered, only "not by destroying information."

### 6.13.5 Status of the knobs

`TDLEAF_FEATURE_DEDUP`, `TDLEAF_FEATURE_RBAR` and `TDLEAF_RBAR_LR_COMP` stay in
tree at their disabled defaults, banner-marked as tested and rejected, on the
same terms as `TDLEAF_STACK_NORM_ALPHA`: byte-exact when off, retained so the
result stays reproducible.  `TDLEAF_REP_HIST` stays as a live diagnostic — it is
a pure observer and the r distribution it measures is now a documented property
of the training signal rather than a defect.

`TDLEAF_FEATURE_DEDUP` was never run in production and does not need to be:
6.12.4 shows it is asymptotically the same update as rbar (the two gradient
streams differ per weight by the constant `rbar_i`, which Adam absorbs), so it
can only differ in the `v` transient.

Two lines are now closed by the same shape of evidence — an intervention that
works exactly as designed at the weight level and loses Elo doing it.  Alpha
(6.4-6.10) redistributed displacement across material buckets; rbar (6.11-6.13)
removed within-game repetition.  Both were aimed at the same intuition, that the
online phase over-weights endgames with repeated positions, and the intuition
has now failed twice under direct test.

## Methodology notes (6.13)

- Run: `learn/m260720-2.5e6g-rbar8lr_final.json`, `<tag>_work/` alongside it.
- Compensation confirmed in-run via the learner banner: `TDLeaf LR0: ...
  (online FT/PSQT x0.68)`, `feature_rbar=8`.
- Displacement ratios: exposure-weighted mean of per-bucket `on/upd` from
  `bucket_phase_analysis.py`, weights = that arm's own `upd_on`.
- The `proj` / `cos` columns were examined and are NOT diagnostic here: alpha=1
  shows far more negative offline-vs-online projection (−0.37 to −0.55) than
  either rbar arm (−0.04 to −0.32) while scoring 51 Elo better, so offline
  reversal of online movement does not track iteration outcome.

## 6.14 Reframing: online play as hypothesis generation — and batch size as the clean magnitude knob (2026-08-15)

With both update-rule lines closed (alpha, 6.4-6.10; per-feature normalisation,
6.11-6.13), D. Homan reframed what the online phase is *for*, and the reframing
selects the next knob.

### 6.14.1 The mechanism, as it now reads

> Online learning is productive in the sense that it moves the gameplay enough
> that the variety in the positions used in offline batches is genuinely an
> advantage for learning how to weight those features across a much larger set
> of games.  I perhaps should not be too worried about some Elo loss during
> online learning (as long as it does not become excessive), as the engine is
> updating features that look good/bad over a small number of games; that
> information then goes into the play in subsequent games, those features are
> presumably stress-tested in the subsequent games, and the offline learning
> sorts it out in the end.
>
> — D. Homan, 2026-08-15

This is the strongest reading yet of the whole Part 6 result set, and it fits
every arm.  The online phase is a **hypothesis generator**: batch-8 Adam steps
revalue features on the evidence of a handful of games, which is far too little
evidence to be right.  Being wrong is not the failure mode — it is the point.
The revaluation immediately changes how the engine *plays*, so the next games
probe wherever the new weights have moved, and the corpus accumulates positions
that test the hypothesis.  The offline pass then adjudicates over 55M rows with
a global shuffle, where 512 unrelated positions per batch average away exactly
the small-sample noise the online phase introduced.

Three previously separate observations become one story under it:

- **Frozen generation nets +7** (6.1) while learning iterations net +24..+95.
  A frozen generator proposes no hypotheses, so the corpus tests none and its
  consolidation is close to a no-op — it labels positions with evaluations the
  seed already produces.
- **Online Elo is reliably negative** (−16.7, −31.6) **and the iteration still
  gains** (+39, +53).  The online phase is mid-experiment; scoring it on its own
  strength is scoring a hypothesis before the data are in.
- **rbar lost at matched displacement** (6.13).  It did not reduce movement; it
  changed *which* hypotheses got proposed, by asserting that a feature true for
  3 plies and one true for 60 are equally supported.  Worse hypotheses, same
  budget.

### 6.14.2 What follows for measurement

The corollary is a rule, not just an interpretation: **do not optimise the
online phase's own Elo.**  It is an intermediate quantity, and three arms have
now shown it moving opposite to the iteration total (alpha=1 improved online Δ
by 15 and lost 14 overall).  The only figure of merit is the iteration total
against a foreign anchor.

The open question the rule leaves is the one word "excessive".  There must be a
level of online displacement beyond which the generator degrades faster than the
corpus informs — rbar8's −190.7 online Δ was certainly past it, though that arm
cannot locate the boundary because its displacement was misdirected as well as
large.  Locating it is a magnitude question, and every magnitude arm so far has
confounded magnitude with something else.

### 6.14.3 Why batch size is the knob, and not the update rule

`TDLEAF_BATCH_SIZE` (8 games) varies displacement magnitude while holding
direction quality exactly fixed.  Nothing is reweighted, nothing normalised
away; each game contributes in proportion to what it contains, exactly as at the
default.  The only change is how many games are averaged before each Adam step.

Magnitude scales close to `1/batch`.  Adam's per-step size is `lr x m_hat /
sqrt(v_hat)`, which is O(lr) per touched weight regardless of gradient scale, and
the learner takes `games/batch` steps — 37 500 per 300k-game iteration at batch
8.  So batch 4 is roughly 2x the displacement and batch 16 roughly 0.5x, in the
one dimension that has never been varied cleanly.

It is also **the honest version of what rbar was reaching for**.  rbar attacked
gradient variance by dividing out each game's within-game repetition count,
which destroyed the persistence signal (6.13.3).  A larger batch attacks the
same variance by averaging over *more independent games* — persistence still
enters in proportion, nothing is discarded, and each feature appears in more
batches so Adam's `v` is estimated from denser coverage.  Same target,
information-preserving mechanism, opposite sign on the displacement it costs.

Contrast the three magnitude interventions to date:

| knob | changes magnitude | changes direction | outcome |
|---|---|---|---|
| freeze | yes (to zero) | n/a | +7 |
| `TDLEAF_STACK_NORM_ALPHA` | yes (0.26x) | **yes** — redistributes across material buckets | +39 |
| `TDLEAF_FEATURE_RBAR` | yes (1.47x / 1.04x) | **yes** — discards persistence | −23 / −12 |
| `TDLEAF_BATCH_SIZE` | yes (~1/batch) | **no** | untested |

### 6.14.4 What the existing data says — and does not

**Batch size: nothing usable.**  It has moved three times, every change bundled
with others and every judgement made before the foreign anchor was adopted:

| date | change | bundled with |
|---|---|---|
| 2026_03_19 | introduced at 4 | mini-batch accumulation itself |
| 2026_04_12 | 4 -> 16, "improve learning stability" | depth -> 8, 5000-game sub-iterations |
| 2026_04_13 | 16 -> 4 | 3-fold-repetition fixes |
| 2026_05_01 | 8 -> 4 | K 240 -> 400, FC LR 0.05 -> 0.10 |

None of it survives as evidence.  Part 3.4 proposed `TDLEAF_BATCH_SIZE` 8 -> 64
as the cheapest mechanism-targeted A/B on its list; it was never run.

**Actor refresh cadence: one clean A/B, measuring the wrong observable.**  The
`alphactl_gpa1000` control of 6.9 — 200k games at d6, 11 actors, seed
`m260720-3e6g_final`, everything else identical:

| cadence | PSQT `on/upd` b0..b7 | b0/b7 | PSQT med\|dw\| |
|---|---|---|---|
| 500 games/actor | 6.85 8.78 8.97 7.87 7.20 7.42 7.50 7.28 | 0.94 | 42.95 |
| 1000 (production) | 7.16 8.67 8.93 7.84 7.33 7.47 7.31 7.33 | 0.98 | 44.09 |

2.7% apart.  That result is close to uninformative for the question 6.14.1
raises, for a mechanical reason: the learner runs `--refresh-scores`, which
rescores every trajectory on current weights before computing TD targets, so
**actor staleness never reaches the labels.**  Cadence acts only on the
behaviour policy — which positions get played — while the measurement above is
weight displacement, downstream of the gradient.  It found nothing because it
was looking downstream of where the knob acts.  Under 6.14.1 cadence controls
precisely the interesting thing (how fast the position distribution tracks the
hypotheses), and we have neither an Elo measurement of it nor a corpus-diversity
observable to measure it with.

Note also that seed val MSE on a run's own corpus does **not** work as a
corpus-novelty proxy: alpha=0 and alpha=1 both read 0.006829 while their offline
gains differed by 29 Elo.  A cadence experiment needs a better observable first.
Every production iteration in the chain ran at 1000; the sidecar JSONs do not
record the field because nothing ever varied it.

### 6.14.5 The proposed arms

Batch 4 and batch 16 against the batch-8 baseline's +53, same frame as every
other Part 6 arm (`--continue m260720-2.2e6g`, 300k games at d8, foreign anchor):

| arm | expected displacement | reads |
|---|---|---|
| batch 4 | ~2x | is more well-directed displacement better, or is "excessive" nearby? |
| batch 8 | 1x (baseline, +53) | — |
| batch 16 | ~0.5x | does halving well-directed displacement cost like alpha's 0.26x did (−14)? |

Two confounds to control:

1. **Warmup is counted in Adam steps.**  `TDLEAF_ADAM_WARMUP = 50` and
   `TDLEAF_FT_SESSION_WARMUP = 100` are step counts, so changing batch size
   silently changes the warmup in *game* terms — 400 games at batch 8, 200 at
   batch 4, 800 at batch 16.  The 2026_04_12 change-log entry flagged this at
   the time.  Negligible against 300k games, but it must not be blamed later.
2. **Measurement power.**  1000-game gauntlets carry ±16 on a difference, so
   only effects of alpha's size or larger will read.  Two arms bracketing the
   default give a shape, which is how every conclusion in Part 6 was actually
   reached.

If the response is flat in both directions, the online phase is insensitive to
magnitude across a 4x range and the "excessive" boundary is far outside it — in
which case the next question is cadence, and it needs a corpus-diversity metric
built first.  If batch 4 gains, displacement is under-supplied at the default
and the same lever can be pushed further (or moved to the LR, which is
equivalent to first order).  If batch 4 loses and batch 16 is flat, the default
already sits at the boundary.

## Methodology notes (6.14)

- Batch-size history: `docs/change_log.txt` entries dated 2026_03_19,
  2026_04_12, 2026_04_13, 2026_05_01.  Current value `TDLEAF_BATCH_SIZE = 8` in
  `src/tdleaf.h`; the batch trigger is in `tdleaf.cpp`
  `tdleaf_update_after_game`, which calls `nnue_apply_gradients` once the
  pending game count reaches it.
- Cadence control: `learn/alphactl_gpa1000.log` and
  `learn/alphactl_gpa1000_work/a0/` against `learn/alphapre_work/a0/`.  The
  cadence flag is `selfplay_run.py --games-per-actor` (default 1000), surfaced
  by `train.py --games-per-actor`.
- Warmup constants: `TDLEAF_ADAM_WARMUP`, `TDLEAF_FT_SESSION_WARMUP` in
  `src/tdleaf.h`; both consumed in `nnue_training.cpp` as `warmup_factor` /
  `ft_session_factor` against the step counters `t_adam` / `t_ft_session`.

## 6.15 The batch-16 arm: batch size is not a magnitude knob — and online Elo loss is the dosage meter (2026-08-15)

`TDLEAF_BATCH_SIZE = 16`, 300k games at d8 from `m260720-2.2e6g_final`,
otherwise identical to every other Part 6 arm.

### 6.15.1 6.14.3's central prediction was wrong

6.14.3 argued displacement scales as `1/batch`, since Adam's per-step size is
O(lr) regardless of gradient scale and the learner takes `games/batch` steps.
Batch 16 halves the steps — 18 750 applied batches against the baseline's
37 500 — so displacement should have halved.  It did not move at all:

| measure | batch 8 (baseline) | batch 16 |
|---|---|---|
| Adam steps | 37 500 | **18 750** |
| mean grad L2 norm | 0.147 | **0.213** (x1.449) |
| net PSQT mean \|dw\| | 319.83 | **319.18** |
| net PSQT L2 | 1.700e5 | **1.688e5** |
| PSQT cells touched | 126 108 | 126 051 |
| per-bucket `on/upd` b0..b7 | 14.85 11.04 11.38 10.10 9.06 8.30 8.14 7.94 | 14.84 11.42 11.63 10.22 8.93 8.76 8.38 7.92 |

Displacement matched to 0.2%.  Half the steps, the same distance travelled.

**Why** (interpretation, but the numbers constrain it tightly).  Gradients are
*summed* across the batch's games, not averaged, and the norm rose by 1.449 ~
sqrt(2) — the signature of adding twice as many near-independent per-game
gradients.  Adam then normalises the step to O(1) whatever that scale is.  So
per-step size is unchanged and only the step *count* halved, which for a pure
random walk would give sqrt(0.5) = 0.707 of the net distance and for perfectly
coherent drift 0.5.  Measured 1.00.  The steps must therefore be roughly twice
as mutually coherent — exactly what averaging more games per step buys, since
the cancelling component is the sampling noise in each step's direction.
**Fewer steps, each cleaner, cancel less, and arrive at the same place.**

Batch size is therefore *not* the magnitude knob 6.14 wanted.  It is a
**signal-to-noise knob at constant displacement.**

### 6.15.2 And cleaner steps lost 34 Elo

Seed = −36.3 vs `Leaf_vclassic_eval`.

| arm | net displacement | Adam steps | online Δ | offline Δ | **total** |
|---|---|---|---|---|---|
| frozen (6.1) | 0 | 0 | 0 | +7 | +7 |
| **batch 16** | **1.00x** | **18 750** | **−7.7** | **+27.0** | **+19.3** |
| alpha=1 | 0.26x | 37 500 | −16.7 | +56.0 | +39.3 |
| **batch 8 (default)** | 1.00x | 37 500 | −31.6 | +84.6 | **+53.0** |
| rbar8 + LR comp | 1.04x | 37 500 | −142.7 | +131.0 | −12 |
| rbar8 | 1.47x | 37 500 | −190.7 | +167.8 | −23 |

The run was healthy in every other respect — draw rate 34.7% flat (baseline
34.6%), zero clip fires, seed val MSE on its own corpus 0.006746 against the
baseline's 0.006829, i.e. marginally *less* novel content.  It simply explored
less and the offline phase found correspondingly less to extract (+27.0 against
+84.6).

**Displacement does not order these outcomes.**  Batch 16 and the baseline sit
at identical net displacement and differ by 34 Elo; alpha=1 at 0.26x beats batch
16 at 1.00x.  Two arms are now dead against weight displacement as the
explanatory variable.

### 6.15.3 What does order them: online Elo damage

Sort the six arms by how much the online phase degrades its own play:

| \|online Δ\| | 0 | 7.7 | 16.7 | 31.6 | 142.7 | 190.7 |
|---|---|---|---|---|---|---|
| iteration total | +7 | +19.3 | +39.3 | **+53.0** | −12 | −23 |
| offline gain / \|online Δ\| | — | 3.51 | 3.35 | 2.68 | 0.92 | 0.88 |

Monotone increasing across the whole well-directed family, and roughly linear —
a fit through frozen and the baseline gives `total ~ 7 + 1.46 x |online Δ|`,
which predicts +18.2 at batch 16 (actual +19.3) and +31.4 at alpha=1 (actual
+39.3), both inside the ±16 measurement band.  Then it collapses on the two
rbar arms.

This sharpens 6.14.1 into something operational.  The online phase's Elo loss is
not a cost to be tolerated — **it is the meter reading how much exploration
happened**, and while the exploration is well-directed it pays back ~2.7-3.5x in
the offline phase.  The rbar arms show the other regime: damage of the wrong
kind returns less than 1x.

The mechanism this suggests (interpretation, untested directly): the value the
corpus carries comes from the **noise** component of the online movement, not
the coherent component.  The coherent part is the average gradient — precisely
what the offline pass computes for itself from 512-position batches, so moving
the generator along it adds nothing the offline phase could not find.  The noisy
part is what perturbs play into regions the current weights would not otherwise
visit.  Batch 16 averaged the noise away and kept the coherence; it kept the
generator strong (online Δ only −7.7) and starved the corpus.  Alpha=1 shrank
every step but kept all 37 500 of them, and 37 500 small noisy perturbations
explored more than 18 750 clean ones — which is why it beat batch 16 despite
one quarter the displacement.

### 6.15.4 The prediction for batch 4

Batch 4 doubles the step count to 75 000 and makes each step noisier (predicted
grad norm ~0.147/sqrt(2) ~ 0.104).  By 6.15.1's accounting net displacement
should again come out near 1.00x — more steps, each less coherent.  If 6.15.3
holds, the arm should show **more** online damage and a **higher** iteration
total: `|online Δ|` in the −55 to −65 range and a total near +85 to +95.

That is a real prediction with a real way to be wrong, and three ways it could
fail informatively:

1. **Displacement moves after all** — then 6.15.1's coherence account is
   incomplete and the compensation is a coincidence of this batch ratio.
2. **Damage rises but the total does not follow** — then the linear relation is
   local and the "excessive" boundary sits between 32 and 143, which is exactly
   the unexplored gap.
3. **The corpus degrades** — watch seed val MSE and the draw rate.  The rbar
   arms reached 0.0072-0.0076 and drifted 35.0 -> 33.2 on draws; batch 4 staying
   near 0.0068 and 34.6% would show the damage is exploratory rather than
   destructive.

Batch 4's noise is *sampling* noise from averaging fewer games — unbiased, and
unlike rbar's, which systematically discarded persistence.  That is the reason
to expect it stays in the well-directed family, and it is an assumption, not a
measurement.

## Methodology notes (6.15)

- Run: `learn/m260720-2.5e6g-batch16_final.json`, `<tag>_work/` alongside.
- Batch size confirmed in-run from the learner banner (`batch=16`) and from the
  applied-batch count: `grep -c "applied batch" <tag>_work/traj/learner.log`
  = 18 750 = 300 000 / 16.
- Net displacement: `psqt_w` delta seed -> post-online over cells with any
  change, via `compare_nnue_learning.read_tdleaf_fc`; both mean \|dw\| and the
  full L2 norm reported because the median-based `on/upd` column of
  `bucket_phase_analysis.py` normalises by update count and would have hidden
  the result (`upd_on` fell ~16%, unevenly by bucket, since a bigger batch makes
  each weight appear in a larger fraction of batches).
- Gradient norms: last `TDLeaf clip stats` block of each learner log.

## 6.16 The batch-4 arm: batch 8 is the optimum, and the dosage relation breaks (2026-08-16)

`TDLEAF_BATCH_SIZE = 4`, same frame as every other Part 6 arm.

### 6.16.1 What 6.15 got right

6.15.1's mechanical account holds precisely across the full 4x range:

| batch | Adam steps | mean grad L2 norm | net PSQT mean \|dw\| | net PSQT L2 |
|---|---|---|---|---|
| 16 | 18 750 | 0.213 | 319.18 | 1.6882e5 |
| 8 | 37 500 | 0.147 | 319.83 | 1.6999e5 |
| **4** | **75 000** | **0.101** | **322.53** | **1.7123e5** |

- **Gradient norm scales as sqrt(batch)** — 0.101 / 0.147 / 0.213, successive
  ratios 1.455 and 1.449 against sqrt(2) = 1.414.  6.15.4 predicted ~0.104 for
  batch 4; measured 0.101.  Gradients sum across games and the per-game
  directions are near-independent, exactly as claimed.
- **Net displacement is invariant to batch size** — 1% across a 4x range, with
  the step count varying 4x.  Whatever the batch, the run travels the same
  distance.  6.15.1's coherence-compensation account is confirmed on a third
  point, and weight displacement is now firmly established as *not* the axis.

### 6.16.2 What it got wrong

6.15.4 predicted `|online Δ|` of −55 to −65 and a total of +85 to +95.  Both
were wrong, and not narrowly:

| batch | online Δ | offline Δ | **total** |
|---|---|---|---|
| 16 | −7.7 | +27.0 | +19.3 |
| **8 (default)** | −31.6 | +84.6 | **+53.0** |
| **4** | **−31.7** | +71.0 | **+39.3** |

**Online damage saturated.**  Batch 4 doubled the step count and made each step
sqrt(2) noisier, and degraded its own play by 31.7 Elo against the baseline's
31.6 — identical to within a rounding error, where 6.15.4 expected roughly
double.  Its post-online net measured −68.0 vs classic against the baseline's
−67.9.

This is not one of the three failure modes 6.15.4 named; it is a fourth.  The
dosage relation `total ~ 7 + 1.46 x |online Δ|` predicted +53.3 for batch 4 and
the arm delivered +39.3, so **online Elo damage is necessary but not sufficient**
— two arms with identical damage differ by 14 Elo in what the offline phase
recovers (+84.6 against +71.0).  The relation should be treated as a rough
ordering across mechanisms, not a law.

A speculative account of the saturation, offered as a hypothesis and nothing
more: the gradient depends on the current weights, so once play degrades far
enough the TD errors grow and the update turns corrective.  That feedback would
cap online damage at a level set by the LR and the loss surface rather than by
the batch size, with batch 16's steps simply too coherent to wander out to the
cap.  It predicts an LR sweep moves the cap where batch size cannot.

### 6.16.3 The batch-size response, stated honestly

| batch | total | vs baseline | significance |
|---|---|---|---|
| 16 | +19.3 | −33.7 | **2.2 sigma — resolved** |
| 8 | +53.0 | — | — |
| 4 | +39.3 | −13.7 | 0.9 sigma — not resolved |

With 1000-game gauntlets at ±11, a difference of two carries ±15.6.  So the
defensible statement is: **batch 16 is clearly worse than batch 8, and batch 4
is not better** (it points lower, but a 0.9 sigma gap is not a measurement).
The three points describe an inverted U peaking at the current default, but only
the upper side is resolved.

`TDLEAF_BATCH_SIZE = 8` stays.  It reached its current value through the
confounded history in 6.14.4 and is now, for the first time, supported.

### 6.16.4 A metric that needs no gauntlet

The angle between each arm's online and offline PSQT displacement, computed
straight from the three `.tdleaf.bin` files:

| arm | cos(online, offline) | corpus novelty (seed val MSE) | total |
|---|---|---|---|
| batch 16 | −0.089 | 0.006746 | +19.3 |
| **batch 8** | **−0.113** | **0.006829** | **+53.0** |
| batch 4 | −0.211 | 0.007009 | +39.3 |
| alpha=1 | −0.244 | 0.006829 | +39.3 |
| rbar8 + comp | −0.305 | 0.007188 | −12 |
| rbar8 | −0.334 | 0.007559 | −23 |

Both columns order the six arms monotonically, and the total peaks in the middle
of each — little disagreement between the phases means the offline pass had
nothing to correct and the iteration gains little; large disagreement means the
online phase went somewhere the corpus does not support and offline spends its
budget undoing it.  The optimum sits at modest disagreement, cos ~ −0.11.

Two caveats, both serious.  There is **no repeat-run noise floor**: two
identically configured runs play different games and would not give cos = 1, and
that baseline was never measured, so the absolute scale is uncalibrated.  And
alpha=1 ties the baseline exactly on val MSE while scoring 14 lower, which is
the same tie that disqualified val MSE as a novelty proxy in 6.14.4.  Six points,
two candidate metrics, no held-out test — this is a hypothesis to check, not an
instrument.  It is worth checking because it costs no games.

### 6.16.5 What is left: the learning rate

Batch size is closed in both directions, and it turned out to vary
signal-to-noise at constant displacement rather than magnitude.  That leaves the
learning rate as **the last clean magnitude knob, and the only one never tested
in isolation**:

| knob | displacement | step count | direction quality |
|---|---|---|---|
| `TDLEAF_STACK_NORM_ALPHA` | 0.26x | unchanged | **changed** (redistributes by bucket) |
| `TDLEAF_FEATURE_RBAR` | 1.04-1.47x | unchanged | **changed** (discards persistence) |
| `TDLEAF_BATCH_SIZE` | **1.00x** | 0.5x - 2x | changed (SNR per step) |
| **LR scale** | **directly proportional** | unchanged | **unchanged** |

Adam's step is `lr x m_hat/sqrt(v_hat)` with the second factor O(1), so scaling
every section's LR by a constant scales displacement by that constant, with the
step count fixed and every direction untouched.  Nothing else in this
investigation has that property — every previous magnitude arm changed direction
quality as a side effect, which is precisely what made all of them
uninterpretable as magnitude tests.

It also tests 6.16.2's saturation hypothesis directly, which predicts online
damage moves with LR where it would not move with batch size.

Suggested arms: LR x1.5 and LR x0.67 on all sections uniformly (the same
compile-time-macro treatment as `TDLEAF_RBAR_LR_COMP`, so the default build
stays byte-exact).  Uniform is the right first cut — the per-section LRs are
calibrated to ~0.001 x median(\|w\|) and their *ratios* are not what is in
question.

## Methodology notes (6.16)

- Run: `learn/m260720-2.5e6g-batch4_final.json`, `<tag>_work/` alongside.
- Batch confirmed from the learner banner (`batch=4`) and the applied-batch
  count, 75 000 = 300 000 / 4.
- Health: draw rate 34.3% flat across fifths (34.4 34.6 34.2 34.1 34.4), zero
  clip fires — the arm is not a degenerate run, it simply consolidated less.
  Epoch ladder +65/+65, picked epoch 1 (baseline: +39/+63, picked 2).
- cos(online, offline): PSQT weight deltas seed -> post-online and post-online
  -> final, flattened, via `compare_nnue_learning.read_tdleaf_fc`.  Reported
  because the per-bucket `proj` column of `bucket_phase_analysis.py` is a
  per-bucket regression coefficient and does not aggregate to a single number.
