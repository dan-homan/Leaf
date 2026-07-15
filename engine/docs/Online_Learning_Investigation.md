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
