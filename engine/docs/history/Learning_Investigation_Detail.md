# Learning Investigation — moved detail

Detail moved out of `docs/Learning_Investigation.md` on 2026-09-27, when that
document was restructured for readability.  Every block below is **verbatim** as
it stood in the synthesis on 2026-09-27 (section and statement references inside
it therefore use the synthesis's numbering of that date: in particular the λ-optimum
statement was then the first "§1 M", now §1 U, and §6 item numbers refer to the
old ranking).  The synthesis keeps each result's conclusion and headline numbers
and points here for the tables, arm designs and derivations.

Same conventions as the rest of `history/`: preserved as it was, not re-verified.
The chronological records these blocks cite (`7.13.3`, `Offline 3.2`, …) are
`Online_Learning_Investigation.md` and `Offline_Learning_Investigation.md` in this
directory.

## D1. The maturity scope box (the synthesis's original preface, 2026-09-15 → 09-19)

> ## ⚠️ Scope: this is what a MATURE net does
>
> **Essentially every measurement in both records was taken on a net with millions
> of games of learning already in it.**  The arms of Parts 1–4 ran from a seed at
> ~5M cumulative games; Part 5 from the `d8t` series; Part 6 from `m260720` at
> 2.2–3.0M; Part 7 and the entire offline investigation from `m260720` at
> 5.5–7.0M.  **No arm anywhere in either document was run on a young net.**
>
> That matters most for the central finding.  **The handoff damage is only
> *apparent* after roughly a million games of learning, sometimes longer**
> (D. Homan, from the chain's own history).  The chain's foreign-anchor
> decomposition agrees as far as it goes: the online Δ read **+18, +7, −2, −1** at
> 100k → 1M games and only turned clearly negative at 2.2M (−27), reaching −151 by
> 6e6 [6.1, 7.1].
>
> **Two readings, and we have never run the measurement that separates them:**
>
> 1. the excursion genuinely does not happen on a young net — the optimum is not
>    yet sharp enough to fall off; or
> 2. **it happens every time and is simply masked**, because early online learning
>    still gains real Elo and the leg total is the *sum* of a handoff loss and a
>    learning gain.  Only once learning saturates does the loss show up naked.
>
> **ANSWERED 2026-09-17 on the `m260916` chain (R9) — reading 2, and smaller than
> either party expected.**  Two 50k-game ladders from an early and a late
> offline-trained state give a weighted mean of **−0.70 ± 3.23 (early)** against
> **−13.46 ± 3.23 (late)**, a difference of +12.70 ± 4.57 (2.8σ).  So the cost is
> masked early and visible later, but at ~13 Elo rather than the ~130 a mature
> `m260720` net paid.  Full numbers and caveats in §1 K.
>
> **The offline half was measured separately on 2026-09-19** (`cons1`, §3): a
> wide consolidation window is NOT the answer on this chain — four legs at a
> matched dose lose −9.2 ± 8.8 to the newest leg alone — and the target and row
> type are both at or past their optimum.  §4 gained three closures.
>
> Read every "−120 to −150 Elo" in this document as **a mature-net number at the
> pre-2026-09-15 LR**, superseded for current work by §1 K.  Treat the §4
> graveyard the same way: a knob that bought nothing on a saturated net at 4× the
> present learning rate has not been tested on a growing one.


---

## D2. The search-budget price list (full tables; §1 N keeps the summary)

**N. The search budget has a measured price list, and DEPTH IS A FLOOR.**  The
single most useful planning artifact in this document, because it converts wall
clock into label quality at a known exchange rate.

Read `selfplay.cpp:470` first: `min_search_depth = cfg.nodes ? cfg.depth : 0`,
and the node break at `search.cpp:548` fires only once `max_ply >=
min_search_depth`.  **`--depth D --nodes N` means "search to at least D, then
stop at the first iteration boundary past N nodes".**  Depth is a floor that
must be paid in every position; the node budget only extends past it, and only
where the position is cheap.  Two consequences that are easy to get backwards:
raising `--nodes` alone barely moves a config whose depth floor already costs
more than the budget, and a config with a LOW floor and a high budget searches
deepest exactly where the position is simplest.

Cost, measured as median nodes per move over 24 positions drawn from the `5e6g`
corpus (n=24, so adjacent rows are within noise of each other):

| config | median depth | median nodes | clock |
|---|---|---|---|
| `d6/800` (the chain to 6e6g) | 6 | 3029 | 1.00× |
| `d6/2000` | 8 (adaptive) | 3341 | 1.10× |
| `d8/2000` | 8 | 4914 | 1.62× |
| `d7/2000` | 7 | 5008 | 1.65× |
| `d8/4000` | 8 | 12013 | 3.97× |

Strength, 600-game matches of one net against itself at two budgets, **fixed
nodes under a non-binding clock (`-tc 600+10`), so these numbers are
hardware-independent and transfer between machines**:

| contrast | Elo |
|---|---|
| `d6/2000` vs `d6/800` | **+190.8 ± 13.4** |
| `d8/2000` vs `d6/2000` | **+182.5 ± 13.1** |
| `d8/4000` vs `d6/2000` | +268.4 ± 15.3 |
| `d8/4000` vs `d8/2000` | **+71.6 ± 11.3** |

Consistent to 14 Elo around the triangle (`E−B` measured 268.4 against `E−C` +
`C−B` = 254.1), which is ordinary Elo non-additivity.  The resulting frontier,
cumulative from the chain's own `d6/800`:

| config | clock | Elo over `d6/800` | Elo per unit of ADDED clock |
|---|---|---|---|
| `d6/2000` | 1.10× | +191 | **1900** |
| `d8/2000` | 1.62× | +373 | **351** |
| `d8/4000` | 3.97× | +445…+459 | **31** |

The exchange rate collapses by ~60× across the ladder.  **`d8/2000` is the knee**:
it captures 72% of what `d8/4000` gains for 41% of its clock, and buys uniform
depth 8 rather than the adaptive depth of `d6/2000` — which matters for learning,
because adaptive depth searches deepest where positions are simplest and so makes
label quality heteroscedastic across the corpus.

⚠️ **The budget is a label-quality knob, not only a strength knob.**  At `d6/800`
the chain has been generating TD targets and corpus labels from a search ~373 Elo
weaker than `d8/2000` of the same net.  Under G, that is the quantity that
governs whether the bootstrap still has headroom.


---

## D3. Corpus statistics are not training hyperparameters — the seven `cons1` calibration arms (§1 O in full)

**O. CORPUS STATISTICS ARE NOT TRAINING HYPERPARAMETERS.  Seven arms, seven
losses, and the losses track one thing.**  The single most decision-relevant
result of the calibration programme (2026-09-20, `cons1`, all offline arms from
the `5e6g` pre-offline seed, 2000 games each, rated against the seed and
`classic_eval`).

Every change derived from fitting the corpus lost on the anchor, across three
different parameters, both scale and shape, and both directions:

| arm | what the fit said to do | anchor vs `null` | paired |
|---|---|---|---|
| `npA` | path-dependent λ-return product, mean-preserving | **−5.7 ± 9.9** | −9.6 ± 8.7 |
| `nK` | flat K 220 → 190 (fitted 188–192) | −7.7 ± 9.9 | −16.5 ± 8.8 |
| `nshape` | K shaped by material (U), mean-preserving | −14.6 ± 9.7 | −7.3 ± 8.8 |
| `nwB` | position-only outcome weight, as measured | −16.0 ± 10.1 | −4.9 ± 8.8 |
| `nwA` | position-only outcome weight, mean-preserving | −19.0 ± 10.0 | −6.6 ± 8.9 |
| `nout` | λ 0.985 → 0.9925 (toward the fitted 0.991) | −20.1 ± 9.9 | −20.5 ± 9.1 |
| `npB` | path-dependent product, as measured | **−33.5 ± 10.0** | −22.1 ± 8.8 |

And the control that moves AWAY from the fit was fine: `ncp`, λ → 0.970 against a
fitted 0.991, read −5.0 ± 10.0 anchor / +6.5 ± 8.8 paired, with a direct
head-to-head of **+9.7 ± 5.9**.  `ncp2` (0.9775) −3.8 ± 9.9.  So this is not
"the loop is insensitive" — it is specifically that the fitted direction is the
wrong one.

**The losses are ordered by HOW MUCH OUTCOME WEIGHT MOVED, not by how cleverly
it was allocated.**  `npB` doubles the mean outcome weight (0.63 against `main`'s
0.32) and loses 33.5; `nout` raises it and loses 20.1; `bout` did the same on
the composite corpus and lost 20.4; the mean-preserving arms cluster at −6 to
−19.  Four independent measurements, one direction: **more outcome weight
loses, roughly in proportion to how much is added.**

**λ^(N−ply) is already close to optimal.**  `npA` is the correct λ-return —
the product of a material-dependent λ walked along the game's real material
trajectory, with α solved so the mean matches `main` exactly (0.3210 against
0.3214 over 68.9M rows).  It reproduces `main` rather than beating it: −5.7 ±
9.9, inside noise.  A materially more sophisticated construction recovers the
constant that was already there.

**But the trajectory carries real signal.**  Compare the three mean-preserving
offline weightings:

| weighting | anchor |
|---|---|
| trajectory + material-aware rate (`npA`) | **−5.7** |
| position-only, matched mean (`nwA`) | −19.0 |
| position-only, as measured (`nwB`) | −16.0 |

Discarding the trajectory costs ~13 Elo relative to keeping it.  The
within-material sd of `λ^(N−ply)` is 0.13–0.25, which looked like trajectory
NOISE and is better read as information: the same position in a game that mated
quickly is a different training example from one in a long grind.

⚠️ **This closes the line, not the measurements.**  The per-bucket fits are real
facts about chess and are kept in §4 — K is U-shaped in material (≈170 at 13–20
pieces, ≈270 in bare endgames) and λ is monotone (20-ply half-life in the
opening, 400+ in the endgame).  What is refuted is the inference from them to a
training hyperparameter.


---

## D4. What the offline pass responds to — game diversity, row type and leaf-row dose (§1 H in full)

**H. What the offline pass responds to is game diversity and row type — but the
diversity half is REGIME-DEPENDENT and reverses on the young chain.**  On the
mature `m260720` chain at 4× the present LR, drawing from 2.5M games instead of
500k at identical rows, epochs, optimizer steps and wall clock was worth **+36
anchor / +45 paired** [Offline 2.4].  On the young R9 chain the same manipulation
is **negative**: the `cons1` arms (below) put a four-leg composite at **−9.2 ±
8.8 paired / −12.1 ± 9.9 anchor** against the newest leg alone, at a matched
68.9M-row dose.  Neither cons1 figure reaches 2σ, so the honest statement is
"not helpful, plausibly ~10 Elo harmful" — but the sign is consistent across two
instruments and opposite the mature-chain result.  **Do not carry the +36/+45
into current work.**

The row-type half survives intact and has now been re-measured on a corpus whose
leaf rows are trustworthy.  At a fixed row budget root rows beat leaf rows by
**+35.6 ± 11.0 paired** [Offline 3.2] pre-R8; `cons1`'s `nleaf`, which is the
first test on an R8 corpus and holds the GAME SET fixed, agrees in sign at
**−3.8 ± 9.0 paired / −18.4 ± 10.0 anchor**.  Smaller, same direction.  The 60 cp
quiet gate is correct: 60/120/200 are flat and removing it costs **−27.9 ± 11.3**
[Offline 4.3] — the discarded tail carries label *information* (ΔMSE_out +52% in
the top bin) that a static evaluator cannot represent, because those labels are
good precisely because search resolved a tactic.

**Dose via leaf rows does not clear the bar either (2026-09-20).**  `nboth`
trained on the newest leg's root rows AND all its leaf rows over the same ~1M
games — 153.7M rows against the null's 68.9M, a 2.23× dose with the game set
unchanged.  On the anchor: `null` −105.3 ± 6.9, `nboth` −114.6 ± 7.2, `nleaf`
−123.6 ± 7.2.  **The ordering is monotone in how much of the corpus is leaf
rows** — root only > root+leaf > leaf only — so leaf rows dilute roughly in
proportion to their share, and 2.2× the dose does not buy past it.  (It won its
direct head-to-head against the null by +17.9 ± 5.9; see §5 for why that does not
decide the question.)  Note the confound that went unresolved because the anchor
closed the question first: one epoch over 2.23× rows is also 2.23× the Adam
steps, so "more data" and "more steps" were never separated here.  **Offline dose
via EPOCHS is untouched by this** and remains available — the chain already ships
`picked_epoch` 2 on six of eight legs.


---

## D5. The offline target's λ optimum — the four-point table (§1 U, formerly the first §1 M)

**M. The offline target sits in a broad λ optimum, and the slope above it is
steep.**  `--bt-td-lambda` sets the outcome's weight as `λ^(N−ply)`; at the
default 0.985 that is 0.08 at ply 0 of a 167-ply game and ~0.37 averaged.  Four
points on the single-leg corpus, one epoch each, 2000 games each:

| td_λ | vs seed | vs classic_eval |
|---|---|---|
| 0.9925 | +0.5 ± 6.5 | −125.4 ± 7.1 |
| **0.9850** (default) | **+21.0 ± 6.3** | **−105.3 ± 6.9** |
| 0.9775 | +13.0 ± 6.1 | −109.1 ± 7.1 |
| 0.9700 | **+27.5 ± 6.1** | −110.2 ± 7.2 |

**Only one of these is a real result: 0.9925 is worse, by −20.5 ± 9.1 paired and
−20.1 ± 9.9 anchor, agreeing to within 0.4 Elo on two independent instruments.**
Everything in 0.970–0.985 is flat within resolution, and the two instruments do
not even agree on the ordering there (paired ranks 0.970 > 0.985 > 0.9775, anchor
ranks 0.985 > 0.9775 > 0.970).  The 0.9775 interior point came in *below both its
neighbours* on the paired column — a shape no smooth curve produces, and the
clearest sign that this interval is noise.  The one trustworthy comparison inside
it is a DIRECT match, 0.970 vs 0.985 = **+9.7 ± 5.9 (1.6σ)**, which hints at the
lower value without establishing it.

Two consequences.  **More outcome weight is the wrong direction**, confirmed
independently on both corpora (−20.5 on the single leg, −10.3 ± 9.0 on the
composite).  And because the cp channel is carrying more of the useful signal
than the default λ credits it with, **this bounds `--bt-rescore` downward**: the
cheap screen that was supposed to license the expensive arm has come back
negative.


---

## D6. The `m260916` chain record — BayesElo scale, leg decomposition, the epoch confound, and the `cons1` arm design (§3 R9 in full)

**The chain to 6e6g, on one BayesElo scale.**  A 38-PGN combined rating (38,000
games, 20 players) puts every `-tdleaf` and `-final` net of the chain on a single
scale, which is a better instrument than the per-leg paired matches below —
those compare each leg only with its predecessor.  Online Δ = `tdleaf(N) −
final(N−1)`, offline Δ = `final(N) − tdleaf(N)`:

| leg | kgames | online Δ | offline Δ | leg total | online per 100k |
|---|---:|---:|---:|---:|---:|
| `2e5g` | 100 | +76 | +11 | +87 | +76.0 |
| `5e5g` | 300 | +56 | +9 | +65 | +18.7 |
| `1e6g` | 500 | +41 | +2 | +43 | +8.2 |
| `2e6g` | 1000 | +39 | +31 | +70 | +3.9 |
| `3e6g` | 1000 | +29 | +8 | +37 | +2.9 |
| `4e6g` | 1000 | +14 | +42 | +56 | +1.4 |
| `5e6g` | 1000 | +16 | +12 | +28 | +1.6 |
| `6e6g` | 1000 | **−3** | +30 | +27 | **−0.3** |

**The online phase has crossed zero, one leg earlier than §6 item 1 predicted**
(that item said 7–8M games).  Per-leg BayesElo errors are ±9–13, so −3 is
"indistinguishable from zero", not "significantly negative" — but the trend
across eight legs is unambiguous and the per-100k column falls by two orders of
magnitude.  Offline is unaffected and is now carrying the entire leg: +30 of the
+27 total.

Under §1 G this is a search-margin failure, not necessarily saturation, and §1 N
prices the fix: the chain generates at `d6/800`, which is ~373 Elo weaker than
`d8/2000` of the same net.  The prediction on record is that restoring the margin
returns online Δ to +20…+40; if it does not, saturation is real and generation
should stop being funded.

**The leg decomposition.**  Each leg is rated by a paired family match against
the previous leg's final net: the online Δ is the post-generation `.tdleaf` state
against that opponent, the leg total is the post-consolidation net against the
same opponent, and offline is the difference.  Recomputed from the `_final.json`
sidecars:

| leg | games | epochs | online Δ | leg total | offline Δ |
|-----|------:|-------:|---------:|----------:|----------:|
| `2e5g` | 100k | 2 | +84.3 ± 8.7 | +104.1 ± 8.8 | +19.8 ± 12.4 |
| `5e5g` | 300k | 2 | +63.6 ± 8.6 | +92.5 ± 8.3 | +28.9 ± 12.0 |
| `1e6g` | 500k | 2 | +50.4 ± 8.3 | +65.4 ± 8.5 | +15.0 ± 11.9 |
| `2e6g` | 1M | 2 | +49.0 ± 8.1 | +90.2 ± 8.6 | +41.3 ± 11.8 |
| `3e6g` | 1M | 2 | +41.2 ± 8.2 | +54.3 ± 8.1 | +13.1 ± 11.6 |
| `4e6g` | 1M | **4** | +6.9 ± 8.5 | +66.1 ± 8.5 | +59.2 ± 12.0 |
| `5e6g` | 1M | **4** | +10.4 ± 8.3 | +41.9 ± 8.1 | +31.5 ± 11.6 |

Three readings, in descending order of confidence.

*The online phase is productive on every leg.*  This is the headline, and it is
the thing that was **not** true of the mature `m260720` chain, where online legs
came in flat or negative and §4's graveyard entries were closed on that basis.
Under R7+R8 on a young net, generation adds Elo every time.

*Across the four 1M legs the online contribution falls and the offline one does
not.*  Weighted least squares on the four equal-size legs: online **−14.9 ± 3.7
per leg (4.1σ)**, leg total **−13.2 ± 3.7 (3.5σ)**, offline **+1.6 ± 5.2
(0.3σ)**.  The offline fit has χ²/dof = 7.9/2, i.e. real leg-to-leg scatter
beyond match error, so "flat" means "no trend resolvable through large noise",
not "steady".

*The epoch change does NOT explain it — checked and dismissed.*  The two legs
where online collapsed are also the two that extended the offline ladder from 2
epochs to 4, which looks like a confound and is not one, for two reasons.
**Timing:** leg *N*'s online phase runs *before* leg *N*'s consolidation, from leg
*N−1*'s final net, so `epochs(N)` cannot reach `online Δ(N)`.  The relevant
quantity is `epochs(N−1)`, and the **onset** of the collapse — `3e6g` +41.2 →
`4e6g` +6.9 — starts from a 2-epoch net on both sides.  **Selection:** `train.py`
ships the best-rated epoch, not the last, and `picked_epoch` is 2 on six of the
eight legs including `5e6g` (whose ladder peaks at e2 +49.3 and *decays* to e4
+26.8).  Only `4e6g` shipped an epoch-4 net (+45.1, within 1σ of its own e2
+41.2).  So the offline dose was effectively constant across the chain.

The one residue is second-order: picking the max of four noisy ladder points
instead of two carries a selection bias of order +4 Elo, which inflates
`final(4e6g)` and therefore deflates `online Δ(5e6g)` — the last point only, by
roughly 4 of its 34-Elo shortfall.  The decline stands.

**The `cons1` consolidation arms (2026-09-19), run within R9.**  Offline-only:
no new games, seven one-epoch arms from a single seed, each rated over 2000
games at 1+0.01 on `holdout_openings.epd` against two opponents — the seed
itself (paired) and `classic_eval` (foreign anchor).  Driver
`run_consolidation_arms.py`, sampler `sample_corpus.py`; both documented in
`SCRIPT_USE.md`.

Three design choices carry the results.  **The seed is the PRE-offline state**
(`m260916-5e6g_work/train/m260916.tdleaf.bin`, exported as
`m260916-5e6g-tdleaf.nnue`), not `5e6g_final`: seeding from the post-offline net
would have made the control measure over-consolidation on data it had already
seen twice, and left the target arms nothing to learn against.  **The dose is one
leg's entire eligible corpus**, 68,934,511 rows, so the control is that corpus
unsampled — the composite at any useful quota is larger than a single leg
(four legs at quota 19 is 75.8M rows), so the dose had to be set by the smaller
side.  **Every arm matches it row for row**, with the composite quota-sampled at
19 rows/game over 4M games and trimmed by lowering the per-game cap.

| arm | corpus | td_λ | vs seed | vs classic_eval |
|---|---|---|---|---|
| `null` | 5e6g alone, unsampled | 0.985 | +21.0 ± 6.3 | −105.3 ± 6.9 |
| `base` | 4 legs, quota 19 | 0.985 | +11.8 ± 6.2 | −117.4 ± 7.1 |
| `bout` | 4 legs | 0.9925 | +1.6 ± 6.5 | −137.8 ± 7.4 |
| `nout` | 5e6g | 0.9925 | +0.5 ± 6.5 | −125.4 ± 7.1 |
| `ncp2` | 5e6g | 0.9775 | +13.0 ± 6.1 | −109.1 ± 7.1 |
| `ncp` | 5e6g | 0.970 | +27.5 ± 6.1 | −110.2 ± 7.2 |
| `nleaf` | 5e6g leaf, same games as `null` | 0.985 | +17.2 ± 6.4 | −123.6 ± 7.2 |

Plus one direct head-to-head, `ncp` vs `null` = +9.7 ± 5.9 (745/689/566).
Conclusions in §1 H and §1 M; three lines closed in §4.

**The pipeline validates against the chain's own numbers.**  `null` is one epoch
on the 5e6g corpus from the 5e6g pre-offline state — exactly what the leg's own
epoch 1 did, rated against the same opponent at the same time control.  The leg
recorded **+16.3 ± 8.9**; `cons1` got **+21.0 ± 6.3**.  The two corpora differ by
three rows out of 68.9M.  Remaining differences are host (Linux vs this Mac, so
different nps at a fixed clock), opening book (`training` vs `holdout`) and n
(1000 vs 2000), which is why the agreement is "consistent", not "reproduced".

Two narrower confounds worth remembering: per-leg Δ is **not normalised per
game** — the early legs are 100k–500k games, so the online yield *per 100k games*
falls far faster than the table's per-leg column (84 → 21 → 10 → 4.9 → 4.1 → 0.7
→ 1.0), which is the shape of an ordinary saturation curve; and **hash 16**
applies only to the `6e6g` leg (everything else ran at 128, and generation has
since reverted).


---

## D7. `eval_noise` — the full record (design, implementation defect, calibration, TD-error measurement)

**Positional-uncertainty perturbation — `eval_noise` (2026-09-20 → 09-23).**

*What it is.*  A zero-mean cp offset on the static eval, keyed on the **pawn
structure alone** plus a per-process salt, so it is constant across every
non-pawn move: it perturbs which structure the engine steers toward, never its
tactics or its material trades.  Actors only (`train.py --eval-noise CP`, each
actor drawing its own field, salt = seed + slot); the learner never carries it.
Two properties matter for learning.  The field is **fixed per actor process**, so
an actor makes the same structural bet ("structure X is worth +12") game after
game — a consistent, repeatable misjudgment rather than per-node noise.  And both
sides of a self-play game **share** the field: it is not one side blundering
against a clean opponent, both misjudge the same structures and the consequences
surface only as the true value asserts itself later.

*The design argument (why it could teach anything).*  Play is perturbed, labels
are not.  The label for a record is the CLEAN value of the leaf of the PV the
noisy search chose, so a perturbed move the net already understands is priced
into that label and produces **no TD error**; only consequences the clean net did
not foresee do, and those are what it can learn from.  σ must be large enough
that the diverted choices have consequences, and small enough that they do not
swamp the gradient's ability to discriminate.  That balance has two costs, both
growing with how often and how badly the noise diverts play: **variance** (label
jitter) and **bias** — with λ = 0.985 per ply each target blends the next ~60–70
plies of the game actually played, which contain further noisy choices, so the
trace learns the value of the σ-player, not of the position.  Self-play symmetry
cancels that on average but not position by position.  The phase-2 Elo cost
counts every consequential mistake, foreseen or not, so it **overcounts** the
useful part; the useful part is measured directly below.

*Implementation defect (2026-09-20 → 09-23, change_log 2026_09_23a).*  The
"labels are clean" half was believed to need no code, and it did.  The actor
records its leaf and root STATICS clean (`nnue_evaluate` bypasses `score_pos`),
but the root SEARCH score carried the PV leaf's offset, and `--refresh-scores`
could not remove it — it shifts the root by (refreshed leaf − recorded leaf),
both clean, so by 0.  `leaf_ok` therefore compared clean against noisy and
deleted ~half the records (128 → 66 per game at σ 20, → 44 at σ 30), keeping
exactly those whose leaf drew near-zero noise; the offline root labels carried
noise of sd ~σ.  **No production leg ever ran with eval_noise.**  Fix: the actor
subtracts the PV leaf's offset from the root score (root-STM POV, mates
untouched) before the gate, the dump or the `.tdg` see it; σ 0 is bit-identical.
Still noisy, second order: the per-iteration `id_scores` behind the ID-variance
weight.

*Calibration (m260916-7e6g, d6/800, frozen; 30k self-play games per arm, 4k-game
noisy-vs-clean matches, fastchess 95% CI).*

| σ (cp) | draw % | mean ply | Elo vs clean |
|---|---|---|---|
| 0 | 22.20 | 132.8 | — |
| 5 | 22.07 | 133.1 | +2.0 ± 9.6 |
| 10 | 22.30 | 132.8 | −0.7 ± 9.7 |
| 15 | 21.84 | 132.3 | −17.0 ± 9.6 |
| 20 | 21.83 | 132.4 | −19.4 ± 9.4 |
| 30 | 21.67 | 131.4 | −45.6 ± 9.6 |
| 40 | 21.50 | 130.6 | −66.0 ± 9.9 |

**Free up to σ 10; the cost turns on exactly where the arithmetic says** — a move
choice compares two independently drawn structures, so the distortion has sd
σ√2 and P(>50 cp) goes 0.04% at σ 10 to 1.8% at σ 15.  **It displaces the
position distribution completely:** at σ 5, 100% of games diverge from the
paired control, median divergence at ply 1.  **It does not move sharpness:** draw
rate 22.20 → 21.50 over σ 0→40 (wrong sign, ~2σ), against the target of
recovering 22.5 → 24.2%.  It changes WHICH games are played while leaving their
character untouched — sharpness is a property of the evaluation function, not
of the net's structural preferences, consistent with M.  (The quiet-fraction
arm, q@60 0.440 → 0.412, went through the defective gate and is biased DOWN
under noise; do not quote it.)  Note that σ 40 cost 66 Elo against a clean
opponent yet moved the self-play draw rate only 0.7 points: the structural
consequences are real but small next to the outcome variance already in these
games.  The costs are d6 figures; deeper search cannot correct a pawn-structure
misjudgment (the structure persists to the leaves), so they should not shrink
much at d8 — unmeasured.

*Does it buy learning signal? (2026-09-23, post-fix,
`scripts/arms/eval_noise_tderr.sh`.)*  3 arms × 8,000 d8 games from the frozen
m260921-2.5e6g net through the real actor/learner pipeline, learner frozen with
`--refresh-scores` and dumping every leaf_ok record, the learner's TD recursion
replayed offline (K, λ, 100 cp clip, terminal term; the ID-variance weight is not
dumped and scales the gradient, not e).  Errors one-sigma from 20 game blocks;
rms errors ±0.4%.

| σ | rec/g | rms e (trace) | rms δ (one-step) | δ pawn / other steps | calib. slope | root rows @60 |
|---|---|---|---|---|---|---|
| 0 | 128.3 | 0.1291 | 0.0272 | 0.0306 / 0.0260 | +0.0423 ± 0.0015 | 0.531 |
| 20 | 128.7 | 0.1279 (−0.9%) | 0.0327 (+20%) | +20.5% / +20.0% | +0.0347 ± 0.0020 | 0.514 |
| 30 | 127.1 | 0.1285 (−0.5%) | 0.0373 (+37%) | +37.6% / +36.9% | +0.0275 ± 0.0018 | 0.497 |

- **The trace error that multiplies every gradient does not rise.**  The
  pre-committed reading: at σ 20–30 the noise buys mistakes the net already
  prices, not new signal.  Caveat — rms e measures the *quantity* of TD error,
  not its information; the arms visit different positions, so equal magnitude
  does not prove equal content.  But the mechanism's signature (extra error where
  the net misjudged a structure) is not visible at ~0.4% resolution.
- **The one-step delta rises uniformly on pawn and non-pawn steps**, so it is
  label jitter from choice distortion — each record is the clean value of a line
  a noisy search chose, off by ~11 cp sd at σ 20 and ~16 cp at σ 30 — not
  structural consequences, and it telescopes away under λ.  (A pre-fix run
  appeared to localise the rise to pawn steps; that was the defective gate's
  record selection.)
- **The bias is measurable.**  The slope of e on (d − 0.5) is positive at σ 0
  (games end more decisively than the net predicts) and falls 3σ at σ 20, 6σ at
  σ 30.  Label jitter alone predicts −0.002 / −0.004 (errors-in-variables); the
  residual −0.005 / −0.011 is the trace learning what the NOISY player achieves —
  about a quarter of the net's existing calibration pull at σ 30.
- **The offline corpus thins slightly** (root rows passing 60 cp −3% / −6%),
  because the clean value of a noisily chosen line disagrees more with the root
  static.

*Status.*  **Kept, defaulted off** — the only knob that diversifies the position
distribution without varying openings.  A leg at σ 20–30 would test
**diversity** (whether reaching structures the net's own policy never visits is
worth anything), not learning from unforeseen mistakes, and would pay the Elo
cost plus the calibration bias to do it; any such leg needs a binary built on or
after 2026_09_23a.  **Reopens if:** structural coverage is shown to be worth
something (Offline 2.4's +36/+45 is about distinct *games*, not distributional
coverage, so it does not transfer), or a σ is found where rms e rises while the
slope bias stays small.


---

## D8. PSQT hypotheses — the frozen-arm record (2026-09-24)

**PSQT hypotheses — directional exploration in parameter space (2026-09-24).**

*Why.*  eval_noise is non-directional: its offsets hash the pawn structure, so
each diverted game lands somewhere unrelated and any misjudgment it exposes is a
one-off.  A perturbation of the WEIGHTS is a hypothesis the net itself could
hold; an actor holding it steers consistently toward a class of positions, and
the correction is something the learner can express.

*What the PSQT holds* (`scripts/psqt_decomp.py`, m260921-2.5e6g, 334k positions,
usage-weighted).  Per (plane, bucket) group, the usage-weighted mean over (king
bucket, square) is material; deviations are positional.  PSQT positional sd
88 cp across positions and 35 cp per quiet move, half king-relative; FC 78 cp
per quiet move; the two uncorrelated (−0.001), so **quiet-move positional
variance is 17% PSQT / 83% FC**.  A plain mean over entries misses the material
by 18.5 cp rms (max 69) and Adam counts by 6.3 (max 35), so the engine takes
usage-weighted material from a reference file.

*Mechanism* (`--psqt-noise FRAC`, change_log 2026_09_24a).  One ε ~ N(0, FRAC) per
(piece type, PSQT bucket), 48 in all, scaling that group's positional part by
(1+ε) on own and enemy planes; material exact.  Labels stay clean with no special
code: the statics include the perturbation and `--refresh-scores` removes it.
Verified to 0.37 cp mean against an independent prediction.  `--psqt-opponent`
chooses who plays it: `same` (both sides), `clean` (+ε vs the current net) or
`anti` (+ε vs −ε), the two-sided modes with per-side PSQT tables and per-side
TT/score hash, side A alternating colour (`--pair-openings`, used by the arms, plays each opening twice; legs default to one opening per game so a seed-paired sibling keeps the identical opening sequence).

*Measurement* (`scripts/arms/psqt_noise_tderr.sh` + `psqt_noise_coherence.py`;
frozen m260921-2.5e6g, 8,000 d8 games per arm, FRAC 0.5, paired openings).  The
derivative of the eval with respect to a pattern's scale is exactly its
positional PSQT contribution, so the learner's TD gradient along all 48 pattern
directions is exact from the leaf dump.  All 200 paired games diverge from clean
play, median at ply 1.

| design | seed | side A Elo | align with own ε (z) | corr(Ḡ, ε) | χ²/48 |
|---|---|---|---|---|---|
| clean self-play | — | — | null ±0.2 | — | 1.85 |
| same | 101 / 202 | — | **+27.5 / +26.2** | +0.62 / +0.70 | 23.0 / 25.7 |
| clean | 101 / 202 | **−67.8 / −87.5** (±3.3) | +0.61 / −2.07 | ±0.09 | 5.6 / 9.9 |
| anti | 101 / 202 | +9.0 / −19.8 (±3.4) | **−7.61 / −6.83** | −0.46 / −0.47 | 5.0 / 4.5 |

(Elo 1σ, pentanomial over opening pairs.  χ²/48 is heavy-tail-deflated; compare
between arms, not with 1.)  Trace error rms e is flat in every design; one-step
δ rises +34–40% uniformly, as with eval_noise.

- **Shared-field play only ENACTS a hypothesis.**  In 19 of the 20 patterns
  shifted >3σ the gradient follows ε's sign, for both signs: when both players
  value a pattern at (1+ε) it is worth about that *in their games*, and TD
  learns values under the policy that played.  The hypothesis is never tested.
- **Against a clean opponent the enactment vanishes** (≈ 0, not the half
  expected), and the cost becomes visible: 50% is −68 / −88 Elo — the shared
  arms hid it because the opponent shared the hypothesis.
- **Antithetic play keeps outcomes balanced** (+9 / −20) and the outcome
  separates the hypotheses sharply (2.6σ, 5.8σ, opposite directions).  The
  PSQT-projected gradient pushes against +ε in BOTH arms regardless of which
  side won, and against the other arm's nearly orthogonal ε too (−2.6 / −4.1,
  cos 0.045) — a property of the antithetic design, not of hypothesis quality.
  Plausible cause, unconfirmed: A-as-White and B-as-Black both prefer the
  positions where they disagree most, so c·ε is tied to A's colour.
- **What this does and does not show.**  In no design does the gradient
  *projected on the 48 PSQT pattern directions* track hypothesis quality.  That
  is a statement about one subspace.  Every parameter's gradient is the same TD
  error times that parameter's sensitivity, so the question is what the errors
  correlate with — and the consequences of a PSQT hypothesis (weak squares, a
  slow attack, king safety) are FC-encoded features, not the pattern's own PSQT
  entries.  The response can land in FC/FT gradients the instrument never
  looked at.  Only a learning leg answers it: TODO **H1** (`anti`, FRAC 0.25,
  a repeat of the latest leg once the chain slows).


---

## D9. Calibrating K, λ and the outcome weight from the corpus — the measurements (closed 2026-09-20)

**Calibrating K, λ or the outcome weight from the corpus (closed 2026-09-20,
`cons1`).**  The measurements are sound and worth keeping; the inference from
them to a hyperparameter is what failed.  Full result in §1 O.

*What was measured.*  Raw outputs are not committed — they regenerate in ~8
minutes and the numbers that matter are here.  From `engine/learn/`:

```sh
python3 calibrate_from_corpus.py --source m260916-5e6g_work \
    --games 60000 --max-lag 60 --quiet-cp 60
```

60k games of the 5e6g leg, gated to the training population — and the gate
changes none of it (overall K 188.5 gated against 191.6 ungated, λ 0.9913
against 0.9915), which is itself worth knowing: the quiet gate does not select
positions whose cp→score mapping differs.

- **K is U-shaped in material.**  Per NNUE stack: 268.7 (1–4 pieces), 200.2,
  175.9, **168.9** (13–16), 177.3, 190.2, 199.4, 185.0 (29–32).  Overall
  maximum-likelihood K is **188–192** against the configured 220.  Mechanism is
  plain: with ≤4 pieces a cp edge is often a dead draw, so the mapping must be
  flatter; the 13–20 piece middlegame converts advantages most reliably.
- **λ is MONOTONE in material**, measured as the decay of `corr(ev_t, ev_{t+k})`
  — position to position, the result never entering: 0.99827 (5–8 pieces, 403-ply
  half-life) falling to **0.96613** (29–32, **19 plies**).  The opening is
  plastic; a simplified endgame is nearly static.  Keyed on material rather than
  game ply on the cross-tab evidence (R² 0.855 vs 0.785; ply's residual effect
  confined to the 25–32 piece rows).
- **Outcome reliability**, `Var(outcome − ev)`, rises monotonically 0.0228 →
  0.1915 across the same range, with `corr(ev, outcome)` falling 0.930 → 0.181.

*Why acting on it failed.*  Seven arms, seven anchor losses (§1 O), while the
one arm moving opposite to the fit was fine.  The likely reading: the sigmoid
temperature and the trace decay are **gradient-shaping** parameters, not
calibration parameters.  K = 220 being flatter than the fitted 190 compresses
targets at large |cp|, shrinking gradients on already-decided positions and
concentrating learning near equality — a training-dynamics property with no
connection to how well the sigmoid predicts results.

**Reopens if** the regime changes the gradient economics rather than the
statistics — a much deeper search, a different loss (`--bt-loss-gamma` away from
1.0), or an optimizer change.  Re-deriving the fits on a new chain is cheap and
they will likely still hold; that is not evidence to act on them.  The branch
`k-by-material` carries all of it behind compile flags defaulting off and
byte-identical to `main` when off: `TDLEAF_K_SHAPE`, `TDLEAF_LAMBDA_SHAPE` (with
the path-dependent λ-return walk), `TDLEAF_W_RELIABILITY`, plus `--bt-w-mean`.
**It is deliberately unmerged.**

⚠️ **One piece is untested and this harness cannot test it.**  The ONLINE half of
`TDLEAF_LAMBDA_SHAPE` — per-material decay in the eligibility trace between
adjacent records — never ran: `cons1` arms are offline-only (`--batch-train`),
so `tdleaf.cpp`'s trace code is never executed.  It is a different mechanism
from everything above (credit assignment, not target construction), and testing
it needs a generation leg.


---

## D10. Retired §6 rationale — raising the search budget, and the generation-sharpness evidence behind it (items 1 and 7 as of 2026-09-21; both since acted on)

**1. RAISE THE SEARCH BUDGET TO `d8/2000` — the next leg, and the test of
whether generation is worth funding at all.**  Online Δ crossed zero at `6e6g`
(§3 R9): +39, +29, +14, +16, −3 across the 1M legs, with offline unaffected and
now carrying the whole leg (+30 of the +27 total).  Under §1 G that is a
search-margin failure before it is saturation, and §1 N prices the repair: the
chain generates at `d6/800`, **~373 Elo weaker than `d8/2000` of the same net**.

`d8/2000` is the knee of the frontier — 72% of what `d8/4000` gains for 41% of
its clock, at 1.62× the current leg — and it keeps 1M games and the full ~69M-row
corpus that the *working* half consumes.  Prefer it over `d8/4000` at 500k games
(1.99× clock, half the corpus, and the extra label quality bought at the worst
exchange rate on the ladder).  It also gives UNIFORM depth 8, where `d6/2000`
would search deepest in the simplest positions and make label quality
heteroscedastic.

**Pre-committed reading.**  Online Δ should return to **+20…+40** — the early
chain gave +39…+76 when the margin was large, less the ~13 handoff of §1 K.  If
online stays near zero with 373 Elo of margin restored, the decline was never a
search-margin problem, saturation is real, and the conclusion is to stop funding
generation beyond corpus production and put the clock into the offline half.
Either outcome is worth the leg.  **Canary: the draw rate**, not gradient norms —
32–33% now, healthy is 35–40%, and the `d10` leg that regressed on the old chain
ran at 52%.  ⚠️ **The "32–33%" is unreconciled.**  A direct count of the
generation games — actor logs and generation PGNs independently, agreeing to
0.01% over 1M games per leg — puts `m260916`'s self-play at **22.5%** at `6e6g`
and 29.9% at the `d8/2000` `7e6g` leg, never above 31% after the first 100k
(series in §1 M).  Whichever is right the sign of the recommendation is
unchanged, but the two numbers measure different things and the difference
should be tracked down before either is quoted.  **Method note:** keep passing `--gauntlet-tdleaf`; this whole
decomposition exists only because R9 passed it on every leg.

**7. Generation sharpness — the evidence from `m260720` that backs item 1.**
Not a separate line of work: §1 M is the *game-level* half of item 1's case, and
this entry records what the old chain shows.  `m260916` has spent all 7M of its
games at d6/800 and therefore its whole life at a **22–25% draw rate, outside
`TRAINING.md`'s own 35–40% band** — a canary that went unread because it is
written for d8 and the chain ran at d6.  `m260720` sat *inside* that band for its
entire productive life: flat at 35.3→36.1% across 4.8M games of d8, against
34.5→26.6% over its own 2M games of d6.  Its quiet fraction decayed ~7× slower at
d8 than at d6.  The `m260916` `7e6g` leg at d8/2000 moved the draw rate
22.5 → 29.9, the same direction and roughly the same size as `m260720`'s
26.6 → 35.3 at its own d6→d8 switch — which is why that leg's small **+15.6
total must not be quoted as "depth bought nothing"**: it was the leg that started
paying for the transition.

**On the node budget, which item 1 sets at 2000.**  Measured on `m260720`'s
`d8/4000` legs: the budget adds depth **in the endgame only** (mean achieved
depth 8.14 at 32 pieces rising monotonically to 14.10 at 3), costs **no quiet
rows** (rows/game −0.5% across the introduction, phase composition shift
<0.25 pp — the quiet-fraction dip is games getting 1.9% longer, not rows being
lost), and does **nothing to the draw rate** (35.88 without, 35.96/35.83 with).
So the budget is not a risk to the corpus, and the earlier worry that it was
gating out endgame rows is disproved.  Whether to run it at all is then purely
§1 N's exchange-rate question, and N's price list is the better guide than the
composition argument: at a d8 floor, 2000 nodes is *below* the median 4914 that
d8 itself costs, so it binds only in cheap positions and `d8/2000` is close to
plain fixed d8.  **The leg launched 2026-09-21 runs `--depth 8 --nodes 0`** —
fixed d8, no extension — which for the draw-rate reading is immaterial (the
budget does not move it) but gives up the endgame extension `d8/2000` would buy.
Worth knowing when setting the config for the leg after it.

**Still true:** depth is an optimum, not a monotone — `m260720`'s d10 leg gave
43% draws and −33.1 — so treat the draw rate as a hard gate in both directions.

---

## D11. Fixed depth as a rating instrument — the full record (§1 P, 2026-09-25)

Added to the synthesis on `main` on 2026-09-25 (commit f7d4d00) and moved here verbatim when it was merged with the 2026-09-27 restructure; the synthesis keeps a condensed §1 P and the ledger rows.

**P. FIXED DEPTH IS AN EVAL INSTRUMENT, NOT A STRENGTH MEASUREMENT — and its
scale is depth-dependent.**  (Companion to N, which prices a search budget for
*generation*; P is about the budget used to *rate*.)  Measured 2026-09-25 on an
M5 Pro (6 P-cores, 18 logical), `m260720` binaries, `training_openings.epd`,
the standard gauntlet adjudication.

*Why the instrument differs from a clock.*  `classic_eval` is within the family
— same search, only the eval differs — but "only the eval" still moves two
search economics.  Measured over 20 midgame corpus positions at depth 12:
`classic_eval` runs **2.48 M nps** against the NNUE net's **1.40 M nps**, yet
needs **1.31× more nodes per ply** (132,952 vs 101,618 median).  At a clock the
two nearly cancel — in-game at `3+0.05` both reached mean depth 13.24
(classical) / 13.42 (NNUE) at the same time per move.  Fixed depth removes both
and strips classical of the ~1.8× node budget the clock was handing it.  The
same channel exists *inside* the NNUE family: nps is flat (1.32–1.40 M across
four chain checkpoints) but nodes-to-depth is not — `1e5g` needs 114,843 median
nodes for depth 12 where `7e6g` needs 101,618 (−12% median, −22% total).
Trained nets prune better, a clock pays them for it, and a fixed depth does not.
**That is the point of the instrument, not a defect**: what survives fixed depth
is how good the evaluation is at a fixed search.

*The anchor moves ~85 Elo.*  Same binary, same opponent, same openings:

| contrast | 3+0.05 | fixed depth 12 |
|---|---|---|
| `2.5e6g-final` vs `classic_eval` | **+16.7 ± 19** (n=1000) | **+101.5 ± 26.8** (n=500) |
| `2.5e6g-final` vs `2.2e6g-final` | **+38.0 ± 16.6** (n=1000) | **+18.1 ± 20.2** (n=634) |

(95% throughout; note `pgn_score`/sidecar `err` is ONE sigma — ×1.96 before
comparing with a fastchess line.)

*The scale is depth-dependent, not a fixed offset.*  The same family pair, only
the limit varying:

| condition | n | Elo (95%) | engine-s/game | draw% | SNR | SNR²/min |
|---|---|---|---|---|---|---|
| d6 | 4000 | +12.25 ± 9.5 | 0.22 | 22.3% | 1.29 | 1.35 |
| **d8** | **4000** | **+20.26 ± 9.0** | **1.10** | **30.2%** | **2.25** | **1.13** |
| d10 | 1818 | +34.32 ± 12.7 | 5.04 | 36.7% | 2.70 | 0.73 |
| d12 | 634 | +18.10 ± 19.8 | 11.15 | 46.2% | 0.91 | — |
| 3+0.05 (~d13) | 1000 | +38.0 ± 16.6 | 11.05 | 43.1% | 2.29 | 0.24 |

d12 is the weakest point (widest bar) and is compatible with the trend; what is
left is roughly linear in depth.  **A d8 reading is ~0.5× the TC reading on this
leg**, and the ratio against `classic_eval` runs 0.25–1.00 along the chain (below),
so the two scales are NOT convertible — re-baseline once and never mix.

*It is calibrated at the null.*  `off-g200-final` vs `7e6g-final`, the offline
leg the TC gauntlet called flat: `3+0.05` +0.7 ± 15.3 (n=1000, ~22 min) against
**d8 −0.35 ± 8.9 (n=4000, 4 min 14 s)**.  Same answer, 1.7× tighter, 5× less
wall clock.  The instrument does not manufacture signal.

*The chain ladder keeps its order but not its scale.*  All checkpoints vs
`classic_eval`, d8/1000 games (~1 min each) against the recorded `3+0.05` column:
−348.6/−175.0 (100k), −198.8/−50.0 (500k), −134.1/−12.5 (1M), −60.7/+33.8 (2M),
−36.3/+43.3 (2.2M), +16.7/+39.8 (2.5M), +180.8/+182.2 (7M).  Monotone and
order-preserving; ratio 0.50 → 0.25 → 1.00.  ⚠️ Also note 2.2M→2.5M reads as
*nothing* by anchor differencing (+43.3 → +39.8) where the direct head-to-head
says +20.3 ± 9.0.  Elo is not additive across anchors; keep using paired family
matches for leg deltas.

*What it buys.*  Cost is ~10× lower per game at d8, and because a fixed-depth
result is reproducible and load-immune, concurrency is free: measured aggregate
throughput 70 pos/s at c=8, 95 at c=12, **130 at c=18 (1.84×)**, saturating past
18.  A clock cannot use those cores — per-process nps falls 1.40 M → 1.15 M
(c=8) → 0.94 M (c=18), silently rewriting the effective time control.  Be
precise about what the precision gain is, though: 4000 games at d8 reaches the
*same* confidence as 1000 games at `3+0.05` (SNR 2.25 vs 2.29), because the
effect is half the size.  What you buy is the fifth of the wall clock.  d6 is
nominally the most efficient per minute but halves the effect again and drops
the draw rate to 22%; **d8 is the operating point** — efficient, and still
clearly eval-decided.

*Reproducibility, and why NOT nodes.*  Repeat runs over the same 20 positions:

| binary | limit | positions differing / 20 |
|---|---|---|
| `2.5e6g-final` | `depth 12` | **0** (also 0 under 16-way CPU load) |
| `2.5e6g-final` | `nodes 115000` | **5–8** (one changed the best move; under load, swings to 7× in node count — d11/45,925 vs d16/343,485 on one position) |
| `classic_eval` | `nodes 115000` | 0 |

So `go nodes` is not reproducible in the **NNUE** build, which contradicts the
claim at `uci.cpp:452` ("Deterministic (single-threaded) and load-independent")
and bears on any fixed-node measurement in this record, §1 N's budget ladder
included.  Likely suspect, unconfirmed: under a node budget
`search_cfg.check_inter` is pinned to 1023 (`search.cpp:592`), so
`SEARCH_INTERRUPT_CHECK` → `inter()` → `uci_check_interrupt()` runs ~100× more
often than under a clock.  Until that is understood, **rate at fixed depth, not
fixed nodes** — fixed nodes would otherwise be the better instrument, since it
keeps the nodes-to-depth channel that fixed depth discards.

*Operational trap.*  Under `go depth`/`go nodes` Leaf sets `time_limit = MAXT`
and ignores the clock (`uci.cpp`, the depth/nodes branch) while the driver keeps
enforcing it: at d12 under `3+0.05` an engine used 2.78 s of a ~6.2 s budget,
and d14 would exceed it.  `match.py` now selects `-tc inf` automatically
whenever a depth or node budget is set.

*What this is wired to.*  `match.py --depth N` / `--nodes N`;
`train.py --gauntlet-depth N`, `--epoch-depth N`, with concurrency defaulting to
the core count under either.  Because the fixed-depth Elo cannot be read against
the recorded chain, `train.py` also runs a **`tc-anchor` continuity match** —
`<tag>-final` against every `--gauntlet-anchors` opponent at `--tc`, 1000 games,
recorded separately in the sidecar as `tc_anchor_gauntlet`.  That column is what
keeps the `3+0.05` ladder alive and the only thing that would catch a change
improving eval-per-node while costing search speed.  Every sidecar now carries a
`rating_conditions` block; an Elo without its budget is unreadable.

*Not established.*  The ~0.5× compression factor rests on ONE real-signal leg.
A second is cheap (4.5 min at d8) and should be measured before d8 deltas are
read quantitatively.

### The rating instrument (2026-09-25)

| claim | evidence | effect | regime | grade |
|---|---|---|---|---|
| A fixed-DEPTH match is bit-reproducible and load-immune | §1 P | 0/20 positions differ across repeat runs, solo and under 16-way load | R9 | **ESTABLISHED** |
| `go nodes` is NOT reproducible in the NNUE build | §1 P | 5–8/20 positions differ, one best-move change; to 7× node count under load; `classic_eval` 0/20 | R9 | **ESTABLISHED** (mechanism open) |
| Fixed depth moves the `classic_eval` anchor by ~85 Elo | §1 P | +16.7 ± 19 at 3+0.05 vs +101.5 ± 26.8 at d12, same pairing (95%) | R9 | **ESTABLISHED** |
| `classic_eval` is 1.8× faster per node but needs 1.31× more nodes per ply | §1 P | 2.48 M vs 1.40 M nps; 132,952 vs 101,618 median to d12 | R9 | **ESTABLISHED** |
| Trained nets prune better — nodes-to-depth falls along the chain | §1 P | `1e5g` 114,843 vs `7e6g` 101,618 median to d12 (−12%; −22% total) | R9 | SUPPORTED (n=20 positions) |
| The fixed-depth reading is depth-dependent, ~0.5× of TC at d8 | §1 P | same pair: +12.3 (d6), +20.3 (d8), +34.3 (d10), +38.0 (3+0.05) | R9 | SUPPORTED (one leg) |
| Fixed depth is calibrated at the null | §1 P | `off-g200` vs `7e6g`: +0.7 ± 15.3 at 3+0.05, −0.35 ± 8.9 at d8 (95%) | R9 | **ESTABLISHED** |
| The d8 chain ladder preserves ORDER but not scale | §1 P | monotone across 7 checkpoints; d8/TC ratio 0.50 → 0.25 → 1.00 | R9 | **ESTABLISHED** |
| Fixed depth frees concurrency | §1 P | 70 → 95 → 130 pos/s at c=8/12/18, saturating past 18 | R9 | **ESTABLISHED** |
| d8/4000 games ≈ 3+0.05/1000 games in confidence, at 1/5 the clock | §1 P | SNR 2.25 vs 2.29; 4.5 min vs ~22 min | R9 | **ESTABLISHED** |
