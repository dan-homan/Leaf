# Learning Investigation — what is known about the hybrid loop

**This is the distilled record: read it for conclusions.**  Three documents in
`docs/history/` hold the provenance:

- `Online_Learning_Investigation.md` (2026-07-14 → 09-15) and
  `Offline_Learning_Investigation.md` (09-02 → 09-03) — the chronological
  research records.  Every citation of the form `[7.13.3]` or `[Offline 3.2]`
  points into them.  Blow-by-blow, with many readings later overturned: read
  them for reproduction, never for conclusions.
- `Learning_Investigation_Detail.md` — full tables, arm designs and write-ups
  moved out of this document on 2026-09-27 so it can stay readable.  Cited here
  as `[Detail D6]`.

`TODO.md` owns the checklist of work; this document owns the rationale.  Where
they disagree about status, `TODO.md` is the one to fix.

| section | what it is | use it when |
|---|---|---|
| **Where things stand** | one page on the current chain and the live question | orienting |
| 1. The model | the loop in present tense, one lettered statement per claim | understanding |
| 2. Evidence ledger | one graded row per claim, with its regime | before quoting a number |
| 3. Regime boundaries | the changes that decide whether an old result still holds | before trusting an old result |
| 4. The graveyard | closed lines, each with **what would reopen it** | before proposing anything |
| 5. Measurement manual | hygiene rules, each with what it cost to learn | before designing an arm |
| 6. Open lines | ranked, with the question each answers | deciding what to run |

---

## Where things stand (2026-09-27)

> **Update 2026-10-07.**  `m260929` (fresh 2026-09-29 on R11) reached +141.0 vs
> `classic_eval` at 3+0.05 at 6M games, ~+21–27 per 500k-game d8 leg, against
> `m260921`'s +4.2 best.  Its depth-8 gain slowed to ~+7 per leg over 5–6M.  One
> offline consolidation over the WHOLE 2M→6M history (10 legs, 382.5M PV-quiet
> leaf-confirmed root rows, every label re-scored on the current net, 2 epochs)
> added **+60.3 at depth 8 and +42.5 ± 14.6 at 3+0.05** over 6e6g-final (§1 W) —
> about eight recent legs' worth at d8, from ~7 h of offline compute and no new
> games.

> **Update 2026-09-29.**  The PV is now dumped on every root row, and selecting
> offline rows by PV quietness instead of the 60 cp residual gate is worth
> **+24.9 ± 5.0** on the anchor at depth 8 (§1 V).  It is the `train.py` default,
> and a fresh chain starts on it.  `m260921` reached +4.2 against `classic_eval`
> at 3+0.05 at 4.5M games — its first positive anchor reading.

**The chain.**  `m260921` is a fresh chain started 2026-09-21 on the current
recipe (R7 + R8 + R10, §3): one LR set for both phases, batch 50, PV repairs,
and the root-window fix that finally makes `--depth 8` mean depth-8 labels.  It
ran d6 to 1.5M games, switched to d8, and has reached 3M games in 500k-game legs.
The foreign anchor (`classic_eval`, 3+0.05) rose −364 → −33.  A further d8 leg
from `3e6g` with default settings is running, to see whether learning picks up
as the net matures.

**What the last week established** (§1 T, all at depth 8 with paired openings):

- **The 3e6g leg was run four ways from the same parent** — default, with
  antithetic PSQT hypotheses at 0.25 and 0.5 (§1 S), and at 4× the online LR.
  **No final net moved the anchor by more than ~1σ at equal depth** (+3.0,
  +2.2, −1.7, +5.9, each ±7).  All four gained against the parent (+13, +8, 0,
  +20).  Family gains are not transferring to the anchor.
- **The knobs change *where* the gain arrives, not how much there is.**  4× LR
  reproduces `m260720`'s old signature exactly (online net −21 on the anchor,
  offline recovers +27 / +35); antithetic hypotheses at 0.25 give a gentler
  online phase that gains +18 over 3e6g's online net against the parent, and an
  offline phase with nothing left to add.  The finals end within noise of each
  other.
- **The time-control anchor is too noisy for these effects.**  Two of eight
  3+0.05 anchor matches were ~2σ excursions (n25 final −59, n50 tdleaf −25) that
  depth 8 did not reproduce.  At equal depth the final nets sit between −5 and
−13 against `classic_eval`; most of the 3+0.05 deficit is classic's speed.  Evaluations are now rated at fixed depth 8 (§1 P) — `train.py
  --gauntlet-depth`, or `match.py --depth 8 --srand` for sibling comparisons.

**The live question.**  Is the flat anchor at 3M games saturation of the d8
bootstrap (§1 G), or a slow patch?  `m260720` was also slow at this point (+17.5
from 2.5M to 3M) and gained in bursts across several legs.  The running default
leg is the first reading; §6 ranks what follows.

---

## 1. The model

The statements are grouped by topic.  **Letters are stable identifiers, not an
order** — code, scripts and `TODO.md` cite them — so new statements take new
letters.  Grades and effect sizes are in §2; regimes in §3.

**Maturity caveat.**  Almost everything in the two chronological records was
measured on a net with 2–7M games already in it, at the pre-2026-09-15 online LR
(4× today's) [Detail D1].  Where a statement depends on maturity it says so; K is
the measurement on a young net, and R9/R10 (§3) are the first young chains since
the records began.

| | statement |
|---|---|
| **The loop and its measurement** | |
| A | The loop; the foreign anchor is the only figure of merit |
| P | Fixed depth is an eval instrument, not a strength measurement; its scale is depth-dependent |
| **The online phase** | |
| B | Online cost is a one-time HANDOFF, not a per-step cost |
| C | Around the optimum is an Elo-flat noise ball; η sets its radius |
| D | Direction quality, not displacement magnitude, is the axis |
| E | Σ (gradient-noise covariance) is the lever, and it works |
| F | The handoff damage lives in the FC block |
| J | Stationary vs scale-finding weight sections |
| K | On a young net the handoff is small, and grows with maturity |
| I | Framing: online play as hypothesis generation |
| **Search, labels and games** | |
| G | The bootstrap has headroom only while search(E) > E |
| N | The search budget's price list; `--depth` is a floor |
| M | Self-play sharpens its own games; depth sets where that stops |
| L | The recorded PV approximates the line that produced the score |
| Q | The root window collapse, and depth-D labels restored |
| **The offline phase** | |
| H | Row type matters; game diversity helped a mature chain, not a young one |
| U | The offline target sits in a broad λ optimum |
| O | Corpus statistics are not training hyperparameters |
| **Exploration and the 3M-game siblings** | |
| S | Exploration knobs: what `eval_noise` and PSQT hypotheses do |
| T | At 3M games, four knobs moved where gain arrives, not the anchor |

### The loop and its measurement

**A. The loop.**  Actor/learner self-play generates games with online TDLeaf
learning on, dumping a quiet-gated corpus; `--batch-train` then consolidates that
corpus into the promoted net.  The only figure of merit is the leg total against
a **foreign anchor** — family matches are non-transitive in both directions (§5).

**P. FIXED DEPTH IS AN EVAL INSTRUMENT, NOT A STRENGTH MEASUREMENT — and its
scale is depth-dependent (2026-09-25).**  `classic_eval` runs 1.8× the NNUE
net's nodes per second but needs 1.31× more nodes per ply; at a clock the two
nearly cancel, and trained nets also prune better than young ones (−12% median
nodes to depth 12 from `1e5g` to `7e6g`).  A fixed depth quotients out both, so
what survives is **how good the evaluation is at a fixed search**.  Measured on
`m260720` binaries [Detail D11]:

- **It moves the anchor**: the same pairing reads +16.7 ± 19 at 3+0.05 and
  +101.5 ± 26.8 at depth 12 (95%).  At depth 8 the `m260921` nets sit at −5 to
  −13 against `classic_eval`, where 3+0.05 reads −33 to −59 (§1 T).
- **Its scale grows with depth**: one family pair reads +12 (d6), +20 (d8), +34
  (d10), +38 (3+0.05) — **a d8 reading is ~0.5× the time-control reading** on
  that leg, and the ratio against `classic_eval` runs 0.25–1.00 along the chain.
  The two scales are not convertible: re-baseline once and never mix them.
- **It is calibrated at the null** (−0.35 ± 8.9 at d8 where 3+0.05 read
  +0.7 ± 15.3), **bit-reproducible and load-immune** (0/20 positions differ across
  repeat runs, also under 16-way load), and frees concurrency.  4000 games at d8
  carry the same confidence as 1000 at 3+0.05 in a fifth of the clock.
- **Anchor differencing under-reads family gains**: `m260720` 2.2M → 2.5M reads
  as nothing by differencing against `classic_eval` at d8, where the direct
  head-to-head says +20.3 ± 9.0.
- ⚠️ **`go nodes` is NOT reproducible in the NNUE build** (5–8 of 20 positions
  differ across repeat runs, up to 7× in node count under load; `classic_eval`
  is clean) — contrary to the comment at `uci.cpp:452`.  Rate at fixed depth,
  not fixed nodes, until that is understood.

`match.py --depth N` sets both engines and defaults to `-tc inf`;
`train.py --gauntlet-depth N` / `--epoch-depth N` rate a leg this way and add a
`tc-anchor` continuity match at 3+0.05, so the recorded ladder continues and a
change that improves eval-per-node while costing speed is still caught.  Every
sidecar carries a `rating_conditions` block.  The ~0.5× compression rests on one
real-signal leg.


### The online phase

**B. The cost of online play is a HANDOFF, not a per-step cost.**  Moving a mature
net out of an offline optimum into online play buys a one-time excursion,
historically −120 to −150 Elo at the pre-restart LR.  Started from an *online*
endpoint instead, the same 30,000 games cost nothing: **+11.1 ± 9.1 against
−129.8 ± 9.6 from the same binary, seed and protocol — +140.9 ± 13.2, 10.7σ**
[7.13.3].  Paid once per offline→online transition regardless of leg length,
which is why 30k / 100k / 500k legs read the same [7.10.1].  **Minimise handoffs,
not steps.**  Size on a young net: K.  A consequence seen again in §1 T: the
offline phase's gain is largely *recovery* of what the online phase disturbed, so
a gentler online phase leaves offline less to do.

**C. What surrounds the optimum is an Elo-flat noise ball.**  After the
excursion the net keeps diffusing at 77–87% of normal speed at no cost, and that
later motion is **orthogonal** to the damaging direction (cos −0.048) [7.13.4].
The ball is stationary — `E|Δw|² ∝ η·Σ/κ` — so damage is flat in time and
proportional to η: displacement ratio 0.520 at an η ratio of 0.25 (√η predicts
0.500), Elo damage ratio 0.273 against 0.520² = 0.270 [7.13.1].  **η sets a
radius, not a destination.**

**D. Direction quality, not displacement magnitude, is the axis.**  Cutting the
largest mover's displacement 1.65× bought nothing [7.12.5]; the undamaged
continuation diffuses for free [7.13.4]; η sets a radius [7.13.1]; and `bs64`
travelled **43% further than `bs8` and lost 46 fewer Elo** [7.15.5].  Any
proposal phrased as "move the weights less" is answering the wrong question.

**E. Σ — the gradient-noise covariance — is the lever, and it works.**  An online
batch of 8 games is ~1,200 sequential, autocorrelated positions; the offline
trainer's 512 come from a global shuffle.  Batch size at **matched Adam steps**:
−140.6 (B=8) → −111.8 (16) → **−76.6 (32)** → −94.3 (64) of handoff damage;
**8 → 32 is +64.0 ± 12.9, 5.0σ** [7.15.3].  Σ also survives by elimination: η,
per-section magnitude, four optimizer mechanisms and the target-difference
reading are all exhausted (§4).  The current recipe runs batch 50.

**F. The handoff damage lives in the FC block.**  Cosine between independent
runs' displacement from one offline start: fc2_b 0.836 … fc0_w 0.361, against
ft_w 0.206 and psqt_w 0.040 [7.13.2] — isotropic noise would give ≈0.0002.  It
is **not** the additive per-bucket constants: the undamaged arm moves them more
[7.13.5].

**J. There are two kinds of weight section, and only one obeys a magnitude LR
rule** [7.12.2].  **Stationary** — `fc0_w`, `ft_w`, `psqt_w`, 99.99% of
parameters — have their scale set by the init constants and barely move across
millions of games; `LR ≈ 0.001 × median|w|` is well defined for them.
**Scale-finding** — `fc2_w` and the five bias sections, 1,672 parameters — start
at or near zero and spend the run finding their scale, so their LR is a *growth
rate*.  Sizing a bias LR off a converged magnitude would freeze a fresh net.

**K. On a young net under the current recipe the handoff is small, and it grows
with maturity.**  Measured on `m260916` (R9): two 50,000-game TDLeaf ladders
(1000 Adam steps at batch 50) from an early and a late offline-trained state,
each point a 1000-game match against its own start:

| | weighted mean over 8 points | vs zero |
|---|---|---|
| early seed (1e6g) | **−0.70 ± 3.23** | 0.2σ |
| late seed (5e6g) | **−13.46 ± 3.23** | **4.2σ** |
| early − late | **+12.70 ± 4.57** | 2.8σ |

So the cost is masked early and visible later — at ~13 Elo, not the ~130 the
mature `m260720` chain paid at 4× the LR.  Neither ladder shows a
dip-then-recovery shape (χ² against a constant non-significant): a flat offset
from the moment the net leaves the offline optimum.  How much of the shrinkage
is the LR cut (η-scaling predicts ~−32) and how much is R8 is not separated.

**I. FRAMING, not mechanism — online play as hypothesis generation.**  Batch-Adam
steps revalue features on a handful of games; that changes how the engine plays,
so the next games probe wherever the weights moved, and the offline pass
adjudicates with a global shuffle [6.14.1].  Kept as a useful interpretation:
it is the only account of why a *frozen* generator's corpus consolidates to
nothing [6.2] while a learning generator's does not.  It is not evidence for any
particular answer, and the exploration work of §1 S was an attempt to supply
hypotheses deliberately rather than through learning noise.

### Search, labels and games

**G. The bootstrap `E ← search_d(E)` has headroom only while search(E) > E.**
d6 saturated around 2e6 games on the old chain; d8 reopened it [4.6] and had
saturated by ~5e6 [7.3, 7.7.4].  d10 was tried once and regressed, confounded by
a 52% draw rate [Offline 1.6].  Saturation is what makes label engineering
fruitless: from one seed the offline pass converges to the same ceiling across
corpora differing in composition, coverage and labelling — a **2.7 Elo band over
four arms** [7.7.4].  Whether `m260921` at 3M is at this ceiling or in a slow
patch is the live question (§6 item 1).

**N. The search budget has a measured price list, and DEPTH IS A FLOOR.**
`--depth D --nodes N` means "search to at least D, then stop at the first
iteration past N nodes" (`selfplay.cpp:470`, `search.cpp:548`).  Depth must be
paid in every position; the node budget only extends past it, and only where the
position is cheap.  Measured with fixed-node self-matches [Detail D2] — ⚠️ a path §1 P has since
found not reproducible in the NNUE build, so treat the rungs as indicative:

| config | clock | Elo over `d6/800` | Elo per unit of ADDED clock |
|---|---|---|---|
| `d6/2000` | 1.10× | +191 | 1900 |
| `d8/2000` | 1.62× | +373 | 351 |
| `d8/4000` | 3.97× | +445…+459 | 31 |

The exchange rate collapses ~60× along the ladder, and `d8/2000` is the knee.
The budget is a **label-quality** knob, not only a strength knob: under G, the
margin between search and static eval is what the bootstrap feeds on.  A node
budget adds depth **in the endgame only** and costs no quiet rows.  `m260921`
runs plain `--depth 8 --nodes 0`.

**M. Self-play sharpens its own games, and DEPTH sets where that stops.**  Young
chains drift toward sharper play — draw rate, game length and quiet fraction all
fall — and the drift saturates by roughly 2–4M games at a level set by depth:
**d6 ~22–27% draws, d8 ~35–36%, d10 43%**.  `m260720` sat at 35.3–36.1% across
4.8M games of d8; `m260916` spent its whole life at d6 at 22–25%, outside
`TRAINING.md`'s 35–40% healthy band.  `m260921` confirms the level on a clean
chain: **25.2% at its last d6 leg, 35.0% at its first d8 leg**, and 35% since.
Depth is an optimum, not a monotone — d10's 43% draws came with a −33 leg — so
the draw rate is a gate in both directions.  Causation from depth to *Elo* is
not established.

**L. The PV the engine records is an APPROXIMATION of the line that produced the
score, and TDLeaf trains at that PV's leaf.**  The leaf position and its
accumulator are exact (`TDLEAF_CHECK_ACC` zero mismatches), but the root search
score is not that leaf's static eval in roughly half of records: a real
alpha-beta search can take its value from a node the triangular PV does not
name.  The repairable causes — the root fail-high stub and TT cutoffs at PV
nodes — were fixed (R8), and the leaf-match gate drops records whose leaf and
propagated score differ by more than 10 cp.  What remains is **symmetric** (leaf
higher 27.4% / lower 27.4%, mean +0.7 cp against sd 74): variance, not bias, and
maturity-invariant in scale.

**Q. The root aspiration window was collapsing, and fixing it restored depth-D
labels (2026-09-21).**  `--depth 8 --nodes 0` should give depth-8 labels, but
`m260916` recorded mean depth 6.86 with 55.8% of rows below the floor.  Cause:
after a root fail-high the search set `alpha = beta`; futility pruning then cut
the very line that had failed high, the re-search failed low, and the iteration
never resolved.  `PV_LAST_RESOLVED` then substituted the previous iteration's
PV, score and depth.  The fix, for learning play only: no alpha raise and no
beta lower at the root, `PV_LAST_RESOLVED` retired, and unresolved stubs skipped
rather than trained on.  Result: **88% of searches resolve, recorded depth 8.00
with 0.00% below floor, and +298.45 ± 12.36 Elo of generation strength at 1.45×
the clock**; the substitution alone had cost +282.63 ± 12.06.  Dropping Houdart's
fail-high depth reduction was tested and **hurts**.  Whether the window fix also
helps *competitive* search is untested (TODO S1).  This is regime R10.

### The offline phase

**H. Row type matters; game diversity helped a mature chain and not a young
one.**  At a fixed row budget **root rows beat leaf rows** — +35.6 ± 11.0 paired
on the mature chain [Offline 3.2], and the same sign on an R8 corpus with the
game set held fixed (`nleaf`, −3.8 paired / −18.4 anchor).  Adding all leaf rows
on top of root rows at 2.2× the dose does not help either (anchor −9.4 ± 10.0;
ordering root > root+leaf > leaf is monotone).  Among *residual* gates the 60 cp
gate was the right width: 60/120/200 are flat, removing it costs −27.9 ± 11.3
[Offline 4.3].  ⚠️ **Superseded by §1 V (2026-09-29)**: the residual is the wrong
axis — selecting rows by PV quietness beats every residual gate by ~25 Elo.
**Game diversity reverses with regime**: on the mature `m260720` chain a
four-corpus window was worth **+36 anchor / +45 paired** [Offline 2.4]; on the
young R9 chain the same manipulation reads −9.2 ± 8.8 paired / −12.1 ± 9.9 anchor.
Do not carry the +36/+45 into current work.  Full arms: [Detail D4, D6].

**U. The offline target sits in a broad λ optimum, and the slope above it is
steep.**  (Lettered M before 2026-09-27; P now belongs to the rating instrument.)  `--bt-td-lambda` sets the outcome's
weight as `λ^(N−ply)`.  λ 0.970–0.985 is flat within resolution; **0.9925 is
worse by −20.5 ± 9.1 paired and −20.1 ± 9.9 on the anchor** — two instruments
within 0.4 Elo, the most replicated number in this document — and it costs on
the composite corpus too.  More outcome weight is the wrong direction, and the
cp labels carry more usable signal than the default credits [Detail D5].

**V. QUIET MEANS NO TACTIC AT THE ROOT MOVE OR THE REPLY — not a small
residual (2026-09-29).**  The 60 cp gate keeps a root row when |search − static|
≤ 60, i.e. it conditions on the label's own residual.  With the PV now dumped on
every root row, the `m260921-4.5e6g` corpus was re-consolidated four ways from
the SAME post-online state, games, trainer seed and epoch-ladder protocol, and
rated at depth 8, 8000 games each against `classic_eval` on identical `--srand`
openings (±3.5 one sigma):

| arm | rows | ladder e1 / e2 | anchor (d8) |
|---|---|---|---|
| tdleaf (no offline) | — | — | +2.0 |
| G60 control (the leg's own run) | 34.9M | +25.8 / +32.1 | +8.3 |
| **P1**: no capture/check/promotion in PV plies 1–2, no residual gate | 34.9M of 41.6M | +101.8 / +89.1 | **+33.2** |
| **P2**: P1 and residual ≤ 200 | 34.9M of 37.8M | +65.4 / +75.9 | **+31.6** |
| P1 ∩ G60 | 24.1M (all) | +2.4 / +15.6 | +4.4 |
| G60 subsample (equal-dose control) | 24.1M | +8.3 / +19.8 | +5.8 |

**P1 beats the gate by +24.9 ± 5.0**; the offline phase goes from +6 to +31 over
the same net.  The gain is entirely in the rows the residual gate *rejected*
while quiet at k = 2 (24.6% of all rows): removing the gate's own loud rows is
neutral (P1 ∩ G60 vs its equal-dose control, −1.4 ± 5), and the >200 cp
residual rows are neutral too (P2 ≈ P1) though they move the net further.  This
also explains [Offline 4.3]: "no gate" admitted the valuable quiet rows *and* the
26% that are loud with a large residual, and those cost more than the others
gave.  Loss curves agree: the control's validation blend MSE moves −1.1% in an
epoch (its rows are where static already ≈ search) and its outcome MSE not at
all; P1 moves −16% and −1.8%.  `train.py --bt-quiet-pv 2` (P1) is the default
from here.  One leg, one seed, depth-8 scale only — the time-control size and a
chain replication are open (§6).

**W. A MATURE CHAIN'S WHOLE HISTORY, RE-LABELLED, IS WORTH MORE THAN ANOTHER
LEG (2026-10-07).**  At 6M games `m260929`'s depth-8 anchor was gaining ~+7 per
leg.  Its last ten legs' raw dumps (2e6g → 6e6g incl. the sibling 5e6g and the
soup leg 5e6gS; 5M distinct games) were consolidated in one offline run from
`6e6g-final`:

- **Rows:** every root row that trains under R11 — PV-quiet (k=2), `leaf_ok` (or,
  pre-`leaf_ok`, a paired leaf row), |cp| ≤ 1500 — 383,977,887 paired; 382,524,119
  after the |cp| cap on the new labels and dedup (964 dropped).
- **Labels:** each root's label replaced by its PV leaf's static on the START net
  (`--bt-rescore`, the retargeting of §4).  Staleness grew with age: mean
  |new − old| 24.5 cp on 6e6g up to 88.6 cp on 2e6g (30.6% moved > 100 cp), with
  no signed bias (±0.4 cp) on any leg.
- **Training:** standard trainer and LRs, 8 threads, one epoch at seed 1000
  (747k optimizer steps, ~5× a normal two-epoch leg), then a second epoch from
  its state at seed 1001.  3.1 h per epoch; 17.9 GB resident.

| net | d8 vs classic (8000 g, paired openings) | d8 head-to-head | 3+0.05 vs classic | held-out MSE blend / outcome |
|---|---|---|---|---|
| 6e6g-final (start) | +127.6 ± 3.7 | — | +141.0 ± 9.8 | 0.011545 / 0.090481 |
| window, epoch 1 | +167.0 ± 3.9 | +43.7 ± 4.5 vs start | — | 0.011012 / 0.089745 |
| **window, epoch 2** | **+187.9 ± 4.0** | +31.0 ± 4.4 vs epoch 1 | **+183.5 ± 10.8** | 0.010765 / 0.089244 |

**+60.3 at depth 8 and +42.5 ± 14.6 at the time control** — the instruments agree
within error, so this is strength, not a depth-8 artefact.  The in-epoch loss
(first-epoch rows are unseen when trained on) fell steadily through all 363M rows
and was still falling at the end; epoch 2 halved the Elo increment (+39 → +21)
with train MSE dipping just below validation — a third pass is probably near the
point of diminishing returns.

**What it does NOT separate**, all moved at once: (1) game diversity and dose —
10× the rows from 5M games, which §1 H found worth ~+36 on a mature chain;
(2) fresh labels — without re-scoring, the oldest legs' labels were 60–90 cp
stale; (3) optimization length — ~5× the steps per epoch.  §6 item 10.

Practicalities: `train.py --assemble-only` and a bounded-memory dedup were added
for this (the in-loop Python dedup set cost ~80 B/row and was OOM-killed at 26 GB);
the pairing/re-labelling ran as one-off scripts (`learn/w26/pair_leg.py`,
`rescore_all.sh`, `relabel.py`), not yet part of `train.py`.  15 trainer threads
were measured SLOWER than 8 (the serial clip-norm tail, 64–71% of batch time,
does not parallelise).

**O. CORPUS STATISTICS ARE NOT TRAINING HYPERPARAMETERS.**  Seven arms fitted K,
λ or the outcome weight to the corpus; **seven lost on the anchor** (−5.7 to
−33.5), across three parameters, both scale and shape, and both directions,
while a control moving *away* from the fit was fine (λ 0.970: head-to-head
+9.7 ± 5.9).  The losses are ordered by **how much outcome weight moved**.  The
path-dependent λ-return (`npA`) reproduces the default rather than beating it,
so `λ^(N−ply)` is already near-optimal — and discarding the game trajectory costs
~13 Elo, so the trajectory is information, not noise.  The fits themselves are
real facts about chess (K U-shaped in material, λ monotone) [Detail D3, D9]; the
inference from them to a hyperparameter is what fails, probably because K and λ
shape gradients rather than calibrate predictions.

### Exploration and the 3M-game siblings

**S. Exploration knobs: what `eval_noise` and PSQT hypotheses do (2026-09-20 →
09-26).**  Two ways of giving the actors a different view of chess than the
learner's, so the learner sees positions shaped by other understanding.  Both
keep labels clean: the learner never carries the perturbation, and
`--refresh-scores` re-derives every label from its own weights.

- **`eval_noise`** — a zero-mean cp offset hashed from the pawn structure, per
  actor.  Free below σ 10, −19 Elo at 20, −66 at 40.  **It displaces the games
  completely** (at σ 5 every game diverges from the clean one by ply 1) **but
  adds no learning signal**: the TD error that multiplies every gradient is flat
  to ±0.4% at σ 20–30, and what it does add is label jitter and a small bias
  toward the noisy player's values.  Non-directional by construction: each
  diverted game lands somewhere unrelated.  Closed as a signal source (§4); kept
  off by default as a diversity knob.  [Detail D7]
- **PSQT hypotheses** (`--psqt-noise FRAC --psqt-opponent`) — scale the
  *positional* part of each (piece type, bucket) PSQT group by 1+ε, material
  untouched: 48 coherent hypotheses of the form "be 50% more sure where knights
  belong in bucket 3", one draw per actor generation.  Directional, and
  expressible by the learner.  Four frozen designs measured the TD gradient
  along the 48 pattern directions [Detail D8]:
  - **Shared-field** play (both sides hold ε) only **enacts** the hypothesis:
    the gradient follows ε's sign for both signs (z ≈ +27).  When both players
    value a pattern at 1+ε it is worth that *in their games*.
  - **Against the clean net** the enactment vanishes, but one side then costs
    −68 / −88 Elo at FRAC 0.5.
  - **Antithetic** (+ε against −ε) keeps outcomes balanced, and the outcome
    separates hypotheses sharply (±3.4 Elo per 8k games).  The PSQT-projected
    gradient tracks hypothesis quality in no design — but PSQT carries only
    ~17% of quiet-move positional variance (83% is FC), so the consequences
    can land in gradients the instrument never looked at.  Only legs answer
    that: see T.

**T. At 3M games on `m260921`, four knobs changed where the gain arrives, not
the anchor (2026-09-24 → 27).**  Four seed-paired sibling legs from
`2.5e6g_final`, 500k d8 games each, identical openings; each rated at fixed
depth 8, 4,000 games per match, one seeded opening order (`--srand 20260925`)
against both `classic_eval` and the parent.  One-sigma errors ±5 per match, ±7
on differences:

| leg | online net: anchor Δ / vs parent | final: anchor Δ / vs parent | draw % |
|---|---|---|---|
| `3e6g` (default) | +1.0 / −7.8 | **+3.0 / +12.9** | 35.1 |
| `n25` (antithetic, 0.25) | 0.0 / +10.4 | **+2.2 / +8.4** | 29.3 |
| `n50` (antithetic, 0.5) | −8.4 / +0.9 | **−1.7 / +0.1** | 21.5 |
| `lr4` (online LR 4×) | −20.7 / −15.2 | **+5.9 / +19.7** | 34.9 |

(Anchor Δ is the change on `classic_eval` from the parent, which reads
−11.3 ± 5.1 at depth 8.)

- **No final net moves the anchor beyond ~1σ.**  All four gain against the
  parent; none carries that into the foreign anchor.
- **The knobs trade online against offline.**  `lr4` reproduces `m260720`'s
  signature: the online net loses 21 on the anchor (3.1σ below 3e6g's) and the
  offline phase recovers +26.6 / +34.9 (3.9σ / 5.4σ) — the largest offline
  effect of the four.  `n25` is the opposite: a gentler online phase (the back
  of the net moved ~30% less) that beats 3e6g's online net by **+18.2 ± 6.4**
  against the parent — replicated at 3+0.05 (+17 ± 11.4) — and an offline phase
  that adds nothing.  The finals end within noise of one another.
- **Final gain against the parent falls with PSQT noise**: +12.9, +8.4, +0.1 at
  0, 0.25, 0.5 (n50 − 3e6g = −12.8 ± 6.4, 2.0σ).  Three points: a trend to note,
  not a finding.
- **The weights moved differently even when the Elo did not.**  n25 and 3e6g
  share well under half their feature-transformer direction of change (cos 0.40);
  n25's online phase shrank the PSQT positional spread 2.4% while lr4's grew it
  5%.  Material is stable to ±6 cp in every leg.  No leg showed output-bias
  drift.

Two cautions before generalising.  §1 P found the same pattern on `m260720` —
anchor differencing reading nothing across a leg that gained +20 head-to-head —
so "no anchor movement" is partly the instrument.  And `m260720` was also slow at this point (+17.5 on the
anchor from 2.5M to 3M) and gained in bursts across legs, so one leg per knob is
thin evidence.
**R. The root window can be opened WHERE IT STUBS but not EVERYWHERE — the
narrow aspiration window is load-bearing, and the draw rate saw it first
(2026-09-27).**  Q left 1.7% of searches exiting on the sequential
fail-high/fail-low break with the PV a stub and the ply skipped.  Two ways to
close that were measured head to head, both learning-play only, both on
`m260925-1e5g_final` at d8, 300 self-play games for telemetry and a 2000-game
fixed-depth match for strength.

*Arm 1 — open the root window for every iteration* (`root_alpha/root_beta =
∓MATE` whenever `pv_learning_mode`).  It works on the stated metric: stubs
1.7% → 0, searches resolved 98.3% → 100%, PV shorter than depth 14.36% → 2.41%,
mean pv_len 8.68 → 9.21, and it is 19% FASTER (166 s vs 205 s / 300 games).
And it costs **−444.74 ± 26.25 Elo** (7.17%, n=2000).  That is the WARNING above
the root window in `search.cpp` cashing out — it predicted ~6:1 in self-play and
this is ~13:1.  The three couplings it names (futility pruning keyed on alpha,
the internal singular-ext guard `hscore>alpha`, the non-first-move re-search
guard `beta>alpha+1`) are all disabled or always-true at `alpha = -MATE`.

*Arm 2 — open it only where the iteration would otherwise stub*
(`PV_WIDEN_UNRESOLVED`, default 1): at the `} else break;` exit, if the
iteration did not resolve and the window is not already full, set ∓MATE and
re-search ONCE.  Stubs 1.7% → 0 and searches resolved → 100% exactly as in arm 1,
at **−6.08 ± 10.74 Elo** (49.12%, n=2000) and +0.3% clock.  The re-search fired on
5.68% / 5.13% of searches (two seeds) — more than the 1.7% stub rate, because it
also rescues intermediate iterations — and **0 were still unresolved after it**,
so one full-width pass always resolves.  ⚠️ Arm 2 buys **no measurable label-quality
improvement**: on a second seed, gate retention and bias sd both land on the
wrong side of baseline (97.90/97.20% against baseline 97.41/97.66%; sd 18.2/21.2
against 20.6/20.8), and the record count follows game length rather than the
recovered plies (+3.6% on one seed, −2.4% on the other).  What arm 2 delivers is
the structural fact — every recorded ply now comes from a resolved search — for
free.  (The single-seed version of this paragraph claimed arm 2 had the best
label quality of the three; a second seed retracted it.)

Two seeds (777 / 31337) where a metric proved seed-sensitive; arm 1 was measured
on seed 777 only, having already been settled by the match.

| metric | baseline | arm 1 (∓MATE always) | arm 2 (widen-if-unresolved) | arm 3 (arm 2, stopgap retired) |
|---|---|---|---|---|
| searches resolved | 98.3% / 98.3% | 100.0% | 100.0% / 100.0% | 100.0% / 100.0% |
| stub plies skipped | 635 / 631 (1.7%) | 0 | 0 | 0 |
| full-width re-searches | — | — | 5.68% / 5.13% | **100.1% / 98.2%** |
| …still unresolved after | — | — | 0 | 0 |
| leaf-match gate retained | 97.41 / 97.66% | 96.79% | 97.90 / 97.20% | **97.84 / 97.81%** |
| label bias sd (cp) | 20.6 / 20.8 | 19.6 | 18.2 / 21.2 | **12.5 / 16.5** |
| PV shorter than depth | 14.36% | **2.41%** | 13.78% | 14.32% |
| draw rate (300 games) | 38.3 / 38.3% | 18.0% | 30.7 / 39.7% | 31.7 / 34.7% |
| clock / 300 games | 204.9 s | **166.1 s** | 205.5 s | 196.2 s |
| **Elo vs baseline, d8, n=2000** | — | **−444.74 ± 26.25** | **−6.08 ± 10.74** | **−8.17 ± 11.97** |

⚠️ **Arm 1's PV-length win was a symptom, not a benefit.**  Its
shorter-than-depth 2.41% did not come from resolving iterations — arm 2 resolves
just as many and stays at 13.78%.  It came from reaching far fewer draw-by-rule
early returns at PV nodes (repetition 47,648 → 31,805, fifty-move 7,364 → 788,
against arm 2's 48,784 / 5,510): the wide window was steering into a different
and much weaker class of position.  A PV-length metric can improve because the
search got worse.  The short PVs that remain are draw-rule early returns, not an
aspiration problem, and closing them is a separate question.

**The draw rate called it before the Elo did.**  38.3% → 18.0% for arm 1 is 5.7σ
on 300 games and far below `TRAINING.md`'s healthy 35–40% band at d8; arm 2's
30.7% is 2.0σ, weak evidence, the 2000-game match then found no strength
difference, and a second seed read 39.7% — noise, as a 2σ result on 300 games
usually is.  This is the canary of §1 M and the online-stability rules working
exactly as specified, on a 3-minute run, ahead of a 2000-game match.  ⚠️ It is a
*detector*, not a measurement: use it to decide whether to spend the match, not
in place of it.

*Arm 3 — can Q's stopgap now be retired?*  `PV_NO_ALPHA_RAISE` /
`PV_NO_BETA_LOWER` are **not** dead machinery under arm 2: they control how OFTEN
the sequential break happens, where arm 2 handles it once it has.  Building arm 2
with both at 0 makes the full-width re-search fire on **100.1% / 98.2% of
searches** instead of ~5% — i.e. essentially every search now resolves its final
iteration at ∓MATE.  That is structurally close to arm 1, so the strength result
is the surprise: **−8.17 ± 11.97 Elo** (48.83%, n=2000), indistinguishable both
from baseline and from arm 2 (Δ = 2.1 ± 16.1).  Only the FINAL iteration runs
full width — the earlier ones still shape the TT and move ordering through the
narrow window — and that is apparently enough to keep arm 1's collapse away.
Arm 3 is also 4.5% faster (196.2 s vs 205.5 s: a collapsed re-search plus a
full-width one is cheaper than the wide re-search the stopgap produces), and it
is the **only arm whose label quality moves consistently** — bias sd 12.5/16.5
against baseline's 20.6/20.8 and gate retention 97.84/97.81% against
97.41/97.66%, same direction on both seeds, though the sd magnitude is unstable.

**Adopted 2026-09-27; `PV_NO_ALPHA_RAISE`/`PV_NO_BETA_LOWER` deleted** (D. Homan's
call, over a recommendation to gate it behind a learning leg first).  The
argument is mechanism, not the measurement: the collapse is what *creates* the
fail-low — futility pruning keys on alpha, so `alpha = beta` prunes the very line
that just failed high — and the stopgap only made that rarer.  Re-searching the
break at full width addresses it directly, which leaves the collapsed search as
a cheap probe and the full-width one as definitive.  The measurement cannot
settle a ±15 Elo question at n=2000, and a 5,000-game handoff test measures
handoff damage rather than generator strength, so no cheap experiment was going
to decide it; the theory does.  Verified after deletion: the shipped source
reproduces the flagged build's telemetry exactly on both seeds (bias sd 12.5 /
16.5, widen 100.12% / 98.17%, 0 unresolved).  What remains open is whether the
label-noise reduction shows up in a net — visible only in a real leg.  Regime
R10.


---

## 2. Evidence ledger

Grades: **ESTABLISHED** — replicated or ≥3σ, and survives the regime changes in §3.
**SUPPORTED** — single arm, or <3σ, or inside the ~26 Elo arm-to-arm variance.
**FRAMING** — fits every arm, never tested directly.  `±` is **one sigma** unless
marked (95%).  Rows are grouped as in §1.  Rows tagged R1–R6 were measured on a
mature net; assume they are untested on a young one unless the row says otherwise.

### The online phase

| claim | evidence | effect | regime | grade |
|---|---|---|---|---|
| The cost is a handoff, paid once per offline→online transition | 7.13.3 `onon` arm | +140.9 ± 13.2, 10.7σ | R6 | **ESTABLISHED** |
| Damage is flat across a 17× range of run length | 7.10.1 | −123.4 / −125.4 / −130.9 at 30k/100k/500k | R5 | **ESTABLISHED** |
| displacement ∝ √η, Elo ∝ displacement², damage ∝ η | 7.13.1, 7.10.2 | 0.520 vs 0.500; 0.273 vs 0.270 | R5 | **ESTABLISHED** |
| Direction quality is the axis, not displacement magnitude | 7.12.5, 7.13.1, 7.13.4, 7.15.5 | four independent | R5–R6 | **ESTABLISHED** |
| Σ reduces handoff damage (batch size at matched steps) | 7.15.3 | 8→32 = +64.0 ± 12.9, 5.0σ | R6 | **ESTABLISHED** |
| Batch peak at 32; 32–64 flat | 7.15.3 | 32→64 = −17.7, 1.4σ | R6 | SUPPORTED |
| Σ is the residual mechanism | 7.14.4 | by elimination only | R6 | **FRAMING** |
| Damage is localised to the FC block | 7.13.2 | cos 0.36–0.84 FC vs 0.21 ft_w, 0.04 psqt_w | R6 | SUPPORTED |
| Not the additive per-bucket eval constants | 7.13.5 | undamaged arm moves them *more* | R6 | SUPPORTED |
| On a young net the handoff is statistically absent | §1 K, `m260916` 1e6g ladder | −0.70 ± 3.23 over 8 points | R9 | **ESTABLISHED** |
| It reappears with maturity, at ~13 Elo not ~130 | §1 K, 5e6g ladder | −13.46 ± 3.23 (4.2σ); early−late +12.70 ± 4.57 | R9 | **ESTABLISHED** |
| The residual is a flat offset, not an excursion-and-return | §1 K | χ² vs constant p > 0.05 for both | R9 | SUPPORTED |
| Under R7+R8 the online phase is productive on every young leg | `m260916` decomposition | +84, +64, +50, +49, +41, +7, +10 | R9 | **ESTABLISHED** |
| …and its contribution falls leg by leg, crossing zero at 6M | [Detail D6] | −14.9 ± 3.7/leg (4.1σ); BayesElo +39, +29, +14, +16, −3 | R9 | **ESTABLISHED** |
| The 2→4 epoch switch does not explain the decline | [Detail D6] | timing and `picked_epoch` | R9 | **ESTABLISHED** |
| Bias sections grow monotonically on a mature chain | 7.12.2 | fc0_b ×15.9, fc2_b ×8.9 at 7M | R5 | SUPPORTED, unexplained |
| Online play as hypothesis generation | 6.14.1 | — | R3 | **FRAMING** (§1 I) |

### Nulls — measured and found not to matter

| claim | evidence | effect | regime | grade |
|---|---|---|---|---|
| The online loss is target-independent | 3.1 | three error formulas, all −50 to −95 | R1 | **ESTABLISHED** |
| Stale Adam moments are not the mechanism | 7.10.7, 7.11.7 | −129.3 ± 9.9 vs −123.4 ± 9.3 | R5 | **ESTABLISHED** |
| Per-section LR miscalibration is not the damage | 7.12.5 | χ² 6.68 / 7 dof after a 1.65× displacement cut | R5 | **ESTABLISHED** |
| The quiet gate is not the target difference between phases | 7.14.3 | χ² 1.90 / 2 dof | R6 | **ESTABLISHED** |
| Session warmup delays the equilibrium, not changes it | 7.11.1 | −131.3 vs −149.7 at 30k | R5 | SUPPORTED |
| The FT bias-correction defect is negligible | 7.11.6 | clip rate ~3e-7 | R5 | **ESTABLISHED** |
| Generator decay within a leg is not differential | 7.3.1 | `rearly` vs `rlate` 0.4 apart | R5 | **ESTABLISHED** |
| Book diversity is not the plateau | 4.4 | in-book −26 ± 14 vs holdout −29 ± 18 | R1 | **ESTABLISHED** |
| Depth-6 endgame labels are the *cleanest* in the corpus | 3.3 | bucket-0 MSE 0.011 vs 0.207 | R1 | **ESTABLISHED** |

### Search, labels and games

| claim | evidence | effect | regime | grade |
|---|---|---|---|---|
| `--depth D --nodes N` makes D a floor, not a ceiling | `selfplay.cpp:470`, `search.cpp:548` | code | — | **ESTABLISHED** |
| Budget Elo per unit clock collapses ~60× along the ladder | §1 N | 1900 → 351 → 31 | R9 | **ESTABLISHED** |
| A node budget adds depth in the endgame only, and costs no quiet rows | `m260720` 7e6 dump | depth 8.14 at 32 pieces → 14.10 at 3; rows/game −0.5% | R5 | **ESTABLISHED** |
| Self-play sharpens at d6 and is flat at d8 | actor logs / gen PGNs, three chains | d6 34.5→26.6%; d8 35.3→35.8% over 4.8M | R3–R10 | **ESTABLISHED** |
| Depth sets the draw level: d6 ~22–27%, d8 ~35%, d10 43% | three chains; `m260921` 25.2 → 35.0 at its switch | — | R3–R10 | **ESTABLISHED** |
| Depth is an optimum, not a monotone | `m260720` 5.5e6 | d10 = 43% draws and −33.1 | R4 | SUPPORTED |
| The recorded PV's leaf is exact; the root score is not its eval in ~half | `tdleaf-pv-telemetry` | `CHECK_ACC` 0 mismatches | R7 | **ESTABLISHED** |
| The residual mis-approximation is symmetric and maturity-invariant | same | hi 27.4% / lo 27.4%; sd 63–72 across 1e5→7e6 | R7 | **ESTABLISHED** |
| `PV_LAST_RESOLVED` overwrote score AND depth; labels fell below the floor | §1 Q | 55.8% below floor at d8/0; `m260720` 0.00% | R8 | **ESTABLISHED** |
| Collapsing the root window is the cause, and it is root-only | §1 Q | no alpha raise → 74.7% resolved; + no beta lower → 88.5% | R8 | **ESTABLISHED** |
| The fix is worth ~298 Elo of generation strength at 1.45× clock | 4000 games, fixed d8 | +298.45 ± 12.36; depth 6.83 → 8.00 | R10 | **ESTABLISHED** |
| Dropping Houdart's fail-high depth reduction hurts | five paired arms | worse resolution and speed | R8 | **ESTABLISHED** |
| Stub rows are worse than resolved rows, and the gate cannot filter them | 4000 games | gate-60 pass 39.4% vs 49.1% | R8 | **ESTABLISHED** |
| Whether the window fix helps competitive search | two matches vs a common opponent | 15.8 ± 17.3 — not measurable that way | R10 | **UNMEASURED** (TODO S1) |

### The offline phase

| claim | evidence | effect | regime | grade |
|---|---|---|---|---|
| Root rows beat leaf rows at a fixed row budget | Offline 3.2; `cons1` `nleaf` | +35.6 ± 11.0 paired; R8: −3.8 paired / −18.4 anchor | R4, R9 | **ESTABLISHED** |
| Leaf rows do not help as a supplement at 2.2× dose | `cons1` `nboth` | anchor −9.4 ± 10.0; monotone in leaf share | R9 | SUPPORTED |
| The 60 cp quiet gate is the right *residual* width; removing it hurts | Offline 4.3 | −27.9 ± 11.3, replicated | R4 | **ESTABLISHED** — but superseded by the next row |
| PV quietness (no tactic in plies 1–2, no residual gate) beats the 60 cp gate | §1 V | +24.9 ± 5.0 at d8, 8000 g, same state/games/budget | R11 | SUPPORTED (one leg) |
| The gain is the gate-rejected quiet rows; the gate's loud rows and >200 cp rows are neutral | §1 V | P1∩G60 vs G60s −1.4 ± 5; P2 vs P1 −1.6 ± 5 | R11 | SUPPORTED |
| Requiring the PV to be confirmed by its leaf (`leaf_ok`) is Elo-neutral | `m260929-3e6g-pvok` | −0.7 ± 5 at d8, 8000 g; 3.1% of P1 rows dropped, mostly short draw PVs | R11 | SUPPORTED — adopted for correctness |
| A whole-history window (10 legs, 382.5M rows, labels re-scored on the start net) beats another generation leg | §1 W | +60.3 at d8 (8000 g, paired), +42.5 ± 14.6 at 3+0.05, vs ~+7 per recent leg | R11 | SUPPORTED (one run; ingredients not separated) |
| A second epoch over that window still pays | §1 W | +31.0 ± 4.4 head-to-head; +20.9 ± 5.6 on the d8 anchor | R11 | SUPPORTED |
| Two sibling legs from one parent agree within error; their weight average is at least as good as the better one | `m260929-4.5e6g` / `-5e6g` / soup | siblings +88.9 / +97.0; soup +100.7 (d8, 8000 g) | R11 | SUPPORTED |
| The trainer does not scale past 8 threads | 3M-row benchmark | 8 thr 32.8k rows/s, 15 thr 25.8k (serial tail 64% → 71%) | — | **ESTABLISHED** |
| Game diversity is worth ~+40 at identical compute — on a mature chain | Offline 2.4 | +36 anchor / +45 paired | R4 | **ESTABLISHED** (R4 only) |
| …and is not helpful on the young chain | `cons1` base vs null | −9.2 ± 8.8 paired, −12.1 ± 9.9 anchor | R9 | SUPPORTED |
| Outcome weight above the default costs ~20 Elo | §1 U | −20.5 ± 9.1 paired, −20.1 ± 9.9 anchor | R9 | **ESTABLISHED** |
| λ is flat across 0.970–0.985 | §1 U | ~5 Elo spread; instruments disagree on order | R9 | SUPPORTED |
| Calibration-derived changes lose: 7 arms, 3 parameters | §1 O | anchor −5.7 to −33.5; none positive | R9 | **ESTABLISHED** |
| More outcome weight loses in proportion to how much is added | §1 O | `npB` −33.5; `nout` −20.1; `bout` −20.4 | R9 | **ESTABLISHED** |
| The game trajectory carries signal | §1 O | `npA` −5.7 vs `nwA` −19.0 at matched mean | R9 | SUPPORTED |
| K is U-shaped in material; λ is monotone | [Detail D9] | K 169@13–16 to 269@1–4; λ 0.966 to 0.998 | R9 | **ESTABLISHED** (as corpus facts) |
| Offline gain peaks at epoch 2 and decays after | `m260916` 5e6g ladder | e1 +16.3, e2 +49.3, e3 +36.6, e4 +26.8 | R9 | SUPPORTED |
| Validation MSE does not rank nets | 7.4; Offline 2.3, 3.3; `nshape` | ten online arms; twice wrong-signed | R4–R9 | **ESTABLISHED** |
| ΔMSE_out prices label information, not usable signal | Offline 4.4 | top bin +52.6% better, costs 28 Elo | R4 | **ESTABLISHED** |
| Retargeting root labels to a rescored PV leaf is worth nothing | 7.7.3 | −2.5 ± 21.0 (95%) | R5 | SUPPORTED |
| The offline pass converges to a ceiling flat against corpus composition | 7.7.4 | four arms in a 2.7 Elo band | R5 | SUPPORTED |

### Exploration and the siblings

| claim | evidence | effect | regime | grade |
|---|---|---|---|---|
| `eval_noise` displaces the games completely, free below σ 10 | [Detail D7] | 100% diverge by ply 1 at σ 5; 0 ± 10 Elo at σ 10 | R9 | **ESTABLISHED** |
| `eval_noise` does not move sharpness | 7 arms × 30k games | draw 22.20 → 21.50 over σ 0→40 | R9 | **ESTABLISHED** |
| `eval_noise` σ 20–30 adds no TD error to the trace | 3 arms × 8k, frozen | rms e −0.9% / −0.5% (±0.4%); slope bias −3σ / −6σ | R10 | **ESTABLISHED** |
| Shared-field PSQT hypotheses are enacted, not tested | 2 frozen arms × 8k, FRAC 0.5 | align z +27.5 / +26.2 | R10 | **ESTABLISHED** |
| One-sided PSQT hypotheses at FRAC 0.5 cost −68 / −88 Elo | 2 frozen arms | ±3.3 each | R10 | **ESTABLISHED** |
| The PSQT-projected gradient does not track hypothesis quality in any design | 4 two-sided frozen arms | clean +0.6 / −2.1; anti −7.6 / −6.8 regardless of winner | R10 | SUPPORTED (one subspace) |
| PSQT is ~17% of quiet-move positional variance, FC ~83% | `psqt_decomp.py`, 2.5e6g | uncorrelated (−0.001) | R10 | **ESTABLISHED** |
| No sibling leg at 3M moved the d8 anchor beyond ~1σ | §1 T | +3.0, +2.2, −1.7, +5.9 (±7) | R10 | SUPPORTED |
| 4× online LR reproduces `m260720`'s online-loss / offline-recovery signature | §1 T | online −21.6 ± 6.9 vs 3e6g; offline +26.6 / +34.9 | R10 | **ESTABLISHED** |
| …but its final is not distinguishable from 1× | §1 T | +3.0 ± 6.8 anchor, +6.9 ± 6.4 vs parent | R10 | SUPPORTED |
| Antithetic hypotheses at 0.25 give a gentler, better online phase | §1 T; d8 and 3+0.05 | +18.2 ± 6.4 vs 3e6g online net on the parent; +17 ± 11.4 at TC | R10 | SUPPORTED (one leg, two instruments) |
| Final-net gain against the parent falls with PSQT noise | §1 T | +12.9 / +8.4 / +0.1 at 0 / 0.25 / 0.5 | R10 | SUPPORTED (three points) |
| At equal depth the final nets sit at −5 to −13 vs `classic_eval`; most of the 3+0.05 deficit is speed | `learn/d8eval/` | d8 −5 to −32; 3+0.05 −33 to −59 | R10 | SUPPORTED |

### PV resolution, round 2 (2026-09-27)

| claim | evidence | effect | regime | grade |
|---|---|---|---|---|
| Opening the root window for EVERY iteration recovers the PV | §1 R arm 1 | stubs 1.7% → 0, resolved → 100%, 19% faster | R10 | **ESTABLISHED** |
| …and costs almost everything | §1 R arm 1 | −444.74 ± 26.25 Elo at d8, n=2000 | R10 | **ESTABLISHED** |
| Opening it ONLY where the iteration stubs is strength-neutral | §1 R arm 2 (`PV_WIDEN_UNRESOLVED`) | −6.08 ± 10.74 Elo at d8, n=2000; +0.3% clock | R10 | **ESTABLISHED** |
| One full-width re-search always resolves | §1 R arm 2 | fired on 5.68% of searches, 0 unresolved after | R10 | **ESTABLISHED** |
| Arm 2 buys no measurable label-quality gain | §1 R arm 2, two seeds | gate 97.90/97.20% vs baseline 97.41/97.66%; sd 18.2/21.2 vs 20.6/20.8 | R10 | **ESTABLISHED** (retracts a single-seed claim) |
| Q's stopgap is what keeps the sequential break rare | §1 R arm 3 | full-width re-searches 5.1-5.7% with it, 98-100% without | R10 | **ESTABLISHED** |
| Retiring the stopgap under arm 2 is strength-neutral | §1 R arm 3 | −8.17 ± 11.97 vs baseline; Δ vs arm 2 = 2.1 ± 16.1 | R10 | SUPPORTED (2000 games cannot exclude −15) — **adopted on mechanism, not on this number** |
| …and is the only arm whose label noise moves consistently | §1 R arm 3, two seeds | sd 12.5/16.5 vs 20.6/20.8; gate 97.84/97.81% vs 97.41/97.66% | R10 | SUPPORTED, magnitude unstable |
| Arm 1's PV-length gain was a symptom of weaker play | §1 R | shorter-than-depth 2.41% vs arm 2's 13.78% at equal resolution; repetition returns 47.6k → 31.8k | R10 | SUPPORTED |
| The draw rate detected arm 1's collapse on 300 games | §1 R | 38.3% → 18.0%, 5.7σ, before any match was run | R10 | **ESTABLISHED** |
| Whether arm 3's label-noise gain reaches the NET | — | needs a leg; telemetry and a 2000-game match cannot see it | R10 | **OPEN** (shipped anyway, on mechanism) |

### The rating instrument (2026-09-25)

| claim | evidence | effect | regime | grade |
|---|---|---|---|---|
| A fixed-DEPTH match is bit-reproducible and load-immune | §1 P | 0/20 positions differ, solo and under 16-way load | R9 | **ESTABLISHED** |
| `go nodes` is NOT reproducible in the NNUE build | §1 P | 5–8/20 differ; to 7× node count under load; `classic_eval` 0/20 | R9 | **ESTABLISHED** (mechanism open) |
| Fixed depth moves the `classic_eval` anchor by ~85 Elo | §1 P | +16.7 ± 19 at 3+0.05 vs +101.5 ± 26.8 at d12 (95%) | R9 | **ESTABLISHED** |
| `classic_eval` is 1.8× faster per node but needs 1.31× more nodes per ply | §1 P | 2.48 M vs 1.40 M nps; 132,952 vs 101,618 median to d12 | R9 | **ESTABLISHED** |
| Trained nets prune better — nodes-to-depth falls along the chain | §1 P | −12% median, −22% total, `1e5g` → `7e6g` | R9 | SUPPORTED (n=20 positions) |
| The fixed-depth reading is depth-dependent, ~0.5× of TC at d8 | §1 P | +12.3 (d6), +20.3 (d8), +34.3 (d10), +38.0 (3+0.05) | R9 | SUPPORTED (one leg) |
| Fixed depth is calibrated at the null | §1 P | −0.35 ± 8.9 at d8 vs +0.7 ± 15.3 at 3+0.05 (95%) | R9 | **ESTABLISHED** |
| The d8 chain ladder preserves ORDER but not scale | §1 P | monotone over 7 checkpoints; d8/TC ratio 0.50 → 0.25 → 1.00 | R9 | **ESTABLISHED** |
| d8 / 4000 games ≈ 3+0.05 / 1000 games in confidence, at 1/5 the clock | §1 P | SNR 2.25 vs 2.29 | R9 | **ESTABLISHED** |
| Two of eight 3+0.05 anchor matches in the sibling programme were ~2σ excursions d8 did not reproduce | §1 T, §5 | n25 final −59.3, n50 tdleaf −24.7 | R10 | SUPPORTED |

### The loop as a whole

| claim | evidence | effect | regime | grade |
|---|---|---|---|---|
| Freeze-generate → consolidate is a no-op | 4.3, 6.1/6.2 | d8 frozen leg +7 on 1.7× the games against +53 | R1, R3 | **ESTABLISHED** ¹ |
| A fresh d8 corpus was worth ~0 in the mature chain | 7.3 | `rnew` −26.2 ± 17.6 (95%) | R5 | SUPPORTED |
| The mature d8 loop reached equilibrium at ~+13 per 500k leg | 7.9.1, 7.9.2 | online −126, offline +139 | R5 | SUPPORTED |
| The actor/learner path is equal-or-better to the merge | 5.5 | +23 ± 17, bit-exactness gate | R3 | **ESTABLISHED** |
| The draw-rate canary detects pathology, not decay | 7.2 | 35.1% flat through a −151 leg | R5 | **ESTABLISHED** |
| A family head-to-head can win while the anchor does not | §5, `nboth`; §1 T | +17.9 ± 5.9 direct vs −9.4 anchor; siblings +8…+20 vs ~0 | R9, R10 | **ESTABLISHED** |
| The d6→d8 switch should follow the ANCHOR's yield, not the parent's | `m260921` 1.5e6g | parent read 166 anc/Mg while the anchor read 33 | R10 | SUPPORTED |

¹ 6.2's evidence was half Elo and half "validation MSE rose at every epoch"; 7.4
later disqualified val MSE as a ranking instrument, so the d8 half now rests on
the Elo alone.  The conclusion stands on thinner support than 6.2 claimed.

---

## 3. Regime boundaries

A result is only as portable as the regime it was measured in.  **Net maturity
is an axis orthogonal to all of these**: a saturated net is a different subject
from a growing one (§1 G, K), and R9/R10 are the first young chains since the
records began.

**R1 — the multi-writer merge (through 2026-07-17).**  13–16 concurrent learning
processes merging weight deltas against stale baselines — roughly W× the
effective LR near a fixed point.  All of Online Parts 1–4.  The swamping online
displacement is why Part 3 recommended retiring online learning, a conclusion
that did not survive R3.

**R2 — the FRC castle accumulator bug (fixed 2026-07-18).**  A phantom enemy
piece corrupted search evals and online gradients after certain castles; every
FRC game before the fix carries it.  Offline corpora were clean.  Treat pre-fix
online numbers as indicative only.

**R3 — actor/learner split, `--refresh-scores`, natural termination
(2026-07-18).**  One optimizer, sole `.tdleaf.bin` writer.  Two collapses bought
the two hard rules: play to natural termination, and keep TD targets on current
weights.  Online Part 6 lives here.

**R4 — the offline recipe (2026-09-02/03).**  `--corpus-window`, `--bt-rows
root`, wide dumps with a `gate` column so any narrower gate is an offline filter.
The whole Offline record sits here.

**R5 — the ± convention correction (2026-09-06).**  `pgn_score` reported a
one-sigma binomial error while fastchess reports a 95% pentanomial interval — a
factor of 1.96, mixed freely before this.  Earlier results are suspect as to
significance, not point estimate.

**R6 — `TDLEAF_ADAM_EPS` 1e-8 → 1e-12 (2026-09-12).**  At 1e-8 the optimizer was
sensitive to accumulated gradient scale, which is the whole mechanism by which
`--grad-norm` worked.  **Invalidates any arm whose intervention moved
accumulated gradient scale** — including batch size, which is why 6.16 and 7.15
disagree.

**R7 — the restart (2026-09-15).**  Batch **50**; **one LR set for both phases
at scale 1.0**, the constants rescaled to 0.25× so offline is unchanged and
**online drops 4×**; warmup 100 steps, live for the first time; gradient clip
2.0; `--corpus-window 1`.  **Every "online cost" figure in Online Parts 6–7 was
measured at 4× today's online LR.**  `--lr-scale 4` reproduces the old online
step exactly (§1 T).

**R8 — the PV repairs (2026-09-17).**  `PV_NO_TT_CUTOFF`, `PV_LAST_RESOLVED`
(since retired by R10) and the 10 cp leaf-match gate on the online trace and
leaf rows.  Label bias −24.9 → +0.2 cp, sd 186 → 80.  **Leaf-row results must
not be pooled across this boundary.**

**R9 — the `m260916` chain (2026-09-15 → 09-21).**  The first chain combining
R7 and R8, and the first young chain: d6 with an 800-node budget to 6M, one d8
leg at 7M.  Its handoff ladders are §1 K, its consolidation arms (`cons1`) are
§1 H/O/P, and its full leg record is [Detail D6].  ⚠️ **It ran with the root
window collapse live** (§1 Q): mean recorded depth fell from 6.13 to 5.88 over
its d6 legs as the below-floor fraction grew 13.6% → 33.1%, and its d8 leg
recorded 7.22 with 47.7% below floor.  Its late-chain decline in online yield therefore
**cannot be separated from that growing defect** (TODO D0) — do not quote it as
evidence of saturation.

**R10 — the root-window fix and the `m260921` chain (2026-09-21, current).**  No
alpha raise and no beta lower at the root during learning play,
`PV_LAST_RESOLVED` retired, stubs skipped (§1 Q): labels are at the requested
depth, 0.00% below floor.  `m260921` is a fresh chain on R7 + R8 + R10 with
`--depth 6/8 --nodes 0`, 500k-game legs from 1M, `--gauntlet-tdleaf` on every
leg.  At the time control (leg_summary.py; `anc/Mg` = anchor gain per million
games):

| leg | cum. games | depth | anchor | anc/Mg | draw % | quiet |
|---|---:|---|---:|---:|---:|---:|
| `1e5g` | 100k | 6 | −364.1 | — | 33.1 | 0.694 |
| `2e5g` | 200k | 6 | −259.9 | 1042 | 27.5 | 0.645 |
| `5e5g` | 500k | 6 | −165.4 | 315 | 26.9 | 0.622 |
| `1e6g` | 1M | 6 | −109.5 | 112 | 27.8 | 0.573 |
| `1.5e6g` | 1.5M | 6 | −92.8 | 33 | 25.2 | 0.546 |
| `2e6g` | 2M | 8 | −89.9 | 6 | 35.0 | 0.483 |
| `2.5e6g` | 2.5M | 8 | −58.6 | 63 | 35.0 | 0.454 |
| `3e6g` | 3M | 8 | −32.8 | 52 | 35.1 | 0.440 |
| `3.5e6g` | 3.5M | 8 | −33.8 | −2 | 35.1 | 0.432 |
| `4e6g` | 4M | 8 | −21.2 | 25 | 34.8 | 0.467 |
| `4.5e6g` | 4.5M | 8 | +4.2 | 51 | 35.0 | 0.465 |

The d6 → d8 switch was made at 1.5M, when the anchor's yield had fallen to 33 per
million (the parent match still read 166 — §5).  The first d8 leg was a
transition leg (+6); the second paid (+63).  The quiet fraction kept falling at
d8 until 4e6g, where `PV_WIDEN_UNRESOLVED` (§1 R) stopped ~10% of plies being
skipped as unresolved stubs — the step up is recording, not calmer games.  The
3e6g siblings are §1 T.  4e6g and 4.5e6g are the first two legs with every
search resolved; on@a was +20.9 on both, after 3.5e6g's −24.7 (on@a by leg from
2.5e6g: +15.1, +8.9, −24.7, +20.9, +20.9).  ⚠️ Before 2026-09-29 `leg_summary.py`
differenced each leg against the previous ROW, which for 3.5e6g was the `n50`
sibling: it printed anc/Mg 47 and on@a 0.0 there.  It now uses the recorded
`parent_tag`.

**R11 — the PV dump and PV-quiet offline rows (2026-09-28/29, current).**  Every
root row carries the walked PV (`pv` column, `.tdg` v3); root rows are dumped
ungated, the |cp| ≤ 1500 cap moved to assembly; offline rows are selected by
`train.py --bt-quiet-pv 2` (no capture/check/promotion in PV plies 1–2, no
residual gate) instead of the 60 cp residual gate (§1 V).  First measured on
`m260921-4.5e6g`'s corpus; a fresh chain starts on it.  Root rows also carry
`leaf_ok` from 2026-10-02 (required in PV mode), and an offline phase whose ladder
is all ≤ 0 keeps the pre-offline net (`train.py`, from 2026-09-29).

`m260929`, from `--init-nnue material` on R11 (d6 to 1.5M, then d8; depth-8
gauntlets with a 3+0.05 continuity match per leg):

| games | 3+0.05 vs classic | `m260921` same games | d8 vs classic (final) | note |
|---|---:|---:|---:|---|
| 100k | −300.0 | −364.1 | −192.5 | |
| 200k | −276.7 | −259.9 | −119.4 | offline rejected |
| 500k | −220.9 | −165.4 | −84.1 | offline rejected |
| 1M | −122.2 | −109.5 | −57.6 | |
| 1.5M | −101.8 | −92.8 | −28.7 | offline rejected |
| 2M | −57.9 | −89.9 | −3.6 | switch to d8 |
| 2.5M | −12.2 | −58.6 | +30.9 | |
| 3M | +14.3 | −32.8 | +33.1 | |
| 3.5M | +25.1 | −33.8 | +57.5 | |
| 4M | +48.6 | −21.2 | +72.2 | |
| 4.5M | +78.1 | +4.2 | +82.3 | sibling 5e6g +87.3 / +86.4; soup +100.7 (d8, 8000 g) |
| 5M | +89.5 | — | +108.1 | from the soup |
| 5.5M | +93.6 | — | +115.8 | |
| 6M | +141.0 | — | +122.3 | |
| window (§1 W) | **+183.5** | — | **+187.9** (8000 g) | 2 epochs over 2M→6M, re-labelled |

Draw rate 34.8 → 35.9% over the d8 legs; depth-8 search time per move fell
9.34 → 7.93 ms (§6 item 9).

---

## 4. The graveyard

Closed lines, each with its numbers and **what would reopen it**.  Almost all
were closed on a mature net: "bought nothing on a saturated net" is the honest
headline for most, and it is not the same claim as "does not work".

**Batch size (closed 6.16, reopened 7.15).**  *The template case.*  6.16 found a
peak at 8 and closed the line; 7.15 moved the peak to 32 by rating *handoff
damage* rather than leg total, matching **Adam steps** rather than games, and
running after R6.  **Reopens if** the peak is re-measured on *leg yield*:
6.16's batch-16 arm cut damage 4× and made the loop worse.

**Per-bucket gradient normalisation — `TDLEAF_STACK_NORM_ALPHA` (6.6–6.10).**
Worked at the weight level, bought nothing (+39.3 against +53.0).  **Reopens
if** re-read against the damage protocol (§6).

**Per-feature vote normalisation — `TDLEAF_FEATURE_RBAR` / `_DEDUP` (6.11–6.13).**
Lost −23, and still −12 at matched displacement.  The premise was wrong: `r`
counts feature **persistence**, and duration is evidence.  **Reopens if** a form
removes variance without removing persistence.

**Learning-target redesign — blend / hybrid / root (Part 2, closed 3.1).**
Three unrelated error formulas all lost 50–95 Elo from one seed: the loss lives
in the shared update machinery.  **Reopens:** never, on this evidence.

**Freeze-generate → consolidate (closed 6.1/6.2).**  A frozen generator labels
positions with evaluations the seed already reproduces.  `TDLEAF_FREEZE=1` stays
the right tool for controls.  **Reopens:** never as a productive loop.

**Endgame over-coherence (6.4–6.10).**  Real at the weight level, not costing
Elo.  An unexplained gap between a small harness and production measurements is
a standing reason to distrust small harnesses.

**Endgame staleness / phase-dependent depth (3.3).**  Contradicted: endgame
labels are the cleanest.  **Reopens:** never as stated.

**Book diversity (4.4).**  No book overfit.  **Reopens if** the opening
generator's *recipe* changes.

**`--opt-reset` (7.10.7, 7.11.7).**  Null for damage; its apparent final-net gain
was a corpus confound (−5.9 ± 8.4 when repeated).

**`--grad-norm` (7.10.4).**  An LR cut in disguise via the eps floor; a third of
it vanished at R6.  **Reopens if** eps is raised again.

**Per-section LR recalibration (7.11.5, 7.12).**  The best candidate section's
LR cut 3.57×, displacement fell as designed, ladder flat.  Produced §1 J, which
matters more than the null.

**Retargeting root labels to a rescored PV leaf (7.7).**  Null twice.  **Reopens
if** the rescored leaf comes from a shallow **re-search** rather than a static
eval.

**Widening the quiet gate (Offline 1 → 4).**  Removing the gate costs 27.9 ±
11.3.  ⚠️ Reopened and resolved differently (§1 V): the loss came from the loud
rows "no gate" also admitted; a PV-quiet filter admits the rest and gains ~25.

**The wide consolidation window (A1 +36/+45 on R4; `cons1` closed it on R9).**
The clearest reversal in the record; the staleness mechanism proposed for it also
failed.  **Reopens if** a diversity manipulation does not also age the labels —
sampling wide *within* one leg (§6).

**Leaf rows as a corpus, alone or as a supplement (Offline 3.2; `cons1`).**
Root > root+leaf > leaf, monotone.  **Reopens if** a mechanism specifically needs
the leaf's static eval — a leaf-targeted auxiliary loss, not more rows.

**Outcome-weighted targets and `--bt-rescore` (`cons1`).**  λ 0.9925 costs ~20
on two instruments; the screen for rescoring came back negative.  **Reopens if**
cp labels become demonstrably worse (much deeper search, much older corpora).

**Calibrating K, λ or the outcome weight from the corpus (`cons1`, 2026-09-20).**
Seven arms, seven losses (§1 O).  The flags live, off and byte-identical, on the
unmerged branch `k-by-material` [Detail D9].  ⚠️ The *online* half of
per-material λ (the eligibility trace) never ran and needs a generation leg.
**Reopens if** the regime changes gradient economics (deeper search, a different
loss or optimizer).

**`eval_noise` as a source of learning signal (2026-09-20 → 09-23).**
Displaces games completely and adds no TD signal (§1 S) [Detail D7].  Its first
implementation left the root search score noisy — fixed 2026-09-23; no
production leg ever ran with it.  **Kept, defaulted off**, as the only knob that
diversifies positions without varying openings.  **Reopens if** structural
coverage is shown to be worth something, or a σ is found where the trace error
rises while the slope bias stays small.

**Depth 10 (Offline 1.1).**  d8→d10 regressed −13.6 with 43–52% draws, on a
mature chain before R4/R6.  Not reopened by anything since; the draw-rate gate
(§1 M) is the first check if it is.

---

## 5. Measurement manual

Each of these cost something to learn.

### Choosing an instrument

**THE DECISION POLICY (D. Homan, 2026-09-20).**  A single opponent is never the
best basis for a decision; the instruments answer different questions:

| instrument | answers | cost / precision |
|---|---|---|
| direct head-to-head between siblings | "is A stronger than B?" | cheap, tight |
| common-opponent subtraction (A−C via B) | the same question, badly | errors add; unreadable under ~20 Elo |
| foreign anchor | "does this generalise past the family?" | the DECISION criterion |

Family matches reward what A learned about positions B misjudges, which need not
generalise — §1 T's four siblings gained +8 to +20 on their parent and ~0 on the
anchor.  `classic_eval` shares Leaf's search and differs in eval only, a real
limit on its independence.  **When the tests do not show a clear win, elegance,
simplicity and expected correctness decide**; complexity must be earned by a
clear win.

**Rate eval changes and sibling legs at FIXED DEPTH 8 (2026-09-25).**  It answers
"did the evaluation improve", not "is the engine stronger" (§1 P); keep a
3+0.05 `tc-anchor` match for the strength question.  For siblings, add a seeded
opening order.
`match.py -tc inf --depth1 8 --depth2 8 --srand <seed>`, 4,000 games per match
(~7 minutes at 15 concurrent), gives ±5 per match and removes nodes-per-second
from the comparison entirely.  Two of eight 3+0.05 anchor matches in the sibling
programme were ~2σ excursions (n25 final −59.3, n50 tdleaf −24.7) that depth 8
did not reproduce.  Use the time control for strength-as-played; use fixed depth
to compare evaluations.  Pairing openings did **not** tighten the differences
(paired ±6.9 against ~±7.0 independent) — at depth 8 the two nets' games diverge
too quickly for shared openings to correlate the results.

**An anchor Elo is only comparable at the same RATING BUDGET.**  `classic_eval`
and an NNUE net differ in nodes per second and nodes per ply, so the gap moves
with the budget: the `m260921` finals read −5 to −13 at depth 8 and −33 to −59 at
3+0.05, and `1+0.01` is different again.  Fixed depth is a third condition, not a
faster clock (§1 P).  Every `train.py` sidecar from 2026-09-25 carries a
`rating_conditions` block for this reason.

**Before blaming machine load, read the depth record.**  fastchess writes every
move's depth into the PGN; at a fixed clock, a node-rate drop shows up as lost
depth for the NNUE side relative to classic.  The n25 −59.3 had normal depths
throughout.  A 10% node-rate loss is ~0.15 ply.

**The d6→d8 handover follows the ANCHOR's yield, not the parent's.**  At
`m260921-1.5e6g` the parent match read 166 Elo per million games while the
anchor read 33.  `leg_summary.py`'s `anc/Mg` is the trigger; `m260720` switched
at 73.

**Arm-to-arm variance is ~26 Elo.**  Two identical 30k configurations with
different seeds read −149.7 and −123.4 [7.11.9].  Any single-arm comparison
quoted at ±10 is under-powered by roughly 2×.  `--seed` removes the
generation-seed part of it, not trajectory divergence.

**Two matches against a common opponent do not subtract.**  Play the arms
against each other.  Worked example (`cons1`, λ 0.970 vs 0.985): via the seed
+6.5 ± 8.8, via the anchor −5.0 ± 10.0, direct **+9.7 ± 5.9** — the subtractions
disagree in sign.  But a head-to-head resolves a contrast; it does not replace
the anchor on a design decision (`nboth` won +17.9 ± 5.9 direct and lost on the
anchor).

**Validation MSE never ranks nets, epochs or arms** — twice wrong-signed, and
`nshape` improved both validation metrics while losing 14.6 Elo.  A smoke test
for optimizer health only.  **`ΔMSE_out` prices label information, not usable
signal.**

**Never pool a decaying series**, and one early ladder point cannot distinguish
"less damage" from "same damage, reached later" [7.11.4, 7.12.5].

**Mind the ± convention** (R5): one sigma from `pgn_score` and the tables here;
95% from fastchess's own `Elo:` line.

### Scoring traps

- **Identify the probe engine by its exact name.**  `pgn_score` matches by
  SUBSTRING; a near-miss matches neither player and silently scores every game
  from Black's side — once −23.3 for a true +296.6.  An ad-hoc scorer that
  guessed the probe by *excluding* names made the same error on the parent's own
  match (−5.2 for −11.3).  Copy the name from the PGN's `[White]` header.
- `match.py` logs contain interim `Elo:` blocks; take the **last**, just before
  `Finished match` [7.12.4].
- `--bt-quiet-cp` is a corpus-**assembly** knob in `train.py`; always verify the
  assembled corpus [7.14.1].

### Cheap protocols

- **Handoff damage in ~50 minutes per arm**: 5,000 online games plus one
  1000-game rating against the start [7.11.8].
- **Weight-level validation minutes into a run**: `--ladder` / `--publish-stamped`
  bake a net every N games; diff against the start before spending gauntlet time.
- **Mid-leg health checks, used through §1 T**: the draw rate per actor
  generation against a seed-paired sibling on the same openings (adjust for the
  generation's ε size when hypotheses are on); and the per-stack output-bias
  change against the parent from a snapshot of the live `.tdleaf.bin` —
  outcome-imbalance drift would move all eight stacks one way.
- **Frozen TD-error arms** (`scripts/arms/eval_noise_tderr.sh`,
  `psqt_noise_tderr.sh`): run the real actor/learner pipeline with both sides
  frozen, dump every trainable record, and replay the learner's TD recursion
  offline — measures what a generation change does to the gradient before
  spending a leg on it.
- **`psqt_decomp.py`** splits a net's PSQT into usage-weighted material and
  positional parts on real positions; with several nets it tracks composition
  across a chain.
- **Match Adam steps, not games**, whenever a knob touches aggregation.
- **Fixed-budget matches are hardware-independent** — how §1 N was priced.  ⚠️
  Use fixed DEPTH: `go nodes` is not reproducible in the NNUE build (§1 P), so
  the transfer claim is unverified for nodes, §1 N's own ladder included.  A
  node-only match also drops the depth floor; pass both `--depth` and `--nodes`
  if nodes are needed at all.
- **Decompose every leg** with `--gauntlet-tdleaf`; recompute from the
  `_final.json` sidecars, not from notes.
- **Pre-commit the reading of an arm**, and state its validity checks separately.

### Traps that cost real time

- The learner loads state from the compiled-in `NNUE_TDLEAF_BIN`, not
  `--tdleaf-out` [7.11.9].
- Nothing that trains or rates executes from `run/`: `main_bk.dat` would feed it
  book moves.  Build training binaries in `learn/`.
- A handoff ladder must carry its `.nnue` into the scratch directory — a missing
  net falls back to classical eval **silently**.
- Verify a setting from the **learner's own startup banner**, not the build log.
- Health canaries: the draw rate at the level its depth sets (§1 M; detects
  pathology, not decay), mean game length, and per-section bias movement.

---

## 6. Open lines, ranked

Ranked by what they would tell us per unit of compute, as of 2026-09-27.
`TODO.md` carries the checklist.

**1. Is 3M a slow patch or saturation? (running.)**  A default d8 leg from
`3e6g` is in progress.  Rate it at depth 8 against `3e6g_final`, `classic_eval`
and the 2.5e6g parent with `--srand 20260925`, so it joins the §1 T table.  If
the anchor resumes gaining, the flat 3e6g was a slow leg of the kind `m260720`
also had; if it stays flat, §1 G's saturation is the reading, and the budget
question (item 3) comes forward.

**2. A second 4× online-LR leg.**  One `lr4` leg reproduced `m260720`'s
signature and gave the best final of the siblings, by only ~1σ.  `m260720`'s
gains came in bursts across several legs, so a single leg is thin evidence
against the LR hypothesis.  A second `lr4` leg continuing from `lr4_final`,
rated like item 1, would show whether the larger online walk compounds.

**3. Search budget beyond plain d8.**  §1 N prices `d8/2000` at nearly the cost
of plain d8, since 2000 nodes binds only in cheap positions, and it extends
endgame depth for free.  If item 1 reads saturation, the next depth-margin step
is here — watching the draw-rate gate, since d10's 43% draws regressed.

**4. Exploration, next form.**  PSQT hypotheses are paused, not closed: antithetic
play at 0.25 produced a measurably better online phase, but no knob so far has
moved the anchor.  PSQT carries ~17% of positional variance; perturbing the FC
layers would reach the rest at the cost of harder sizing.  Revisit after items
1–2 say whether the chain is still learning at all.

**5. Does the root-window fix help competitive search? (TODO S1.)**  Measured so
far only through a common-opponent subtraction, which §5 rules out.  An ungated
build at a real time control against the current release answers it directly.

**6. Sample wide within one leg.**  Every `cons1` diversity arm confounded more
games with older labels.  `sample_corpus.py --quota 17` over one leg's ~1M games
draws the same dose from 4× the games at the same label age — separating the two
cleanly and cheaply.

**7. ~~A quiet gate tighter than 60 cp.~~**  Superseded by §1 V: the residual
is the wrong axis.  What is open instead is **PV quietness at time control and on
a chain** — the +25 is one leg at depth 8 — and its parameters (k = 1 and 3;
P2's residual cap as the conservative variant).

**9. Does the offline phase make the net cheaper to search?**  `m260929-2e5g`'s
offline phase (PV-quiet rows) LOST at fixed depth — ladder ep1 −32.8 / ep2 −30.3
(±4.7), depth-8 anchor −15.7 ± 7.5 — yet the same ep2 net GAINED at 3+0.05:
continuity anchor −218.9 ± 11.6 vs the pre-offline net's −276.7 ± 12.7
(+57.8 ± 17.2), and a head-to-head stopped at 244 games read +37.2 ± 17.3.
Opposite signs, not compression.  The candidate is search efficiency: depth 8
quotients out nodes-to-depth (the eval also shapes pruning, so depth 8 is not a
pure eval measure either).  The chain keeps deciding on depth 8 — it generates at
fixed depth, so a net that is cheaper to search but evaluates worse gives it
nothing — and `train.py` rejected ep2 on that basis.  Open for later, because a
reliable offline effect on search cost would matter for the SHIPPED net: measure
nodes-to-depth (or time-to-depth) for the two 2e5g nets on a fixed position set,
and whether the sign split recurs on other rejected legs.

*Supporting evidence across legs (2026-10-04).*  Generation searches get cheaper
as the net matures.  From the generation PGNs' per-move times (wall clock, 14
actors, every 10th game): m260929's depth-8 legs ran 9.34 → 8.63 → 8.35 → 8.14 →
8.16 → 8.13 → 7.98 ms per move (2e6g → 5e6g, **−14.6%**), while games shortened
only 147.2 → 144.0 plies (−2.2%) and the draw rate stayed at 34.8–35.5%.  Time
per game fell 16%, almost all of it from a smaller depth-8 tree.  m260921 shows
the same signature (8.75 → 8.22 ms over its d8 legs, one noisy 9.78), and both
chains converge near 8.0–8.2 ms — a property of maturing nets, not of the R11
recipe.  It is invisible to the depth-8 instrument and would show at a time
control.  Caveats: wall clock under a constant actor load, not node counts.  The
clean measurement is nodes to depth 8 on a fixed position set per leg's final.

**10. Separate the window's ingredients (§1 W).**  The +60 at depth 8 moved three
things at once.  Paired offline arms from the same start (6e6g-final), each rated
at d8 8000 g on the same openings, would separate them: (a) the same window
WITHOUT re-labelling (stale labels, same rows and steps); (b) re-labelled, but
only the most recent 2–3 legs at the same step count (diversity vs freshness);
(c) the last leg alone at ~5× its usual steps (optimization length).  Worth doing
before the window becomes a routine step in the loop — and the pairing/re-label
scripts belong in `train.py` if it does.

**8. Attack Σ directly.**  Shuffle records across a pool of games before forming
a learner batch — the offline-style decorrelation, never tried, and the only way
to confirm Σ positively rather than by elimination.

**Backlog** — each worth doing once its blocker clears:
- *Rate the PV repairs (R8) in isolation* — superseded in urgency by R10, whose
  Elo is measured (§1 Q); matched Adam steps, two replicates per side if run.
- *Re-read alpha and rbar against the damage protocol* — both closed on leg
  total before R6; ~50 minutes each.
- *Actor refresh cadence* — blocked on a corpus-diversity observable with a
  repeat-run noise floor.
- *A1b: drop the stalest corpus from a wide window* — after item 6.
- *The root-vs-mix confound* [Offline 3.4] — root-only at 86M rows would settle
  whether leaf rows were harmful or merely diluting.
- *m260916's late decline* (TODO D0) — confounded with the growing window
  defect (R9); do not rely on it either way.

---

## Reading map

| record | dates | regime | net maturity | what it settled |
|---|---|---|---|---|
| Online 1–4 | 07-14 → 07-17 | R1, R2 | ~5M | targets exonerated; bootstrap saturation; d8 reopens it; book diversity retired |
| Online 5 | 07-17/18 | R2 → R3 | `d8t` series | actor/learner split; the FRC castle bug; the two stability rules |
| Online 6 | 08-13/16 | R3 | 2.2–3.0M | frozen generation closed at d8; alpha, rbar; online-as-hypothesis-generator |
| Online 7.1–7.12 | 09-05 → 09-13 | R4, R5 | 5.5–7.0M | corpus worth zero; val MSE ranks nothing; five optimizer nulls; the noise ball |
| Online 7.13–7.15 | 09-13/15 | R6 | 7.0M | **the handoff finding**; Σ works (8→32 = +64) |
| Offline 1–4 | 09-02/03 | R4 | 5.5M | root rows; the 60 cp gate; game diversity worth ~+40 on a mature chain |
| Detail D6 (`m260916`, `cons1`) | 09-15 → 09-21 | R9 | young → 7M | handoff scales with maturity; calibration closed; diversity reverses |
| §1 Q (PV resolution) | 09-21 | R8 → R10 | 7M | the root window collapse; +298 of generation strength |
| Detail D11 (rating instrument) | 09-25 | R9 | `m260720` ladder | fixed depth measures eval; `go nodes` not reproducible |
| Detail D7–D8 (exploration) | 09-20 → 09-24 | R9, R10 | 2.5M, 7M | `eval_noise` adds no signal; PSQT hypotheses enact, not test |
| §1 T (siblings) | 09-24 → 09-27 | R10 | 2.5 → 3M | four knobs, no anchor movement at d8 |

The online and offline chronological records never measured a young net; the
last four rows are the first that did.
