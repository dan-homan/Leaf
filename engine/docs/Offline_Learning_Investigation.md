# Offline Learning Investigation — why the hybrid loop plateaued

> Companion to `docs/Online_Learning_Investigation.md`, which covers the online
> (TDLeaf) half of the loop.  This document covers the **offline consolidation**
> half: `--batch-train` and the corpora it consumes.
>
> Chain under study: `m260720` (actor/learner split), 5.5M cumulative games —
> 0–2M at depth 6, 2–5M at depth 8, the last 500k at depth 10.

---

## Part 1 — The plateau, and where the signal went (2026-09-02)

### 1.1 The plateau is real, and depth 10 made it worse

Foreign-anchor Elo (vs `Leaf_vclassic_eval`, 1000 games at 3+0.05), read from the
per-iteration sidecar JSONs.  `online` is the post-generation net, `offline` the
promoted consolidated net:

| iteration | depth | online | offline | offline gain | iteration total |
|---|---|---|---|---|---|
| 2.5e6g | 8 | −67.9 | +16.7 | +84.6 | +77.4 |
| 3e6g   | 8 | +11.8 | +34.2 | +22.3 | +17.5 |
| 3.5e6g | 8 | +20.9 | +111.0 | +90.1 | +76.8 |
| 4e6g   | 8 | +47.9 | +112.9 | +65.0 | **+1.9** |
| 4.5e6g | 8 | +56.1 | +103.4 | +47.3 | **−9.5** |
| 5e6g   | 8 | +63.6 | +109.8 | +46.2 | **+6.4** |
| 5.5e6g | 10 | +43.0 | +96.2 | +53.2 | **−13.6** |

The loop stopped netting anything after 3.5e6g.  Each iteration still shows the
familiar shape — online loses ~40–65 Elo, offline puts back ~46–53 — but the two
now cancel exactly.  The depth-10 iteration was a regression, and it lost its
head-to-head against its own parent: `5.5e6g-final` vs `5e6g-final` was
**283 W / 378 L / 339 D**.

**Depth is no longer the lever.**  `Online_Learning_Investigation.md` 3.6 predicted
that raising search depth restores headroom by making `search(E) > E` again; that
worked at d6→d8 and has now failed at d8→d10.  This investigation asks what the
offline phase is actually extracting, and why more of it stopped helping.

### 1.2 Validation outcome-MSE says offline overfits after epoch 1

`--batch-train` reports two validation numbers per epoch (split by game).
`MSE(blend)` is against the training target, which is mostly the net's own eval —
a poor progress metric.  `MSE(outcome)` is against the game result and is the one
that measures generalising signal.  Trajectory across the chain (baseline → ep1 → ep2):

| iteration | depth | MSE(outcome) base → ep1 → ep2 |
|---|---|---|
| 1e5g | 6 | 0.098479 → 0.095605 → **0.094137** |
| 1e6g | 6 | 0.101541 → 0.099714 → **0.099530** |
| 2e6g | 6 | 0.101320 → 0.099647 → **0.099464** |
| 2.5e6g | 8 | 0.088605 → 0.087922 → **0.087853** |
| 3.5e6g | 8 | 0.086221 → 0.085423 → **0.085381** |
| 4.5e6g | 8 | 0.086784 → 0.085784 → **0.085739** |
| 5e6g | 8 | 0.086625 → **0.085656** → 0.085676 ↑ |
| 5.5e6g | 10 | 0.072770 → **0.071737** → 0.071778 ↑ |

Early in the chain epoch 2 still improved held-out outcome prediction.  From 5e6g
on it makes it **worse**: the second pass is memorising the training games'
outcomes.  (`MSE(blend)` has been rising at epoch 2 since 2.2e6g, but that number
also rises when the net simply moves away from the labels' generator, so it is not
by itself evidence of overfitting.  `MSE(outcome)` is.)

Note the levels are only comparable *within* a corpus: the 5.5e6g baseline is
much lower because that corpus is 52% draws (see 1.5), which shrinks outcome
variance.

The net explains a flat ~41% of outcome variance across the whole late chain
(1 − MSE(outcome)/(0.25·(1−drawrate))), and has not moved.

### 1.3 The measurement: `--bt-diag`

To ask "how much is left in this corpus for this net?" without training, the batch
trainer gained a read-only diagnostic pass (`--bt-diag`).  For every row it compares
three predictors of the game outcome:

- `d` — the net's own static eval, as a probability, `sigmoid(score/K)`
- `p_lab` — the corpus `cp` label as a probability (root rows: the generator's
  **search** score; leaf rows: the generator's **static** eval)
- `result` — the actual outcome

and reports `delta = MSE_out(net) − MSE_out(label)`.  **The bootstrap
`E ← search(E)` has headroom only while the label predicts the outcome better than
the net does.**  `delta ≤ 0` means distilling those labels cannot make the net a
better outcome predictor — that channel is saturated.  It also reports the
correlation between the label residual `(p_lab − d)` and the outcome residual
`(result − d)`: positive correlation means following the label moves the net in the
outcome-correct direction.

Run against the `5.5e6g` corpus (1-in-8 thinning, 11.9M rows) with the state that
generated it (`m260720-5e6g_final.tdleaf.bin`):

```sh
./Leaf_vdiag --batch-train corpus.tsv --bt-diag --bt-threads 16
```

### 1.4 Result: the corpus has almost nothing left to teach — except in the tail that gets filtered out

Aggregate:

| rows | share | dcp rms | ΔMSE_out | corr |
|---|---|---|---|---|
| ROOT (search labels) | 46.4% | 45.7 cp | +2.08% | +0.145 |
| LEAF (static labels) | 53.6% | 43.9 cp | **+0.41%** | +0.087 |

Leaf rows are self-distillation by construction — their `cp` label *is* a static
eval — and they behave that way: 54% of every offline pass buys +0.41%.  Root rows,
the real `E ← search(E)` channel, carry the label's whole 2% advantage.

The decisive breakdown is by **how far the label is from the net** (`|dcp|`):

ROOT rows:

| \|label − net\| | share | ΔMSE_out | corr |
|---|---|---|---|
| 0–10 cp | 20.9% | +0.04% | +0.021 |
| 10–20 | 19.0% | +0.34% | +0.058 |
| 20–30 | 15.4% | +0.80% | +0.089 |
| 30–40 | 12.1% | +1.57% | +0.125 |
| 40–50 | 9.3% | +2.55% | +0.160 |
| 50–60 | 7.0% | +3.66% | +0.191 |
| 60–70 | 5.0% | +4.51% | +0.213 |
| > 70 | 11.4% | **+8.42%** | **+0.293** |

Monotone, and steep: **the informative content of a search label is essentially
proportional to how much it disagrees with the net.**  The 40% of rows where
search and static agree within 20 cp are, as data, nearly worthless — they tell
the net what it already says.  All the signal lives in the disagreement tail.

Leaf rows show the same shape but flat until the very end (+0.00% through 50 cp,
+4.68% only in the `>70` bin) — consistent with those rows being the net's own
output plus accumulated online displacement.

### 1.5 …and the corpus is built by throwing that tail away

`tdleaf.cpp` gates every dumped row on agreement between the static eval and the
search score, `TDLEAF_DUMP_QUIET_CP`, **default 60 cp**:

```c
if (abs(r.root_static - r.score_root_stm) <= dump_quiet_cp) {   // root rows
if (abs(r.score_stm  - root_leaf_pov)    <= dump_quiet_cp) {    // leaf rows
```

Measured on the `5.5e6g` run: 500,000 games, 79.58M recorded plies, **44.1M root
rows survive — the gate discards 44.6% of root positions** (and 36% of leaf
positions).  It discards them ranked by exactly the quantity 1.4 shows the signal
is proportional to.

This is the central finding.  The filter's stated rationale is sound — "unresolved
tactics show up as static-vs-search disagreement", and a static evaluator cannot
learn a tactic — but it is a *proxy* for tacticality, and it is perfectly
correlated with informativeness.  At the current maturity it removes the
mis-evaluated quiet positions along with the tactical ones, and those are the only
positions with anything to say.

The `ΔMSE_out` column partly discriminates the two families: in the `>70` bin the
label is **8.4% better than the net at predicting the actual game outcome**, with
residual correlation +0.29.  That is not tactical noise — a label that is merely
seeing an unlearnable tactic would not be a systematically better predictor of how
the game ends.  Most of the discarded tail is real, learnable evaluation error.

(Caveat, stated honestly: "better outcome predictor" is necessary but not
sufficient for "learnable by a static net".  Some of that tail *is* tactics.  The
fix is therefore to widen the gate and add a *direct* tactical exclusion, not to
remove filtering altogether — see 1.7.)

> **⚠️ RETRACTED by Part 4 (2026-09-03).**  The caveat immediately above was the
> right one and this section did not weight it heavily enough.  The discarded tail
> does carry the information — Part 4.2 confirms it at far larger effect size —
> but training on it makes the net **28 Elo worse** against a foreign anchor
> (Part 4.3).  Those labels are good *because search resolved a tactic*, which is
> exactly why a static evaluator cannot absorb them.  **The claim in this section
> that "most of the discarded tail is real, learnable evaluation error" is wrong,
> and the 60 cp gate is vindicated.**  `ΔMSE_out` prices label information, an
> upper bound on usable signal — never read it as a training recommendation.
> See Part 4.4.

Note also that this measurement understates the truncation.  The gate is applied
against the **generating** net's static eval at dump time; the diagnostic compares
against the promoted seed, which differs by 500k games of online displacement.
That displacement is what puts any rows at all above 60 cp in the table above.  In
the gate's own coordinates the retained distribution is hard-truncated at 60.

### 1.6 The other channel — outcomes — is information-starved, and depth 10 starved it further

With the eval-bootstrap channel gated down to +2%, the remaining offline signal is
the outcome term.  Under the λ-return target
`p = w·result + (1−w)·sigmoid(cp/K)`, `w = λ_eff · td_λ^(N−ply)`, the trainer
reports mean `w` = 0.300 on this corpus — so ~30% of the target weight is outcome
and ~70% is the (now nearly empty) eval bootstrap.

Draw rate across the chain's corpora (whole-corpus, from the `result` column):

| corpus | depth | W | D | L | mean outcome weight |
|---|---|---|---|---|---|
| 2.2e6g | 8 | 0.302 | 0.396 | 0.301 | 0.320 |
| 3e6g | 8 | 0.300 | 0.400 | 0.300 | 0.317 |
| 4e6g | 8 | 0.295 | 0.411 | 0.295 | 0.317 |
| 5e6g | 8 | 0.294 | 0.413 | 0.292 | 0.316 |
| 5.5e6g | **10** | 0.242 | **0.519** | 0.239 | 0.300 |

Depth 10 pushed the draw rate from 41% to **52%**, well outside the 35–40% healthy
band `TRAINING.md` records for d8.  Each drawn game contributes essentially no
outcome information, so the depth-10 iteration bought sharper labels for a channel
that was already gated shut, while cutting the information in the only channel that
was still open by ~25%.  That is a sufficient explanation for the d10 regression on
its own.

Meanwhile the outcome channel is asked to carry the load with 500k games of
labels against 23M FT parameters — which is precisely why validation
`MSE(outcome)` now rises at epoch 2 (1.2).

### 1.7 What follows

The offline phase is not misconfigured in its optimizer, its LRs, or its target
algebra.  It is **data-starved on both channels at once**:

1. the eval-bootstrap channel is gated shut by `TDLEAF_DUMP_QUIET_CP = 60`,
   which discards 45% of root positions ranked by informativeness
   (**superseded — Part 4 shows that tail is unlearnable and the gate is
   correct**);
2. the outcome channel has one 500k-game corpus per pass and overfits in two epochs.

Both are addressable, and (2) is addressable *entirely offline on data already on
disk*: the `m260720` chain has **1.04 billion positions across 5.5M games**
archived in `<tag>_work/corpus.tsv.gz` (642M positions / 3.5M games from the d8+d10
era alone).  Every offline pass to date has used one iteration's ~90M.

Ranked experiment plan in `docs/TODO.md`.

---

## Methodology notes (Part 1)

- Elo figures are the sidecar `final_gauntlet` / `tdleaf_gauntlet` entries against
  `Leaf_vclassic_eval` (a foreign anchor), 1000 games at 3+0.05, ±11 Elo.
- `--bt-diag` was added to `src/nnue_batch_train.cpp` for this investigation.  It
  is a read-only pass: it loads the corpus and the paired state exactly as a
  training run does, evaluates every record once, prints the decomposition, and
  exits without touching a weight.  Binning is by ply-gap from game end, by piece
  count (NNUE material stack), and by `|label − net|` in 10 cp bins.
- The diagnostic binary was built as
  `perl comp.pl diag NNUE=1 TDLEAF=1 NNUE_NET=m260720.nnue` and run from a scratch
  directory holding `m260720.nnue` plus the state under test copied to
  `m260720.tdleaf.bin` — never from `run/` (the `run/` invariant).
- The corpus sample is a deterministic 1-in-8 line thinning of
  `m260720-5.5e6g_work/corpus.tsv.gz` preserving the `# tdleaf-corpus
  axis=game-ply` marker; `gid`/`endply` are untouched, so the result decay and the
  by-game validation split behave exactly as on the full corpus.
- Draw rates are whole-corpus tallies over the `result` column via a 1-in-211
  stratified `awk` pass (so they are row-weighted, i.e. length-biased toward long
  games; the direction and size of the d8→d10 change is far larger than that bias).
- The retained-fraction figure compares root rows in the corpus against
  `Σ endply` over the 500,000 distinct `gid`s in the same corpus — the engine
  records every ply under internal self-play, so `Σ endply` is the candidate count.

---

## Part 2 — A1: is the offline phase starved for *games*? (2026-09-02, running)

### 2.1 The question, and why the arms are row-matched

Part 1 left the outcome channel as the prime suspect: validation `MSE(outcome)`
now rises at epoch 2, i.e. the second pass memorises 500k games' outcomes.  If
that is the binding constraint, then **game diversity** — not corpus size, not
epochs, not LR — is what the offline phase is short of, and the fix costs no new
generation at all: the chain has 1.04B archived positions across 5.5M games.

The naive arm ("train on everything") confounds three things at once: more games,
more rows, and more optimizer steps at fixed LR.  So the arms are **row-matched**:
identical row count, identical epochs, identical steps, identical wall clock —
the *only* variable is how many distinct games those rows come from.

| arm | corpora | games | rows | rows/game |
|---|---|---|---|---|
| **ctl** | `5.5e6g` only | 500k | ~95.0M | ~190 |
| **a1m** | `3.5e6g` + `4e6g` + `4.5e6g` + `5e6g` + `5.5e6g` | ~2.5M | ~95.2M | ~38 |

Both start from the *same* seed — `m260720-5.5e6g_work/train/m260720.tdleaf.bin`,
the post-online / pre-consolidation state of the 5.5e6g iteration — and run
2 epochs at the production settings (`--bt-K 220 --bt-lr 0.25 --bt-batch 512
--bt-threads 8`).  `ctl` additionally reproduces the production 5.5e6g
consolidation, whose promoted net scored **+96.2 ± 11.4** vs `Leaf_vclassic_eval`;
that is the external validity check on the whole pipeline.

**Why these five corpora and not all thirteen.**  Corpus labels distil their
generator (`Online_Learning_Investigation.md` 3.6), so mixing in a corpus whose
generator was 130 Elo weaker would pull the net backwards through the 70% of the
target weight that sits on the eval bootstrap.  The d6 era and `2.2e6g`–`2.5e6g`
are excluded on that ground.

> **Correction (made after the arms ran).**  The selection above was justified at
> setup time by each iteration's *own* consolidated Elo (+111.0 / +112.9 / +103.4 /
> +109.8 / +96.2, spread ~17), but that is the wrong quantity: a corpus is labelled
> by its **generator**, which is the *previous* iteration's promoted net.  The
> generator Elos are:
>
> | corpus | generator | generator Elo vs classic |
> |---|---|---|
> | `3.5e6g` | `3e6g_final` | **+34.2** |
> | `4e6g` | `3.5e6g_final` | +111.0 |
> | `4.5e6g` | `4e6g_final` | +112.9 |
> | `5e6g` | `4.5e6g_final` | +103.4 |
> | `5.5e6g` | `5e6g_final` | +109.8 |
>
> The real spread is **78 Elo**, not 17, and one fifth of `a1m`'s data carries
> labels from a net 75 Elo weaker than the rest.  The arm won decisively anyway
> (2.4), which bounds how much that stale fifth can have cost — but dropping
> `3.5e6g` is now the obvious refinement (arm A1b in `TODO.md`).

### 2.2 Two corpus-construction issues found while building the arms

**Cross-corpus duplicate rate is negligible.**  Distinct FENs from the first 4M
rows of `4e6g` and `5e6g` overlap on 23,563 of ~3.75M — **0.63%**.  Games diverge
from the shared FRC book almost immediately because each iteration's generator has
different weights, so unioning corpora does not need the dedup pass that
frozen-pair generation requires.  No dedup was applied; the leakage into any
by-game validation split is far below the resolution of anything measured here.

**The trainer's `gid` handling does not survive a corpus union — and already
wastes GBs on a single corpus.**  Dump gids are `(pid & 0xFFF) << 20` plus a
counter (`tdleaf.cpp`), so they are sparse and run to ~4.3e9.  `bt_load_file`
does `gid_N.resize(r.gid + 1)` — a `uint16_t` per *gid value*, not per game:

| corpus | max gid | `gid_N` cost |
|---|---|---|
| `4e6g` | 3.60e9 | **7.2 GB** |
| `3.5e6g` | 3.00e9 | 6.0 GB |
| `5.5e6g` | 7.08e8 | 1.4 GB |

Production runs have been paying this all along.  Worse, `gid_base = gid_max + 1`
accumulates across a comma-separated file list, so five such corpora would reach
~1.6e10 and **overflow the `uint32_t` gid** outright.  Both arms therefore
renumber gids to compact sequential ids at corpus-build time (2.5M games → 5 MB),
which also makes the two arms' by-game validation splits directly comparable.
A fix in the loader (hash-map or sort-based compaction) is filed in `TODO.md`.

### 2.3 The epoch ladder said nothing — and that was not evidence of a null

Validation `MSE(outcome)`, per arm (levels are **not** comparable between arms: the
by-game split holds out different games, so each arm has its own scale):

| arm | baseline | epoch 1 | epoch 2 |
|---|---|---|---|
| `ctl` (500k games) | 0.072215 | 0.071243 (−1.35%) | 0.071246 **↑** |
| `a1m` (2.5M games) | 0.083285 | 0.082205 (−1.30%) | 0.082208 **↑** |

Epoch 1 buys ~1.3% in both arms; epoch 2 buys nothing in both, by an identical
+0.000003.  Taken at face value this looks like a clean null — 5× the games
reproduced the epoch-2 stall exactly, so whatever stops epoch 2 is not game
diversity.

**That reading was wrong, and it is worth recording why.**  The gauntlet (2.4)
shows `a1m` is ~40 Elo stronger than `ctl`, while `a1m`'s validation MSE *level* is
higher (0.082 vs 0.071) and its epoch curve is the same shape.  Validation MSE had
no power to see a 40 Elo difference here — it did not merely under-report it, it
ordered the arms backwards.  This is the strongest case yet for `TRAINING.md`'s
standing rule: **rate by gauntlet, never by validation-MSE level.**  The epoch-2
stall is real and reproducible in the loss, and it is simply not the thing that
governs strength.

### 2.4 Result: game diversity is worth ~40 Elo at identical compute

1000 games each vs the foreign anchor `Leaf_vclassic_eval`, 3+0.05, FRC openings,
two matches at a time on 32 cores:

| net | W/L/D | Elo vs `classic_eval` |
|---|---|---|
| **`a1m_ep1`** (2.5M games) | 624/222/150 | **+148.7 ± 12.0** |
| `a1m_ep2` | 618/229/153 | +142.7 ± 11.9 |
| `ctl_ep1` (500k games) | 588/274/138 | +112.9 ± 11.6 |
| `ctl_ep2` | 561/267/172 | +105.3 ± 11.5 |
| *production `m260720-5.5e6g-final` (ep2)* | *—* | *+96.2 ± 11.4* |

- **`a1m` − `ctl` = +35.8 ± 16.7 (2.1σ) at epoch 1, +37.4 ± 16.6 (2.3σ) at
  epoch 2** — two independent replications, same sign, same magnitude.
- **Pipeline validity:** `ctl_ep2` at +105.3 ± 11.5 reproduces the production
  5.5e6g consolidation (+96.2 ± 11.4) within noise (Δ 9.1 ± 16.2).  The two differ
  only in which 5% of games the gid renumbering holds out.

Paired head-to-head, 1000 games each, `a1m_ep1` as the reference engine — no anchor
noise:

| matchup | W/L/D | Elo |
|---|---|---|
| `a1m_ep1` vs `ctl_ep1` | 348/218/434 | **+45.4 ± 11.1** |
| `a1m_ep1` vs `m260720-5.5e6g-final` | 363/220/417 | **+50.0 ± 11.1** |

Four measurements, one direction.  Note the direct family matches read *higher*
than the anchor difference (+45.4 vs +35.8), not lower — the ~5× family-match
compression recorded in `Online_Learning_Investigation.md` 4.6 did not appear here.

**The finding.**  At identical row count, identical epochs, identical optimizer
steps and identical wall clock, drawing those rows from 2.5M games instead of 500k
is worth **+36 Elo against a foreign anchor and +45 head-to-head**.  Against the net
actually promoted for that iteration, the best `a1m` snapshot is **+50 Elo for zero
extra compute and zero new games** — the data was already on disk.

The offline phase was starved of *game diversity*, not of rows, epochs, or learning
rate.  It has been re-consolidating one 500k-game corpus per iteration while 1.04B
positions across 5.5M games sat archived beside it.

### 2.5 Two secondary findings

**Epoch 1 beat epoch 2 in both arms** — +112.9 vs +105.3 (`ctl`), +148.7 vs +142.7
(`a1m`).  Individually each gap is inside noise, but the sign replicates across two
independent arms and agrees with the validation curve.  Production promoted epoch 2
on the strength of a *family* epoch ladder that scored it 48.3 vs 56.4 for epoch 1 —
i.e. the family ladder **inverted** the ordering the foreign anchor gives.  The
existing guidance ("select the epoch by ladder, not by assuming the last epoch") is
sound, but the ladder opponent matters: a family opponent is not just compressed,
it can be wrong-signed.

**The corpus quiet gate remains unaddressed.**  Nothing in Part 2 touches Part 1's
central finding — the +2.08% / +0.41% label headroom and the 44.6% of root positions
discarded by `TDLEAF_DUMP_QUIET_CP = 60`.  A1 improved how the *existing* corpora are
used; A2/A3 attack what goes into them.  The two are independent and should compound.

### 2.6 Reproduction

```sh
# corpora (gids renumbered compactly — see 2.2)
#   ctl : m260720-5.5e6g corpus, unthinned              95,028,415 rows / 500k games
#   a1m : 3.5e6g+4e6g+4.5e6g+5e6g+5.5e6g, 1-in-2 then
#         Bresenham-thinned to the same row count       95,028,415 rows / 2.5M games

# both arms, from a dir holding m260720.nnue + the 5.5e6g pre-consolidation state
# copied to m260720.tdleaf.bin:
./Leaf_va1 --batch-train <corpus>.tsv --bt-out <arm>     --bt-epochs 2 --bt-K 220 --bt-lr 0.25 --bt-batch 512     --bt-threads 8 --bt-seed 1000 --bt-max 95028415

# rating binaries (plain NNUE=1 from the self-contained epoch snapshots), then
python3 match.py Leaf_va1x_<arm>_ep<N> Leaf_vclassic_eval -n 1000 -c 8 -tc 3+0.05     --openings training_openings.epd --fischer-random --pgn-out <out>.pgn
```

Artifacts live in `learn/offline_a1/` (`ctl/`, `a1m/`, gauntlet + head-to-head PGNs
and logs); the two corpus builders are `build_ctl.sh` / `build_union.sh` /
`build_a1m.sh` recorded in the same directory.

---

## Part 3 — Root vs leaf rows: the search label is the whole story (2026-09-03)

### 3.1 The arms

Part 1 measured the two row types' label quality directly: root rows (search-score
labels) predicted the game outcome **+2.08%** better than the net itself, leaf rows
(the generator's own static eval) only **+0.41%** — yet leaf rows are 54% of every
corpus and every offline pass.  Part 3 tests that prediction at the level that
matters.

Three arms, each **exactly 190,000,000 rows, one epoch**, from the same seed (the
5.5e6g post-online / pre-consolidation state) and the same hyperparameters as the
A1 arms (`--bt-K 220 --bt-lr 0.25 --bt-batch 512 --bt-threads 8 --bt-seed 1000`):

| arm | composition | games | rows/game |
|---|---|---|---|
| `root` | 100% `depth > 0` (search labels) | 2,499,993 | 76 |
| `leaf` | 100% `depth == 0` (static labels) | 2,500,000 | 76 |
| `mix` | natural 45.3% root / 54.7% leaf | 2,500,000 | 76 |

Sources: the same five corpora `a1m` used (`3.5e6g`…`5.5e6g`).  The 4-corpus default
window could not supply the root arm — exact counts are **207.3M root** and **249.8M
leaf** rows across five corpora, but only 166.3M root across four.

### 3.2 Result: root-only wins, by a lot

1000 games each vs `Leaf_vclassic_eval`, 3+0.05, FRC openings:

| arm | W/L/D | Elo vs `classic_eval` |
|---|---|---|
| **`root`** | 633/214/153 | **+155.1 ± 12.1** |
| `leaf` | 597/253/150 | +124.6 ± 11.7 |
| `mix` | 580/276/144 | +109.1 ± 11.5 |

Paired head-to-heads, 1000 games each, no anchor noise:

| matchup | W/L/D | Elo |
|---|---|---|
| `root` vs `leaf` | 362/260/378 | **+35.6 ± 11.0 (3.2σ)** |
| `root` vs `a1m_ep1` | 332/283/385 | +17.0 ± 11.0 (1.5σ) |
| `mix` vs `a1m_ep2` | 294/331/375 | −12.9 ± 11.0 (−1.2σ) |

**At a fixed row budget, training on root rows alone is worth ~+36 Elo over leaf
rows alone**, and beats the natural mix by +46 on the anchor (2.8σ).  Part 1's
label diagnostic predicted the direction before any of this ran, which is what
makes the result credible rather than merely 3σ.

`root` is also the strongest net in the chain to date: **+17.0 ± 11.0** over
`a1m_ep1`, the best A1 net — suggestive at 1.5σ, not settled.

The practical consequence is large and cheap: **54% of every offline pass has been
spent on rows that are worse than useless at the margin.**  `--bt-rows root` halves
the corpus for free, or buys 2× the games at the same cost.

### 3.3 Validation MSE was wrong-signed for the third time

| arm | val MSE(outcome) Δ | Elo |
|---|---|---|
| `leaf` | **−1.40%** (best) | +124.6 (worst) |
| `mix` | −1.32% | +109.1 |
| `root` | **−1.18%** (worst) | **+155.1** (best) |

The arm whose validation loss fell *most* played *worst*, by 30 Elo.  Together with
Part 2.3 (`a1m` had the higher MSE level and won by 40 Elo) and the epoch-2 story
(MSE flat while ep1 beat ep2 on the gauntlet), that is three independent occasions
where validation MSE either could not see the difference or ordered it backwards.

**Standing rule, now with three confirmations: validation MSE is a smoke test for
optimizer health (smooth = fine, oscillating = step-size trouble) and nothing
more.  It must never be used to select a net, an epoch, or an arm.**

### 3.4 What the mix arm does and does not show — and a retraction

On the anchor, `mix` (+109.1) sits *below both* pure arms, and 33.6 ± 16.6 (2.0σ)
below `a1m_ep2`, which used the same five corpora, the same seed and the same
371k optimizer steps, differing only in 76 rows/game seen once vs 38 seen twice.
That was written up mid-run as an anomaly needing explanation.

**The paired match retracts most of it.**  `mix` vs `a1m_ep2` measured directly is
**−12.9 ± 11.0 (1.2σ)** — unresolved, not a 2σ anomaly.  Most of the apparent gap
was anchor-match noise in `mix`'s single 1000-game gauntlet.  The lesson repeats
Part 2's: differences between two nets should be read from a paired match, not
from the difference of two independent anchor matches, which carries √2 the error
and evidently more than that in practice.

What survives:

- `root` > `mix` (+46.1, 2.8σ on the anchor) and `root` > `leaf` (3.2σ paired).
- `leaf` ≈ `mix` (−15.5 ± 16.4, 1.0σ) — not separated.
- The **density** question (76 rows/game once vs 38 twice, at matched steps) is
  **unresolved**, leaning slightly against more density.  It is not settled by
  this run and should not be quoted as settled.

**A confound that remains open.**  `mix` carries 34 root rows/game against `root`'s
76, so `root` > `mix` may be nothing more than "more root rows" — no interaction
between the row types required.  The discriminating arm is **root-only at 86M rows**,
matching the root-row count *inside* `mix`: landing near +155 would mean the leaf
rows in `mix` were actively harmful, near +109 that `mix` was merely root-starved.
Not yet run.

Note also that the `root` arm drew 190M of the 207.3M root rows available (92%), so
it is close to exhaustive — scaling root-only training further needs **more games**,
not more sampling of the ones we have.

### Methodology notes (Part 3)

- One decompression pass over the five archives emitted all three arm corpora, each
  with its own per-type Bresenham rate (38M rows per corpus per arm) so every game
  contributes rather than a prefix of the shards.  Composition verified on a 1-in-50
  stratified pass: leaf 100.0%/0.0%, root 0.0%/100.0%, mix 54.7%/45.3% against the
  corpus-natural 54.6/45.4.
- gids renumbered per (arm, corpus); all three arms cover all 2.5M games (the root
  arm 2,499,993 — seven games contributed no root row surviving the quiet gate).
- Cross-corpus dedup was **not** applied to these arms: the rate measured 0.63%
  (Part 2.2), it falls equally on all three, and a 190M-key set would have cost
  ~7 GB competing with the trainer's memory.
- Arms ran sequentially, not concurrently: each needs ~10.4 GB against ~21 GB
  available and 2 GB of swap.  Throughput 29.7–30.7k pos/s, ~1.7 h per arm.
- Before trusting the surprising `mix` result, all four nets were checked for
  distinct md5s and each rating binary confirmed to reference its own `.nnue`.

### 3.5 Open confound: the outcome/eval blend was tuned on the *mixture*

Raised by Daniel Homan when the Part 3 result landed, and it is the right
objection to make.

The target is `p = w·outcome + (1−w)·σ(cp/K)` with `w = λ_eff · td_λ^(N−ply)`.
Every constant in it — `K = 220` (MLE over 58M positions), `td_λ = 0.985`
(offline convergence testing), and the `λ` ceilings — was calibrated on corpora
containing **both** row types, and `--bt-leaf-lambda` (which exists precisely so
leaf rows can carry their own outcome weight) has always been left at parity with
the root ceiling.  So the blend is optimal for the mixture, and **possibly optimal
for neither pure type**.

The mechanism is easy to see and cuts differently for each type:

- **Leaf rows.**  `cp` *is* the generator's own static eval, so `σ(cp/K) ≈ d` and
  the error collapses to roughly `w·(outcome − d)`.  A leaf row is therefore
  already close to pure outcome regression at strength `w ≈ 0.30`, plus a weak
  anchor pulling the net back toward the generator.  Nothing about `w = 0.30` is
  obviously right for that.
- **Root rows.**  The `(1−w)` term carries genuine search-distillation signal, so
  its best `w` is set by a real bias/variance trade-off against the outcome term.

There is no reason one `w` optimises both, and Part 3 ran them at the same one.

**What this threatens, and what it does not.**  It does *not* threaten the
production conclusion: `root` beat the natural mix and beat `leaf` **at the blend
production actually uses**, so `--bt-rows root` is the right default today
regardless.  It *does* threaten the stronger reading — "leaf rows carry no
signal" — because a leaf-specific `λ` might lift the leaf arm.  Nobody should
quote Part 3 as proof that leaf positions are worthless *in principle*.

**Why the effect is probably modest** (Homan's judgement, and the mechanism agrees).
Because a leaf row's bootstrap term is near-self-consistent, raising the leaf
ceiling mostly *scales the leaf gradient* rather than redirecting it — closer to a
per-row-type learning-rate change than to a different target.  This chain's
history is unkind to pure magnitude knobs: stack-norm alpha (6.10), per-feature
vote normalisation (6.12–6.13) and batch size (6.15–6.16) were each rejected on
production A/B, with direction quality, not displacement magnitude, as the axis
(`Online_Learning_Investigation.md`).  The K/λ calibration also showed a broad flat
region near its optimum, so moderate mis-tuning should cost little.

**The test, if it is ever worth running:** a leaf arm at the same 190M budget with
`--bt-leaf-lambda` swept over a few values (and/or a leaf-specific `--bt-td-lambda`).
Worth doing only when deciding whether to keep *generating* leaf rows at all — see
the generation questions in `TODO.md` — not before.

---

## Part 4 — A2: the quiet gate is doing its job (2026-09-03)

### 4.1 The enabling change: the gate became re-cuttable

`TDLEAF_DUMP_QUIET_CP` was applied at dump time, so rows it rejected were never
written and the only way to ask "was the gate too tight?" was to regenerate — but
online learning stays on during generation, so two runs from one seed diverge and
the gate is confounded with different games.

Both dump files now carry an 8th **`gate`** column: the value the quietness test
compared `cp` against, in the same POV.  The condition is uniform across both
files — `|cp − gate| ≤ QUIET_CP` — where

- **root** rows: `gate` = root **static** eval, `cp` = root **search** score;
- **leaf** rows: `gate` = propagated root **search** score, `cp` = leaf **static** eval.

So one wide dump re-cuts to any narrower gate offline (`--bt-quiet-cp`), and
`--bt-diag` prices each width (`|cp − gate|` bins) before training anything.  The
dump default moved to `QUIET_CP = 1000` (effectively open) in both the engine and
`train.py --quiet-cp`; the gate is consulted **only** in the dump path, never in
the TD update, so widening costs the online phase nothing.

Measured on a fresh 100k-game d8 run, this recovers a lot of previously-destroyed
data — **1.76× the root rows** and 1.39× the leaf rows the 60 cp gate admitted:

| file | rows | /game | ≤60 | ≤120 | ≤200 |
|---|---|---|---|---|---|
| root | 13.80M | 138 | 56.7% | 77.9% | 88.6% |
| leaf | 13.95M | 139 | 72.1% | 86.2% | 93.0% |

### 4.2 The diagnostic says the discarded rows are the informative ones

`--bt-diag` on the wide root dump, against the net that generated it, binned by the
gate quantity:

| \|cp − gate\| | share | ΔMSE_out | corr |
|---|---|---|---|
| 0–20 | 23.8% | **−1.17%** | +0.01 |
| 20–40 | 18.9% | −0.57% | +0.05 |
| 40–60 | 13.4% | +0.91% | +0.12 |
| 60–80 | 9.5% | +3.32% | +0.19 |
| 80–100 | 6.9% | +6.93% | +0.26 |
| 100–120 | 5.1% | +10.96% | +0.33 |
| 120–140 | 3.8% | +15.71% | +0.40 |
| **> 140** | **18.5%** | **+52.63%** | **+0.73** |

Priced as whole corpora with `--bt-quiet-cp`: gate 60 → **−0.46%**, 120 → +1.29%,
200 → +3.50%, 400 → +6.84%, none → **+12.45%**.  The rows the production gate
*keeps* have labels that predict game outcomes no better than the net already
does; all the label information sits in the tail it discards.

### 4.3 …and the training arms say that information is unusable

Four arms, each **exactly 7,829,099 rows** (the 60 cp arm's ceiling) drawn from the
**same 100k games**, Bresenham-spread so every arm sees all 100k games, same seed
(the run's own post-online state), same hyperparameters, 1 epoch.  Only the
admitted row population differs — a perfectly paired experiment.

| arm | mean \|cp − gate\| | anchor Elo vs `classic_eval` | paired vs `g60` |
|---|---|---|---|
| **`g60`** (production) | 25.8 | **+103.7 ± 11.5** | — |
| `g120` | 42.1 | +100.7 ± 11.5 | −16.0 ± 11.0 |
| `g200` | 55.7 | +100.3 ± 11.4 | +4.2 ± 11.0 |
| **`gnone`** | 93.6 | **+67.5 ± 11.2** | +6.9 ± 11.0 |

**No gate width beats 60.**  `g120` and `g200` are flat on both measurements
(their two paired reads disagree in sign, −16.0 and +4.2, which is the noise floor
of two ±11 matches between near-identical nets).  Removing the gate is clearly
worse.

**The `gnone` anchor/family conflict, and its resolution.**  The anchor said
`gnone` was 36 Elo *worse*; the paired family match said +6.9 *better* — a 43 Elo
disagreement in sign.  Both matches were clean (no time losses, crashes or illegal
moves; comparable adjudication rates).  Rather than pick the convenient one, both
anchors were replicated with independent fastchess seeds:

| arm | run 1 | run 2 | **pooled (2000 games)** |
|---|---|---|---|
| `g60` | +103.7 ± 11.5 | +98.1 ± 11.4 | **+100.9 ± 8.1** |
| `gnone` | +67.5 ± 11.2 | +78.4 ± 11.3 | **+73.0 ± 7.9** |

**`gnone` − `g60` = −27.9 ± 11.3 (2.5σ) on the foreign anchor, versus +6.9 ± 11.0
in family.**  The anchor result replicates; the disagreement is genuine
non-transitivity, not noise.  Two nets separated by a 15k-step update from one
common seed share their blind spots, so the family match cannot see a difference
that a foreign evaluator punishes.  This is the sharpest illustration yet of the
chain's standing rule — **the foreign anchor is the figure of merit** — and a
warning that a family head-to-head can read *zero* on a real 28 Elo regression.

### 4.4 Retraction: Part 1 over-read its own diagnostic

Part 1.5 concluded that the gate "removes the mis-evaluated quiet positions along
with the tactical ones, and those are the only positions with anything to say",
and argued from `ΔMSE_out` that "most of the discarded tail is real, learnable
evaluation error."

**The second claim is wrong and the experiment says so.**  The discarded tail does
carry the information — 4.2 confirms that at much larger effect size than Part 1
could see — but training on it makes the net *worse*, by 28 Elo against a foreign
anchor.  Those are positions where the search score is high-quality precisely
*because search resolved a tactic*, and a static evaluator cannot represent what
search saw.  Feeding them in injects targets the network cannot fit, and it pays
for the attempt.

**The methodological lesson, which now attaches permanently to `--bt-diag`:
`ΔMSE_out` prices label *information*, which is an upper bound on usable signal,
not a substitute for it.**  A label can know more about how the game ends and still
be a worse thing to train on.  Part 1's caveat ("necessary but not sufficient for
learnable by a static net") was correct and should have been weighted more heavily
than the monotone table it accompanied.

**The gate at 60 cp is vindicated.**  It is not a legacy accident; it is doing the
job it was designed to do.

### 4.5 What is settled and what is not

Settled: widening the gate does not help, and removing it hurts.  The 60/120/200
band is flat, so there is no cheap Elo in this knob.

**Not settled — the scale.**  These arms are 7.8M rows and ~15k optimizer steps,
4% of the Part 3 arms.  The test had the power to resolve `gnone`'s 28 Elo but
would not resolve a 5–10 Elo difference between 60, 120 and 200.  A tighter gate
than 60 was not tested at all, and the diagnostic's negative `ΔMSE_out` for
`|cp − gate| < 40` hints the optimum could sit *below* 60.

The infrastructure now makes any of that a filter away: the gate is a training-time
hyperparameter, dumps are wide by default, and re-asking costs one `--bt-quiet-cp`
sweep on a corpus already on disk.  That is the durable result of A2 even though
its headline answer was "no".

### Methodology notes (Part 4)

- Generation: `train.py --tag a2gate --continue m260720-5.5e6g --games 100000
  --depth 8 --concurrency 17 --recompile --skip-train`.  `--recompile` is
  **mandatory** — `train.py` reuses an existing `Leaf_vtrain_hl_a` when the baked
  net matches, and a stale binary would have dumped no `gate` column.
- Arms built by one pass over the root dump per arm, filtering on `|cp − gate|`
  then Bresenham-sampling to the common 7,829,099-row budget; verified each arm's
  max `|cp − gate|` equals its nominal gate and each covers all 100,000 games.
- Seed for all four arms was the a2gate post-online state, md5-verified identical
  across arms.
- Anchor matches are 1000 games at 3+0.05 with FRC openings; the `g60`/`gnone`
  replicates used fresh fastchess seeds (`match.py` fixes none), so the pooled
  figures are 2000 independent games per arm.


---

## Chain head: `m260720-5.5e6gR` (2026-09-03)

The Part 3 `root` arm is the strongest net the chain has produced and was promoted
to the head so `--continue` starts from it:

| net | Elo vs `Leaf_vclassic_eval` |
|---|---|
| **`m260720-5.5e6gR`** (new head) | **+155.1 ± 12.1** |
| `m260720-5.5e6g` (previous head) | +96.2 ± 11.4 |
| `a1m_ep1` (Part 2 best) | +148.7 ± 12.0 |

**+58.9 Elo over the superseded head, from games already on disk** — no new
generation.  It beat `a1m_ep1` by +17.0 ± 11.0 head-to-head.

Artifacts: `m260720-5.5e6gR_final.{nnue,tdleaf.bin,json}` plus the rating binary
`Leaf_vm260720-5.5e6gR-final`.  The `.tdleaf.bin` pairing hash (`0x0A3B39CB`)
matches the previous head, i.e. it pairs with the base `m260720.nnue` as every
consolidated state in this chain does.

Two sidecar details worth knowing, because both are easy to get wrong:

- **`parent_tag` is `m260720-5.5e6g`, not `m260720-5e6g`.**  This net is an
  *alternative consolidation of the same 5.5e6g online phase* (it was seeded from
  that phase's post-online state), so `5.5e6g_final` is a sibling rather than an
  ancestor.  Naming the grandparent instead would have been defensible
  genealogically but breaks the corpus window: `chain_corpora` walks `parent_tag`,
  so it would silently **skip `5.5e6g`'s corpus** — the freshest and least stale
  one — while still reaching back to the stale `3.5e6g`.  With the parent set
  correctly, `--corpus-window 4` resolves to `5.5e6g / 5e6g / 4.5e6g / 4e6g`,
  generators +109.8 / +103.4 / +112.9 / +111.0, all within 10 Elo.
- **`games_this_iter` is 0 and `depth` is null.**  This iteration generated no
  games; it re-consolidated existing ones, all already counted.  `cumulative_games`
  stays 5,500,000.

Because it was produced by the investigation's arms rather than a `train.py` run,
`epoch_ladder` is empty and `corpus_window.rows_used` records the arm's build
quotas.  The provenance is in the sidecar's `note` field.
