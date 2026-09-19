# Learning Investigation — what is known about the hybrid loop

**This is the distilled record.**  It replaces
`history/Online_Learning_Investigation.md` (4,854 lines, 2026-07-14 → 09-15) and
`history/Offline_Learning_Investigation.md` (796 lines, 2026-09-02 → 09-03) as the
thing to read.  Those two moved to `docs/history/` on 2026-09-15, unedited, as the
chronological record: every number here cites a section in them (`7.13.3`, `Offline
3.2`), and their methodology notes point at the artifacts on disk that back it.
Read them for provenance and reproduction, not for conclusions — both are
blow-by-blow, both contain confident intermediate readings that their own
follow-ups overturned (41 and 10 retraction markers respectively), and neither
front page survived contact with its own Part 7.

---

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

Six sections carry the load and they are meant to be used differently:

| section | what it is | use it when |
|---|---|---|
| 1. The model | what the loop is believed to be, in present tense | orienting |
| 2. Evidence ledger | one row per standing claim, graded, with the regime it was measured in | before quoting any number |
| 3. Regime boundaries | the eight changes — plus net maturity — that decide whether an old result still holds | before trusting an old result |
| 4. The graveyard | closed lines, each with **what would reopen it** | before proposing anything |
| 5. Measurement manual | hygiene rules, each with what it cost to learn | before designing an arm |
| 6. Open lines | ranked, with the arm and its pre-committed reading | deciding what to run |

`TODO.md` owns the checklist of work; this document owns the rationale.  Where
they disagree about status, `TODO.md` is the one to fix.

---

## 1. The model

Twelve statements.  Each is tagged with its strongest evidence; grades and effect
sizes are in §2.  Statement I is explicitly an interpretation, not a mechanism.
**Most of it describes a mature net at the pre-restart LR** — see the scope box;
K is the statement about young nets under the current recipe, and it is now
measured rather than open.

**A. The loop.**  Actor/learner self-play generates games with online TDLeaf
learning on, dumping a quiet-gated corpus; `--batch-train` then consolidates that
corpus into the promoted net.  The only figure of merit is the iteration total
against a **foreign anchor** — family matches are non-transitive in both
directions (§5).

**B. The cost of online play is a HANDOFF, not a per-step cost.**  Moving a mature
net out of an offline optimum into online play buys a one-time excursion,
historically −120 to −150 Elo at the pre-restart online LR.  Started from an
*online* endpoint instead, the same 30,000 games cost nothing: **+11.1 ± 9.1
against −129.8 ± 9.6 from the same binary, seed and protocol — +140.9 ± 13.2,
10.7σ** [7.13.3].  The cost is paid once per offline→online transition regardless
of leg length, which is why 30k / 100k / 500k legs all read the same [7.10.1].
**Minimise handoffs, not steps.**  Maturity qualifier in K.

**C. What surrounds the optimum is an Elo-flat noise ball.**  After the excursion
the net keeps diffusing at 77–87% of normal in the weight sections and costs
nothing, and that later motion is **orthogonal** to the damaging direction
(cos −0.048) [7.13.4].  The ball is stationary — `E|Δw|² ∝ η·Σ/κ` — so damage is
flat in time and proportional to η, which is exactly what was measured.  Three
numbers close the loop [7.13.1]: displacement ratio 0.520 at an η ratio of 0.25
(√η predicts 0.500), Elo damage ratio 0.273, and 0.520² = 0.270.  **η sets a
radius, not a destination**; relaxation toward a fixed point is excluded by a
factor of two.

**D. Direction quality, not displacement magnitude, is the axis.**  Four
independent demonstrations: cutting the net's largest mover's displacement 1.65×
bought nothing [7.12.5]; the undamaged continuation diffuses at 77–87% for free
[7.13.4]; η sets a radius [7.13.1]; and `bs64` travelled **43% further than `bs8`
and lost 46 fewer Elo** [7.15.5].  Net displacement is also invariant to batch
size across a 4× range [6.16.1].  Any proposal phrased as "move the weights less"
is answering the wrong question.

**E. Σ — the gradient-noise covariance — is the lever, and it works.**  An online
batch is 8 games ≈ 1,200 sequential, highly autocorrelated positions; the offline
trainer's 512 come from a global shuffle across millions of games.  Batch size at
**matched Adam steps** is the intervention (gradients are summed and Adam
normalises the step, so B changes samples-per-step, not step size):
−140.6 (B=8) → −111.8 (16) → **−76.6 (32)** → −94.3 (64) of handoff damage.
**8 → 32 is +64.0 ± 12.9, 5.0σ** — the first knob in Parts 6–7 to survive a
controlled measurement [7.15.3].  Σ additionally survives *by elimination*: η,
per-section magnitude, four optimizer mechanisms and the target-difference reading
are all exhausted (§4).

**F. The damage lives in the FC block.**  Cosine between independent runs'
displacement from the same offline start: fc2_b 0.836 … fc0_w 0.361, against
ft_w 0.206 and psqt_w 0.040 [7.13.2].  Isotropic noise in 21.6M dimensions gives
cos ≈ 0.0002, so 0.473 mean requires a genuine systematic component.  It is **not**
the additive per-bucket eval constants — the *undamaged* arm shifts them more
(3.7 cp against 2.4 / 2.5) [7.13.5].  The systematic direction exists only from an
offline-trained start and is traversed once.

**G. The bootstrap `E ← search_d(E)` has headroom only while search(E) > E.**
d6 saturated around 2e6 games; d8 reopened it [4.6] and had saturated by ~5e6
[7.3, 7.7.4].  d10 was tried once and regressed, confounded by a 52% draw rate
[Offline 1.6].  Saturation is what makes label engineering fruitless: from one
seed the offline pass converges to ~+151 across corpora differing in composition,
coverage and labelling — a **2.7 Elo band over four arms** [7.7.4].

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

**I. FRAMING, not mechanism — online play as hypothesis generation.**  Batch-Adam
steps revalue features on the evidence of a handful of games, far too little to be
right; that immediately changes how the engine plays, so the next games probe
wherever the weights moved and the corpus accumulates positions that test the
hypothesis; the offline pass then adjudicates with a global shuffle [6.14.1].
**Retained deliberately as a useful interpretation, not as an established
mechanism.**  Its original support has weakened in two places — the monotone
+7 / +39 / +53 ladder it was read off is now known to be ordered by a quantity that
includes handoff cost [7.13], and a learning generator's fresh corpus measured at
zero in the late chain [7.3].  What it still explains, and nothing else does as
well, is why a *frozen* generator's corpus consolidates to nothing by construction
[6.2] while a learning generator's does not.  It is the right frame for asking
what the online phase is worth; it is not evidence for any particular answer.

**J. There are two kinds of weight section, and only one obeys a magnitude LR
rule** [7.12.2].  **Stationary** — `fc0_w`, `ft_w`, `psqt_w`, 99.99% of parameters
— have their scale set by the init constants and do not move across 7M games; this
is where the noise ball lives and where `LR ≈ 0.001 × median|w|` is well defined.
**Scale-finding** — `fc2_w` and the five bias sections, 1,672 parameters — all
initialise at or near zero and spend the run finding their own scale, so the LR is
a *growth rate* and a fixed absolute LR is a self-annealing schedule.  Sizing a
bias LR off a converged magnitude would leave a fresh net unable to move it.
**L. The PV the engine records is an APPROXIMATION of the line that produced the
score, and TDLeaf trains at that PV's leaf.**  The leaf position and its
accumulator are provably exact (`TDLEAF_CHECK_ACC` reports zero mismatches; an
off-by-one probe matches the leaf 43% against one-ply-back 2%), but the root
search score is *not* that leaf's static eval in roughly half of records — a real
alpha-beta search with a TT, extensions, reductions and pruning can take its
value from a node the triangular-array PV does not name.  Two repairable causes
were found and fixed on `tdleaf-pv-telemetry`: the root **fail-high stub**, where
`pc[0]` is written explicitly as `{move, TT-guessed reply, NOMOVE}` when an
aspiration iteration never resolves (35.9% of searches), and **TT bound cutoffs
at PV nodes**.  What remains is irreducible and **symmetric** — leaf higher 27.4%
/ lower 27.4%, mean +0.7 cp against sd 74 — so it is variance, not bias, and
therefore a Σ contributor rather than a gradient-direction problem.  Crucially
the error *scale* is maturity-invariant (sd 63–72 across 1e5→7e6 games) even
though the exact-match spike grows 38%→55%, which is what makes a single
threshold implementable.

**K. On a young net under the current recipe the handoff is small, and what
remains scales with maturity.**  This was the open question of the scope box and
it is now measured on the `m260916` chain (R7+R8, d6/800 nodes).  Two 50,000-game
TDLeaf ladders — 1000 Adam steps at batch 50, so past the ~300-step equilibration
— run from an EARLY offline-trained state (`1e6g_final`) and a LATE one
(`5e6g_final`), each point a direct 1000-game match against its own starting net:

| | weighted mean over 8 points | vs zero | χ² vs a constant |
|---|---|---|---|
| early seed (1e6g) | **−0.70 ± 3.23** | 0.2σ | 13.5 / 7 dof, p > 0.05 |
| late seed (5e6g) | **−13.46 ± 3.23** | **4.2σ** | 8.1 / 7 dof, p > 0.05 |
| early − late | **+12.70 ± 4.57** | **2.8σ** | |

Three readings.  **The excursion is an order of magnitude smaller** than the
−123…−150 of Parts 7.10–7.13 — the worst single point anywhere is −29.  **At the
early seed it is statistically absent**, and at the late seed it is real but
small.  So "masked early, naked later" is the right description, with the caveat
that the late-seed cost is 13 Elo rather than the ~130 the mature `m260720`
chain paid.  And **neither ladder shows a dip-then-recovery shape**: χ² against a
constant is non-significant for both, so this is a flat offset from the moment
the net leaves the offline optimum, held through 1000 steps — an equilibrium in
the sense of C, at a tenth the radius, not a transient.

How much of the shrinkage is the 4× LR cut (damage ∝ η, 7.13.1, predicts ~−32
from −130) and how much is R8 is **not separated** — the restart moved both at
once.  The measured −13.5 is below even the η-scaling prediction, which is
suggestive but not attributable.

Caveats: 8 points at ±9 each resolve a 13 Elo offset but not a 10 Elo dip, so
"no excursion shape" is a statement about what this instrument can see.  And
7.11.9's ~26 Elo arm-to-arm variance applies to any single ladder point; only
the pooled means are tight.

---

## 2. Evidence ledger

Grades: **ESTABLISHED** — replicated or ≥3σ, and survives the regime changes in §3.
**SUPPORTED** — single arm, or <3σ, or inside the ~26 Elo arm-to-arm variance.
**FRAMING** — fits every arm, never tested directly.

Regimes are the short names from §3.  `±` convention: everything below is quoted as
**one sigma** unless marked (95%).

**Every row below was measured on a mature net** (≥2.2M cumulative games, most at
5–7M).  Where a claim is specifically known to depend on maturity it says so; where
it is silent, assume the claim is *untested* on a young net rather than established
for one.

### Mechanism of the online phase

| claim | evidence | effect | regime | grade |
|---|---|---|---|---|
| The cost is a handoff, paid once per offline→online transition | 7.13.3 `onon` arm | +140.9 ± 13.2, 10.7σ | R6, mature | **ESTABLISHED** |
| The handoff is only *apparent* after ~1M games of learning | D. Homan, from the chains; 6.1's decomposition (+18 / +7 / −2 / −1 at 100k–1M, −27 at 2.2M) | — | R1/R2 for the early rows | SUPPORTED, mechanism open ² |
| Damage is flat across a 17× range of run length | 7.10.1 | −123.4 / −125.4 / −130.9 at 30k/100k/500k | R5 | **ESTABLISHED** |
| displacement ∝ √η, Elo ∝ displacement², damage ∝ η | 7.13.1, 7.10.2 | 0.520 vs 0.500; 0.273 vs 0.270 | R5 | **ESTABLISHED** |
| Direction quality is the axis, not displacement magnitude | 7.12.5, 7.13.1, 7.13.4, 7.15.5 | four independent | R5–R6 | **ESTABLISHED** |
| Σ reduces handoff damage (batch size at matched steps) | 7.15.3 | 8→32 = +64.0 ± 12.9, 5.0σ | R6 | **ESTABLISHED** |
| Damage is localised to the FC block | 7.13.2 | cos 0.36–0.84 FC vs 0.21 ft_w, 0.04 psqt_w | R6 | SUPPORTED |
| Not the additive per-bucket eval constants | 7.13.5 | undamaged arm moves them *more* (3.7 vs 2.4/2.5 cp) | R6 | SUPPORTED |
| Batch peak at 32; 32–64 flat | 7.15.3 | 32→64 = −17.7, 1.4σ — right side unresolved | R6 | SUPPORTED |
| Σ is the residual mechanism | 7.14.4 | by elimination only | R6 | **FRAMING** |

### Nulls — things measured and found not to matter

| claim | evidence | effect | regime | grade |
|---|---|---|---|---|
| The online loss is target-independent | 3.1 | three different error formulas, all −50 to −95 | R1 | **ESTABLISHED** |
| Stale Adam moments are not the mechanism | 7.10.7, 7.11.7 | −129.3 ± 9.9 vs −123.4 ± 9.3; the apparent +36 on the final net retracted at −5.9 ± 8.4 | R5 | **ESTABLISHED** |
| Per-section LR miscalibration is not the damage | 7.12.5 | χ² 6.68 on 7 dof, p = 0.46, after a verified 1.65× displacement cut | R5 | **ESTABLISHED** |
| The quiet gate is not the target difference between the phases | 7.14.3 | χ² 1.90 on 2 dof, p = 0.39 | R6 | **ESTABLISHED** |
| Session warmup delays the equilibrium, it does not change it | 7.11.1 #3 | −131.3 ± 9.3 vs −149.7 ± 10.2 at 30k | R5 | SUPPORTED |
| The FT bias-correction defect is real but negligible | 7.11.6 | typical FT step 0.25–0.35, clip rate ~3e-7 | R5 | **ESTABLISHED** |
| Hash 16 vs 128 cannot carry a 151 Elo collapse | 7.5 | +8.86 ± 11.36 (95%) at fixed depth | R5 | SUPPORTED |
| Generator decay within a leg produces no differential Elo | 7.3.1 | `rearly` vs `rlate` differ by 0.4 | R5 | **ESTABLISHED** |
| Book diversity is not the plateau | 4.4 | in-book −26 ± 14 vs holdout −29 ± 18 | R1 | **ESTABLISHED** |
| Depth-6 endgame labels are the *cleanest* in the corpus | 3.3 | bucket 0 MSE 0.011 vs bucket 7 0.207 | R1 | **ESTABLISHED** |

### The offline phase

| claim | evidence | effect | regime | grade |
|---|---|---|---|---|
| Game diversity is worth ~+40 at identical compute | Offline 2.4 | +35.8 ± 16.7 / +37.4 ± 16.6 anchor; +45.4 ± 11.1 paired; +50.0 ± 11.1 over the promoted net | R4 | **ESTABLISHED** |
| Root rows beat leaf rows at a fixed row budget | Offline 3.2 | +35.6 ± 11.0 paired (3.2σ), +46 anchor over the natural mix | R4 | **ESTABLISHED** |
| The 60 cp quiet gate is correct; removing it hurts | Offline 4.3 | −27.9 ± 11.3 pooled over 2000 games/arm, replicated | R4 | **ESTABLISHED** |
| ΔMSE_out prices label *information*, an upper bound on usable signal | Offline 4.4 | the top bin is +52.6% better at predicting outcomes and training on it costs 28 Elo | R4 | **ESTABLISHED** |
| Validation MSE does not rank nets | 7.4; Offline 2.3, 3.3 | ten online arms; three independent offline occasions, two of them wrong-*signed* | R4–R5 | **ESTABLISHED** |
| The offline pass converges to a ceiling flat against corpus composition | 7.7.4 | four arms in a 2.7 Elo band at 167–190M rows | R5 | SUPPORTED |
| Epoch 2 is harmful from an undamaged seed | 7.7.4, 7.9.3; Offline 2.5 | −14 and −20; replicated sign, individually <2σ | R5 | SUPPORTED |
| Retargeting root labels to a rescored PV leaf is worth nothing | 7.7.3 | −2.5 ± 21.0 (95%); refresh between epochs −6.0 ± 20.8 | R5 | SUPPORTED |
| Corpus size returns are flat between 12M and 20M rows | 7.14.4 | identical nets and identical handoff | R6 | SUPPORTED |

### The loop as a whole

| claim | evidence | effect | regime | grade |
|---|---|---|---|---|
| Freeze-generate → consolidate is a no-op | 4.3 (d6), 6.1/6.2 (d8) | d6 flat at −3 / −10; d8 frozen leg returned +7 on 1.7× the games against +53 | R1, R3 | **ESTABLISHED** ¹ |
| A fresh d8 corpus from a learning generator was worth ~0 in the late chain | 7.3 | `rnew` −26.2 ± 17.6 (95%); `rold190` − `rall` = +2.1 ± 21.3 | R5 | SUPPORTED |
| The d8 loop reached equilibrium at ~+13 ± 7 per 500k-game leg | 7.9.1, 7.9.2 | online −126, offline +139, net the small difference | R5 | SUPPORTED |
| The single-optimizer actor/learner path is equal-or-better to the merge | 5.5 | +23 ± 17 over its seed, bit-exactness gate | R3 | **ESTABLISHED** |
| The draw-rate canary detects pathology, not decay | 7.2 | 35.1% flat across ten deciles through a −151 leg | R5 | **ESTABLISHED** |
| Bias sections grow monotonically and were still rising at 7M games | 7.12.2 | fc0_b ×15.9, fc2_b ×8.9, ft_b ×6.4, fc1_b ×3.3 | R5 | SUPPORTED, unexplained |
| Online play as hypothesis generation | 6.14.1 | — | R3 | **FRAMING** (§1 I) |
| The recorded PV's leaf is exact; the root score is not its static eval in ~half of records | branch `tdleaf-pv-telemetry` | `CHECK_ACC` 0 mismatches; off-by-one 43% vs 2% | R7 | **ESTABLISHED** |
| The root fail-high stub writes a 2-ply PV by construction | same | 35.9% of searches | R7 | **ESTABLISHED** |
| The residual mis-approximation is symmetric, not directional | same | hi 27.4% / lo 27.4%, mean +0.7 cp vs sd 74 | R7 | **ESTABLISHED** |
| Its scale is maturity-invariant | same, m260720 ladder 1e5→7e6 | sd 63–72, ≤10 cp band 78–82% | R7 | **ESTABLISHED** |
| Six candidate causes of the residual are excluded | same | aspiration clamping, score hash, leaf quiescence, off-by-one, accumulator rebuild, fail-hard clamping | R7 | **ESTABLISHED** |
| Batch 50 from a fresh `--init-nnue` chain: 500k games in, ahead of `m260720` at the same point | D. Homan, current run | not clearly significant | **R7** | SUPPORTED |
| On a young net under R7+R8 the handoff is statistically absent | §1 K, `m260916` 1e6g ladder | −0.70 ± 3.23 over 8 points | R9 | **ESTABLISHED** |
| It reappears with maturity, but at ~13 Elo not ~130 | §1 K, `m260916` 5e6g ladder | −13.46 ± 3.23 (4.2σ); early−late +12.70 ± 4.57 (2.8σ) | R9 | **ESTABLISHED** |
| The residual is a flat offset, not an excursion-and-return | §1 K | χ² vs constant 13.5 and 8.1 on 7 dof, both p > 0.05 | R9 | SUPPORTED |
| Under R7+R8 the online phase is PRODUCTIVE on every leg | `m260916` decomposition | online Δ +84, +64, +50, +49, +41, +7, +10 — never negative | R9 | **ESTABLISHED** |
| …but the online contribution falls across the four 1M legs while the offline one does not | §3 R9 table | online −14.9 ± 3.7/leg (4.1σ); total −13.2 ± 3.7 (3.5σ); offline +1.6 ± 5.2 (0.3σ) | R9 | **ESTABLISHED** (the decline itself) |
| The 2→4 epoch switch does NOT explain the decline | §3 R9 | `epochs(N)` postdates `online Δ(N)`; onset leg `4e6g` starts from a 2-epoch net; `picked_epoch` = 2 on six of eight legs | R9 | **ESTABLISHED** |
| Offline gain peaks at epoch 2 and decays after | `5e6g` epoch ladder | e1 +16.3, e2 +49.3, e3 +36.6, e4 +26.8 | R9 | SUPPORTED (one leg) |
| ⚠️ A WIDE consolidation window is not helpful on the young chain — it reverses A1 | §1 H, `cons1` base vs null | −9.2 ± 8.8 paired, −12.1 ± 9.9 anchor, at a matched 68.9M-row dose | R9 | SUPPORTED (1.0σ, 1.2σ; consistent in sign) |
| Outcome weight above the default costs ~20 Elo | §1 M, `cons1` nout vs null | −20.5 ± 9.1 paired, −20.1 ± 9.9 anchor — two instruments within 0.4 Elo | R9 | **ESTABLISHED** |
| …and it costs on the composite too, so it is not a corpus artifact | `cons1` bout vs base | −10.3 ± 9.0 paired, −20.4 ± 10.2 anchor | R9 | SUPPORTED |
| λ is FLAT across 0.970–0.985 | §1 M | spread ~5 Elo on the anchor at ±7; the two instruments rank the three points differently | R9 | SUPPORTED |
| Leaf rows do not beat root rows, on an R8 corpus at a fixed game set | §1 H, `cons1` nleaf | −3.8 ± 9.0 paired, −18.4 ± 10.0 anchor | R9 | SUPPORTED |
| The offline pipeline reproduces the chain's own epoch 1 | `cons1` null vs 5e6g ladder e1 | +21.0 ± 6.3 against +16.3 ± 8.9 (different host, book and n) | R9 | **ESTABLISHED** |

² Absent on a young net, or present and masked by concurrent learning gains?
**Answered on R9: masked** — §1 K.  These older rows additionally sit in R1/R2 and
were not measured the same way as the late ones (7.11.5), so they support the
*shape* of the observation, not a number; the numbers to quote are §1 K's.

¹ 6.2's evidence was half Elo and half "validation MSE rose at every epoch"; 7.4
subsequently disqualified val MSE as a ranking instrument, so the d8 half now
leans on the Elo alone (+7 on 1.7× the games).  The conclusion stands; its
support is thinner than 6.2 claimed.

---

## 3. Regime boundaries

A result is only as portable as the regime it was measured in.  These eight
changes are what decide whether an old number still means anything.  Every ledger
row and graveyard entry above and below carries one of these tags.

**There is an eighth axis, orthogonal to all seven: net maturity.**  Every arm in
both records ran on a net with ≥2.2M cumulative games of learning, most at 5–7M —
Parts 1–4 from a ~5M seed, Part 6 from `m260720` at 2.2–3.0M, Part 7 and the whole
offline investigation from 5.5–7.0M.  A saturated net is a *different experimental
subject* from a growing one: the bootstrap `E ← search_d(E)` is closed (§1 G), the
offline pass converges to a fixed ceiling regardless of what it is fed [7.7.4], and
the online phase has no remaining learning gain to offset its handoff cost (§1 K).
**Nothing in the graveyard was tested on a young net.**  R7 is the first fresh
chain since these records begin, so maturity and regime are for once varying
together — which is a reason to re-ask cheap questions on the new chain, not to
assume the old answers carry.

**R1 — the multi-writer merge (through 2026-07-17).**  13–16 concurrent learning
processes merging weight deltas against stale baselines, i.e. roughly W× the
effective single-writer LR near a fixed point.  All of Parts 1–4 sits here.  Online
displacement was large enough to swamp whatever signal the drift carried, which is
why Part 3.5's `seedctl` read the offline gain as pure repair and Part 3.7/4.5
recommended retiring online learning — a conclusion that did not survive R3.

**R2 — the FRC castle accumulator bug (fixed 2026-07-18, Part 5.3).**  On castles
whose destination held the castling side's own rook, `nnue_record_delta` subtracted
a phantom enemy piece from the opponent-perspective accumulator, corrupting search
evals until the opponent's next king move — and online TDLeaf gradients throughout.
**Every FRC game played before this fix carries it**, which is every training game
in Parts 1–4.  Offline corpora were clean (FEN rebuilds).  Treat pre-fix online
numbers as indicative only.

**R3 — actor/learner split, `--refresh-scores`, natural termination (2026-07-18).**
One optimizer, sole `.tdleaf.bin` writer, bit-exact against a single-process run.
Two collapses bought the two hard rules (play to natural termination; keep TD
targets on current weights).  Part 6 lives here.  The R1→R3 change is what flipped
Part 3's conclusion about freezing the generator.

**R4 — the offline recipe wins (2026-09-02/03).**  `--corpus-window` (multi-corpus
consolidation), `--bt-rows root`, wide dumps with a `gate` column so any narrower
gate is an offline filter.  The whole Offline document is at this boundary; the
online document's Part 7 is downstream of it.

**R5 — the ± convention correction (2026-09-06, 7.8.1).**  `pgn_score` reported a
**one-sigma binomial** error; fastchess's own line is a **95% pentanomial
interval** — a factor of 1.96, and the two were mixed freely in the same
discussions.  Everything before this is suspect as to significance, not as to
point estimate.  `pgn_score` now reports pentanomial one sigma.  Also in this era:
`5.5e6gR`'s +155.1 anchor reading proved an outlier that inverted a leg's apparent
sign (7.8.2), so any conclusion resting on a single 1000-game anchor match from
before it should be re-derived.

**R6 — `TDLEAF_ADAM_EPS` 1e-8 → 1e-12 (2026-09-12).**  At 1e-8 the optimizer was
demonstrably sensitive to the *accumulated gradient scale* — that is the entire
mechanism by which `--grad-norm` worked, and a third of its effect vanished when
eps dropped.  **This invalidates any arm whose intervention moved accumulated
gradient scale.**  Batch size moves it as √B, which is why 6.16 closed the batch
line at a peak of 8 and 7.15 reopened it at 32.  Check every closed magnitude knob
against this boundary before accepting that it is closed.

**R7 — the restart (2026-09-15, current).**  A fresh `--init-nnue` chain with:
batch **50** (`TDLEAF_BATCH_SIZE_DEFAULT`); **one LR set used by both phases at
scale 1.0** — the `TDLEAF_ADAM_*_LR0` constants rescaled to 0.25× so offline is
unchanged and **online drops 4×** (FC0 0.00035, FC1 0.00125, FC2 0.0175,
FC_bias 0.375, FT 0.00375, FT_bias 0.005, PSQT 3.25), with `fc0_w` additionally
carrying the 7.12 recalibration; `TDLEAF_ADAM_WARMUP` 1000 → **100** and **live for
the first time ever** (it is keyed on the persisted `t_adam`, which is 0 only on a
fresh net); `TDLEAF_GRAD_CLIP_NORM` 1.0 → **2.0** (the accumulated norm grows as
√B); the FT double-ramp fix; and `--corpus-window` **1**, deliberately giving up
R4's ~45 Elo of game diversity so the new chain starts clean.

**The R7 consequence that matters most: every "online cost" figure in Parts 6 and
7 was measured at 4× the current online LR.**  Since damage ∝ η [7.13.1], the
handoff on the new chain should be roughly a quarter the size before Σ is counted
— and η and Σ moved *at the same time*, so the restart cannot attribute between
them.

**R8 — the PV repairs (merged to `main` 2026-09-17).**
`PV_NO_TT_CUTOFF=1`, `PV_LAST_RESOLVED=1` (both gated to TDLeaf learning play;
competitive search verified node-identical) and `TDLEAF_LEAF_MATCH_CP=10`, which
gates both the online trace and the dumped **leaf** rows on
`|leaf_static − propagated root search| ≤ 10 cp`.  Root rows are untouched — they
are gated on root *quietness*, a different quantity.  Effect on label quality:
coherent bias **−24.88 → +0.15 cp**, sd **185.9 → 79.7**, records reaching full
depth **42% → 86%**, ~74% of records retained.  This is a regime boundary for
corpora: those generated before it contain fail-high stubs and 60 cp-gated leaf
rows, those after do not, so **leaf-row results must not be pooled across it**.
No isolated Elo measurement exists — the `m260916` chain (R9) runs R7 and R8
together and cannot attribute between them.

**R9 — the `m260916` chain (current).**  A fresh material-only seed carried to
5M+ games at **depth 6 with an 800-node budget**, batch 50, one LR set at scale
1.0 both phases, PV repairs on, `--corpus-window 0`, `--bt-rows root`,
`--bt-quiet-cp 60`, 2 offline epochs (**4 at the `4e6g` and `5e6g` legs** — see
the confound below).  It is the first chain to combine R7 and R8, and the first
young chain since the records begin, which is what makes the ladders in §1 K
possible.  Its shallower search (d6/800n against d8/4000n) means throughput
figures and absolute Elo levels do not compare with the `m260720` chain; the
decomposition does.  Unlike R7, **`--gauntlet-tdleaf` was passed on every leg**,
so the online/offline split is measured the same way from end to end.

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

## 4. The graveyard

Closed lines.  Each entry keeps its numbers — several of these produced the clues
the standing conclusions rest on — and each names **what would reopen it**.

**Read the whole section with §3's maturity caveat in hand.**  Every one of these
was closed on a net at 2.2–7M games, where the bootstrap was saturated and the
online phase had no learning gain left to show.  "Bought nothing on a saturated
net" is the honest headline for most of them, and it is not the same claim as
"does not work".  That does not make them live — the mechanisms are still refuted
where the entry says a mechanism was refuted — but a knob whose entry reads
"bought no *Elo*" rather than "its mechanism is wrong" is a candidate for one cheap
re-ask on the young R7 chain.

**Batch size (closed 6.16, REOPENED 7.15).**  *The template case; read it before
closing anything else.*  6.16 found an inverted U peaking at 8 (+19.3 at batch 16,
2.2σ worse; +39.3 at batch 4, 0.9σ) and closed the line.  7.15 reopened it and
moved the peak to 32.  Three things separated them: 6.16 rated *leg total* where
7.15 rates *handoff damage*; 6.16 matched **games**, so step count varied 4× and
"cleaner steps" and "fewer steps" cancelled; and 6.16 predates R6.  **Reopens
again if:** the peak is re-measured on *leg yield* rather than damage — 6.16's
batch-16 arm cut damage 4× and made the loop worse (offline recovery +27.0 against
+84.6), which is the standing warning against adopting batch 32/50 on damage alone.

**`TDLEAF_STACK_NORM_ALPHA` — per-game per-bucket gradient normalisation (6.6–6.10).**
Worked exactly as designed at the weight level (deep-endgame displacement 8×
suppressed, bucket profile inverted from 1.87 to 0.60) and bought nothing:
iteration total +39.3 against the baseline's +53.0.  Byte-exact no-op at its
default 0.0, kept in tree for reproduction.  **Reopens if:** the rejection
criterion is revisited — it was judged on leg total in R3, partly through "online
Δ", which R6-era work reinterpreted as handoff cost.  It is Σ-adjacent (it changes
gradient aggregation), so it deserves a re-read against the damage protocol even
though nothing suggests it will win.

**Per-feature vote normalisation — `TDLEAF_FEATURE_RBAR` / `_DEDUP` (6.11–6.13).**
Also worked as designed, also lost: −23 uncompensated, and **still −12 at matched
displacement**, i.e. ~85% of the 76 Elo gap was the reweighting itself.  The
premise was wrong in an instructive way: `r` counts feature **persistence**, not
position repetition (mass-weighted mean r = 15.9; 92% of PSQT gradient mass sits at
r ≥ 2), and **duration is evidence** — a feature true for 60 plies of a won game is
better support than one true for 3.  Averaging it away destroys real signal.  Dedup
is asymptotically the same update as rbar (Adam absorbs the per-weight constant),
so it was never worth running.  **Reopens if:** someone finds a form that removes
variance without removing persistence — but note 6.12.4's constraint, that the step
rise is not an implementation choice and only the LR can compensate it.

**Learning-target redesign — blend / hybrid / root (Part 2, closed 3.1).**  Three
completely different error formulas (65-record eligibility trace / local one-step
blend / prediction-gated short trace with root distillation) lost 50–95 Elo online
from the identical seed.  The loss is systematic and lives in the shared update
machinery, not the target math.  Code removed in simplification Phase 1; the env
knobs now hard-error.  **Reopens if:** never, on this evidence — three-for-three
across mechanically unrelated targets is as clean as an exoneration gets.

**Freeze-generate → consolidate (3.7/4.5 recommended it; 6.1/6.2 closed it).**
Closed at both depths: d6 flat at −3 / −10 [4.3, after the duplication landmine was
removed], and at d8 the frozen leg used 1.7× the games of the one before it and
returned **+7 against +53**, with validation MSE rising at every epoch.  A frozen
generator labels positions with evaluations the seed already reproduces, so there
is no descent direction to find.  `TDLEAF_FREEZE=1` remains the right tool for
*controls* and is mandatory-with-dedup if ever used for production.  **Reopens if:**
never as a productive loop — but note that Part 7 found the *learning* generator's
corpus also worth zero in the late chain, so "frozen is closed" is not an argument
that learning generation pays.

**Endgame over-coherence (3.2/3.4, 6.4/6.5; retracted and re-retracted in 6.8/6.9;
closed by 6.10).**  Real at the weight level and replicated across two chains
(deep-endgame per-update PSQT violence 1.87× the opening bucket, identical ratio
across the R1→R3 regime change), controllable by alpha — and **not costing Elo**.
Note the unresolved loose end: the `alpha_pretest.sh` harness measured b0/b7 ≈
0.94–0.99 where every production iteration measured 1.32–2.12, across two seeds
158 Elo apart, after excluding binary, flags, dump env, cadence, game count, book
and shuffle seed.  **Never explained; the instrument was deleted.**  **Reopens if:**
a small-harness measurement is ever needed for absolute levels again — that gap is
a standing reason to distrust one.

**Endgame *staleness* / phase-dependent depth limits (3.3).**  Directly tested and
contradicted: depth-6 endgame labels are the cleanest in the corpus by an order of
magnitude (bucket-0 MSE 0.011 against the opening's 0.207; a ≥150 cp endgame
advantage converts 92% of the time).  The stalest labels are in the *opening*.
**Reopens if:** never as stated.  The historical d6→d8 bump has a different
explanation — global bootstrap saturation [3.6].

**Book diversity (4.4).**  In-book −26 ± 14 against out-of-book −29 ± 18 on a
disjoint holdout generated with a different RNG seed: no book overfit at all.  A
side lesson that has kept paying: the scary-looking −29 was the **time control**,
not the book.  **Reopens if:** the opening generator's *recipe* changes, not merely
its seed — a wider book from the same recipe samples themes the net has mastered.

**Optimizer state across the phase boundary — `--opt-reset` (7.10.7, 7.11.7).**
Null for online damage (−129.3 ± 9.9 against −123.4 ± 9.3), and its apparent
+36.0 ± 12.6 improvement to the *final* net was a corpus confound — repeated with
both arms consolidating the identical 190M-row file it reads **−5.9 ± 8.4**.
**Reopens if:** nothing pending.  Note `TDLEAF_ADAM_WARMUP` has been a hard no-op
since the first session ever run (keyed on the persisted `t_adam`) and goes live
for the first time under R7.

**`--grad-norm` (7.10.4, corrected 7.10.7).**  Appeared to cut damage by 91 Elo;
the mechanism turned out to be the `TDLEAF_ADAM_EPS` floor, i.e. a learning-rate
cut in disguise, and a third of the effect vanished when eps went 1e-12.
Deliberately **not** defaulted under R7: its strength scales with positions per
batch (~1,200 at B=8, ~7,500 at B=50), it is applied after clipping so it gives no
relief from the clip, and the scale mismatch it was written for is inert now.
**Reopens if:** eps is ever raised again.

**Per-section LR recalibration / "find the badly calibrated section" (7.11.5,
7.12).**  The cleanest of the five nulls: `fc0_w` was the best candidate the
investigation had (3.7× hot against two stationary sections that agree to 6%, *and*
the largest mover in the net by a factor of two), its LR was cut 3.57× in a
seed-paired arm, its displacement fell 1.65× as designed, and the seven-point
ladder came back flat (χ² 6.68 on 7 dof, p = 0.46).  The FC biases absorbed the
slack (fc2_b +38%, fc0_b +26%) — D. Homan's "loss in one part of the net can be
compensated by another", visible inside 1000 games.  The LR change was kept on
hygiene grounds.  **Reopens if:** nothing — but §1 J (the stationary/scale-finding
split) came out of this work and matters more than the null.

**Retargeting root labels to a rescored PV leaf (7.7).**  Null twice: retargeting
−2.5 ± 21.0, refreshing the target between epochs −6.0 ± 20.8.  The diagnostics
explain it: the seed→generator correction moved labels by 31.8 cp and bought zero,
and the epoch-to-epoch refresh moved them only 14.0 cp.  **Reopens if:** the
rescored leaf comes from a **shallow re-search** rather than a static eval — that
keeps the `E ← search_d(E)` channel alive instead of re-expressing existing labels.
Measure PV stability first (re-search stored roots, check the leaf is still the
stored leaf); it bounds how far any leaf retargeting can travel.

**Widening the quiet gate (Offline Part 1 proposed it, Part 4 closed it).**  Part
1.5's "the gate throws away the signal" was retracted by its own follow-up: row-
matched arms over the same 100k games at gate 60 / 120 / 200 / none read
+100.9 / flat / flat / +73.0, i.e. **removing the gate costs 27.9 ± 11.3**.  The
tail carries the information and it is unlearnable — those labels are good because
search resolved a tactic.  **Reopens for a gate *tighter* than 60**, which was never
tested, and which `--bt-diag`'s negative ΔMSE_out below 40 cp actively hints at
(§6).

**Depth as the lever (4.5/4.6 established it; Offline 1.1 closed it at d10).**
d6→d8 reopened the bootstrap decisively (the first positive consolidation of the
late chain).  d8→d10 regressed: −13.6 on the leg and a head-to-head loss to its own
parent.  **Reopens** — see §6; the d10 leg is confounded by a 52% draw rate and
predates R4 and R6.

**The wide consolidation window (A1 opened it at +36/+45; `cons1` closed it on
R9).**  The single clearest reversal in the record.  A1 measured a four-corpus
window worth **+36 anchor / +45 paired** on the mature chain at 4× the present
LR; `cons1` ran the same manipulation on the young R9 chain at a matched 68.9M
row dose and got **−9.2 ± 8.8 paired / −12.1 ± 9.9 anchor**.  Both cons1 figures
are ~1σ, so this closes on "not helpful", not on "harmful".  What makes it a
closure rather than a null is that the *mechanism* proposed for it also failed:
if old legs' rows hurt because their cp labels are stale, raising outcome weight
(which is age-proof) should rescue them, and instead it cost a further
−10.3 ± 9.0.  **Reopens if** a leg is run at a depth or LR that moves the regime
again, or if someone finds a diversity manipulation that does not also age the
labels — sampling wide *within* one leg, for instance, which cons1 did not test.

**Leaf rows as a training corpus (Offline 3.2 closed it pre-R8; `cons1` closed
it again post-R8).**  3.2's −35.6 ± 11.0 was confounded: it predates R8, so its
leaf rows came from the broken corpus, and the doc's own rule says leaf results
must not be pooled across that boundary.  `cons1`'s `nleaf` is the clean rerun —
R8 corpus, same games as the root arm, same 68.9M dose — and it agrees in sign
at **−3.8 ± 9.0 paired / −18.4 ± 10.0 anchor**.  Smaller than 3.2 claimed, same
direction.  **Reopens** only as a BLEND: pure-leaf-vs-pure-root is now answered
twice, but no one has tested leaf rows as a *supplement* to root rows at matched
total dose, which is a different question and the one §6's old item 11 meant.

**Outcome-weighted targets above the default, and with them the case for
`--bt-rescore` (`cons1` closed both).**  λ = 0.9925 costs −20.5 ± 9.1 paired and
−20.1 ± 9.9 anchor — two independent instruments within 0.4 Elo of each other,
the most replicated single number in this document.  It costs on the composite
corpus too (−10.3 ± 9.0), so it is not an artifact of which rows were used.  The
screen was built so that a slope toward the outcome would license the expensive
rescoring arm; the slope runs the other way, meaning the cp labels carry more
usable signal than the default λ credits.  **Reopens if** a future regime makes
cp labels demonstrably worse — a much deeper search, or a corpus deliberately
aged beyond four legs — since staleness is the only mechanism that would flip
the sign.

---

## 5. Measurement manual

Each of these cost something to learn.  They are regime-independent.

**Rate against a foreign anchor, and know that family matches are non-transitive
in both directions.**  Family matches *understate* real improvement — ~+65 of
style-robust gain showed as +13 within the family [4.6] — and they can also read
**zero on a real 28 Elo regression** [Offline 4.3], where the anchor replicated
across independent seeds and the family match said the losing arm was +6.9 better.
Two nets separated by one update from a common seed share their blind spots.

**Arm-to-arm variance is ~26 Elo.**  Two identical 30k configurations with
different seeds read −149.7 ± 10.2 and −123.4 ± 9.3 [7.11.9].  **Any single-arm
comparison quoted at ±10 is under-powered by roughly 2×.**  This is the most
under-used number in the investigation; `train.py --seed` exists to remove the
generation-seed part of it, but not trajectory divergence or per-point match noise.

**Two direct matches against a common opponent do not subtract.**  Paired said one
arm was better by 24.3 ± 11.8, the anchor said the other by 10.4 ± 16.3, the direct
match said 0.7 ± 8.3 [7.10.6].  **Play the arms against each other**; 1000 games
costs 23 minutes.

`cons1` produced the cleanest worked example on record.  The contrast
λ=0.970 vs λ=0.985, estimated three ways on the same two nets:

| instrument | estimate |
|---|---|
| through the paired opponent (the seed) | +6.5 ± 8.8 |
| through the foreign anchor (`classic_eval`) | −5.0 ± 10.0 |
| **direct head-to-head, 2000 games** | **+9.7 ± 5.9** |

The two subtractions disagree in SIGN, and the anchor's estimate excludes the
direct measurement.  The direct match is also the tightest (±5.9 against ±8.8),
because subtracting two ratings adds their errors while a head-to-head does not.
**When a contrast matters, spend the games on the head-to-head.**  Twenty-five
minutes of direct match here settled what four arms and roughly two hours of
common-opponent rating had left ambiguous.

**Never pool a decaying series.**  Pooling three points of a decaying transient
gave a spurious 5.0σ [7.11.4].  Relatedly, **one early ladder point cannot
distinguish "less damage" from "same damage, reached later"** — an arm that read
+15.1 at 1000 games was *below* its control by 2000 [7.12.5].

**Validation MSE never ranks nets, epochs or arms.**  Ten online arms [7.4] and
three independent offline occasions [Offline 2.3, 3.3, 4] — twice **wrong-signed**,
once ordering a 40 Elo difference backwards.  It is a smoke test for optimizer
health and nothing more.  The one legitimate use is ranking two checkpoints on the
same trajectory, where it merely detects overfitting [7.9.3].

**`ΔMSE_out` prices label information, not usable signal.**  A label can know more
about how the game ends and still be a worse thing to train on [Offline 4.4].
Never read `--bt-diag` as a training recommendation.

**An anchor Elo is only comparable at the SAME time control.**  `classic_eval`
and an NNUE net differ in nodes per second, so the gap between them moves with
the clock: a number measured at `3+0.05` cannot be read against one measured at
`1+0.01`.  This binds the chain's recorded `final_gauntlet` anchors (3+0.05,
`train.py`'s `--tc` default) against anything rated at the epoch-ladder default
(1+0.01) — including the composite-corpus arms.  Within one programme it costs
nothing as long as every arm *and its control* are rated identically; across
programmes it is a trap.  The same caveat applies to `5e6g`'s e2→e3 ladder step
(−12.7): same TC as the arms, but a different opponent (the pre-offline net,
not the seed), which is why §6 item 2 measures its null directly instead of
importing that number.

**Mind the ± convention** (R5): `pgn_score` reports one sigma, fastchess's own line
reports a 95% interval — a factor of 1.96.

**Cheap protocols that changed the economics:**

- **Handoff damage in ~50 minutes per arm**: damage equilibrates by 2–3k games and
  is path-independent, so a 5,000-game online run plus one 1000-game rating against
  the starting net is sufficient [7.11.8].  Measure damage only; take survivors to a
  full leg with a real corpus.
- **Weight-level validation five minutes into a run**: `--publish-stamped` /
  `train.py --ladder` bake a `.nnue` every N games; diff the 1000-game net against
  the start.  The 7.12.4 table confirmed both that the intervention worked and that
  the isolation held (ft_w identical to four significant figures) before any
  gauntlet time was spent.  **Any future optimizer arm should do this first.**
- **Match Adam *steps*, not games**, whenever the knob touches aggregation — this
  is what separated 7.15 from 6.16.
- **Decompose every leg — it costs one flag.**  `--gauntlet-tdleaf` rates the
  post-generation `.tdleaf` state alongside the post-consolidation net, both as
  paired family matches against the previous leg's final.  Online Δ is the first,
  leg total the second, offline Δ the difference.  R9's entire decomposition (§3)
  exists only because the flag was passed on every leg; R7 dropped it partway and
  the corresponding question there is unanswerable [7.11.5].  Recompute from the
  `<tag>_final.json` sidecars rather than from notes — the sidecars carry W/L/D
  and the match error for each entry.
- **The handoff ladder, as run for §1 K**: build a TDLEAF binary per rung, seed it
  from the offline-trained state under test, run 50k self-play games at batch 50
  (≈1000 Adam steps, well past the ~300-step equilibration), stamp a net every
  ~6k games, and rate each stamp in a 1000-game match **against its own starting
  net**.  Two traps, both paid for: the `.nnue` must travel with the binary into
  the scratch directory — a missing net does **not** error, the engine silently
  falls back to classical eval and the whole ladder rates a different program —
  so guard the load and abort on `NNUE: not found`.  And build in `learn/`, never
  `run/`, or `main_bk.dat` feeds book moves into the ladder.  Driver scripts are
  in `scripts/arms/`.

**Pre-commit the reading of an arm** (6.7, 6.12.5, 6.17.4 all did) so the
interpretation is not chosen after the fact.  And state the arm-validity checks
separately from the substantive ones.

**Traps that cost real time:**

- `--bt-quiet-cp` is a corpus-**assembly** knob in `train.py`, never passed to
  `nnue_batch_train`.  **Always verify the assembled corpus, not the input file** —
  the drop is silent [7.14.1].
- The learner loads state from the compiled-in `NNUE_TDLEAF_BIN`, **not** from
  `--tdleaf-out` [7.11.9].
- `train.py` runs binaries from `learn/` while `comp.pl` writes to `run/`; a manual
  rebuild must copy, or the old binary keeps running silently.  And nothing training
  or rating ever executes from `run/` — it holds `main_bk.dat`.
- `match.py` logs contain periodic interim `Elo:` blocks; `grep -m1` returns the
  first, which at 1000 games into a match read −47.7 ± 79.9 against a final of
  −91.7 ± 18.0.  Take the **last** one, after `Finished match` [7.12.4].
- A transient playing out over ~1000 Adam steps cannot be seen in a 24-game harness
  (3 applies).  Scale is the only test.
- Verify the intended setting from the **learner's own startup banner**, not the
  build log — `train.py` skips a recompile when the baked net matches [7.15.2].
- Health canaries: draw rate 35–40% at d8 (detects pathology, **not** decay —
  7.2), plus mean game length, plus — nearly free and *not* blind to uniform decay
  — per-section bias RMS per leg [7.12.2].

---

## 6. Open lines, ranked

Ranked by expected value per unit of compute.  `TODO.md` carries the checklist;
this is the rationale.  Items marked ⚠️ were previously ruled out under a regime
or criterion that has since changed.

**1. The ONLINE DECLINE is real — decide what to do about it.**  `m260916` ran
seven legs with the online phase productive on every one, which the mature chain
never managed.  But across the four 1M legs online falls −14.9 ± 3.7/leg (4.1σ)
while offline does not, and the obvious confound — the 2→4 epoch switch — is
dismissed on timing and on `picked_epoch` (§3 R9).  Per 100k games the series is
84 → 21 → 10 → 4.9 → 4.1 → 0.7 → 1.0, the shape of ordinary saturation.  With the
handoff at ~−13 (§1 K), **online Δ crosses zero within 2–3 legs, around 7–8M
games** — at which point generation is paying 13 Elo for rows and the loop
inverts.  Two responses, not exclusive: make generation productive again (depth,
which R9's d6/800n makes cheap to raise — item 7), or shift the yield to the
offline half, which is what item 2 and the composite-corpus programme are for.
**Method note:** keep passing `--gauntlet-tdleaf`; this decomposition exists only
because R9 passed it on every leg, and 7.11.5 records R7 legs where dropping it
hid exactly this drift.

**2. Sample WIDE WITHIN one leg — the one diversity manipulation `cons1` did
not test.**  The composite-corpus programme answered its own question and closed
it (§4): four legs at a matched dose is −9.2 ± 8.8 paired against the newest leg
alone, and the staleness mechanism that would have explained a fix failed too.
But every `cons1` arm confounded two things — more distinct GAMES and older
LABELS — because more games could only come from older legs.  The clean
separation is available and cheap: one leg holds ~1M games at 68.9 eligible rows
each, so a quota of 17 over all 1M games draws the same dose from **4× the games
at the same label age**.  If that also fails, diversity per se is not the lever
and H's mature-chain +36/+45 was a label-age effect all along; if it wins, the
wide window failed on staleness after all and the composite was simply the wrong
way to buy diversity.  `sample_corpus.py --quota 17` over one leg is the whole
experiment.

**3. Rate the PV repairs (R8) IN ISOLATION.**  `m260916` runs R7 and R8 together
and cannot attribute between them: the handoff fell from ~−130 to −13.5, but
damage ∝ η alone predicts ~−32 from the 4× LR cut, so R8's share is the gap
between −32 and −13.5 — suggestive, not attributed.  The label-quality case is
strong on its own terms — bias essentially eliminated, variance halved, residual
filtered rather than trained on — and the mechanism is visible in the source
rather than inferred statistically.  But no isolated Elo exists, and four prior
interventions in this investigation (alpha 6.10, rbar 6.13, `fc0_w` 7.12, root
fallback) were correct at the label level and bought nothing.  **Arm:** `both`
against `base` from a state with known handoff damage, **matched Adam steps**
(~15–31k games at batch 50 — the batch counts games, so games-matched is
steps-matched), two replicates per side against the ~26 Elo arm-to-arm variance.
A 5k-game arm at 98 steps is *not* sufficient — that mistake was made once
already.  Note the residual mis-approximation is a Σ contributor, and Σ is the
one lever 7.14 left standing and 7.15 showed pays.

**4. ⚠️ The quiet gate *tighter* than 60 cp — now the only untested offline
knob.**  Was item 8.  `--bt-diag`'s negative ΔMSE_out below 40 cp hints that the
near-quiet band is where the label information is, and this is the one offline
lever `cons1` did not touch.  It was deliberately excluded there because it
fights the matching rules — tightening to 20 cp drops root to 36.8 rows/game, so
a fixed quota stops filling and the game population shifts — but at a quota of
17 or below that objection disappears, because 80% of games still fill it.  With
the window closed, the target flat and leaf rows answered, this is where the
remaining offline yield would have to be.  (`--bt-rescore` moved to §4: the
screen for it came back negative.)

**5. Attack Σ directly, not through batch size.**  Batch size is a proxy: the
mechanism is decorrelation, and B only buys it by averaging more whole games.  The
learner consumes `.tdg` files, so it could shuffle *records across a pool of games*
before forming a batch — the actual offline-style intervention, never tried.  This
is the sharpest live line, and it is the only one that would confirm Σ positively
rather than by elimination (§2 marks Σ as FRAMING for exactly that reason).
**Reading:** if record-level shuffling at fixed B reproduces the 8→32 gain, Σ is
demonstrated; if it does not, "Σ" is standing in for something else about whole-game
aggregation.

**6. ⚠️ Re-read alpha and rbar against the damage protocol.**  Both were rejected
on *leg total* in R3, judged partly through "online Δ" — which 7.13 has since shown
is a handoff cost rather than a dosage meter — and both predate R6.  Both are
Σ-adjacent.  Nothing suggests they will win, and 6.13.3's finding that duration is
evidence is a real reason to expect rbar not to.  But they were closed on an
observable now known to be the wrong one, and the 5k-game protocol makes the
re-read cost ~50 minutes each rather than a full iteration.

**7. ⚠️ Depth, on the new chain.**  d10 was rejected on a single leg (−13.6) that
was confounded by a 52% draw rate — cutting outcome information ~25% — and that
predates R4 (corpus window, root rows) and R6.  `E ← search_d(E)` remains the only
mechanism known to restore headroom by construction.  **Precondition:** treat the
draw rate as a hard gate and buy decisiveness from the opening book rather than
from depth [Offline A6].  Fixed-nodes generation gives phase-adaptive depth for
free.

**8. ⚠️ A1b — drop the STALEST corpus, not the whole window.**  The surviving
fragment of the old "deferred offline wins" item; the rest is closed in §4.  A1b
was never run: keep a wide window but drop its oldest leg, whose labels came
from a generator 75 Elo weaker.  `cons1` gives it a reason to exist again — the
composite lost, and if that loss is label age rather than diversity, then the
window minus its stalest leg should sit between the two.  Cheap on the archives
already on disk, and it reads directly against `cons1` base and null.  Do it
after item 2, which separates the same two factors more sharply.

**9. Actor refresh cadence.**  The one clean A/B measured the wrong observable:
the learner's `--refresh-scores` means actor staleness never reaches the labels, so
cadence acts only on the behaviour policy — which positions get played — while the
measurement was weight displacement, downstream of the gradient [6.14.4].  Under
the §1 I framing cadence controls precisely the interesting thing (how fast the
position distribution tracks the hypotheses), and there is neither an Elo
measurement of it nor a corpus-diversity observable to measure it with.  **Blocked
on:** building that observable.  `cos(online, offline)` [6.16.4] is the candidate
and costs no games, but it has **no repeat-run noise floor** — two identically
configured runs were never compared, so its absolute scale is uncalibrated.  Build
that floor first.

**10. Leaf rows as a SUPPLEMENT, not a substitute.**  Pure leaf against pure
root is now answered twice and closed (§4): `cons1`'s `nleaf` reran it on an R8
corpus at a fixed game set and got −3.8 ± 9.0 paired.  What remains is the
question the old wording actually meant — leaf rows *added to* root rows at a
matched total dose, where they might supply coverage rather than replace labels.
Low priority: the two pure comparisons both came out negative, so a blend would
have to beat its own better component.

**11. The bias-growth canary.**  `fc0_b` ×15.9, `fc2_b` ×8.9, `ft_b` ×6.4,
`fc1_b` ×3.3 across the chain, monotone and **still rising at 7M games** — `fc2_b`
+28% in a leg whose total was +6.6 ± 8.2 [7.12.2].  These are the constant-capable
channels `TRAINING.md` flags for outcome-imbalance absorption.  Self-play is
supposed to be immune, so this is unexplained rather than known-pathological.  It
is nearly free to log per leg and, unlike the draw rate, not blind to uniform
decay.  Carry it through the R7 chain from the start.

**12. The root-vs-mix confound.**  `mix` carried 34 root rows/game against `root`'s
76, so `root > mix` may be nothing more than "more root rows".  The discriminating
arm is root-only at 86M rows, matching the root-row count *inside* `mix`: landing
near +155 means the leaf rows were actively harmful, near +109 that `mix` was
merely root-starved [Offline 3.4].  Also unresolved from the same arms: the density
question (76 rows/game once vs 38 twice at matched steps), which leans slightly
against more density and **should not be quoted as settled**.

---

## Reading map to the chronological record

| Part | dates | regime | net maturity | what it settled | status |
|---|---|---|---|---|---|
| Online 1 | 07-14 | R1, R2 | ~5M | Ruled out corruption; framed the plateau as a fixed-LR noise ball | conclusion superseded; the weight-level forensics stand |
| Online 2 | 07-14/15 | R1, R2 | ~5M | blend / hybrid / root targets designed and A/B'd | all rejected in 3.1; code removed |
| Online 3 | 07-16 | R1, R2 | 5–6M | Online loss is target-independent; `seedctl`; bootstrap saturation; endgame *correlation* not staleness | 3.1/3.3/3.6 stand; 3.5's consequence reversed in 6.3 |
| Online 4 | 07-16/17 | R1, R2 | 5–6.5M | `TDLEAF_FREEZE`, the duplication landmine, book diversity retired, d8 reopens the bootstrap, family-match compression | stands |
| Online 5 | 07-17/18 | R2→R3 | `d8t` series | Internal self-play, equivalence study, actor/learner split, the FRC castle bug, the two stability rules | stands |
| Online 6 | 08-13/16 | R3 | 2.2–3.0M | Frozen generation closed at d8; alpha, rbar, batch size all closed; online-as-hypothesis-generator | batch size reopened by 7.15; the rest stands |
| Online 7.1–7.9 | 09-05/10 | R4, R5 | 5.5–7.0M | The trade inverted: corpus worth zero, val MSE ranks nothing, the ± mismatch, equilibrium at ~+13/leg | stands |
| Online 7.10–7.12 | 09-11/13 | R5 | 7.0M | Five optimizer mechanisms, five nulls; the noise-ball reading; the stationary/scale-finding split | stands |
| Online 7.13–7.15 | 09-13/15 | R6 | 7.0M | **The handoff finding**; the gate is not the difference; Σ works (8→32 = +64.0) | current |
| Offline 1 | 09-02 | R4 | 5.5M | `--bt-diag`; the plateau is data starvation on both channels | 1.5's gate reading retracted by 4.4 |
| Offline 2 | 09-02 | R4 | 5.5M | Game diversity worth ~+40 at identical compute | stands |
| Offline 3 | 09-03 | R4 | 5.5M | Root rows are the whole story (+36 paired) | stands; the density and root-count confounds remain open |
| Offline 4 | 09-03 | R4 | 5.5M + 100k fresh | The 60 cp gate is correct; widening buys nothing, removing costs 28 | stands |

Not one row of that table is a young net.  The column is there to make that impossible to miss.
