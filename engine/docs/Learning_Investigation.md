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
> Reading 2 is the more conservative one and fits the handoff model without
> modification.  Nothing in either document discriminates them, because the
> instrument that would — a within-leg checkpoint ladder over the first few
> thousand games [7.11.4] — was only ever pointed at mature states.  §6 item 2
> makes this a first-class arm on the new chain, where young states exist again
> for the first time since the records begin.
>
> Read every "−120 to −150 Elo" in this document as **a mature-net number of
> unknown applicability to a young one**, and treat the §4 graveyard the same way:
> a knob that bought nothing on a saturated net has not been tested on a growing
> one.

---

Six sections carry the load and they are meant to be used differently:

| section | what it is | use it when |
|---|---|---|
| 1. The model | what the loop is believed to be, in present tense | orienting |
| 2. Evidence ledger | one row per standing claim, graded, with the regime it was measured in | before quoting any number |
| 3. Regime boundaries | the seven changes — plus net maturity — that decide whether an old result still holds | before trusting an old result |
| 4. The graveyard | closed lines, each with **what would reopen it** | before proposing anything |
| 5. Measurement manual | hygiene rules, each with what it cost to learn | before designing an arm |
| 6. Open lines | ranked, with the arm and its pre-committed reading | deciding what to run |

`TODO.md` owns the checklist of work; this document owns the rationale.  Where
they disagree about status, `TODO.md` is the one to fix.

---

## 1. The model

Eleven statements.  Each is tagged with its strongest evidence; grades and effect
sizes are in §2.  Statement I is explicitly an interpretation, not a mechanism.
**All of it describes a mature net** — see the scope box above; K is the statement
about young ones, and it is mostly a statement about what is not known.

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

**H. What the offline pass responds to is game diversity and row type.**  At
identical rows, epochs, optimizer steps and wall clock, drawing from 2.5M games
instead of 500k is worth **+36 anchor / +45 paired** [Offline 2.4].  At a fixed
row budget, root rows (search labels) beat leaf rows (the generator's own static
eval) by **+35.6 ± 11.0 paired** [Offline 3.2].  The 60 cp quiet gate is correct:
60/120/200 are flat and removing it costs **−27.9 ± 11.3** [Offline 4.3] — the
discarded tail carries label *information* (ΔMSE_out +52% in the top bin) that a
static evaluator cannot represent, because those labels are good precisely because
search resolved a tactic.

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
**K. Everything above is a mature-net statement, and the young-net case is
unmeasured.**  The handoff becomes apparent only after ~1M games of learning,
sometimes longer; the chain's own decomposition read +18 / +7 / −2 / −1 at
100k–1M games and only turned clearly negative at 2.2M [6.1].  Whether the
excursion is *absent* on a young net or merely *masked* by concurrent learning
gains is the open question of §6 item 2 — and B's "minimise handoffs" follows from
the mature case either way, since a masked cost is still a cost.  Two further
cautions on reading the early numbers: they sit in R1/R2 (multi-writer merge, and
the FRC castle bug that corrupted every pre-2026-07-18 online gradient), and 7.11.5
independently warns that **"the online damage grew across the chain" is
unestablished** because `--gauntlet-tdleaf` stopped being passed after `3e6`, so
the early and late figures were not measured the same way.  The shape of the
observation is D. Homan's from running the chains; the numbers that would settle it
do not exist yet.

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
| Batch 50 from a fresh `--init-nnue` chain: 500k games in, ahead of `m260720` at the same point | D. Homan, current run | not clearly significant | **R7** | SUPPORTED |

² Absent on a young net, or present and masked by concurrent learning gains?  No
measurement discriminates them — §1 K, §6 item 2.  The early rows additionally sit
in R1/R2 and were not measured the same way as the late ones (7.11.5), so they
support the *shape* of the observation, not a number.

¹ 6.2's evidence was half Elo and half "validation MSE rose at every epoch"; 7.4
subsequently disqualified val MSE as a ranking instrument, so the d8 half now
leans on the Elo alone (+7 on 1.7× the games).  The conclusion stands; its
support is thinner than 6.2 claimed.

---

## 3. Regime boundaries

A result is only as portable as the regime it was measured in.  These seven
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

Two narrower confounds worth remembering: **hash 16** applies only to the `6e6g`
leg (everything else ran at 128, and generation has reverted); and
**`--gauntlet-tdleaf` stopped being passed after `3e6`**, so "the online damage
grew across the chain" was never measured the same way at both ends and should be
treated as unestablished [7.11.5].

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

**1. Batch 50 on a full leg — does the damage reduction survive as leg yield?**
7.15 rates damage, not yield, and 6.16's batch-16 arm cut damage 4× while making
the loop *worse* (offline recovery +27.0 against +84.6).  The R7 chain has adopted
batch 50 ahead of this test, and is currently 500k games in and ahead of `m260720`
at the same point — encouraging, not yet significant.  **Arm:** the chain is the
arm; the missing measurement is a decomposed leg (`--gauntlet-tdleaf` on *every*
leg, which stopped being passed after `3e6` and is how the online drift went
unseen).  **Reading:** if recovery falls in proportion to damage, Σ bought
nothing at the loop level and the line returns to 6.16's warning.

**2. ⚠️ Does the handoff happen on a YOUNG net?  Absent, or masked?**  The scope
box states the problem: the damage is only *apparent* after ~1M games of learning,
and the chain's own decomposition read +18 / +7 / −2 / −1 at 100k–1M before turning
negative at 2.2M [6.1].  Either the excursion does not occur on an unsaturated net,
or it occurs every time and is netted out by concurrent learning gains.  **The
distinction decides whether "minimise handoffs" is a rule for the whole chain or
only for its late life** — and therefore how often the restart should alternate
online and offline phases.  **Arm:** the 5k-game damage protocol [7.11.8] plus a
within-leg checkpoint ladder [7.11.4] applied at several young states of the R7
chain — 100k, 500k, 1M, 2M cumulative games — each rated against its own starting
net.  The R7 chain is the first opportunity since these records begin: every state
in them is already mature.  **Reading:** a ladder that dips and recovers means the
excursion is present and masked, and the leg total is a sum of two effects that
should be optimised separately; a ladder that never dips means the excursion is a
property of a converged optimum, and handoff cost is a late-chain concern only.
**Cost:** ~50 min per state, and the states are produced by the chain anyway.
Cheap enough that not running it is the expensive choice.

**3. Attack Σ directly, not through batch size.**  Batch size is a proxy: the
mechanism is decorrelation, and B only buys it by averaging more whole games.  The
learner consumes `.tdg` files, so it could shuffle *records across a pool of games*
before forming a batch — the actual offline-style intervention, never tried.  This
is the sharpest live line, and it is the only one that would confirm Σ positively
rather than by elimination (§2 marks Σ as FRAMING for exactly that reason).
**Reading:** if record-level shuffling at fixed B reproduces the 8→32 gain, Σ is
demonstrated; if it does not, "Σ" is standing in for something else about whole-game
aggregation.

**4. ⚠️ Re-read alpha and rbar against the damage protocol.**  Both were rejected
on *leg total* in R3, judged partly through "online Δ" — which 7.13 has since shown
is a handoff cost rather than a dosage meter — and both predate R6.  Both are
Σ-adjacent.  Nothing suggests they will win, and 6.13.3's finding that duration is
evidence is a real reason to expect rbar not to.  But they were closed on an
observable now known to be the wrong one, and the 5k-game protocol makes the
re-read cost ~50 minutes each rather than a full iteration.

**5. ⚠️ Depth, on the new chain.**  d10 was rejected on a single leg (−13.6) that
was confounded by a 52% draw rate — cutting outcome information ~25% — and that
predates R4 (corpus window, root rows) and R6.  `E ← search_d(E)` remains the only
mechanism known to restore headroom by construction.  **Precondition:** treat the
draw rate as a hard gate and buy decisiveness from the opening book rather than
from depth [Offline A6].  Fixed-nodes generation gives phase-adaptive depth for
free.

**6. ⚠️ A quiet gate *tighter* than 60 cp.**  Never tested in either direction
below 60, and `--bt-diag` reads **negative** ΔMSE_out for `|cp − gate| < 40`,
which hints the optimum sits under 60.  The A2 arms had power for 28 Elo, not for
5–10 [Offline 4.5].  Cheap: dumps are wide by default and the gate is now an
offline filter, so this is one `--bt-quiet-cp` sweep over a corpus already on disk.

**7. Restore the deferred offline wins on the new chain.**  `--corpus-window` was
turned down to 1 for R7 deliberately, giving up a measured **+36 anchor / +45
paired** [Offline 2.4]; `--bt-rows root` is worth **+35.6 paired** [Offline 3.2].
These are not retired, they are deferred, and the point of recording them here is
that they get revisited rather than forgotten.  A1b — dropping the stalest corpus
from the window, whose labels came from a generator 75 Elo weaker — was never run
and bounds what label staleness costs.

**8. Actor refresh cadence.**  The one clean A/B measured the wrong observable:
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

**9. Leaf rows — settle the blend confound before dropping them.**  `--bt-rows
root` won at the blend production actually uses, but every constant in
`p = w·outcome + (1−w)·σ(cp/K)` was calibrated on the *mixture* and
`--bt-leaf-lambda` has always sat at parity with the root ceiling.  A leaf row's
bootstrap term is near-self-consistent, so its error collapses toward pure outcome
regression at `w ≈ 0.30` [Offline 3.5].  Nobody should quote Part 3 as proof that
leaf positions are worthless *in principle*.  Worth running only when deciding
whether to keep *generating* leaf rows (54% of dump I/O).

**10. The bias-growth canary.**  `fc0_b` ×15.9, `fc2_b` ×8.9, `ft_b` ×6.4,
`fc1_b` ×3.3 across the chain, monotone and **still rising at 7M games** — `fc2_b`
+28% in a leg whose total was +6.6 ± 8.2 [7.12.2].  These are the constant-capable
channels `TRAINING.md` flags for outcome-imbalance absorption.  Self-play is
supposed to be immune, so this is unexplained rather than known-pathological.  It
is nearly free to log per leg and, unlike the draw rate, not blind to uniform
decay.  Carry it through the R7 chain from the start.

**11. The root-vs-mix confound.**  `mix` carried 34 root rows/game against `root`'s
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
