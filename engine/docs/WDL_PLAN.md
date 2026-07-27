# WDL Head — Phased Implementation Plan

Status: **Phases 0–1 complete (2026-07-25); Phase 2 next.**
Supersedes the design in the `WDL-Head` branch's `WDL_HEAD.md` (mid-June).  That
branch predates the actor/learner split, the v12 `.tdleaf.bin` format, the env
guardrail, and the `hybrid_loop.py` → `train.py` rename — its diffs are ported
**by hand as reference material** into a fresh branch from `main`; nothing is
merged.  Its Phase 2 (multi-writer additive delta-merge) is dropped entirely:
the learner is the sole `.tdleaf.bin` writer on `main`.

---

## Settled design decisions

| Decision | Choice |
|---|---|
| Head structure | Per-material-bucket (×8), fp32, **34 → 3** + softmax. Inputs: `fc2_in[32]` + material (STM cp, scaled) + fifty counter (scaled). |
| Constraint | `p_w + p_d + p_l = 1` by softmax construction. |
| Search score (Stage C) | `score = K·logit(v)`, `v = p_w + c·p_d`, `K = TDLEAF_K = 220`, computed **in logit space** via stable log-sum-exp — never through post-softmax probabilities (tail saturation). Asymptotically linear in material ⇒ retains the unbounded material spine (no shuffling in won positions). |
| POV | Head is STM-POV like everything else; side flip swaps `l_w ↔ l_l`, negating the score (negamax-safe at `c = 0.5`). |
| Material/PSQT | PSQT feeds the head as an input (per-bucket learned weight `w_mat`); PSQT stays fully trainable — the don't-freeze-PSQT constraint is honored *through* the head. Soft gauge (PSQT scale × `w_mat`) accepted; canary watches the product, not the factors. |
| Fifty-move counter | Head input (late injection), **not** an FT feature (accumulator/TT-key blast radius). Search-side exact draw handling stays. Ships with a **TT-cutoff gate at high counters** (no TT cutoff when `pos.fifty` ≳ 90) in the same change that makes search-visible eval fifty-dependent (Phase 4). |
| Repetition counts | **Not** an eval input — path-dependent, search handles exactly. *(Under reconsideration — see "Draw-tracking diagnosis" below: reps are the dominant draw and are invisible to the static head; a rep-count input is Fix 2, deferred until the λ change is settled.)* |
| Contempt | Side-relative `c = 0.5 ∓ δ` (root side devalues its own draws — restores exact negation). δ = 0 **structurally pinned in all training paths**; UCI option for play; default δ set by even-play gauntlet (expected: 0). Targets never see δ. |
| Training target | Distributional λ-return: `π_t = (1−λ^dply)·P_{t+1} + λ^dply·π_{t+1}`, terminal one-hot, softmax-CE gradient (per the original `WDL_HEAD.md` math, White-POV recursion, STM conversion per ply). |
| Outcome-conditioned λ | `λ_draw` vs `λ_dec` allowed; **`λ_win = λ_loss` mandatory** (win/loss asymmetry = designed-in outcome-imbalance drift). Constants fitted offline first (Phase 0), not guessed. |
| Trunk commitment | Staged: A (head-only, gradient stops at `fc2_in`, scalar byte-identical) → B (WDL CE primary into trunk, FC2 scalar auxiliary) → C (search on WDL). FC2 dropped only after Stage C proves it unnecessary. |
| Head init | Reproduce current play at step zero: `w_mat = (PSQT→cp scale)/K`, positional row seeded from FC2, `l_d` from global draw-rate prior. WDL analogue of `--init-nnue-classical`. |
| Knob policy | No new `TDLEAF_*` env vars unless allowlisted in `tdleaf_check_env()`; prefer compile-time constants (loss weights, λs) and UCI options (δ). |

---

## Phase 0 — Offline λ calibration study (no C++)

Fit the target constants from data before any engine change.

- Extend the `extract_positions.py` / `analyze_calibration.py` pipeline: on
  existing PGN corpora, split by game outcome, fit (`λ_dec`, `λ_draw`) by
  held-out cross-entropy of the blended target vs realized outcome as a
  function of plies-to-end.  Reliability curves by distance-to-end, per
  outcome class, as the diagnostic output.
- Measure the global draw-rate prior for head init (`l_d` seed).
- **Exit:** fitted (`λ_dec`, `λ_draw`) with evidence, or a measured null
  (λ_draw ≈ λ_dec ⇒ single λ, idea retired cheaply).

### Phase 0 results (2026-07-25 — `scripts/fit_outcome_lambda.py`)

Data: `learn/corpus_r_dedup.tsv` (root corpus, d6 generation, early-training
net era, 78k games sampled / 25M rows).  Method: future-eval-anchored blend
fit — held-out CE against the outcome itself is circular (the outcome is
inside the target, λ→1 always wins), so the target is scored on predicting
`σ(cp/K)` at horizon `h` plies ahead; closed-form optimal blend weight
`w*(d)` per distance bin + geometric fit, confirmed by direct λ scan.
Split-half agreement ≤ 0.0025 everywhere.

| anchor | λ_dec | λ_draw |
|---|---|---|
| h = 8  | 0.975 | 0.958 |
| h = 16 | 0.985 | 0.978 |
| h = 32 | 0.993 | 0.988 |

- **λ_draw < λ_dec at every horizon** (Δ ≈ 0.005–0.017): draw outcomes decay
  *faster* — very informative near the end (`w*` 0.7–0.9 at d≈10), weakly
  informative far back.  Absolute λ rises with anchor horizon (proxy
  contamination — future eval approaches the outcome as h→d), so the robust
  readout is the class *separation*; the h=16 anchor reproduces the tuned
  `TDLEAF_LAMBDA = 0.985` for decisive games.
- **Starting constants:** `λ_dec = 0.985` (unchanged), `λ_draw ≈ 0.9775`,
  to be validated by a cheap batch-train A/B; **refit on mature-net d8 data**
  before Stage C (era caveat: fitted on early-net d6 corpus, 14% draw rate).
- **Draw-rate prior** for head init, this era: `p_d ≈ 0.14` (recompute at
  init time from the then-current corpus).
- **Reliability by distance** (WDL motivation confirmed in-house): near the
  end (d<20) moderate evals over-claim — `σ(cp/K) = 0.75` realizes only
  `z ≈ 0.64` ("ahead but drawish"); at long distance mid evals are slightly
  under-confident.  Full curves in `learn/wdl_phase0/`.

## Phase 1 — Branch + head scaffolding (Stage A, offline)

Fresh branch from `main` (suggested: `wdl-head-2`).  Port by hand from
`WDL-Head` diffs: head structs/constants (`define.h`, `nnue.h`), forward pass,
`wdl` CLI read-out, TD(λ)-over-distributions (`tdleaf.cpp`,
`nnue_training.cpp`), batch-trainer WDL loss (`nnue_batch_train.cpp`),
`train.py --wdl-head` plumbing (rewritten against `train.py`, not
`hybrid_loop.py`).

- Head inputs are the new 34-vector (branch code had 33: `fc2_in` + material;
  add fifty).
- Gradient stops at `fc2_in`.  **Scalar path must be byte-identical** to a
  non-WDL build — verify with the strict-FP byte-exact regression technique
  from simplification Phase 1 (eval trace + scalar `.tdleaf.bin` sections).
- Persistence: `.tdleaf.bin` **v13** — WDL weights/biases + Adam state as a new
  section; loader accepts v12.
- `.nnue` extension: head weights ride in the net file as a **versioned
  trailing section** (play binaries are non-TDLEAF and must load the head
  without `.tdleaf.bin`; readers that predate it hit clean EOF).  Extend
  `nnue_write_nnue` + loader now so every later phase round-trips;
  `NNUE_EMBED` inherits it for free.
- Batch trainer: distributional target from corpus rows (they already carry
  `result`/`ply`/`endply`); wire Phase-0 (`λ_dec`, `λ_draw`) as compile-time
  constants.
- Telemetry: calibration canary — predicted vs realized W/D/L on a held-out
  corpus slice (extend `diff_tdleaf_checkpoints.py` or a small new script).
- **Exit:** head trains on an existing corpus; calibration curves sane;
  scalar bit-identity confirmed; a gauntlet of WDL build vs plain build is a
  statistical no-op.

### Phase 1 results (2026-07-25)

Implemented as planned with three deviations worth recording:
- **Head is 34→3** (the plan's "35" was an arithmetic slip): `fc2_in[32]` +
  material + fifty.  The head forward is a SEPARATE call
  (`nnue_wdl_head_forward`) after `nnue_forward_fp32`, so the scalar forward
  is textually untouched.
- **Init is exactly zero + priors** (no symmetry-breaking noise): a linear
  output layer needs none, and against O(100) `fc2_in` activations even
  std-0.02 noise adds ±4–5 logits of garbage that swamped the principled
  init (caught in smoke testing).  At init from any freshly loaded `.nnue`
  the FP32 shadows are integer-valued, so the head's material input equals
  the int-path eval exactly; the two paths drift apart only as training
  makes weights fractional (long-standing FP32-shadow property, and the head
  trains/serves through the same FP32 activations, so it is self-consistent).
- **Fifty plumbing:** the TSV dump now writes the real halfmove clock into
  the FEN (old corpora carry 0 there); `.tdg` needed nothing (`pos.fifty`
  ships inside the stored position).
Verified: scalar sections of `.tdleaf.bin` (v13) and exported `.nnue`
byte-identical to a non-WDL build after identical batch-train runs; val
MSE/NLL identical at every epoch; WDL Brier 0.484 (init) → 0.435 (ep3) on a
400k-row corpus sample; online `--selfplay` smoke trains the head and saves
v13; `.nnue` trailer + v12↔v13 cross-loads all round-trip.  The Phase-1
"gauntlet no-op" check is subsumed by byte-identity (play binaries are
non-WDL builds anyway); `train.py --wdl-head` builds TDLEAF binaries with
the head and records the flag in the sidecar.

## Phase 2 — Actor/learner integration (still Stage A)

- Learner (`--learn-stream`): compute WDL targets post-game (outcome known at
  `tdleaf_update_after_game()`); `--refresh-scores` discipline applies to the
  head's bootstrap `P` exactly as to scalar TD targets (online-stability rule 2).
- Verify `.tdg` replay reproduces `pos.fifty` per record — add a check
  alongside `TDLEAF_CHECK_ACC`.  (Expected: yes, positions are rebuilt by
  replay; no `.tdg` format change anticipated.)
- Frozen actors are untouched ⇒ generated games identical; the learner just
  also trains the head.  Draw-rate canary must be unchanged by construction.
- **Exit:** head trained online matches offline-corpus calibration on the
  held-out slice.

### Phase 2 results (2026-07-25)

Zero learner code changes needed beyond a fifty sanity guard: the `.tdg`
record ships the full `position` (fifty included), `tdleaf_rebuild_record`
fills the rest, and the WDL λ-return bootstrap is computed by forward passes
at update time — so the head's targets are on current weights **by
construction**, independent of `--refresh-scores` (which remains required
for the scalar targets, as before).  The learner now rejects trajectory
files whose positions carry a fifty counter outside [0,100].

Verified (m260720 net, d5, natural termination, disjoint opening chunks):
- Frozen actor (400 games) → `.tdg` → learner `--refresh-scores`: 397/397
  consumed, 0 rejected, v13 saved; dumped corpus carries real fifty values
  (observed up to 99; 27 fifty-move draws in generation).
- **Actor generation is byte-identical** between WDL and non-WDL builds
  (20-game `.tdg` byte comparison, pid-gid excluded) — the draw-rate canary
  is unchanged by construction.
- **Held-out calibration** (Brier on a 150-game disjoint chunk):
  fresh-init 0.597 → online-trained 0.527 vs offline-trained (2 epochs on
  the same games' corpus) 0.520.  Online matches offline within ~1.3%; the
  small offline edge is the expected two-full-passes effect.

## Phase 3 — Trunk commitment (Stage B)

- WDL CE becomes the primary loss into FT/FC0/FC1 (port the branch's Phase-4
  trunk-grad code as reference); FC2 scalar loss retained as auxiliary with a
  compile-time weight.
- **LR recalibration pass**: softmax-CE gradient magnitudes differ from
  sigmoid-TD; re-check per-section steps with `TDLEAF_LOG_STEP_CLIPS`
  telemetry before any long run.
- Search still runs on the scalar ⇒ clean A/B: gauntlet Stage-B net (scalar
  play) vs Stage-A net.  Judge by **foreign anchor** (family matches compress
  gains ~5×).
- One full `train.py` hybrid iteration (generate → consolidate → gauntlet)
  with both losses active.
- **Exit:** scalar-play gauntlet non-regressing; head calibration improves
  (trunk features now serve the head).

### Phase 3 status (2026-07-25 — code + offline A/B done, gauntlet pending)

`WDL_TRUNK_GRAD=1` ports the trunk backprop (fc2_in → FC1 → FC0 → FT, with
FC2 weights, the FC0 passthrough, and PSQT all excluded; wdl_mat/wdl_fifty
stop-gradient).  `TDLEAF_WDL_TRUNK_WEIGHT` (default 0.1) is compile-time,
sweepable via `comp.pl TDLEAF_WDL_TRUNK_WEIGHT=<x>` — no new env vars.
- **Regression:** a weight-0 trunk-grad build reproduces the Stage-A
  `.tdleaf.bin` bit-exactly (whole file, head section included).
- **LR recalibration check:** serial-apply step-clip telemetry over 743
  batches — zero clips, per-category max |step| unchanged vs no-trunk
  (FC 2.4 vs 2.5, FT 22.9 vs 22.5); Adam's per-weight normalisation absorbs
  the gradient mix.  No LR changes needed at weight 0.1.
- **Offline A/B** (2M-row corpus slice, 2 epochs, 100k-row held-out val):

  | trunk weight | scalar MSE(blend) | WDL Brier |
  |---|---|---|
  | 0 (Stage A) | 0.01766 | 0.4298 |
  | 0.1         | 0.01781 | 0.4257 |
  | 0.5         | 0.01782 | 0.4258 |
  | 1.0         | 0.01785 | 0.4258 |

  Brier gains ~1% at 0.1 and saturates; scalar MSE degrades slightly and
  monotonically.  0.1 confirmed as default.  (Unlike the old branch's
  report, weight 1.0 does not hurt Brier here — but it buys nothing over
  0.1 and costs the most scalar MSE.)
- Landmine fixed during the port: the online WDL pass must copy the FT
  backprop fields (acc_raw/ft_idx/n_ft) into the activations when the trunk
  gradient is on — Stage A's head-only path never read them.
- **Full-corpus A/B** (25M-row root corpus, 2 epochs, 1.26M-row held-out
  val): trunk weight 0.1 → Brier 0.4290 → 0.4208 (−1.9%), scalar MSE(blend)
  0.01301 → 0.01335 (+2.6%) — both effects larger at scale than on the 2M
  slice.
- **Scalar-play A/B gauntlet** (800 games/arm vs `Leaf_vclassic_eval`,
  3+0.05, FRC openings, one BayesElo frame): w0 −80 ±25, w0.1 −95 ±26 —
  a −15 ±36 point estimate.  **Inconclusive within error**: no confirmed
  regression, but no evidence the trunk co-training helps scalar play
  either, consistent with the small monotone offline MSE cost.  Resolving
  ±10 Elo would need ~10k games/arm.
- **Recommendation recorded:** `WDL_TRUNK_GRAD` stays opt-in (default off)
  for scalar-play-bound training; use it (weight 0.1) on the Stage-C track,
  where the calibration gain is what matters and the scalar-play cost is
  moot (search will run on the head).  Weight 0.1 is the confirmed sweet
  spot in either case.
- **Full hybrid iteration** (2026-07-26, operator-run):
  `m260720wdl-1e6g` (`--wdl-head`, Stage A — no trunk grad) vs the directly
  comparable baseline `m260720-1e6g` (same parent `m260720-5e5g`, 500k games
  d6, 2 epochs):
  - Draw-rate canary identical (25.6% vs 25.9%, 28k games sampled each) —
    generation unaffected, as the byte-identity guarantee requires.
  - Scalar consolidation curves essentially identical (ep2 val MSE(blend)
    0.008573 vs 0.008618; differences are the games themselves — learner
    arrival-order nondeterminism, present in any two reruns).
  - Gauntlets within noise, mixed signs: final vs classic −151.8 ±12.1 (wdl)
    vs −134.1 ±11.8 (base), Δ −17.7 ±17 (~1σ); vs parent-final +88.4 ±11.3
    vs +91.0 ±11.4, Δ −2.6 ±16; epoch ladder favored the WDL run (+91.7 vs
    +80.3).  Consistent with Stage A's structural no-op on scalar play.
  - Head deliverable: Brier 0.4826 (online-trained, entering consolidation)
    → 0.4493 after 2 offline epochs (4.9M-row val) — offline consolidation
    sharpens the head just as it does the scalar net; the trained head ships
    in `m260720wdl-1e6g_final.nnue`'s trailer + the v13 state.

**Phase 3 complete.**  Stage-B trunk co-training remains opt-in per the
recommendation above; the production `--wdl-head` path is Stage A.

## Phase 4 — Search on WDL (Stage C)

- Score dispatch in `score.cpp` switches to the logit conversion.  Fast
  approximation of the two log-sum-exps (piecewise/rational; linear tails);
  perf budget: NPS regression ≲ 2–3%, measured, so a slow `logf` is never
  misread as an Elo regression.
- **TT-cutoff gate** for high fifty counters lands here (eval becomes
  fifty-dependent only now).  Audit the NNUE/score hash caches in `score.cpp`
  for the same staleness (cached eval keyed by position but fifty-dependent —
  gate or key adjustment).
- Contempt plumbing: UCI option, side-relative `c`, default δ = 0; training
  drivers (`selfplay_run.py`, `train.py`) never set it.
- Won-position behavior: targeted endgame conversion tests (KQK, KRK, up-a-
  piece shuffling at long TC) to confirm the logit-space material spine.
- Gauntlet vs Stage-B binary + foreign anchor at matched conditions.
- Then flip generation: actors play on WDL scores; watch draw-rate **and**
  calibration canaries through the first full hybrid iteration.
- **Exit:** non-regression vs scalar search; healthy first co-evolved
  iteration.

### Phase 4 results (2026-07-26 — machinery complete + validated; flip deferred)

Implementation (`WDL_SEARCH`, commit 0a1b5e7): head weights moved to
nnue.cpp as the single fp32 authority (play binaries load the .nnue trailer,
hard-error without it); **train == serve** — the head's canonical inputs are
the INT inference path's activations, captured via an optional out-param on
`nnue_evaluate` (the fp32-activation path survives only for WDL_TRUNK_GRAD);
logit-space conversion at K=220 with side-relative contempt
(`WDLContempt` UCI option, default 0, never set by training); fifty-keyed
score-hash; TT score-cutoff gate at fifty ≥ 90.

Validated:
- Conversion hand-check exact; NPS cost 2.8% (12s search — within budget,
  fast-approximation option unused).
- fp32→int input skew measured small at converged states (Brier 0.4496 int
  vs 0.4493 fp32); head refit on int inputs → 0.4485.
- Won endgames: KQK +1462 / KRK +901 (tail linearity keeps the material
  spine and piece distinctions); KQK at fifty=80 discounts to +480 (the
  trained fifty input is live, smooth damping); WDL search finds a SHORTER
  KQK mate than scalar search in equal time.
- Trailer guard + all five build combinations verified.

**Same-net A/B gauntlet (the exit test): NOT passed yet.**  refit_ep1 net,
800 games/arm vs `Leaf_vclassic_eval`, 3+0.05 FRC: scalar search −20 ±22,
WDL search −70 ±22 → **WDL search costs ~50 ±31 Elo** with the current
head.  Draw rates equal (~19–20%) — not a contempt/draw-avoidance artifact.
Reading: the head today is a 34→3 read-out trained for CALIBRATION
(softmax-CE), one iteration deep, on an early-strength net; its
`l_w − l_l` is approximately the scalar eval plus a linear fc2_in
correction whose move-ranking sharpness CE never optimized.  The machinery
is proven; the head isn't strong enough to search on yet.

**Path to the flip** (in recommended order):
1. Keep search scalar; continue `--wdl-head` iterations (head deepens every
   cycle for free) and re-run this same-net A/B each iteration — it is two
   compiles + one gauntlet, and directly measures the remaining gap.
2. Add trunk co-training (WDL_TRUNK_GRAD, weight 0.1) on a Stage-C-track
   state so trunk features start serving the head (the Phase-3
   recommendation anticipated exactly this use).
3. If the gap persists, the endgame is the plan's original Stage-C bet:
   head as terminal layer with full-strength trunk gradients and FC2
   demoted — a training-recipe change, not more machinery.
Generation stays on scalar search until the A/B reaches non-regression
(flipping actors early would degrade generated-game quality by the same
~50 Elo).

## Draw-tracking diagnosis + length-targeted λ (2026-07-26)

Triggered by a from-scratch WDL run (`m260726wdl`, material init, `--wdl-search
--wdl-trunk-grad`, d6) where iteration 2 (100k→200k games) went flat-to-worse
vs iteration 1 (head-to-head −10 ±11; pooled vs anchors ≈ −23 Elo; losses to
the *weak* material anchor 43→108) while the self-play draw rate drifted 33%→25%
and grad-clip fires rose 7%→11%.  The head itself carries forward and calibrates
fine (Brier 0.529→0.501 across iters) — the regression is a **decisiveness
drift**, not a carry bug.

**Root cause — the head cannot tell a won material-up position from a drawn
one.** Probing the head on corpus positions the engine scored **+150…+500 cp**
(White to move):

| actual outcome | head p_win | head p_draw | head p_loss |
|---|---|---|---|
| games that WON  | 0.520 | 0.306 | 0.174 |
| games that DREW | 0.510 | 0.372 | 0.118 |

Near-identical.  Why: **threefold repetition is the dominant draw** (≈60% of
draws, 15.5% of games; 50-move ≈3%, insufficient-material ≈7%), and reps are a
*path* property — invisible to a static head (not an input), and the fifty
counter can't proxy (draws end at mean fifty ≈10; only 0.6% of drawn positions
are both material-up and fifty≥40).  The head can only learn "drawn" from the TD
terminal, but (a) the old `λ_draw = 0.9775 < λ_dec = 0.985` gave the draw
terminal a *shorter* reach than a win/loss, and (b) the λ-return bootstraps on
the head's *own* (material-up ⇒ win-leaning) prediction (tdleaf.cpp), a
self-consistent wrong fixed point.  Net play effect: the head over-values
material-up-but-drawn positions ⇒ search plays greedily for material ⇒ the
decisiveness drift and the doubled losses to the weak anchor.

**Fix 1 (done) — length-targeted, order-flipped λ.** `fit_outcome_lambda.py`
re-run split by class × game-length on this corpus (future-eval anchor, h∈{8,16,
32}) robustly shows the OPPOSITE of the original Phase-0 constants: **draws want
λ ≥ decisive at every horizon**, and **long draws want the highest λ** (the draw
terminal must propagate proportionally further in a long drawish/rep endgame),
while long *decisive* games want slightly lower λ (uncertain-until-late grinds).
Implemented (`tdleaf.h` `tdleaf_wdl_lambda_draw`, used by both the online
`tdleaf.cpp` recursion and the offline `nnue_batch_train.cpp` decay):
- decisive: fixed `TDLEAF_WDL_LAMBDA_DEC = 0.985` (validated value, UNCHANGED);
- draw: fraction-of-game normalised `λ_draw(N) = clamp(ρ^(1/N), 0.985, 0.993)`,
  `ρ = 0.163`, `N` = game length in plies — longer game ⇒ slower per-ply decay,
  and every drawn game now decays ≥ decisive (the flip).

A first attempt also LOWERED decisive to 0.975 (the fit's absolute value at
h=16), but the offline A/B showed that worsened aggregate Brier (0.5055 vs
0.5011) — the fit's absolute λ is horizon-ambiguous and unreliable; the
robust readouts are the class ORDERING and the LENGTH trend, not the absolute.
Keeping decisive at 0.985 and confining the change to draws is the clean win.

Era caveat (reconciles with Phase 0): the original constants were fitted on a
14%-draw early-net d6 corpus whose draw population differed (few, later-decided);
the from-scratch WDL corpus is rep-heavy with long drawish tails, which want the
reversed ordering.  Re-examine as the net matures.

**Validation (offline A/B, `m260726wdl-2e5g` corpus, same start/hyperparams,
only λ differs):**

| metric | OLD λ | length-targeted λ |
|---|---|---|
| aggregate val WDL_Brier | 0.5011 | **0.4952** |
| won-vs-drawn p_win gap (matched +150..+500 cp) | 0.010 | **0.039** (~4×) |
| p_loss on material-up WON positions | 0.174 | **0.058** |
| p_draw on material-up DREW positions | 0.372 | **0.399** |

The head discriminates won from drawn material-up positions ~4× better, stops
hedging spurious loss on winning positions, and aggregate calibration improves.

Outcome of the online confirmation run (m260726wdlL, still `--wdl-search`): the
new λ did NOT ease the decisiveness drift or close the head-to-head regression —
because the "lost ground" root cause was NOT head calibration at all, but the
WDL-search generation attractor (next section).  The λ change remains a real,
keepable OFFLINE calibration win (and improved head Brier online too: 0.459→0.442
across the scalar-gen chain), orthogonal to the generation fix.

**Fix 2 — repetition count as a head input — REJECTED (2026-07-27).** The plan was
a rep-count input for the residual rep-path draws.  On inspection it earns
nothing: the search returns a draw on the FIRST repetition (a 2-fold) in both pvs
(search.cpp:1199) and qsearch (search.cpp:2101), BEFORE the NNUE eval runs, so the
head essentially never evaluates a repeated position — it always sees rep=0.  A
rep-count input would be constant 0 where it is read, and the diagnosed
material-up-but-drawn positions are themselves rep=0 (the repetition is a future
property, not a current one).  The "Repetition counts — not an eval input" settled
decision therefore stands.

## Root cause of the from-scratch "lost ground": WDL-search generation (2026-07-27)

The λ fix above improved head calibration but did NOT resolve the from-scratch
"lost ground" (m260726wdlL: Stage-2 still −21 Elo vs Stage-1, draw rate still
33→25%).  Localising the loss BY PIPELINE STAGE found the real mechanism.

Under `--wdl-search` the engine eval **is** the head's cp (score.cpp:80), so both
the search AND the online scalar TD targets are driven by the young head.  Stage-2
broken down by stage (Elo vs the material anchor):

| Stage-2 checkpoint | wdlL (`--wdl-search`) | wdlB (scalar gen) |
|---|---|---|
| Stage-1 consolidated (= Stage-2 start) | +472 | +584 |
| Stage-2 online (post-generation) | **+285** (−187) | **+541** (held) |
| Stage-2 consolidated (final) | +379 | +610 |
| Stage-2 final vs Stage-1 final | **−21** | **+96** |

Under WDL-search, Stage-2's online phase dragged the trunk from its +472
consolidated start DOWN to +285 — the head's own operating point (Stage-1's online
net was +279).  **The young head is a ceiling: WDL-search generation distills the
strong consolidated trunk down to what the head can score, every online phase,
regardless of where it started.**  Consolidation re-lifts (real outcomes let it
exceed the head) but from a corpus generated at the head's level, so Stage-2 lands
below Stage-1.  This is the ~50 Elo WDL-search cost from Phase 4, compounding —
exactly what the "generation stays scalar until non-regression" rule (Path to the
flip) exists to prevent.  It shows up at 200k games because it is the head-coupling
attractor, NOT the d6 depth ceiling (which waits for ~2M games / classical
strength — verified by the engine author's experience).

**Fix (confirmed) — scalar generation (Stage B: `--wdl-head --wdl-trunk-grad`, NO
`--wdl-search`).**  Scalar search generates strong games (trunk stays on its
compounding trajectory); the head trains every iteration as a passenger.  Result
(wdlB column): the online attractor is gone (Stage-2 online +541, holds/builds),
Stage-2 EXCEEDS Stage-1 by +96 (was −21), absolute strength ~+230 Elo higher vs
material and ~+160 vs classic, draw rate healthy (34→30% with a RISING within-run
trend, vs the falling 30→25% drift), and the head develops BETTER as a passenger
(consolidation Brier 0.442 vs 0.495 when it drove its own weak search) — the head
gets strong by learning from strong play, which is the bootstrapping thesis of the
Path to the flip.

**Standing rule (now empirically proven):** generation runs on scalar search; the
head trains as a passenger; flip actors to `--wdl-search` only once the same-net
A/B (Phase 4) reaches non-regression.

## Phase 5 — Contempt calibration + consolidation

- Even-play δ gauntlet (δ-variants vs fixed δ = 0, same binary/net; W/D/L
  composition is the sensitive readout, not just Elo).  Non-flat curve ⇒
  suspect head calibration before adopting a nonzero δ.
- One δ sweep vs weaker anchors to document the UCI knob's value.
- Decide FC2's fate (drop auxiliary if Stage C is proven without it; net
  format cleanup).
- Docs: rewrite `WDL_HEAD.md` to as-built, update `TRAINING.md`,
  `SCRIPT_USE.md`, `change_log.txt`.

---

## Cross-cutting rules

1. Byte-exact scalar regression wherever the scalar path is claimed untouched
   (Phases 1–2), strict-FP technique.
2. `λ_win = λ_loss` always; only draw-vs-decisive asymmetry is legal.
3. δ = 0 pinned in every training path; contempt is UCI-only.
4. Draw-rate canary (35–40% @ d8) remains authoritative through Phase 3; from
   Phase 4 the calibration canary joins it as co-equal.
5. Env guardrail: any new `TDLEAF_*` env var must be allowlisted; default to
   compile-time constants per the simplification-Phase-1 convention.
