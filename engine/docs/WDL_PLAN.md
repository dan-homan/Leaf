# WDL Head — Phased Implementation Plan

Status: **plan agreed 2026-07-25; work not started.**
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
| Head structure | Per-material-bucket (×8), fp32, **35 → 3** + softmax. Inputs: `fc2_in[32]` + material (STM cp, scaled) + `fifty/100`. |
| Constraint | `p_w + p_d + p_l = 1` by softmax construction. |
| Search score (Stage C) | `score = K·logit(v)`, `v = p_w + c·p_d`, `K = TDLEAF_K = 220`, computed **in logit space** via stable log-sum-exp — never through post-softmax probabilities (tail saturation). Asymptotically linear in material ⇒ retains the unbounded material spine (no shuffling in won positions). |
| POV | Head is STM-POV like everything else; side flip swaps `l_w ↔ l_l`, negating the score (negamax-safe at `c = 0.5`). |
| Material/PSQT | PSQT feeds the head as an input (per-bucket learned weight `w_mat`); PSQT stays fully trainable — the don't-freeze-PSQT constraint is honored *through* the head. Soft gauge (PSQT scale × `w_mat`) accepted; canary watches the product, not the factors. |
| Fifty-move counter | Head input (late injection), **not** an FT feature (accumulator/TT-key blast radius). Search-side exact draw handling stays. Ships with a **TT-cutoff gate at high counters** (no TT cutoff when `pos.fifty` ≳ 90) in the same change that makes search-visible eval fifty-dependent (Phase 4). |
| Repetition counts | **Not** an eval input — path-dependent, search handles exactly. |
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

- Head inputs are the new 35-vector (branch code had 33: `fc2_in` + material;
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
