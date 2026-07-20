# Simplification Plan — Consolidating Around Actor/Learner Self-Play

**Status:** Phases 1–2 landed (2026-07-20); Phase 3 proposed.

> **Phase 2 done.** `train.py` now defaults to the actor/learner split; the two
> other modes are demoted behind explicit legacy flags — `--selfplay-gen`
> (internal self-play, multi-writer merge) and the new `--uci-pair-gen` (the old
> `match.py`/fastchess UCI pair). `hl_b` is compiled only for `--uci-pair-gen`.
> Docs consolidated on actor/learner (TRAINING.md Quick Start + Generation Modes,
> CLAUDE.md, SCRIPT_USE.md, `selfplay_run.py` docstring); TODO.md Phase D/E marked
> delivered. `Online_Learning_Investigation.md` was **kept in place** (not moved to
> `history/`) — it is still actively cited from TRAINING.md, so a move would only
> create dangling references.


> **Phase 1 done.** Removed the experimental target modes (`TDLEAF_TARGET`
> blend/hybrid, `TDLEAF_ROOT`/`ROOT_WEIGHT`/`TRACE_LAMBDA`/`QUIET_CP`) and the
> LR-sweep knobs (`TDLEAF_LR_*`, `TDLEAF_FREEZE_PASSTHROUGH`); dropped the
> hybrid-target gate keys + root gradient snapshot from `TDRecord` and the `.tdg`
> format (`TDTRAJ_VERSION` 1→2); added `tdleaf_check_env()` (startup config banner
> + hard-error on stray/retired `TDLEAF_*` vars). **Regression gate passed:** a
> 16-game d6 single-process self-play run gives a **byte-identical** `.tdleaf.bin`
> before/after under strict IEEE FP; default `-ffast-math` builds differ only by
> ≤5e-7 weight-space FP-reassociation noise. `TDLEAF_CHECK_ACC=1` clean.

**Goal:** Retire testing-era scaffolding and consolidate the training workflow around the
internal **actor/learner** self-play model, so the project is easier for others to
understand and harder to misuse in a training run.

**Two guiding decisions (settled):**

1. The two non-actor/learner generation modes (UCI-pair via `match.py`, and
   `--selfplay-gen`) are **demoted, not deleted** — kept working behind flags, but
   actor/learner becomes the default and the only documented-as-supported path.
2. The training binary **hard-errors** at startup if any unknown / retired `TDLEAF_*`
   environment variable is set. Leftover env vars are the exact "mistake under
   unexpected conditions" this plan is meant to prevent.

---

## Background: what's there today

### Runtime `getenv` knobs, grouped

| Group | Env vars | Disposition |
|---|---|---|
| **Experimental training-target modes** | `TDLEAF_TARGET` (blend/hybrid), `TDLEAF_QUIET_CP`, `TDLEAF_TRACE_LAMBDA`, `TDLEAF_ROOT`, `TDLEAF_ROOT_WEIGHT` | **RETIRE.** Silently change the learning target if set; no script sets them; production uses the default plain-trace target (mode 0). |
| **LR-sweep multipliers** | `TDLEAF_LR_{FC,FC2,FC_BIAS,FT,FT_BIAS,PSQT}`, `TDLEAF_FREEZE_PASSTHROUGH` | **RETIRE.** All default to 1.0/off; settled; no script sets them. |
| **Driver-set plumbing** (not user knobs) | `TDLEAF_FREEZE` (actors), `TDLEAF_DUMP_TSV`, `TDLEAF_DUMP_QUIET_CP`, `TDLEAF_DUMP_MAX_CP` | **KEEP.** Set by `selfplay_run.py` / `train.py`; part of the loop. |
| **Diagnostics** | `TDLEAF_CHECK_ACC`, `TDLEAF_TRACE_UPDATE`, `NNUE_DEBUG`, `NNUE_DEBUG_VERBOSE` | **KEEP.** Read-only debug aids; no silent effect on training. |

### Generation modes in `train.py`

Three modes existed pre-Phase-2: UCI-pair via `match.py` (was the *default*),
`--selfplay-gen`, `--actor-learner-gen`. The settled recipe (TRAINING.md) already
ran actor/learner + offline consolidation, so Phase 2 aligned the default with
practice (see the Phase 2 status note above).

`match.py` itself is retained regardless — it is also the gauntlet / rating tool
(`train.py:891,1009`). Only its *generation* role is demoted.

### Doc reconciliation note

A stored working-memory note ("online TDLeaf retired; frozen generation → offline
only", 2026-07-17) is **stale**: the newer docs (2026-07-18) describe a settled recipe
with a *live* learner. Update that memory when Phase 1 lands.

---

## Phase 1 — Remove silent-behavior env knobs (highest value)

**Rationale:** These are the knobs that can silently change *what the net learns* if
set by accident or left over from a prior shell. Removing them shrinks the gradient
code and eliminates the worst mistake class.

### 1.1 Delete experimental target modes
- Keep only the plain eligibility-trace target (current mode 0).
- Remove from `src/tdleaf.cpp`: the `blend`/`hybrid` target branches, the online
  root-learning path, `tdleaf_target_mode()`'s env reads, and the associated
  telemetry counters (`td_blend_*`, `td_pred_*`, `td_gate_*`, `td_root_*`).
- Remove the prediction-gate keys `root_key` / `key_own` / `key_reply` from:
  - `TDRecord` (in-memory) — `src/tdleaf.h` / `src/tdleaf.cpp`
  - `TDTrajRecord` (on-disk `.tdg`) — `src/selfplay_traj.h`; **bump `TDTRAJ_VERSION 1→2`**.
    `.tdg` files are ephemeral handoff artifacts (`train.py` runs `--delete-consumed`),
    so an on-disk format bump is safe.
  - the copy sites in `src/selfplay.cpp` (emit + rebuild) and the fill sites in
    `tdleaf.cpp`.
- Remove the now-dead hyperparameters `TDLEAF_TRACE_LAMBDA`, `TDLEAF_ROOT_WEIGHT`,
  `TDLEAF_QUIET_CP` defaults from `src/tdleaf.h` (keep the quiet-gate constant only if
  still referenced by the plain path / dumping — verify before deleting).

### 1.2 Delete LR-sweep multipliers
- Remove `TDLeafLRMultipliers`, `tdleaf_lr_multipliers()`, and `TDLEAF_FREEZE_PASSTHROUGH`
  from `src/nnue_training.cpp`.
- Collapse every `lr_mul.*` factor (all currently multiply by 1.0) so each per-section
  LR is just `lr_scale * warmup_factor * TDLEAF_ADAM_*_LR0` (FT also keeps its
  `ft_session_factor`). Two apply paths use these (the scalar and the parameterized
  `TDLeafApplyParams`) — update both.

### 1.3 Startup config banner + hard-error guardrail
- On learner startup, print one authoritative line of the *effective* training config:
  `K`, `λ`, per-section Adam LR0s, batch size, weight decay, grad-clip norm.
- **Hard-error** if any `TDLEAF_*` env var outside a known allowlist is set. Allowlist =
  the surviving driver-set + diagnostic vars (`TDLEAF_FREEZE`, `TDLEAF_DUMP_TSV`,
  `TDLEAF_DUMP_QUIET_CP`, `TDLEAF_DUMP_MAX_CP`, `TDLEAF_CHECK_ACC`, `TDLEAF_TRACE_UPDATE`).
  Message should name the offending var and point here.

### 1.4 Regression gate
- Build before/after; run a short fixed-seed actor/learner generation and confirm the
  resulting `.tdleaf.bin` is **byte-for-byte identical** (the removed knobs are all
  no-ops at their defaults, so output must not move).
- Run with `TDLEAF_CHECK_ACC=1` to confirm walked-vs-rebuilt accumulators still match
  after the `.tdg` format change.

**Risk:** moderate (touches the gradient path and the `.tdg` format). The byte-exact
check plus `TDLEAF_CHECK_ACC=1` are the guardrails.

---

## Phase 2 — Consolidate generation on actor/learner (demote the rest)

**Decision:** demote, keep behind flags — do **not** delete the alternate modes.

- `train.py`: make `--actor-learner-gen` the **default** generation mode. Keep
  `--selfplay-gen` and the `match.py`-pair path selectable via explicit flags, but mark
  them `(legacy / unsupported)` in `--help` and docs. The `Leaf_vtrain_hl_b` binary is
  only needed by the legacy `match.py`-pair path, so compile it **lazily** — only when
  that mode is explicitly selected.
- Docs:
  - TRAINING.md: rewrite the "three generation modes" section so actor/learner is *the*
    workflow; move the UCI-pair and `--selfplay-gen` writeups to a short "legacy modes"
    subsection or to `docs/history/`. Update the **Quick Start** (it still leads with the
    `match.py` UCI pair) to actor/learner.
  - CLAUDE.md (root + `engine/`): update the "Three generation modes" paragraph and the
    hyperparameter/env-var list to drop the retired knobs.
  - SCRIPT_USE.md: mark `--selfplay-gen` legacy; trim the long `older/training_run.py`
    section to an "archived" note.
  - Fix `selfplay_run.py`'s docstring line "inherits `TDLEAF_TARGET` etc." (that var is
    gone after Phase 1).
- Prune stale docs:
  - `Online_Learning_Investigation.md` → `docs/history/` (reads as a finished writeup).
  - TODO.md: Phase D/E internal-self-play items are largely delivered by actor/learner;
    close them out.

**Risk:** low. Behavior of the default path is unchanged; this is mostly defaults + docs.

---

## Phase 3 — Deeper cleanups (evaluate separately; do not bundle)

- **In-engine concurrent-write merge.** With actor/learner as sole writer, the
  POSIX-lock + delta-merge machinery in the `.tdleaf.bin` save/load hot path
  (`nnue_training.cpp`) is no longer exercised by the default pipeline — it existed for
  `--selfplay-gen`'s N-writer merge. Candidate to simplify to a plain atomic write, but
  it is correctness-critical *and* still used by the (now-legacy but retained)
  `--selfplay-gen` mode. **Treat as its own reviewed change; keep for now.**
  (`merge_tdleaf.py`, the offline run-merger, is independent and stays.)
- **Compile-flag audit.** `TDLEAF_REPLAY_K` is hard-disabled (=0): retire the replay
  ring-buffer unless it's being resurrected. Confirm `TDLEAF_LOG_STEP_CLIPS` and
  `MATERIAL_ONLY` still earn their keep. `TDLEAF_READONLY` stays (inference builds).

---

## Sequencing

1. **Phase 1** — env-knob removal + guardrail + banner, gated by the byte-exact check.
2. **Phase 2** — default flip to actor/learner + docs consolidation.
3. **Phase 3** — evaluated individually, later.

Update `docs/change_log.txt` with each landed phase, and refresh the working-memory
note about the settled recipe once Phase 1 is in.
