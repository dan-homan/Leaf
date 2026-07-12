# Historical Documentation

This directory holds the "how we got here" record for Leaf's design: experiments,
abandoned approaches, superseded hyperparameter values, and one-off implementation
plans. Everything here is preserved for context — to explain *why* the current
design looks the way it does, and to stop dead ends from being silently
re-attempted — but none of it describes the system as it exists today. For that,
see the living docs one level up in `engine/docs/`: `NNUE.md` (network
architecture), `TRAINING.md` (online + offline training, the hybrid loop),
`SCRIPT_USE.md` (script/CLI reference), `TODO.md` (open work), and
`MAINSTREAM_PLAN.md` (the one still-pending roadmap, internal self-play).

## What's here

- **`TRAINING_HISTORY.md`** — the training-system experiment log: the retired
  dense-piece-value channel and its gauge-anchoring machinery, the PSQT-freezing
  failure, the offline-consolidation sweep history (gen-1 through gen-3+), K/λ
  hyperparameter calibration history, the epoch-replay ablation, the
  outcome-imbalance-drift incident, threaded-batch-trainer tuning history, the
  completed Phases A–C of the pure-PSQT mainstreaming effort, a superseded Adam
  hyperparameter regime (explicitly marked — do not use), and the full
  resolved/implemented changelog carried over from `TODO.md`.
- **`NNUE_HISTORY.md`** — the original NNUE port's file-change record and its
  performance-optimization history (NPS benchmarks, early match results).
- **`BT_PARALLEL_PLAN.md`** — a completed, self-contained implementation plan
  (threaded batch trainer replacing multi-process sharding), moved here as-is.
  Marked `Status: IMPLEMENTED` in its own header.
- **`TRAINING_RUN1.md`** — a one-off record of the very first training run
  (`nn-fresh-260309`, 2026-03-09). Superseded in every particular by later work;
  kept as an early data point.

## Conventions

Files here move in largely as-is from wherever they originated — they aren't
rewritten to match current terminology or re-verified against current code,
since the point is to preserve what was true and known *at the time*. If a
passage in a living doc needs the backstory, it links here rather than
duplicating it.
