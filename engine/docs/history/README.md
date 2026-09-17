# Historical Documentation

This directory holds the "how we got here" record for Leaf's design: experiments,
abandoned approaches, superseded hyperparameter values, and one-off implementation
plans. Everything here is preserved for context — to explain *why* the current
design looks the way it does, and to stop dead ends from being silently
re-attempted — but none of it describes the system as it exists today. For that,
see the living docs one level up in `engine/docs/`: `NNUE.md` (network
architecture), `TRAINING.md` (online + offline training, the hybrid loop),
`SCRIPT_USE.md` (script/CLI reference), `TODO.md` (open work),
and `Learning_Investigation.md` (what is known about the hybrid loop, distilled
from the two chronological investigation records that now sit in here).

## What's here

- **`Online_Learning_Investigation.md`** and **`Offline_Learning_Investigation.md`**
  — the chronological research records of the two halves of the hybrid loop
  (2026-07-14 → 09-15, ~5,650 lines between them). Moved here 2026-09-15 when
  `docs/Learning_Investigation.md` was written to carry their conclusions. They are
  kept for provenance: every number in the synthesis cites a section here, and the
  per-part methodology notes name the artifacts on disk that back each measurement.
  Read them for reproduction, never for conclusions — both are blow-by-blow and
  several of their confident intermediate readings were overturned by their own
  later sections.
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
- **`SIMPLIFICATION_PLAN.md`** — the consolidation around actor/learner self-play
  (Phases 1–3, all landed 2026-07-20): retired TDLeaf env knobs, actor/learner made
  the sole generation mode, and the in-engine multi-writer `.tdleaf.bin` merge
  replaced by a direct atomic write. Marked `COMPLETE` in its own header; the one
  item never closed is a compile-flag audit. Its per-phase notes are chronological,
  so an "still open" remark inside an earlier note refers to that moment, not today.
- **`Generation_Throughput.md`** — why self-play generation scaled badly on the
  Linux box and what fixed it (2026-09-03). Resolved: the memory/NUMA hypothesis
  did not hold, the hash hypothesis did, and `clear_hash()` replacing the
  per-game realloc is worth ~25%. Note the 16 MB hash it validated was later
  reverted to 128 MB after a fixed-depth A/B — see
  `Online_Learning_Investigation.md` 7.5, which supersedes §3 of this file.
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
