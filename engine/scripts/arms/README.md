# Experiment arms

One-off drivers for specific experiments, kept as **reproduction handles** for
results that are cited in the research record.  They are not production tooling —
for that see `scripts/` one level up (`train.py`, `match.py`, `selfplay_run.py`).

Each encodes the exact protocol of the arm it ran: flags, seeds, game counts and
the pre-committed reading.  They are frozen at the state that produced the
published numbers; re-running one against today's `main` will not reproduce it
without checking the constants it assumed, several of which have since changed
(notably the 2026-09-15 LR/batch restart — see `docs/Learning_Investigation.md`
§3 "Regime boundaries").

| script | experiment | where the result lives |
|---|---|---|
| `run_lrfc0_arm.sh` | `fc0_w` LR recalibration, seed-paired 30k ladder | `Online_Learning_Investigation.md` 7.12 — null |
| `run_onon_arm.sh` | online-from-online: is the damage a handoff cost? | 7.13 — **+140.9 ± 13.2, the largest single result** |
| `run_gate_test.sh` | the ungated test, arms G and U | 7.14 — null; the gate is not the difference |
| `run_gate_test2.sh` | corrected arm W (the first "ungated" arm was not) | 7.14.1 — documents the trap inline |
| `run_batch_ladder.sh` | the Σ ladder, batch size at matched Adam steps | 7.15 — **8 → 32 = +64.0 ± 12.9, the one knob that worked** |
| `sweep_td.sh` | outcome-weight ceilings under distance-decayed result weight | `TRAINING_HISTORY.md` (K/λ calibration) |

Moved here from `engine/learn/` on 2026-09-17.  `learn/` is gitignored wholesale,
so these had to be force-added there and `sweep_td.sh` was never tracked at all
despite being cited twice.  Living here they are tracked normally.
