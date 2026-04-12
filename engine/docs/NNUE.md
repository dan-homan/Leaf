# Leaf NNUE Evaluation — Implementation Notes

## Overview

Leaf can use a Stockfish-compatible NNUE (Efficiently Updatable Neural Network) for
position evaluation in place of its classical hand-crafted eval. The implementation
supports the **HalfKAv2_hm** format used by Stockfish 15/16 era networks.

Build with NNUE enabled (from `run/`):

```sh
# Basic NNUE build (loads .nnue file from disk at runtime)
perl ../src/comp.pl <version> NNUE=1

# With a specific net file
perl ../src/comp.pl <version> NNUE=1 NNUE_NET=nn-to-be-trained.nnue

# Embed the .nnue file directly into the binary (no external file needed at runtime)
perl ../src/comp.pl <version> NNUE=1 NNUE_EMBED=1 NNUE_NET=nn-to-be-trained.nnue
```

The network file defaults to `to-be-trained.nnue` and must be in the same directory as the binary,
or in the directory from which the engine is launched.  A fresh network can be generated with
`--init-nnue --write-nnue <file>` before training.  The Stockfish 15.1 release network
`nn-ad9b42354671.nnue` (47 MB) is the historical reference against which Leaf's forward pass
was validated; any Stockfish 15.1–era HalfKAv2_hm network is compatible and can be specified
at compile time with `NNUE_NET=filename`.

When NNUE is not compiled in (`-D NNUE=0` or omitted), the classical eval is used
unchanged.  When NNUE is compiled in but the network file is not found (and not embedded),
the engine falls back to classical eval automatically.

### Embedded NNUE (`NNUE_EMBED=1`)

When compiled with `NNUE_EMBED=1`, the `.nnue` file specified by `NNUE_NET` is embedded
directly into the binary at compile time using [incbin](https://github.com/graphitemaster/incbin).
The `comp.pl` script automatically resolves the net file's absolute path (searching the current
directory and `../run/`) and passes it as `NNUE_NET_PATH` to the compiler.

At runtime, the embedded binary loads the network from memory without needing any external
`.nnue` file.  This is useful for distribution — a single self-contained binary with no
data file dependencies (other than `search.par` and `main_bk.dat` for the opening book).

The binary size increases by the size of the `.nnue` file (~26 MB for the default net).
The `.tdleaf.bin` weights file is NOT embedded and must still be provided externally if
TDLeaf training is used.

---

## Network Architecture (HalfKAv2_hm)

| Component | Detail |
|-----------|--------|
| Feature set | HalfKAv2_hm: 32 king-buckets × 704 piece-square indices = **22,528 features** |
| Feature transformer (FT) | 22,528 → 1,024 int16 per perspective + 8 int32 PSQT per perspective |
| Layer stacks | 8 stacks selected by `(piece_count − 1) / 4`, each: |
| FC0 | 1,024 → 16 (SqrCReLU-only input: 512 per perspective × 2 = 1,024) |
| FC1 | 30 → 32 (dual-activation of FC0 outputs 0–14) |
| FC2 | 32 → 1 (output; FC0 output-15 adds directly as passthrough) |
| Activation | SqrCReLU (outputs 0–14) + CReLU (outputs 0–14, appended) |

**King orientation (HalfKAv2_hm):** each perspective horizontally mirrors the board
when the own king is on the queen side (files a–d), so the king always appears on files
e–h.  The Leaf convention uses `orient = ((ksq_f & 7) < 4) ? 7 : 0` where `ksq_f`
is the king square after rank-flip for the BLACK perspective.

**Own king as feature:** the own king IS included as a feature (PS_KING = 640 slot),
identical to the enemy king entry.  Both kings contribute to the accumulator for their
respective perspectives.

---

## Score Formula

The output follows the Stockfish 15.1 formula exactly:

```
value_internal = (psqt_diff / 2 + positional) / OutputScale(16)   [Stockfish Value units]
centipawns     = value_internal × 100 / NormalizeToPawnValue(361)
               = (psqt_diff / 2 + positional) × 100 / 5776
piece_val_cp   = nnue_dense_piece_val(pos, stm, pc)   [TDLeaf only; 0 otherwise]
final_cp       = centipawns + piece_val_cp
```

where:
- `psqt_diff = psqt[stm][stack] − psqt[opp][stack]`
- `positional = fc2_out + fwdOut`
- `fwdOut = fc0_raw[15] × 9600 / 8128`  (passthrough; `600×OutputScale / (127×WeightScaleBits)`)
- `5776 = 16 × 361`
- `piece_val_cp` = dense piece value correction from `piece_val[6][8]` (48 TDLeaf-trained
  floats; 6 piece types × 8 PSQT buckets).  Sums `piece_val[pt][bucket]` for each STM piece
  and subtracts for each opponent piece.  Initialized to zero; learns material adjustments
  via dense gradients (~200 updates/game vs ~8/5000g for sparse PSQT features).

Relevant quantization constants in `nnue.h`:

| Constant | Value | Meaning |
|----------|-------|---------|
| `NNUE_WEIGHT_SHIFT` | 6 | FC weight quantization: raw int32 >> 6 → [0,127] range |
| `NNUE_SQR_SHIFT` | 7 | SqrCReLU shift: `fc0_raw² >> (2×6+7=19)` → [0,127] |

**SqrCReLU detail:** the raw int32 fc0 value is squared *before* clamping:
`output = clamp(0, 127, fc0_raw² >> 19)`.  This matches Stockfish's `SqrClippedReLU`
exactly — negative fc0 values produce non-zero squared outputs.

`nnue_evaluate()` returns a score from the **side-to-move's perspective** (positive =
good for the side to move).  The score hash stores values in **White's perspective**;
conversion at store/retrieve: `score_w = wtm ? score : -score`.

---

## Files Added / Modified

| File | Change |
|------|--------|
| `src/nnue.h` | New: architecture constants, `NNUEAccumulator` struct, public interface |
| `src/nnue.cpp` | New: FT load/update, FC0–FC2 forward pass, NEON optimizations, MemStream abstraction for file/memory loading |
| `src/nnue_embed.cpp` | New: incbin wrapper for embedding `.nnue` file into binary (compiled when `NNUE_EMBED=1`) |
| `src/incbin.h` | New: public-domain header for cross-platform binary embedding (Dale Weiler) |
| `src/define.h` | Added `#ifndef NNUE / #define NNUE 0 / #endif` guard, `NNUE_EMBED` default |
| `src/chess.h` | Added `NNUE_ACC_PARAM/DEF/ARG/NULL` macros; `NNUEAccumulator acc` in `search_node`; updated `score_pos` declaration |
| `src/score.cpp` | Added NNUE branch at top of `score_pos`: score-hash probe/store, dirty-accumulator refresh, `nnue_evaluate` call |
| `src/search.cpp` | Added accumulator init at search root (with forced dirty=true), copy+update at all three `exec_move` sites, `NNUE_ACC_ARG` at `score_pos` call sites |
| `src/main.cpp` | Added `nnue_load()` call at startup; fixed `score` command to build a temporary accumulator |
| `src/Leaf.cc` | Added `#if NNUE #include "nnue.cpp" #endif` to unity build |

---

## TDLeaf(λ) Online Training

All NNUE layers (FC0/FC1/FC2, FT biases, FT weights, PSQT, dense piece values) can be
trained via TDLeaf(λ) self-play.  Build with `NNUE=1 TDLEAF=1`.
See [`TDLEAF.md`](TDLEAF.md) for the full algorithm reference, hyperparameters, and
training workflow.

---

## Optimization & Development History

The following sections document the NNUE implementation history, including
optimization work, correctness fixes, and early match results.

### Optimizations Applied

1. **Score hash integration** (+22% NPS, 528K → 646K) — `score_pos` probes `score_table`
   before `nnue_evaluate`; ~26–38% of calls served from hash.
2. **FC0 vdotq reordering** (+7% NPS, 646K → 692K) — weights reordered at load time
   for NEON `vdotq_s32` dot-product instructions.
3. **NEON fused dual-activation + vdotq FC1** (+4% NPS, 692K → ~720K) — single fused
   loop for SqrCReLU + CReLU; FC1 uses same vdotq scheme.
4. **Root-accumulator dirty fix** (correctness) — force `dirty=true` at search root to
   prevent stale accumulator reuse across positions.
5. **Lazy accumulator evaluation** (+17% NPS, ~720K → ~840K) — `nnue_record_delta` stores
   feature-index deltas at `exec_move`; `nnue_apply_delta` materializes only when
   `score_pos` is called.  Cut nodes (~58%) skip accumulator updates entirely.
6. **King-capture lazy accumulator fix** (correctness) — captured piece subtraction was
   missing from the opponent's incremental accumulator in king-capture moves.
7. **Own-king feature inclusion** (correctness) — own king included as PS_KING=640
   feature, matching Stockfish; was a major source of evaluation error.
8. **SqrCReLU: square before clamp** (correctness) — `clamp(0, 127, raw² >> 19)` instead
   of `clamp(0,127, raw>>6)²>>7`; negative pre-activations now contribute correctly.

### NPS Benchmarks

8-second `analyze` from starting position, Apple M1 (arm64), single thread:

| Binary | NPS | Notes |
|--------|-----|-------|
| EXchess_classic (no NNUE) | 1,645,247 | baseline |
| v2026_03_01b | 528,348 | NNUE, no optimizations |
| v2026_03_01c | 645,539 | + score hash (+22%) |
| v2026_03_01e | 691,200 | + vdotq FC0 (+31% total) |
| v2026_03_02b | ~720,000 | + NEON dual-act + vdotq FC1 (+36% total) |
| v2026_03_02c | ~840,000 | + lazy accumulator (+59% total) |
| v2026_03_07a | ~870,000 | + correctness fixes (own-king, SqrCReLU) |

### Early Match Results

Self-play matches at 1 min + 0.1 s/move, 100 games each:

| Match-up | Score | Notes |
|----------|-------|-------|
| v2026_03_07a vs classical | **96.0% (92W 8D 0L)** | NNUE with SF15.1 net |
| v2026_03_07a vs v2026_03_06z | 98.0% (96W 4D 0L) | own-king + SqrCReLU fix |

---

See `docs/TODO.md` for planned improvements and open investigations.
