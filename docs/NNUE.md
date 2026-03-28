# Leaf NNUE Evaluation — Implementation Notes

## Overview

Leaf can use a Stockfish-compatible NNUE (Efficiently Updatable Neural Network) for
position evaluation in place of its classical hand-crafted eval. The implementation
supports the **HalfKAv2_hm** format used by Stockfish 15/16 era networks.

Build with NNUE enabled:

```sh
g++ -o Leaf src/Leaf.cc -O3 -D VERS="dev" -D TABLEBASES=1 -D NNUE=1 -pthread
```

or via the build script:

```sh
perl comp.pl 2026_03_07a NNUE=1 
```

The network file defaults to `to-be-trained.nnue` and must be in the same directory as the binary,
or in the directory from which the engine is launched.  A fresh network can be generated with
`--init-nnue --write-nnue <file>` before training.  The Stockfish 15.1 release network
`nn-ad9b42354671.nnue` (47 MB) is the historical reference against which Leaf's forward pass
was validated; any Stockfish 15.1–era HalfKAv2_hm network is compatible and can be specified
at compile time with `NNUE_NET=filename`.

When NNUE is not compiled in (`-D NNUE=0` or omitted), the classical eval is used
unchanged.  When NNUE is compiled in but the network file is not found, the engine falls
back to classical eval automatically.

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
| `src/nnue.cpp` | New: FT load/update, FC0–FC2 forward pass, NEON optimizations |
| `src/define.h` | Added `#ifndef NNUE / #define NNUE 0 / #endif` guard |
| `src/chess.h` | Added `NNUE_ACC_PARAM/DEF/ARG/NULL` macros; `NNUEAccumulator acc` in `search_node`; updated `score_pos` declaration |
| `src/score.cpp` | Added NNUE branch at top of `score_pos`: score-hash probe/store, dirty-accumulator refresh, `nnue_evaluate` call |
| `src/search.cpp` | Added accumulator init at search root (with forced dirty=true), copy+update at all three `exec_move` sites, `NNUE_ACC_ARG` at `score_pos` call sites |
| `src/main.cpp` | Added `nnue_load()` call at startup; fixed `score` command to build a temporary accumulator |
| `src/Leaf.cc` | Added `#if NNUE #include "nnue.cpp" #endif` to unity build |

---

## Optimizations Applied

### 1. Score hash integration (+22% NPS, 528K → 646K)

The NNUE branch in `score_pos` probes the existing `score_table` before calling
`nnue_evaluate`, and stores results after.  About 26–38% of evaluation calls are served
from the hash, avoiding both the forward pass and the dirty-accumulator refresh.

### 2. FC0 vdotq reordering (+7% NPS, 646K → 692K)

FC0 weights are reordered at load time to a "vdotq-friendly" layout:

```
l0_weights[s][ib*64 + ob*16 + k*4 + j]
```

where `ib = i/4`, `j = i%4`, `ob = o/4`, `k = o%4`.  The forward pass uses four
`vdotq_s32` NEON calls per 4-input block, accumulating into all 16 output registers
simultaneously (vs. 16 separate passes the compiler would generate).  Requires
`-D __ARM_FEATURE_DOTPROD` (available automatically on Apple M1/M2/M3 with the system
toolchain; no `-march=native` needed).

### 3. NEON fused dual-activation + vdotq FC1 (+4% NPS, 692K → ~720K)

**Dual activation (step 1):** The two scalar loops producing 3,072 int8 values from the
int16 accumulator were replaced with a single fused NEON loop per perspective.  One pass
reads `a[0..511]` and `a[512..1023]` together and writes all three output slices
(SqrCReLU and both CReLU halves) in 64 `int16x8` iterations instead of 1,536 scalar
iterations.

**FC1 vdotq (step 4):** FC1 weights (32×32 int8) are reordered at load time using the
same `ib*128 + ob*16 + k*4 + j` scheme.  The forward pass uses 8 `int32x4` accumulators
and 8 iterations of 8 `vdotq_s32` calls, replacing the 32×32 scalar loop.

### 4. Root-accumulator dirty fix (correctness, ~0% NPS)

At both search-root initialisation sites in `search.cpp` the accumulator dirty flags are
explicitly forced to `true` before calling `nnue_init_accumulator`.  Without this, if the
engine searched a position and then the game advanced to a new position, the stale
accumulator values from the previous position would be silently reused (dirty flags were
still `false` from the previous search).

### 5. Lazy accumulator evaluation (+17% NPS, ~720K → ~840K)

`nnue_apply_delta` replaces the temporary full-refresh stub with true one-level-lazy
incremental evaluation.  The search wiring was already in place (v2026_03_02b added
`nnue_record_delta` calls at all `exec_move` sites and the `nnue_apply_delta` guard before
`score_pos`); this optimization completes the implementation.

**At every `exec_move`:** `nnue_record_delta` stores ≤4 feature-index integers per
perspective into `acc.add/sub` arrays (no `ft_weights` access) and sets `acc.computed=false`.

**At `score_pos` time only:** `nnue_apply_delta` copies the parent's accumulator (2KB per
perspective) and applies the stored deltas via `add_feat`/`sub_feat`.  For king moves
(`need_refresh[p]=true`), the king's own perspective is rebuilt from scratch; the opponent
perspective is still incremental.

Cut nodes — roughly 58% of all nodes — now pay only a handful of integer assignments
instead of a 4KB copy plus 1K `add_feat`/`sub_feat` calls into the 46 MB `ft_weights`
table.

### 6. King-capture lazy accumulator fix (correctness, v2026_03_06z)

When a king *captures* a piece (`Kxf7` etc.), `nnue_record_delta` was only recording the
king's movement (sub from-square, add to-square) but omitting the subtraction of the
captured piece from the opponent's incremental accumulator.  Fix: added `if (capt_pt)`
capture subtraction in the king-move branch of both `nnue_record_delta` and
`nnue_update_accumulator`.  Verified via `NNUE_CHECK_LAZY` across multiple positions
including king-capture sequences.

### 7. Own-king feature inclusion (correctness, v2026_03_07a)

`halfkav2_feature()` was returning −1 (skip) for the own king, but Stockfish includes the
own king as a feature in the PS_KING = 640 slot, the same as the enemy king.  Removing
the exclusion gives correct feature indices and accumulator values for all positions.
This was a major source of evaluation error.

### 8. SqrCReLU: square raw value before clamping (correctness, v2026_03_07a)

The dual-activation loop was computing `v = clamp(0,127, fc0>>6); sq = v²>>7` — clamping
to `[0,127]` *before* squaring, which zeroed all negative fc0 values.  Stockfish's
`SqrClippedReLU` squares the raw int32 value first:
`output = clamp(0, 127, fc0_raw² >> 19)`.  Many fc0 outputs are strongly negative (e.g.
−13,000 to −8,000 in typical middlegame positions); these now contribute 127 to FC1 input
instead of 0, dramatically improving the positional component of the evaluation.

---

## NPS Benchmarks

8-second `analyze` from the starting position, Apple M1 (arm64), single thread:

| Binary | NPS | Notes |
|--------|-----|-------|
| EXchess_classic (no NNUE) | 1,645,247 | baseline |
| v2026_03_01b | 528,348 | NNUE, no optimizations |
| v2026_03_01c | 645,539 | + score hash (+22%) |
| v2026_03_01e | 691,200 | + score hash + vdotq FC0 (+31% total) |
| v2026_03_02b | ~720,000 | + NEON dual-act + vdotq FC1 + root dirty fix (+36% total) |
| v2026_03_02c | ~840,000 | + lazy accumulator (+59% vs baseline) |
| v2026_03_02f | ~870,000 | + singular-extension accumulator fix (correctness) |
| v2026_03_07a | ~870,000 | + own-king fix + SqrCReLU fix + unified scale (correctness) |

Remaining gap vs. classical: **~1.9×**.  At a 1-minute time control the NNUE version
typically searches approximately 1–2 plies shallower than the classical version.

---

## Match Results

Self-play matches at 1 min + 0.1 s/move, 100 games each:

| Match-up | Score | Notes |
|----------|-------|-------|
| v2026_03_02g vs EXchess_classic | 10.0% (0W 2D 18L) | old code, broken scales |
| v2026_03_06z vs EXchess_classic | 22.5% (4W 1D 15L) | fixed lazy acc, split scale |
| v2026_03_07a vs v2026_03_06z | **98.0% (96W 4D 0L)** | own-king + SqrCReLU + unified scale |
| **v2026_03_07a vs classical** | **96.0% (92W 8D 0L)** | current best |

---

## TDLeaf(λ) Online Training

All NNUE layers (FC0/FC1/FC2, FT biases, FT weights, PSQT, dense piece values) can be
trained via TDLeaf(λ) self-play.  See `docs/TDLEAF.md` for the full reference.
Build with `NNUE=1 TDLEAF=1`.

The `scripts/training_run.py` interactive script manages a full training run: net
selection or random init, building both a training and a read-only binary, running
N iterations of M-game self-play matches via `scripts/match.py`, tracking cumulative game
counts across sessions, and exporting the trained weights to a new `.nnue` file named
`<net_base>-<total_games>g.nnue`.  Fischer Random (Chess960) is the default opening
randomisation method.

### Fresh network initialization (`--init-nnue`)

When starting from scratch, run:
```sh
perl src/comp.pl init NNUE=1 TDLEAF=1 OVERWRITE
./run/Leaf_vinit --init-nnue --write-nnue learn/nn-fresh.nnue
```

**FC / FT weights:** drawn from zero-mean Gaussian distributions with stds calibrated
to keep the initial positional output near zero (classical material dominates early play).

| Layer | Distribution | Notes |
|-------|-------------|-------|
| FT weights (int16) | N(0, 44) | acc std ≈ √30 × 44 ≈ 241; ~40% CReLU active |
| FC0 weights (int8) | N(0, 4) | FC0 CReLU ≈ 3.8; keeps FC1→FC2 chain active |
| FC1 weights (int8) | N(0, 3) | Moderate — fan-in 30, low saturation risk |
| FC2 weights (int8) | N(0, 2) | Small — keeps initial positional output ≈ 0 cp |
| All biases | 0 | Zero-init throughout |
| PSQT | Pure material (no PSQ bonuses) | Same value across all 8 buckets |

All int8 weights use **rejection sampling**: values outside ±127 are discarded and
redrawn rather than clipped, avoiding density spikes at the int8 boundaries.

**Design philosophy: start quiet, let TDLeaf build structure from signal.**  Initial NNUE
positional output should be near zero so that classical material dominates early play
(reasonable game quality from game 1).  The network gradually grows its influence as
TDLeaf learns real patterns from self-play.  Zero means (He/Kaiming principle) are the
correct starting point — non-zero means from a trained network are endpoints, not priors.

**PSQT weights:** initialised with pure classical material values from `score.h`, with
no piece-square bonuses.  All 8 buckets receive the same value.  Sign convention: own
pieces (`pside == persp`) receive `+V`, opponent pieces receive `−V`.  TDLeaf learns
positional adjustments on top of this material prior.

Material values (same as classical `value[]` in score.h):

| Piece  | Material (cp) | int32 units (`cp × 5776/100`) |
|--------|--------------|-------------------------------|
| Pawn   | 100          | 5,776                         |
| Knight | 377          | 21,776                        |
| Bishop | 399          | 23,046                        |
| Rook   | 596          | 34,425                        |
| Queen  | 1197         | 69,144                        |
| King   | 0            | 0                             |

Key hyperparameters (in `src/tdleaf.h`):

| Constant | Value | Notes |
|----------|-------|-------|
| `TDLEAF_LAMBDA_DECISIVE` | 0.8 | Eligibility trace decay for wins/losses |
| `TDLEAF_LAMBDA_DRAW` | 0.5 | Eligibility trace decay for draws |
| `TDLEAF_K` | 400 | Sigmoid temperature (cp) |
| `TDLEAF_ADAM_LR0` | 0.13 | Adam step size for FC layers + FT biases |
| `TDLEAF_ADAM_FT_LR0` | 0.2 | Adam step size for FT weights (sparse; higher LR) |
| `TDLEAF_ADAM_PSQT_LR0` | 1.6 | Adam step size for PSQT (int32 scale) |
| `TDLEAF_ADAM_PV_LR0` | 1.6 | Adam step size for dense piece values |
| `TDLEAF_WEIGHT_DECAY` | 1e-4 | AdamW decoupled weight decay (FC + FT weights only) |
| `TDLEAF_GRAD_CLIP_NORM` | 1.0 | Global gradient L2 norm clip threshold |

Adam normalises gradient magnitude — the per-step size in weight-space is governed by
the LR constants, not raw gradient scale.
See `docs/TDLEAF.md` for the full optimizer reference.

---

See `docs/TODO.md` for planned improvements and open investigations.
