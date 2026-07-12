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
perl ../src/comp.pl <version> NNUE=1 NNUE_NET=nn-leaf-260414.nnue

# Embed the .nnue file directly into the binary (no external file needed at runtime)
perl ../src/comp.pl <version> NNUE=1 NNUE_EMBED=1 NNUE_NET=nn-leaf-260414.nnue
```

The network file defaults to `nn-leaf-260414.nnue` and must be in the same directory as the binary,
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
data file dependencies (other than `main_bk.dat` for the opening book).

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
```

where:
- `psqt_diff = psqt[stm][stack] − psqt[opp][stack]`
- `positional = fc2_out + fwdOut`
- `fwdOut = fc0_raw[15] × 9600 / 8128`  (passthrough; `600×OutputScale / (127×WeightScaleBits)`)
- `5776 = 16 × 361`

An earlier version of the network had a second, densely-updated material channel
(`piece_val[6]`) added to this score alongside PSQT. It was fully removed from the
codebase — PSQT is now the sole material channel, with nothing added on top; see
`docs/history/TRAINING_HISTORY.md` for why it existed and why it was retired.

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

## TDLeaf(λ) Online Training

All NNUE layers (FC0/FC1/FC2, FT biases, FT weights, PSQT — the sole trainable material
channel) can be trained via TDLeaf(λ) self-play.  Build with `NNUE=1 TDLEAF=1`.
See [`TRAINING.md`](TRAINING.md) for the full algorithm reference, hyperparameters, and
training workflow.

---

See [`docs/history/NNUE_HISTORY.md`](history/NNUE_HISTORY.md) for the implementation
history — file additions and performance-optimization notes from the original NNUE port.

---

See `docs/TODO.md` for planned improvements and open investigations.
