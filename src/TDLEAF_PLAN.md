# TDLeaf(λ) NNUE Learning — Implementation Plan

## Overview

TDLeaf(λ) is a temporal-difference reinforcement learning algorithm adapted for minimax
search by Baxter, Tridgell, and Weaver ("Learning to Play Chess Using Temporal
Differences", 2000).  It uses the game result and the sequence of root search scores to
form TD errors, then backpropagates those errors through the NNUE network to update
weights.

The implementation is entirely optional and gated by a compile-time flag:

```sh
perl comp.pl 2026_03_09a NNUE=1 TDLEAF=1
```

When `TDLEAF=0` (default), none of the learning code is compiled in.

---

## Algorithm

For a game of T half-moves:

- Let `d_t` = root search score at ply t, normalised to (0, 1) via a sigmoid:
  `d_t = 1 / (1 + exp(-score_t / K))` where K ≈ 400 cp (controls sigmoid steepness).
- Let `z` = game result from White's perspective: 1.0 = White wins, 0.5 = draw, 0.0 =
  Black wins.
- All `d_t` are from White's perspective: if it is Black's turn, negate the raw score.

**Temporal difference errors (backward view):**

```
e_T   = z - d_T
e_t   = (d_{t+1} - d_t) + λ * e_{t+1}      for t = T-1 … 1
```

**Weight update:**

```
Δw = α * Σ_t  e_t * ∇_w d_t
```

where `∇_w d_t = d_t * (1 - d_t) * ∇_w score_t` (sigmoid derivative times network
gradient at position t).

Recommended defaults: `λ = 0.7`, `α = 1e-5`.

---

## Scope: Which Weights to Train

**Phase A (initial implementation):** FC layers only.

The FC layers are small (≈18 K weights × 8 stacks) and can be updated quickly.  The
feature transformer (FT) is 46 MB and requires sparse gradient accumulation; it is
deferred to Phase B.

| Layer | Parameters per stack | Total (8 stacks) |
|-------|---------------------|-----------------|
| FC0 weights | 1,024 × 16 int8 | 131,072 |
| FC0 biases  | 16 int32 | 128 |
| FC1 weights | 32 × 32 int8 | 8,192 |
| FC1 biases  | 32 int32 | 256 |
| FC2 weights | 32 × 1 int8 | 256 |
| FC2 bias    | 1 int32 | 8 |
| **Total** | | **≈ 140 K** |

**Phase B (optional, future):** FT weights.  Each active feature touches 1,024 int16
weights.  A position activates ≈ 30 features per perspective, so ~60 K of the 23 M FT
weights are touched per position — sparse enough to be tractable.

---

## New Files

| File | Contents |
|------|----------|
| `src/tdleaf.h` | Public interface, `TDGameRecord` struct, constants |
| `src/tdleaf.cpp` | Score recording, TD error computation, gradient accumulation, weight update |

`src/EXchess.cc` gains `#if TDLEAF #include "tdleaf.cpp" #endif`.

---

## Data Structures

### `TDRecord` (one per ply, stored in `TDGameRecord`)

```cpp
struct TDRecord {
    int16_t acc[2][NNUE_HALF_DIMS];   // accumulator snapshot (2 KB)
    int32_t psqt[2][NNUE_PSQT_BKTS]; // PSQT snapshot
    int     score_cp;                  // raw centipawn score from White's POV
    int     stack;                     // layer stack index used
    bool    wtm;                       // White-to-move at this position
};
```

Memory: ≈ 2,120 bytes × 400 plies = ≈ 848 KB.  Acceptable.

### `TDGameRecord`

```cpp
struct TDGameRecord {
    TDRecord   plies[MAX_GAME_PLY];
    int        n_plies;
    float      result;      // 1.0 / 0.5 / 0.0 (White POV)
};
```

One global instance; reset at the start of each game.

---

## FP32 Weight Copies

The stored weights are quantized int8 (FC) and int16 (FT).  Gradient updates require
FP32 precision to avoid rounding to zero on small steps.

```cpp
// In nnue.cpp (inside #if TDLEAF guard):
static float l0_weights_f32[NNUE_LAYER_STACKS][NNUE_L0_INPUT * NNUE_L0_SIZE];
static float l0_biases_f32 [NNUE_LAYER_STACKS][NNUE_L0_SIZE];
static float l1_weights_f32[NNUE_LAYER_STACKS][NNUE_L1_PADDED * NNUE_L1_SIZE];
static float l1_biases_f32 [NNUE_LAYER_STACKS][NNUE_L1_SIZE];
static float l2_weights_f32[NNUE_LAYER_STACKS][NNUE_L2_PADDED];
static float l2_bias_f32   [NNUE_LAYER_STACKS];
```

Initialised from the quantized int8 arrays when `nnue_load()` is called (multiply by
`1.0f / (1 << NNUE_WEIGHT_SHIFT)` to dequantize).

After each gradient step, `nnue_requantize_fc()` writes the updated FP32 values back to
the int8 arrays used by the forward pass:
`l0_weights[s][i] = (int8_t)clamp(-127, 127, roundf(f32 * (1 << NNUE_WEIGHT_SHIFT)))`.

---

## Implementation Steps

### Step 1 — Score recording

**`src/define.h`**
```cpp
#ifndef TDLEAF
#define TDLEAF 0
#endif
```

**`src/chess.h`** — extend `game_rec`:
```cpp
#if TDLEAF
  TDGameRecord td_game;          // accumulator+score history for current game
#endif
```

**`src/main.cpp`** — after each `ts.search()` call (line 476), add:
```cpp
#if TDLEAF
  if (nnue_available)
      tdleaf_record_ply(game.td_game, game.pos,
                        game.ts.g_last, game.pos.wtm);
#endif
```

At game start (`new_game` / `setboard` sites), reset:
```cpp
#if TDLEAF
  game.td_game.n_plies = 0;
#endif
```

**`src/tdleaf.cpp`** — `tdleaf_record_ply()`:
- Copy `acc.acc[0..1][0..1023]` and `acc.psqt` from the current root node's accumulator
  (available in `game.ts.tdata[0].n[0]` — the root search node after search returns).
- Convert raw score to White POV: `score_w = wtm ? score : -score`.
- Append to `td_game.plies[n_plies++]`.

### Step 2 — Game result extraction

**`src/main.cpp`** — at each site where `game.over = 1` is set, determine result:
```cpp
#if TDLEAF
  float td_result = 0.5f;           // default: draw
  if (strstr(game.overstring, "1-0"))        td_result = 1.0f;
  else if (strstr(game.overstring, "0-1"))   td_result = 0.0f;
  if (nnue_available && game.td_game.n_plies > 0)
      tdleaf_update_after_game(&game.td_game, td_result);
#endif
```

Also hook the `"result"` xboard command (already sets `game.over = 1`) to parse the
result string from xboard (format: `"result 1-0 {White mates}"`).

### Step 3 — Forward pass with saved activations

Add to `src/nnue.cpp` (inside `#if TDLEAF`):

```cpp
struct NNUEActivations {
    float l0_in [NNUE_L0_INPUT];     // SqrCReLU output (FP32)
    float fc0_raw[NNUE_L0_SIZE];     // FC0 pre-activation
    float fc1_in [NNUE_L1_PADDED];   // dual-activation output
    float fc1_raw[NNUE_L1_SIZE];     // FC1 pre-activation
    float fc2_in [NNUE_L2_PADDED];   // CReLU output
    float fc2_raw;                    // FC2 output (scalar)
    float fwdOut;                     // passthrough
    float positional;
    int   stack;
};

// Recomputes the forward pass in FP32 from a saved accumulator,
// filling 'act' with all intermediate activations.
void nnue_forward_fp32(const TDRecord &rec, NNUEActivations &act);
```

This function mirrors `nnue_evaluate()` but in FP32 (no NEON), saving every
intermediate value needed for backprop.

### Step 4 — Backpropagation

`tdleaf_backprop()` takes a single `TDRecord`, runs `nnue_forward_fp32()`, then
computes gradients from output back to FC0 inputs using the chain rule.  Results are
accumulated into per-stack gradient arrays:

```cpp
static float grad_l0_w[NNUE_LAYER_STACKS][NNUE_L0_INPUT * NNUE_L0_SIZE];
static float grad_l0_b[NNUE_LAYER_STACKS][NNUE_L0_SIZE];
static float grad_l1_w[NNUE_LAYER_STACKS][NNUE_L1_PADDED * NNUE_L1_SIZE];
// ... etc.
```

Activation derivatives:
- **SqrCReLU**: `d/dx[clamp(0,127, x²>>19)]` = `2x/2^19` when `x²>>19 ∈ (0,127)`, else 0.
  In practice use a straight-through estimator through the clamp for simplicity.
- **CReLU**: `d/dx[clamp(0,127, x)] = 1` when `x ∈ (0,127)`, else 0.

Score-to-sigmoid gradient:
```
∂loss/∂score = e_t * d_t * (1 - d_t) * (1.0f / K) * (100.0f / 5776.0f)
```
where `d_t = sigmoid(score_white / K)`.

### Step 5 — Weight update and requantization

After processing all plies in the game:
```cpp
void tdleaf_apply_gradients(float alpha) {
    for (int s = 0; s < NNUE_LAYER_STACKS; s++) {
        for (int i = 0; i < NNUE_L0_INPUT * NNUE_L0_SIZE; i++)
            l0_weights_f32[s][i] -= alpha * grad_l0_w[s][i];
        // ... FC1, FC2 similarly
    }
    nnue_requantize_fc();   // update int8 arrays from FP32
}
```

`nnue_requantize_fc()` clamps and rounds each FP32 weight back to int8, then writes into
the `l0_weights / l1_weights / l2_weights` arrays used by the live forward pass.  The
score hash must be cleared after requantization (call `clear_score_table()`).

### Step 6 — Weight persistence

```cpp
// Append updated FC layer bytes to a separate delta file, or overwrite
// the original .nnue.  The latter is simpler but destructive.
bool nnue_save_fc_weights(const char *path);
```

**Recommended:** save to a companion file `nn-ad9b42354671.tdleaf.bin` containing only
the 8 × (FC0+FC1+FC2) layer stacks (≈ 140 KB).  At load time, if this file exists,
load the FT from the original .nnue and the FC layers from the companion file.  This
avoids corrupting the original network and makes it easy to reset learning.

---

## Hooks in Existing Code

| Location | Change |
|----------|--------|
| `src/define.h` | Add `#ifndef TDLEAF / #define TDLEAF 0 / #endif` |
| `src/chess.h` — `game_rec` | Add `TDGameRecord td_game` inside `#if TDLEAF` |
| `src/nnue.cpp` — `nnue_load()` | After loading: init FP32 copies inside `#if TDLEAF` |
| `src/main.cpp` — after `ts.search()` | Call `tdleaf_record_ply()` inside `#if TDLEAF` |
| `src/main.cpp` — `game.over = 1` sites | Call `tdleaf_update_after_game()` inside `#if TDLEAF` |
| `src/main.cpp` — `"result"` command | Parse result string; call update inside `#if TDLEAF` |
| `src/main.cpp` — `new_game` / `setboard` | Reset `td_game.n_plies = 0` inside `#if TDLEAF` |
| `src/EXchess.cc` | Add `#if TDLEAF #include "tdleaf.cpp" #endif` |
| `src/comp.pl` | Support `TDLEAF=1` flag → append `-D TDLEAF=1` |

The changes to existing files are all small, inside `#if TDLEAF` guards, and do not
affect compilation or behaviour when `TDLEAF=0`.

---

## Hyperparameters

Exposed via `setvalue` command (already supported by EXchess) or environment variables:

| Parameter | Default | Notes |
|-----------|---------|-------|
| `td_lambda` | 0.7 | Eligibility trace decay |
| `td_alpha` | 1e-5 | Learning rate |
| `td_K` | 400 | Sigmoid temperature (cp) |
| `td_min_plies` | 10 | Skip very short games |
| `td_save_every` | 1 | Save weights after every N games |

---

## Testing Strategy

1. **Gradient check**: For one position, compute `∂score/∂w` numerically
   (`(score(w+ε) - score(w-ε)) / 2ε`) and compare against the analytical gradient.
   Should agree to within 1%.

2. **Smoke test**: Play 10 self-play games with `TDLEAF=1`.  Verify weights change, no
   crash, scores remain in a sane range.

3. **Sanity match**: After 100 self-play training games, run a 100-game match of the
   trained net vs the untrained net.  Expect a small but non-zero positive score.

4. **Long-run match**: After 1,000+ games, match trained vs `EXchess_vtest` (classical).
   Goal: maintain or improve over the 96.0% baseline.

---

## What NOT to Do

- **Do not update FT weights in Phase A**: the 46 MB sparse update is correct in theory
  but hard to implement and quantize correctly; the FC layers are where the
  position-specific signal lives anyway.
- **Do not use a high learning rate**: int8 quantization means FP32 changes smaller than
  `1/(64)` ≈ 0.016 are invisible; but large learning rates will cause catastrophic
  forgetting.
- **Do not clear the score hash between every ply**: clear it once after
  `nnue_requantize_fc()` at end-of-game, not during.
- **Do not train during time-critical games**: the backprop + update is cheap (< 1 ms
  per game) but should be called only after `game.over = 1`, not during search.
