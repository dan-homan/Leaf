# NNUE Implementation — Historical Record

This document preserves the implementation history of Leaf's NNUE port: the
file additions from the original port and the performance-optimization /
correctness-fix history that followed, including NPS benchmarks and early
match results. See [`../NNUE.md`](../NNUE.md) for the current, living
architecture reference (feature set, network layout, quantization, and score
formula).

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
