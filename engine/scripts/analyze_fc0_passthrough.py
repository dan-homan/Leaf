#!/usr/bin/env python3
"""
Per-row drift analysis for FC0 — isolates the passthrough row (output 15)
from the 15 regular rows.

Background: FC0 has 16 outputs.  Outputs 0..14 feed through SqrCReLU → FC1 → FC2.
Output 15 is the "passthrough" — its raw value bypasses FC1/FC2 and contributes
directly to the positional score via `fwdOut = fc0_raw[15] * 9600/8128`.
The passthrough row is zero-initialised by design to suppress eval noise at
game 1; any drift in row 15 produces a near-uniform additive bias on the
positional component of the score.

Reports per-row Δ statistics for each of the 8 layer stacks so you can see
whether the passthrough row is drifting disproportionately to the other rows.
Also prints FC2 bias drift (the other channel that can carry a global offset).

Usage:
    python3 analyze_fc0_passthrough.py <baseline.nnue> <trained.tdleaf.bin>
"""

import sys
import os
import numpy as np

# Reuse the file-format readers from compare_nnue_learning.py
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
from compare_nnue_learning import (  # noqa: E402
    read_nnue_fc, read_tdleaf_fc, L0_SIZE, L0_INPUT, N_STACKS,
)


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    nnue_path, tdleaf_path = sys.argv[1], sys.argv[2]

    print(f"Reading FC0 from {nnue_path} ...")
    base = read_nnue_fc(nnue_path)
    print(f"Reading FC0 from {tdleaf_path} ...")
    upd = read_tdleaf_fc(tdleaf_path)

    # Build deltas as int (post-trained - baseline).  Both come back as int8
    # for fc0_w; promote to int16 to avoid overflow on subtraction.
    print()
    print("=" * 76)
    print("FC0 weight drift, broken down by output row")
    print("=" * 76)
    print(f"  Rows 0..14: SqrCReLU → FC1 → FC2 (normal path)")
    print(f"  Row 15:     passthrough (fc0_raw[15] × 9600/8128 added directly to score)")
    print()

    # Aggregate across stacks: for each row, accumulate Δ stats.
    # Also keep per-stack breakdown for row 15.
    n_per_row = L0_INPUT
    row_totals = np.zeros((L0_SIZE,), dtype=np.float64)        # sum of Δ
    row_abs_totals = np.zeros((L0_SIZE,), dtype=np.float64)    # sum of |Δ|
    row_sqsum = np.zeros((L0_SIZE,), dtype=np.float64)         # sum of Δ²
    row_min = np.full((L0_SIZE,), +127, dtype=np.int16)
    row_max = np.full((L0_SIZE,), -127, dtype=np.int16)
    row_nonzero = np.zeros((L0_SIZE,), dtype=np.int64)
    row_count = np.zeros((L0_SIZE,), dtype=np.int64)

    passthrough_per_stack = []  # list of (stack, mean_delta, sum_delta, n_nonzero)

    for s in range(N_STACKS):
        b = base['fc0_w'][s].astype(np.int16)    # shape (16, 1024)
        t = upd['fc0_w'][s].astype(np.int16)
        d = t - b
        for row in range(L0_SIZE):
            drow = d[row]
            row_totals[row]     += drow.sum()
            row_abs_totals[row] += np.abs(drow).sum()
            row_sqsum[row]      += (drow.astype(np.float64) ** 2).sum()
            row_min[row]         = min(int(row_min[row]), int(drow.min()))
            row_max[row]         = max(int(row_max[row]), int(drow.max()))
            row_nonzero[row]    += int((drow != 0).sum())
            row_count[row]      += n_per_row
        # Passthrough row per-stack breakdown
        prow = d[15]
        passthrough_per_stack.append({
            'stack':       s,
            'mean':        float(prow.mean()),
            'sum':         int(prow.sum()),
            'abs_mean':    float(np.abs(prow).mean()),
            'nonzero':     int((prow != 0).sum()),
            'min':         int(prow.min()),
            'max':         int(prow.max()),
        })

    # Per-row aggregate table (across all 8 stacks)
    print(f"{'Row':>4}  {'N':>6}  {'%moved':>7}  {'mean Δ':>9}  {'|Δ| mean':>9}  "
          f"{'std Δ':>9}  {'min':>5}  {'max':>5}")
    for row in range(L0_SIZE):
        n = row_count[row]
        nz = row_nonzero[row]
        mean = row_totals[row] / n
        absmean = row_abs_totals[row] / n
        std = float(np.sqrt(row_sqsum[row] / n - mean ** 2))
        label = f"{row}" + ("*" if row == 15 else "")
        print(f"{label:>4}  {n:>6}  {100*nz/n:>6.1f}%  "
              f"{mean:>+9.4f}  {absmean:>9.4f}  {std:>9.4f}  "
              f"{int(row_min[row]):>5}  {int(row_max[row]):>5}")
    print("  *row 15 = passthrough (direct-to-score path)")

    # Passthrough per-stack
    print()
    print("=" * 76)
    print("Passthrough row (FC0 output 15) — per-stack breakdown")
    print("=" * 76)
    print(f"{'Stack':>5}  {'mean Δ':>9}  {'sum Δ':>9}  {'|Δ| mean':>9}  "
          f"{'min':>5}  {'max':>5}  {'%moved':>7}")
    for r in passthrough_per_stack:
        print(f"{r['stack']:>5}  {r['mean']:>+9.4f}  {r['sum']:>+9d}  "
              f"{r['abs_mean']:>9.4f}  {r['min']:>5}  {r['max']:>5}  "
              f"{100*r['nonzero']/L0_INPUT:>6.1f}%")

    # Compare passthrough drift magnitude to other rows
    print()
    pt_abs_mean   = row_abs_totals[15] / row_count[15]
    other_abs_mean = (row_abs_totals[:15].sum()) / (row_count[:15].sum())
    pt_mean        = row_totals[15] / row_count[15]
    other_mean     = (row_totals[:15].sum()) / (row_count[:15].sum())
    print(f"Row 15 (passthrough) |Δ| mean = {pt_abs_mean:.4f}")
    print(f"Rows 0..14 mean      |Δ| mean = {other_abs_mean:.4f}  "
          f"(ratio passthrough/others = {pt_abs_mean/max(other_abs_mean,1e-9):.2f}×)")
    print(f"Row 15 (passthrough) mean Δ   = {pt_mean:+.4f}  "
          f"(directional drift; non-zero indicates systematic bias)")
    print(f"Rows 0..14 mean      mean Δ   = {other_mean:+.4f}")

    # FC0 bias row 15 specifically
    print()
    print("=" * 76)
    print("FC0 bias — per-output drift (row 15 bias is the passthrough constant)")
    print("=" * 76)
    print(f"{'Output':>6}  ", end="")
    for s in range(N_STACKS):
        print(f"S{s:>2}Δ  ", end="")
    print(f"{'mean Δ':>9}")
    for o in range(L0_SIZE):
        deltas = []
        for s in range(N_STACKS):
            deltas.append(int(upd['fc0_bias'][s][o]) - int(base['fc0_bias'][s][o]))
        label = f"{o}" + ("*" if o == 15 else "")
        print(f"{label:>6}  " + "  ".join(f"{d:>+4d}" for d in deltas)
              + f"  {np.mean(deltas):>+9.4f}")
    print("  *row 15 = passthrough bias (fc0_raw[15] = bias + Σ w·acc; direct score add)")

    # FC2 bias drift (the other "global offset" channel)
    print()
    print("=" * 76)
    print("FC2 bias drift (one scalar per stack, added directly to fc2_raw)")
    print("=" * 76)
    deltas = []
    for s in range(N_STACKS):
        # fc2_bias appears to be 1 int32 per stack
        d = int(upd['fc2_bias'][s][0]) - int(base['fc2_bias'][s][0])
        deltas.append(d)
        print(f"  Stack {s}: Δ = {d:+d}")
    print(f"  Mean Δ across stacks: {np.mean(deltas):+.4f}  std: {np.std(deltas):.4f}")
    print(f"  All stacks negative?  {all(d < 0 for d in deltas)}")
    print(f"  All stacks positive?  {all(d > 0 for d in deltas)}")


if __name__ == "__main__":
    main()
