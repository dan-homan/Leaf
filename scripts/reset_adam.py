#!/usr/bin/env python3
"""
reset_adam.py -- zero (or decay) the Adam optimiser state in a .tdleaf.bin file.

Zeroing the Adam v (second-moment) and m (first-moment) arrays lets the
optimiser take full-sized steps again after training has plateaued due to
accumulated v values damping all updates.  All weight data (FC FP32 shadow
weights, FT weights, PSQT, FT biases, piece_val) is preserved exactly.

Usage:
    python3 scripts/reset_adam.py <file.tdleaf.bin> [options]

Options:
    --decay F     multiply v and m by F instead of zeroing (0 < F < 1,
                  e.g. 0.1 for a 'soft' reset that keeps directional hints)
    --out PATH    write result to PATH instead of overwriting the input
                  (input is always backed up to <file>.bak first)

Examples:
    # Full zero reset, overwrites in place (backup kept as .bak):
    python3 scripts/reset_adam.py learn/nn-fresh.tdleaf.bin

    # Soft reset — keep 10 % of accumulated v/m:
    python3 scripts/reset_adam.py learn/nn-fresh.tdleaf.bin --decay 0.1

    # Write to a new file, leave original untouched:
    python3 scripts/reset_adam.py learn/nn-fresh.tdleaf.bin --out learn/nn-fresh-reset.tdleaf.bin
"""

import argparse
import os
import shutil
import sys
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from merge_tdleaf import TDLeafFile  # reuse the tested v8 parser/writer


def parse_args():
    p = argparse.ArgumentParser(
        description="Zero or decay Adam optimizer state in a .tdleaf.bin file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("tdbin", help=".tdleaf.bin file to modify")
    p.add_argument("--decay", type=float, default=0.0,
                   help="multiply v and m by this factor instead of zeroing "
                        "(0 = full reset, 0 < F < 1 = partial decay)")
    p.add_argument("--out", default=None,
                   help="output path (default: overwrite input after backing up to .bak)")
    return p.parse_args()


def adam_stats(td):
    """Return (max_v, max_abs_m) across all FC arrays for reporting."""
    max_v = 0.0
    max_m = 0.0
    for fc in td.fc:
        for arr in (fc.v_l0_bias, fc.v_l0_weight, fc.v_l1_bias,
                    fc.v_l1_weight, fc.v_l2_bias, fc.v_l2_weight):
            v = float(arr.max()) if arr.size else 0.0
            if v > max_v:
                max_v = v
        for arr in (fc.m_l0_bias, fc.m_l0_weight, fc.m_l1_bias,
                    fc.m_l1_weight, fc.m_l2_bias, fc.m_l2_weight):
            m = float(np.abs(arr).max()) if arr.size else 0.0
            if m > max_m:
                max_m = m
    return max_v, max_m


def apply_reset(td, decay):
    """Zero or decay all Adam state in-place."""
    zero = np.float32(0.0)
    factor = np.float32(decay)

    # --- FC v and m ---
    for fc in td.fc:
        for arr in (fc.v_l0_bias, fc.v_l0_weight, fc.v_l1_bias,
                    fc.v_l1_weight, fc.v_l2_bias, fc.v_l2_weight):
            arr *= factor

        for arr in (fc.m_l0_bias, fc.m_l0_weight, fc.m_l1_bias,
                    fc.m_l1_weight, fc.m_l2_bias, fc.m_l2_weight):
            arr *= factor

    # --- FT bias v/m ---
    td.v_ft_bias   *= factor
    td.m_ft_bias   *= factor

    # --- piece_val v/m ---
    td.v_piece_val *= factor
    td.m_piece_val *= factor

    # --- sparse PSQT v/m ---
    if decay == 0.0:
        td.psqt_v_rows.clear()
        td.psqt_m_rows.clear()
    else:
        for fi in td.psqt_v_rows:
            td.psqt_v_rows[fi] *= factor
        for fi in td.psqt_m_rows:
            td.psqt_m_rows[fi] *= factor

    # --- sparse FT v (v8) ---
    # Zero case: drop all rows so ft_v_warmed is not set on load.
    # A warmed row with v=0 would produce bc2≈1 and v≈ε, giving ~10,000×
    # oversized FT steps — catastrophic. Dropping the rows is safe.
    if decay == 0.0:
        td.ft_v_rows.clear()
    else:
        for fi in td.ft_v_rows:
            td.ft_v_rows[fi] *= factor
        # After decay, some rows may become effectively zero — drop them to
        # avoid the bc2_warm catastrophe if decay is very small.
        min_nonzero = 1e-20
        td.ft_v_rows = {fi: v for fi, v in td.ft_v_rows.items()
                        if float(v.max()) > min_nonzero}

    # --- t_adam ---
    td.t_adam = 0


def main():
    args = parse_args()

    if not os.path.isfile(args.tdbin):
        sys.exit(f"Error: file not found: {args.tdbin}")
    if args.decay < 0.0 or args.decay >= 1.0:
        sys.exit(f"Error: --decay must be in [0, 1) (got {args.decay})")

    print(f"Loading {args.tdbin} ...", flush=True)
    td = TDLeafFile.load(args.tdbin)

    max_v_before, max_m_before = adam_stats(td)
    n_psqt_v = len(td.psqt_v_rows)
    n_psqt_m = len(td.psqt_m_rows)
    n_ft_v   = len(td.ft_v_rows)

    print(f"  Version   : v{td.version}")
    print(f"  t_adam    : {td.t_adam:,}")
    print(f"  FC max v  : {max_v_before:.4g}")
    print(f"  FC max |m|: {max_m_before:.4g}")
    print(f"  PSQT v rows: {n_psqt_v}  PSQT m rows: {n_psqt_m}  FT v rows: {n_ft_v}")

    mode = f"decay × {args.decay}" if args.decay > 0.0 else "full zero"
    print(f"\nApplying Adam reset ({mode}) ...", flush=True)
    apply_reset(td, args.decay)

    max_v_after, max_m_after = adam_stats(td)
    print(f"  t_adam    : {td.t_adam}")
    print(f"  FC max v  : {max_v_after:.4g}")
    print(f"  FC max |m|: {max_m_after:.4g}")
    print(f"  FT v rows : {len(td.ft_v_rows)}")

    out_path = args.out or args.tdbin
    if out_path == args.tdbin:
        bak = args.tdbin + ".bak"
        shutil.copy2(args.tdbin, bak)
        print(f"\nBackup written to {bak}")

    print(f"Writing {out_path} ...", flush=True)
    td.save(out_path)
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"Done. ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
