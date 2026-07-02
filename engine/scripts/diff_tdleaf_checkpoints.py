#!/usr/bin/env python3
"""Diff two .tdleaf.bin checkpoints section by section.

Usage: python3 diff_tdleaf_checkpoints.py old.tdleaf.bin new.tdleaf.bin
"""
import sys
import numpy as np

sys.path.insert(0, '../scripts')
import importlib.util
spec = importlib.util.spec_from_file_location("cnl", "../scripts/compare_nnue_learning.py")
cnl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cnl)

PIECES = ['PAWN', 'KNIGHT', 'BISHOP', 'ROOK', 'QUEEN', 'KING']


def load(path):
    return cnl.read_tdleaf_fc(path)


def sec_stats(name, old, new):
    o = np.concatenate([np.asarray(a, dtype=np.float64).ravel() for a in old])
    n = np.concatenate([np.asarray(a, dtype=np.float64).ravel() for a in new])
    d = n - o
    print(f"  {name:14s}  med|w|old={np.median(np.abs(o)):9.1f}  med|w|new={np.median(np.abs(n)):9.1f}  "
          f"mean_dw={d.mean():+9.2f}  med|dw|={np.median(np.abs(d)):8.2f}  max|dw|={np.abs(d).max():9.1f}")


def main(p_old, p_new):
    a = load(p_old)
    b = load(p_new)

    print(f"OLD: {p_old}   t_adam={a.get('t_adam')}")
    print(f"NEW: {p_new}   t_adam={b.get('t_adam')}")
    print()

    print("piece_val (raw units; cp = raw * 100/5776 * 0.5 per count-diff unit... shown raw and cp-equiv):")
    cp_per_unit = 0.5 * 100.0 / 5776.0 * 5776.0 / 100.0  # keep raw; also print engine cp = raw*100/5776/2? use script convention
    for i, name in enumerate(PIECES):
        if 'piece_val' in a and 'piece_val' in b:
            va, vb = a['piece_val'][i], b['piece_val'][i]
            # engine: piece_val contributes (pv_diff/2) * 100/5776 cp
            cpa = va * 0.5 * 100.0 / 5776.0
            cpb = vb * 0.5 * 100.0 / 5776.0
            print(f"  {name:7s} raw {va:10.1f} -> {vb:10.1f}   ({cpa:7.1f}cp -> {cpb:7.1f}cp,  d={cpb-cpa:+7.1f}cp)")
    print()

    print("FC sections (int-equivalent FP32 shadow space):")
    for key in ['fc0_bias', 'fc0_w', 'fc1_bias', 'fc1_w', 'fc2_bias', 'fc2_w']:
        # reload raw floats: read_tdleaf_fc rounds to ints; good enough for magnitude diffs
        sec_stats(key, a[key], b[key])
    print()

    if 'ft_bias_learned' in a and 'ft_bias_learned' in b:
        o, n = a['ft_bias_learned'], b['ft_bias_learned']
        d = n - o
        print(f"  ft_bias        med|w|old={np.median(np.abs(o)):9.1f}  med|w|new={np.median(np.abs(n)):9.1f}  "
              f"mean_dw={d.mean():+9.2f}  med|dw|={np.median(np.abs(d)):8.2f}  max|dw|={np.abs(d).max():9.1f}")

    # PSQT: match rows by feature index
    if a.get('n_ft_rows', 0) and b.get('n_ft_rows', 0):
        ia = {fi: k for k, fi in enumerate(a['ft_fi'])}
        common = [(ia[fi], k) for k, fi in enumerate(b['ft_fi']) if fi in ia]
        ka = np.array([c[0] for c in common]); kb = np.array([c[1] for c in common])
        po = a['psqt_w'][ka]; pn = b['psqt_w'][kb]
        d = pn - po
        print(f"  psqt ({len(common)} common rows of {a['n_ft_rows']}/{b['n_ft_rows']}):")
        print(f"                 med|w|old={np.median(np.abs(po)):9.1f}  med|w|new={np.median(np.abs(pn)):9.1f}  "
              f"mean_dw={d.mean():+9.2f}  med|dw|={np.median(np.abs(d)):8.2f}  max|dw|={np.abs(d).max():9.1f}")
        fo = a['ft_w'][ka]; fn = b['ft_w'][kb]
        d = fn - fo
        print(f"  ft_w  (common) med|w|old={np.median(np.abs(fo)):9.1f}  med|w|new={np.median(np.abs(fn)):9.1f}  "
              f"mean_dw={d.mean():+9.2f}  med|dw|={np.median(np.abs(d)):8.2f}  max|dw|={np.abs(d).max():9.1f}")


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
