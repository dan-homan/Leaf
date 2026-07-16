#!/usr/bin/env python3
"""Per-material-bucket phase analysis of a seed -> post-online -> final chain.

Usage:
    python3 bucket_phase_analysis.py <seed.tdleaf.bin> <post_online.tdleaf.bin> <final.tdleaf.bin>

Breaks PSQT and FC-stack movement down by HalfKAv2_hm material bucket
(bucket = (piece_count-1)/4; 0 = 1-4 pieces = deep endgame, 7 = 29-32 =
opening), using the per-weight update counts persisted in the .tdleaf.bin to
separate exposure (updates per bucket) from per-update violence (|dw|/sqrt(n)).

For each bucket also reports how much of the ONLINE displacement the OFFLINE
phase reverses:
    proj = dot(dw_offline, dw_online) / dot(dw_online, dw_online)
negative proj = offline repairs/undoes online's movement in that bucket,
positive = offline confirms and extends it.

First used in Part 3 of docs/Online_Learning_Investigation.md (2026-07-16).
"""
import os
import sys
import numpy as np
import importlib.util

_here = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "cnl", os.path.join(_here, "compare_nnue_learning.py"))
cnl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cnl)


def common_rows(a, b, c):
    ia = {fi: k for k, fi in enumerate(a['ft_fi'])}
    ib = {fi: k for k, fi in enumerate(b['ft_fi'])}
    ic = {fi: k for k, fi in enumerate(c['ft_fi'])}
    fis = [fi for fi in a['ft_fi'] if fi in ib and fi in ic]
    ka = np.array([ia[fi] for fi in fis])
    kb = np.array([ib[fi] for fi in fis])
    kc = np.array([ic[fi] for fi in fis])
    return ka, kb, kc


def main(p_seed, p_online, p_final):
    seed = cnl.read_tdleaf_fc(p_seed)
    post = cnl.read_tdleaf_fc(p_online)
    fin  = cnl.read_tdleaf_fc(p_final)
    ka, kb, kc = common_rows(seed, post, fin)

    ps_s = seed['psqt_w'][ka]; ps_o = post['psqt_w'][kb]; ps_f = fin['psqt_w'][kc]
    cnt_s = seed['psqt_cnt'][ka].astype(np.int64)
    cnt_o = post['psqt_cnt'][kb].astype(np.int64)
    cnt_f = fin['psqt_cnt'][kc].astype(np.int64)

    d_on  = ps_o - ps_s          # online displacement
    d_off = ps_f - ps_o          # offline movement

    print(f"\n===== PSQT by material bucket "
          f"(bucket 0 = 1-4 pieces ... 7 = 29-32 pieces) =====")
    print(f"{'bkt':>3} {'upd_on(M)':>10} {'upd_off(M)':>10} "
          f"{'med|dw|on':>10} {'med|dw|off':>11} "
          f"{'on/upd':>8} {'off/upd':>8} "
          f"{'proj':>7} {'cos':>6}")
    for b in range(8):
        u_on  = (cnt_o[:, b] - cnt_s[:, b]).sum()
        u_off = (cnt_f[:, b] - cnt_o[:, b]).sum()
        don = d_on[:, b]; doff = d_off[:, b]
        med_on  = np.median(np.abs(don))
        med_off = np.median(np.abs(doff))
        # per-update violence: median |dw|/sqrt(updates) over rows updated in b
        m_on = cnt_o[:, b] - cnt_s[:, b]
        m_off = cnt_f[:, b] - cnt_o[:, b]
        sel_on = m_on > 0; sel_off = m_off > 0
        per_on  = np.median(np.abs(don[sel_on])  / np.sqrt(m_on[sel_on]))  if sel_on.any() else 0
        per_off = np.median(np.abs(doff[sel_off]) / np.sqrt(m_off[sel_off])) if sel_off.any() else 0
        denom = float(np.dot(don, don))
        proj = float(np.dot(doff, don)) / denom if denom > 0 else 0.0
        cos = float(np.dot(doff, don)) / (np.linalg.norm(doff) * np.linalg.norm(don) + 1e-12)
        print(f"{b:>3} {u_on/1e6:>10.1f} {u_off/1e6:>10.1f} "
              f"{med_on:>10.1f} {med_off:>11.1f} "
              f"{per_on:>8.2f} {per_off:>8.2f} "
              f"{proj:>+7.2f} {cos:>+6.2f}")

    print(f"\n----- FC stacks by bucket -----")
    print(f"{'bkt':>3} {'fc2b_on':>9} {'fc2b_off':>9} "
          f"{'fc0b med|dw| on':>16} {'off':>7} {'proj0':>7} "
          f"{'fc1b med|dw| on':>16} {'off':>7} {'proj1':>7}")
    for b in range(8):
        f2_on  = float(post['fc2_bias'][b][0] - seed['fc2_bias'][b][0])
        f2_off = float(fin['fc2_bias'][b][0] - post['fc2_bias'][b][0])
        d0on = (post['fc0_bias'][b] - seed['fc0_bias'][b]).astype(np.float64)
        d0off = (fin['fc0_bias'][b] - post['fc0_bias'][b]).astype(np.float64)
        d1on = (post['fc1_bias'][b] - seed['fc1_bias'][b]).astype(np.float64)
        d1off = (fin['fc1_bias'][b] - post['fc1_bias'][b]).astype(np.float64)
        p0 = float(np.dot(d0off, d0on)) / (float(np.dot(d0on, d0on)) + 1e-12)
        p1 = float(np.dot(d1off, d1on)) / (float(np.dot(d1on, d1on)) + 1e-12)
        print(f"{b:>3} {f2_on:>+9.0f} {f2_off:>+9.0f} "
              f"{np.median(np.abs(d0on)):>16.1f} {np.median(np.abs(d0off)):>7.1f} {p0:>+7.2f} "
              f"{np.median(np.abs(d1on)):>16.1f} {np.median(np.abs(d1off)):>7.1f} {p1:>+7.2f}")


if __name__ == '__main__':
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2], sys.argv[3])
