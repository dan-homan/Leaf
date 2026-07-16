#!/usr/bin/env python3
"""Per-material-bucket label quality from a hybrid-loop corpus TSV.

Reads sampled rows from stdin (fen, cp_white, result, ply, depth, gid, endply).
Root rows (depth>0) carry depth-6 search scores; leaf rows depth==0.
Reports, per HalfKAv2_hm material bucket ((pieces-1)//4):
  - n, mean pieces
  - MSE(outcome) of sigmoid(cp/K) vs result
  - decisive-score conversion: P(advantaged side wins | |cp|>=150), draw rate there
  - remaining plies to game end (endply - ply) as a horizon measure
"""
import sys
import math
import numpy as np

K = 220.0

buckets = [dict(n=0, se=0.0, adv_n=0, adv_win=0, adv_draw=0,
                pieces=0, rem=0, n_leaf=0, se_leaf=0.0) for _ in range(8)]

for line in sys.stdin:
    if line.startswith('#') or line.startswith('fen'):
        continue
    parts = line.rstrip('\n').split('\t')
    if len(parts) < 7:
        continue
    fen, cp_s, res_s, ply_s, depth_s, gid, endply_s = parts[:7]
    board = fen.split(' ', 1)[0]
    pieces = sum(1 for ch in board if ch.isalpha())
    b = min(7, max(0, (pieces - 1) // 4))
    cp = int(cp_s); res = float(res_s)
    depth = int(depth_s)
    d = 1.0 / (1.0 + math.exp(-cp / K))
    B = buckets[b]
    if depth > 0:   # root row: search-score label
        B['n'] += 1
        B['se'] += (res - d) ** 2
        B['pieces'] += pieces
        B['rem'] += max(0, int(endply_s) - int(ply_s))
        if abs(cp) >= 150:
            B['adv_n'] += 1
            win = (res == 1.0) if cp > 0 else (res == 0.0)
            B['adv_win'] += 1 if win else 0
            B['adv_draw'] += 1 if res == 0.5 else 0
    else:           # leaf row: static-eval label
        B['n_leaf'] += 1
        B['se_leaf'] += (res - d) ** 2

print(f"{'bkt':>3} {'n_root(k)':>10} {'MSE(out)':>9} {'MSEleaf':>8} "
      f"{'|cp|>=150: n(k)':>15} {'conv%':>6} {'draw%':>6} {'loss%':>6} "
      f"{'rem_plies':>9}")
for b in range(8):
    B = buckets[b]
    if B['n'] == 0:
        continue
    mse = B['se'] / B['n']
    msel = B['se_leaf'] / B['n_leaf'] if B['n_leaf'] else float('nan')
    conv = 100.0 * B['adv_win'] / B['adv_n'] if B['adv_n'] else float('nan')
    drw = 100.0 * B['adv_draw'] / B['adv_n'] if B['adv_n'] else float('nan')
    loss = 100.0 - conv - drw if B['adv_n'] else float('nan')
    rem = B['rem'] / B['n']
    print(f"{b:>3} {B['n']/1e3:>10.1f} {mse:>9.4f} {msel:>8.4f} "
          f"{B['adv_n']/1e3:>15.1f} {conv:>6.1f} {drw:>6.1f} {loss:>6.1f} "
          f"{rem:>9.1f}")
