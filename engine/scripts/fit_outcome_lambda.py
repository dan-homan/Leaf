#!/usr/bin/env python3
"""
fit_outcome_lambda.py — WDL plan Phase 0 (docs/WDL_PLAN.md).

Fit the outcome-conditioned result-decay constants (lambda_dec, lambda_draw)
of the batch trainer's blended target

    y_t = w * z + (1 - w) * sigmoid(cp_t / K),      w = lambda_class^d,
    d   = plies to game end (endply - ply),  z = game outcome (white POV)

from an existing training corpus — no training runs required.

Method
------
Held-out cross-entropy of y against the outcome itself is circular (z is
inside the target, so lambda -> 1 always wins).  The target is instead
anchored on FUTURE EVALS: y_t should predict p_{t+h} = sigmoid(cp_{t+h}/K)
at horizon h plies ahead.  Per outcome class and distance bin, the
closed-form optimal blend weight is

    w*(d) = E[(p_fut - p_t)(z - p_t)] / E[(z - p_t)^2]

i.e. how much of the eval's future movement the final outcome already
explains at distance d, beyond the current eval.  A geometric no-intercept
fit w*(d) ~ lambda^d per class (matching the trainer's functional form,
w(0) = 1) yields lambda; a direct 1-D scan over lambda minimising
MSE[(y_t - p_fut)^2] per class confirms.  Both are reported at several
horizons plus a split-half (by game id) stability check.

Cross-checks in the same report:
  * corr(p_t, z) decay vs distance for decisive games and the normalised
    draw-bias decay E|p_t - 0.5| for draws (the analyze_calibration.py
    estimators, recomputed here).
  * eval reliability (binned p_t vs realised z) per distance band.
  * W/D/L rates per game — the draw-rate prior for WDL head init (l_d seed).

Input
-----
Root-corpus TSV as dumped by TDLEAF_DUMP_TSV (root rows carry search labels):
    fen \t cp \t result \t ply \t depth \t gid \t endply
cp is the root search score in centipawns, WHITE POV; result is White's
result (1 / 0.5 / 0); ply is the 1-based game ply; endply the final ply.
Lines starting with '#' are ignored.  Rows without an endply column are
rejected (the fit needs exact distance-to-end).

Usage
-----
  python3 fit_outcome_lambda.py corpus_r_dedup.tsv
  python3 fit_outcome_lambda.py corpus_r_dedup.tsv \
      --K 220 --sample 0.25 --horizons 8 16 32 \
      --plots wdl_phase0/ --out-json wdl_phase0/lambda_fit.json
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_K        = 220.0
DEFAULT_SAMPLE   = 0.25          # fraction of games kept
DEFAULT_HORIZONS = [8, 16, 32]   # plies ahead for the future-eval anchor
MAX_DIST         = 160           # largest distance-to-end analysed
DIST_BIN         = 4             # distance bin width (>= 2 smooths ply parity)
LAMBDA_GRID      = np.round(np.arange(0.900, 1.0001, 0.0025), 4)
MIN_BIN_ROWS     = 200           # skip w*(d) bins with fewer rows

PLY_KEY_MULT = 100_000           # gid*MULT + ply sort key (ply << MULT)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_corpus(path: Path, sample: float, seed: int) -> pd.DataFrame:
    print(f'Loading {path} ...', flush=True)
    df = pd.read_csv(
        path, sep='\t', comment='#', header=None,
        usecols=[1, 2, 3, 5, 6],
        names=['cp', 'result', 'ply', 'gid', 'endply'],
        dtype={'cp': np.int32, 'result': np.float32, 'ply': np.int32,
               'gid': np.int64, 'endply': np.int32},
        engine='c', na_filter=False)
    n_rows_all  = len(df)
    n_games_all = df['gid'].nunique()
    print(f'  {n_rows_all:,} rows, {n_games_all:,} games')

    if sample < 1.0:
        # Deterministic per-game sampling (whole games kept — the future-eval
        # pairing needs intact within-game sequences).
        h = (df['gid'].to_numpy(np.uint64) * np.uint64(2654435761)
             + np.uint64(seed)) % np.uint64(1 << 32)
        keep = (h.astype(np.float64) / float(1 << 32)) < sample
        df = df[keep]
        print(f'  sampled {sample:.2f} of games -> {len(df):,} rows, '
              f'{df["gid"].nunique():,} games')

    df = df.sort_values(['gid', 'ply'], kind='stable').reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Future-eval pairing
# ---------------------------------------------------------------------------

def pair_future(df: pd.DataFrame, horizon: int):
    """
    For each row, find the first later row of the SAME game at least
    `horizon` plies ahead.  Returns (row_index, future_p_index) as int arrays.
    Fully vectorised via searchsorted on a combined (gid, ply) key.
    """
    gid = df['gid'].to_numpy(np.int64)
    ply = df['ply'].to_numpy(np.int64)
    key = gid * PLY_KEY_MULT + ply
    tgt = gid * PLY_KEY_MULT + (ply + horizon)
    idx = np.searchsorted(key, tgt, side='left')
    n = len(df)
    valid = idx < n
    idx_c = np.minimum(idx, n - 1)
    valid &= (gid[idx_c] == gid)               # future row is same game
    valid &= (ply[idx_c] >= ply + horizon)     # actually >= horizon ahead
    rows = np.nonzero(valid)[0]
    return rows, idx[rows]


# ---------------------------------------------------------------------------
# Per-class lambda estimators
# ---------------------------------------------------------------------------

def wstar_by_distance(d, num, den):
    """
    Closed-form optimal blend weight per distance bin:
        w*(bin) = sum(num) / sum(den),  num = (p_fut-p)(z-p), den = (z-p)^2
    Returns (bin_centers, w*, counts).
    """
    bins = np.arange(0, MAX_DIST + DIST_BIN, DIST_BIN)
    which = np.digitize(d, bins) - 1
    nb = len(bins) - 1
    centers, ws, counts = [], [], []
    for b in range(nb):
        m = which == b
        n = int(m.sum())
        if n < MIN_BIN_ROWS:
            continue
        s_den = float(den[m].sum())
        if s_den <= 1e-9:
            continue
        centers.append(0.5 * (bins[b] + bins[b + 1]))
        ws.append(float(num[m].sum()) / s_den)
        counts.append(n)
    return (np.array(centers), np.array(ws), np.array(counts))


def fit_lambda_geometric(centers, ws, counts):
    """
    Weighted no-intercept fit of log w* = d log(lambda), matching the
    trainer's w = lambda^d (w(0) = 1).  Only bins with w* > 0 participate.
    """
    m = ws > 1e-4
    if m.sum() < 3:
        return float('nan')
    d = centers[m].astype(float)
    lw = np.log(ws[m])
    wt = counts[m].astype(float)
    lam_log = np.sum(wt * d * lw) / np.sum(wt * d * d)
    return float(np.exp(lam_log))


def scan_lambda_mse(d, z, p, p_fut):
    """
    Direct scan: for each lambda on the grid, MSE of
        y = lambda^d * z + (1 - lambda^d) * p   against  p_fut.
    Returns (grid, mse_array, argmin_lambda).
    """
    dz = z - p
    dv = p_fut - p
    mses = np.empty(len(LAMBDA_GRID))
    for i, lam in enumerate(LAMBDA_GRID):
        w = np.power(lam, d)
        r = w * dz - dv                      # (y - p_fut)
        mses[i] = float(np.mean(r * r))
    return LAMBDA_GRID, mses, float(LAMBDA_GRID[int(np.argmin(mses))])


def corr_decay_lambda(d, p, z, decisive: bool):
    """
    analyze_calibration.py-style cross-check.
    Decisive: corr(p, z) per distance bin, normalised to bin 0, geometric fit.
    Draw:     E|p - 0.5| per distance bin, normalised to the FAR end (raw
              decay toward the game end), geometric fit on the reversed axis.
    """
    bins = np.arange(0, MAX_DIST + DIST_BIN, DIST_BIN)
    which = np.digitize(d, bins) - 1
    nb = len(bins) - 1
    vals = np.full(nb, np.nan)
    cnts = np.zeros(nb)
    for b in range(nb):
        m = which == b
        n = int(m.sum())
        if n < MIN_BIN_ROWS:
            continue
        cnts[b] = n
        if decisive:
            pp, zz = p[m], z[m]
            pm = pp - pp.mean(); zm = zz - zz.mean()
            den = np.sqrt((pm * pm).sum() * (zm * zm).sum())
            if den > 1e-12:
                vals[b] = float((pm * zm).sum() / den)
        else:
            vals[b] = float(np.abs(p[m] - 0.5).mean())
    centers = 0.5 * (bins[:-1] + bins[1:])
    ok = ~np.isnan(vals)
    if ok.sum() < 3:
        return float('nan'), centers, vals
    if decisive:
        ref = vals[ok][0]
        if ref <= 1e-6:
            return float('nan'), centers, vals
        norm = vals / ref
        lam = fit_lambda_geometric(centers[ok], norm[ok], cnts[ok])
    else:
        ref = vals[ok][-1]
        if ref <= 1e-6:
            return float('nan'), centers, vals
        norm = vals / ref
        # decay toward the game end: reverse the axis so it is lambda^d form
        c = centers[ok]; v = norm[ok][::-1]
        lam = fit_lambda_geometric(c - c[0], v, cnts[ok][::-1])
    return lam, centers, vals


# ---------------------------------------------------------------------------
# Reliability table
# ---------------------------------------------------------------------------

def reliability(p, z, d, bands=((0, 20), (20, 60), (60, 160))):
    out = []
    edges = np.linspace(0.0, 1.0, 11)
    for lo, hi in bands:
        m = (d >= lo) & (d < hi)
        pp, zz = p[m], z[m]
        rows = []
        for i in range(10):
            bm = (pp >= edges[i]) & (pp < edges[i + 1])
            n = int(bm.sum())
            if n < 100:
                rows.append((0.5 * (edges[i] + edges[i + 1]), np.nan, n))
            else:
                rows.append((float(pp[bm].mean()), float(zz[bm].mean()), n))
        out.append(((lo, hi), rows))
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description='Fit outcome-conditioned result-decay lambdas '
                    '(WDL plan Phase 0).')
    ap.add_argument('corpus', type=Path,
                    help='Root-corpus TSV (fen cp result ply depth gid endply)')
    ap.add_argument('--K', type=float, default=DEFAULT_K,
                    help='Sigmoid temperature in cp (default 220)')
    ap.add_argument('--sample', type=float, default=DEFAULT_SAMPLE,
                    help='Fraction of games to load (default 0.25)')
    ap.add_argument('--horizons', type=int, nargs='+', default=DEFAULT_HORIZONS,
                    help='Future-eval anchor horizons in plies (default 8 16 32)')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--plots', type=Path, default=None,
                    help='Directory for diagnostic PNGs (optional)')
    ap.add_argument('--out-json', type=Path, default=None,
                    help='Write fitted constants + diagnostics as JSON')
    args = ap.parse_args()

    df = load_corpus(args.corpus, args.sample, args.seed)
    if (df['endply'] <= 0).any():
        sys.exit('ERROR: corpus has rows without a valid endply column — '
                 'the fit needs exact distance-to-end.')

    p_all = sigmoid(df['cp'].to_numpy(np.float64) / args.K)
    z_all = df['result'].to_numpy(np.float64)
    d_all = (df['endply'] - df['ply']).to_numpy(np.int64)

    # ---- W/D/L prior ------------------------------------------------------
    g = df.groupby('gid', sort=False)['result'].first()
    n_games = len(g)
    rate_w = float((g == 1.0).mean())
    rate_d = float((g == 0.5).mean())
    rate_l = float((g == 0.0).mean())
    print(f'\nGames: {n_games:,}   White W/D/L rates: '
          f'{rate_w:.4f} / {rate_d:.4f} / {rate_l:.4f}')
    print(f'Draw-rate prior for WDL head init (l_d seed): p_d = {rate_d:.4f}')

    # ---- split halves by gid (stability check) ----------------------------
    gid_arr = df['gid'].to_numpy(np.int64)
    half_b = (gid_arr % 2).astype(bool)

    results = {'K': args.K, 'n_games': n_games,
               'wdl_rates': [rate_w, rate_d, rate_l],
               'horizons': {}}

    for h in args.horizons:
        rows, fut = pair_future(df, h)
        d  = d_all[rows].astype(np.float64)
        z  = z_all[rows]
        p  = p_all[rows]
        pf = p_all[fut]
        keep = d <= MAX_DIST
        d, z, p, pf = d[keep], z[keep], p[keep], pf[keep]
        rowmask_b = half_b[rows][keep]

        print(f'\n=== horizon h = {h} plies   ({len(d):,} paired rows) ===')
        hres = {}
        for cls, cname in ((z != 0.5, 'decisive'), ((z == 0.5), 'draw')):
            dc, zc, pc, pfc = d[cls], z[cls], p[cls], pf[cls]
            num = (pfc - pc) * (zc - pc)
            den = (zc - pc) ** 2
            centers, ws, counts = wstar_by_distance(dc, num, den)
            lam_geo = fit_lambda_geometric(centers, ws, counts)
            grid, mses, lam_scan = scan_lambda_mse(dc, zc, pc, pfc)

            # split-half stability on the scan
            mb = rowmask_b[cls]
            _, _, lam_a = scan_lambda_mse(dc[~mb], zc[~mb], pc[~mb], pfc[~mb])
            _, _, lam_b = scan_lambda_mse(dc[mb],  zc[mb],  pc[mb],  pfc[mb])

            print(f'  {cname:9s}  n={len(dc):>9,}   '
                  f'lambda_geo={lam_geo:.4f}   lambda_scan={lam_scan:.4f}   '
                  f'halves: {lam_a:.4f} / {lam_b:.4f}')
            wtab = '    d:  ' + ' '.join(f'{c:>5.0f}' for c in centers[:12])
            wval = '    w*: ' + ' '.join(f'{w:>5.2f}' for w in ws[:12])
            print(wtab + '\n' + wval)
            hres[cname] = {
                'n': int(len(dc)),
                'lambda_geometric': lam_geo,
                'lambda_scan': lam_scan,
                'lambda_half_a': lam_a, 'lambda_half_b': lam_b,
                'wstar_d': centers.tolist(), 'wstar': ws.tolist(),
                'wstar_n': counts.tolist(),
                'scan_grid': grid.tolist(), 'scan_mse': mses.tolist(),
            }
        results['horizons'][h] = hres

    # ---- cross-check: correlation/bias decay estimators -------------------
    print('\n=== cross-check (analyze_calibration.py estimators) ===')
    dec = z_all != 0.5
    lam_cd, cd_c, cd_v = corr_decay_lambda(
        d_all[dec].astype(float), p_all[dec], z_all[dec], decisive=True)
    lam_db, db_c, db_v = corr_decay_lambda(
        d_all[~dec].astype(float), p_all[~dec], z_all[~dec], decisive=False)
    print(f'  decisive corr(p,z)-decay lambda : {lam_cd:.4f}')
    print(f'  draw     |p-0.5|-decay   lambda : {lam_db:.4f}')
    results['crosscheck'] = {'lambda_corr_decisive': lam_cd,
                             'lambda_bias_draw': lam_db}

    # ---- reliability ------------------------------------------------------
    print('\n=== eval reliability: mean sigmoid(cp/K) vs realised z ===')
    for (lo, hi), rows_ in reliability(p_all, z_all, d_all):
        print(f'  distance {lo:>3}-{hi:<3} plies:')
        line_p = '    p:  ' + ' '.join(
            f'{r[0]:>5.2f}' if not np.isnan(r[1]) else '    -' for r in rows_)
        line_z = '    z:  ' + ' '.join(
            f'{r[1]:>5.2f}' if not np.isnan(r[1]) else '    -' for r in rows_)
        print(line_p + '\n' + line_z)

    # ---- plots ------------------------------------------------------------
    if args.plots is not None:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        args.plots.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, len(args.horizons),
                                 figsize=(5 * len(args.horizons), 4),
                                 squeeze=False)
        for ax, h in zip(axes[0], args.horizons):
            for cname, col in (('decisive', 'tab:red'), ('draw', 'tab:blue')):
                r = results['horizons'][h][cname]
                dd = np.array(r['wstar_d']); ww = np.array(r['wstar'])
                ax.plot(dd, ww, 'o', ms=3, color=col, label=f'{cname} w*(d)')
                lam = r['lambda_scan']
                if not np.isnan(lam):
                    xx = np.linspace(0, MAX_DIST, 200)
                    ax.plot(xx, lam ** xx, '-', color=col, alpha=0.6,
                            label=f'{cname} λ={lam:.3f}')
            ax.set_title(f'h = {h} plies')
            ax.set_xlabel('plies to end'); ax.set_ylabel('blend weight w')
            ax.set_ylim(-0.05, 1.05); ax.legend(fontsize=8); ax.grid(alpha=0.3)
        fig.tight_layout()
        out = args.plots / 'wstar_lambda_fit.png'
        fig.savefig(out, dpi=130)
        print(f'\nPlot written: {out}')

    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, 'w') as f:
            json.dump(results, f, indent=1)
        print(f'JSON written: {args.out_json}')


if __name__ == '__main__':
    main()
