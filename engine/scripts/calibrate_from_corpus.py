#!/usr/bin/env python3
# Leaf — GPL v3-or-later.  Copyright (C) 2026 Daniel C. Homan.
"""Calibrate K and lambda straight from a leg's raw root dump, by game stage.

Why this and not analyze_calibration.py
---------------------------------------
That script wants a PGN-derived parquet, its ``--stage`` means TRAINING stage
(net maturity), and its constants predate the current recipe (K=290,
lambda_decisive/draw = 0.8/0.5 per RECORD).  This one reads the raw
``*.root.tsv.gz`` dump directly -- which already carries cp, result, ply,
endply and the FEN -- and splits by GAME stage (material remaining).

Conventions, verified against tdleaf.cpp
---------------------------------------
``cp`` is WHITE POV (``cp_white = root_wtm ? score_root_stm : -score_root_stm``)
and ``result`` is the white-POV game score, constant within a game.  The model
the trainer uses is ``ev = 1/(1+exp(-cp/K))`` with ``TDLEAF_K = 220``, and the
offline target is ``w*outcome + (1-w)*ev`` with ``w = td_lambda^(endply-ply)``
(``nnue_batch_train.cpp:bt_target``).

The raw dump is gated at ``TDLEAF_DUMP_QUIET_CP`` (default 1000 = effectively
open), so it holds ALL positions, not just the quiet ones the trainer keeps.
``--quiet-cp`` re-cuts to the training population; the default keeps everything
so the two can be compared.

What can and cannot be identified
---------------------------------
**K is identified.**  It is a straight calibration: pick the K whose sigmoid
best predicts observed game scores.  Reported as the maximum-likelihood fit plus
a reliability table.

**Lambda is NOT identified by the data alone.**  The outcome and the eval both
estimate the same unobserved quantity V(s_t), so choosing between them needs a
bias/variance assumption.  Three measurements are reported, each stating the
assumption that turns it into a lambda:

  (A) corr(ev_t, outcome) against plies-to-end -- where the outcome stops being
      predictable, hence stops being informative.
  (B) corr(ev_t, ev_{t+k}) against lag k, fitted as lambda^k -- the eligibility
      trace's own decorrelation rate, on the GAME-PLY axis the trainer uses.
  (C) inverse-variance weight.  Treating outcome and eval as independent
      estimates of V(s_t), the optimal weight on the outcome is
      Var(e_t)/(Var(e_t)+Var(o_n)).  Var(outcome - ev_t) is measurable per
      plies-to-end bucket and is their sum; the split is what needs the
      assumption, so this is reported as the total and as the implied w under
      the stated split.

Usage
-----
    python3 calibrate_from_corpus.py --source m260916-5e6g_work --games 100000
"""

import argparse
import gzip
import math
import sys
from collections import defaultdict
from pathlib import Path

PIECE_CHARS = set("pnbrqkPNBRQK")


def log(m):
    print(m, file=sys.stderr, flush=True)


def find_dump(src, kind="root"):
    p = Path(src)
    if p.is_file():
        return p
    hits = sorted(p.glob(f"*.{kind}.tsv.gz")) + sorted(p.glob(f"*.{kind}.tsv"))
    if not hits:
        sys.exit(f"no *.{kind}.tsv[.gz] under {p}")
    return hits[0]


def board_counts(fen):
    """(total pieces incl. kings, pawns) from the board field of a FEN."""
    board = fen.split(" ", 1)[0]
    total = pawns = 0
    for c in board:
        if c in PIECE_CHARS:
            total += 1
            if c in "pP":
                pawns += 1
    return total, pawns


def read_games(path, max_games, quiet_cp):
    """Yield one game at a time as a list of (cp, result, ply, endply, np, npawn).

    Games are contiguous in the dump (the learner writes one at a time), which
    is what makes the lag analysis possible without holding the file in memory.
    """
    op = gzip.open(path, "rt") if str(path).endswith(".gz") else open(path)
    cur, rows, n = None, [], 0
    with op as fh:
        for line in fh:
            if line.startswith("#") or line.startswith("fen\t"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 8:
                continue
            gid = f[5]
            if gid != cur:
                if rows:
                    yield rows
                    n += 1
                    if n >= max_games:
                        return
                cur, rows = gid, []
            cp = int(f[1])
            if quiet_cp > 0 and abs(cp - int(f[7])) > quiet_cp:
                continue
            tot, pw = board_counts(f[0])
            rows.append((cp, float(f[2]), int(f[3]), int(f[6]), tot, pw))
    if rows and n < max_games:
        yield rows


def fit_K(hist, lo=40.0, hi=900.0):
    """K maximising sum[r*log p + (1-r)*log(1-p)], p = sigmoid(cp/K).

    Draws enter as r=0.5, i.e. half a win and half a loss, which is the usual
    convention for a score-predicting calibration.

    HIST is {cp: [n, sum_r]}.  cp is an integer, so histogramming is exact and
    turns the fit from O(positions) per iteration into O(distinct cp) -- the
    difference between minutes and milliseconds at 6M rows."""
    items = list(hist.items())

    def nll(K):
        s = 0.0
        for cp, (n, sr) in items:
            z = cp / K
            # log(sigmoid(z)) and log(1-sigmoid(z)), overflow-safe
            lp = -math.log1p(math.exp(-z)) if z > -700 else z
            lq = -math.log1p(math.exp(z)) if z < 700 else -z
            s -= sr * lp + (n - sr) * lq
        return s
    for _ in range(60):                      # golden-section
        m1 = lo + 0.382 * (hi - lo)
        m2 = lo + 0.618 * (hi - lo)
        if nll(m1) < nll(m2):
            hi = m2
        else:
            lo = m1
        if hi - lo < 0.5:
            break
    return 0.5 * (lo + hi)


def reliability(hist, K, edges):
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        n = pw = act = 0.0
        for cp, (c, sr) in hist.items():
            if a <= cp < b:
                n += c
                pw += c / (1.0 + math.exp(-cp / K))
                act += sr
        if n < 200:
            continue
        out.append((a, b, int(n), pw / n, act / n))
    return out


def acc_add(d, key, x, y):
    a = d[key]
    a[0] += 1; a[1] += x; a[2] += y
    a[3] += x * x; a[4] += y * y; a[5] += x * y


def acc_corr(a, minn=2000):
    """Pearson r from running sums.  Using sums rather than stored pairs is
    what lets every pair count: a per-cell cap would silently bias the sparse
    cells (high material late, low material early) that the cross-tab exists
    to compare."""
    n = a[0]
    if n < minn:
        return float("nan"), int(n)
    mx, my = a[1] / n, a[2] / n
    vx, vy = a[3] / n - mx * mx, a[4] / n - my * my
    cv = a[5] / n - mx * my
    if vx <= 0 or vy <= 0:
        return float("nan"), int(n)
    return cv / math.sqrt(vx * vy), int(n)


def acc_var_diff(a):
    """Var(y - x) and mean(y - x) from the running sums.

    With x = ev and y = outcome this is the quantity the reliability weighting
    needs: Var(outcome - ev) = sigma^2_outcome + sigma^2_eval, the two errors
    around the position's true value.  The SUM is measurable; the split is not,
    which is why the derived weights below state their assumption."""
    n = a[0]
    if n < 2:
        return float("nan"), float("nan"), 0
    mx, my = a[1] / n, a[2] / n
    m = my - mx
    v = (a[4] - 2.0 * a[5] + a[3]) / n - m * m
    return v, m, int(n)


def acc_lambda(d, key_prefix, lags, minn=2000):
    """Median lambda^(1/k) over lags for one cell."""
    lams = []
    nmax = 0
    for k in lags:
        r, n = acc_corr(d.get(key_prefix + (k,), [0.0] * 6), minn)
        if r == r and r > 0:
            lams.append(r ** (1.0 / k))
            nmax = max(nmax, n)
    if not lams:
        return float("nan"), 0
    return sorted(lams)[len(lams) // 2], nmax


def corr(xs, ys):
    n = len(xs)
    if n < 50:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    return sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="work dir or .tsv[.gz]")
    ap.add_argument("--games", type=int, default=100000)
    ap.add_argument("--quiet-cp", type=int, default=0,
                    help="re-cut to the training population (0 = keep all)")
    ap.add_argument("--K-current", type=float, default=220.0)
    ap.add_argument("--lambda-current", type=float, default=0.985)
    ap.add_argument("--max-lag", type=int, default=60)
    args = ap.parse_args()

    dump = find_dump(args.source)
    log(f"reading {dump.name} ({args.games:,} games, quiet_cp={args.quiet_cp})")

    def newhist():
        return defaultdict(lambda: [0, 0.0])
    allp = newhist()                             # cp -> [n, sum_result]
    by_pieces = defaultdict(newhist)
    by_pawns = defaultdict(newhist)
    npos = 0
    lag_pairs = defaultdict(lambda: ([], []))    # k -> (ev_t, ev_{t+k})
    lag_stage = defaultdict(lambda: [0.0] * 6)   # (stack, k)
    lag_ply   = defaultdict(lambda: [0.0] * 6)   # (plybucket, k)
    lag_cross = defaultdict(lambda: [0.0] * 6)   # (stack, plybucket, k)
    LAGS = []
    rel_mat   = defaultdict(lambda: [0.0] * 6)   # stack -> (ev, outcome)
    rel_cp    = defaultdict(lambda: [0.0] * 6)   # |cp| bucket
    rel_cross = defaultdict(lambda: [0.0] * 6)   # (stack, |cp| bucket)
    toend = defaultdict(lambda: ([], []))        # n bucket -> (ev, outcome)
    toend_stage = defaultdict(lambda: ([], []))  # (stage, n) -> (ev, outcome)
    ngames = 0

    K0 = args.K_current
    for rows in read_games(dump, args.games, args.quiet_cp):
        ngames += 1
        for cp, r, ply, endply, tot, pw in rows:
            npos += 1
            for h in (allp, by_pieces[min(max(tot - 1, 0) // 4, 7)],
                      by_pawns[min(pw // 2, 8)]):
                e = h[cp]
                e[0] += 1
                e[1] += r
        evs = [1.0 / (1.0 + math.exp(-cp / K0)) for cp, _, _, _, _, _ in rows]
        plies = [t[2] for t in rows]
        res = rows[0][1]
        end = rows[0][3]
        for i, (cp, r, ply, endply, tot, pw) in enumerate(rows):
            st_i = min(max(tot - 1, 0) // 4, 7)
            acp = abs(cp)
            cb = (0 if acp < 25 else 1 if acp < 50 else 2 if acp < 100 else
                  3 if acp < 200 else 4 if acp < 400 else 5)
            acc_add(rel_mat, st_i, evs[i], res)
            acc_add(rel_cp, cb, evs[i], res)
            acc_add(rel_cross, (st_i, cb), evs[i], res)
            n = end - ply
            nb = min(n // 20, 7)
            toend[nb][0].append(evs[i])
            toend[nb][1].append(res)
            toend_stage[(min(max(tot - 1, 0) // 4, 7), nb)][0].append(evs[i])
            toend_stage[(min(max(tot - 1, 0) // 4, 7), nb)][1].append(res)
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                k = plies[j] - plies[i]
                if k > args.max_lag:
                    break
                if k % 4:                       # subsample lags for speed
                    continue
                a, b = lag_pairs[k]
                if len(a) < 200000:
                    a.append(evs[i])
                    b.append(evs[j])
                # Same decorrelation, split by the material stage of the EARLIER
                # position.  Unlike the (A) stage table this is not confounded
                # with distance-to-end: it asks how fast the value signal decays
                # from a position of given material, not how far that position
                # happens to sit from the result.
                st = min(max(rows[i][4] - 1, 0) // 4, 7)
                pb = min(plies[i] // 20, 7)
                acc_add(lag_stage, (st, k), evs[i], evs[j])
                acc_add(lag_ply,   (pb, k), evs[i], evs[j])
                acc_add(lag_cross, (st, pb, k), evs[i], evs[j])

    log(f"{ngames:,} games, {npos:,} positions")

    print("=" * 74)
    print(f"CALIBRATION from {dump.name}")
    print(f"  {ngames:,} games / {npos:,} positions   "
          f"quiet_cp={args.quiet_cp or 'all'}")
    print(f"  current: K={args.K_current:g} cp, td_lambda={args.lambda_current:g}/game-ply")
    print("=" * 74)

    # ---------------- (1) K ----------------
    K = fit_K(allp)
    print(f"\n(1) SIGMOID TEMPERATURE K")
    print(f"    overall max-likelihood K = {K:6.1f} cp    "
          f"(current {args.K_current:g})")

    edges = [-2000, -600, -400, -300, -200, -150, -100, -50,
             0, 50, 100, 150, 200, 300, 400, 600, 2000]
    print(f"\n    reliability at the FITTED K={K:.0f}:")
    print(f"      {'cp bin':>14} {'n':>10} {'predicted':>10} {'actual':>8} {'err':>7}")
    for a, b, n, pred, act in reliability(allp, K, edges):
        print(f"      {a:>6}..{b:<6} {n:>10,} {pred:>10.3f} {act:>8.3f} "
              f"{act - pred:>+7.3f}")

    print(f"\n    K by NNUE MATERIAL STACK ((piece_count-1)/4, nnue.cpp:506):")
    print(f"      {'stack':>5} {'pieces':>9} {'n':>11} {'K_fit':>8} {'vs cur':>8}")
    for b in sorted(by_pieces):
        v = by_pieces[b]
        nv = sum(e[0] for e in v.values())
        if nv < 5000:
            continue
        kb = fit_K(v)
        print(f"      {b:>5} {b*4+1:>4}-{b*4+4:<4} {nv:>11,} {kb:>8.1f} "
              f"{kb - args.K_current:>+8.1f}")

    print(f"\n    K by PAWN COUNT:")
    print(f"      {'pawns':>10} {'n':>11} {'K_fit':>8} {'vs current':>11}")
    for b in sorted(by_pawns):
        v = by_pawns[b]
        nv = sum(e[0] for e in v.values())
        if nv < 5000:
            continue
        kb = fit_K(v)
        print(f"      {b*2:>4}-{b*2+1:<5} {nv:>11,} {kb:>8.1f} "
              f"{kb - args.K_current:>+11.1f}")

    # ---------------- (2) lambda ----------------
    print(f"\n(2) LAMBDA")
    print(f"\n    (A) corr(ev_t, outcome) by plies-to-end"
          f"   [outcome informativeness]")
    print(f"      {'plies to end':>14} {'n':>11} {'corr':>8} "
          f"{'Var(out-ev)':>12}")
    for b in sorted(toend):
        ev, out = toend[b]
        if len(ev) < 1000:
            continue
        c = corr(ev, out)
        m = sum(o - e for e, o in zip(ev, out)) / len(ev)
        var = sum((o - e - m) ** 2 for e, o in zip(ev, out)) / len(ev)
        print(f"      {b*20:>5}-{b*20+19:<8} {len(ev):>11,} {c:>8.3f} "
              f"{var:>12.4f}")

    print(f"\n    (B) corr(ev_t, ev_t+k) by lag k"
          f"   [trace decorrelation; lambda^k fit]")
    print(f"      {'lag k':>8} {'n':>10} {'corr':>8} {'implied lambda':>16}")
    fits = []
    for k in sorted(lag_pairs):
        a, b2 = lag_pairs[k]
        if len(a) < 5000 or k == 0:
            continue
        c = corr(a, b2)
        if c > 0:
            lam = c ** (1.0 / k)
            fits.append(lam)
            print(f"      {k:>8} {len(a):>10,} {c:>8.4f} {lam:>16.5f}")
    if fits:
        mid = sorted(fits)[len(fits) // 2]
        print(f"\n      median implied lambda = {mid:.5f} per game-ply "
              f"(current {args.lambda_current:g})")

    print(f"\n    (C) outcome weight w = lambda^(plies to end), current vs (A):")
    print(f"      {'plies to end':>14} {'w @ current':>12} {'corr(ev,out)':>13}")
    for b in sorted(toend):
        ev, out = toend[b]
        if len(ev) < 1000:
            continue
        n_mid = b * 20 + 10
        print(f"      {b*20:>5}-{b*20+19:<8} "
              f"{args.lambda_current ** n_mid:>12.3f} {corr(ev, out):>13.3f}")

    LAGS = [k for k in sorted(lag_pairs) if k > 0]

    print(f"\n    (B2) lambda by MATERIAL STACK   [decay between NEARBY "
          f"positions; the result never enters]")
    print(f"      {'stack':>5} {'pieces':>9} {'n_pairs':>11} {'lambda':>8} "
          f"{'vs current':>11}")
    mat = {}
    for st in range(8):
        lam, n = acc_lambda(lag_stage, (st,), LAGS)
        if lam == lam:
            mat[st] = lam
            print(f"      {st:>5} {st*4+1:>4}-{st*4+4:<4} {n:>11,} "
                  f"{lam:>8.5f} {lam - args.lambda_current:>+11.5f}")

    print(f"\n    (B3) lambda by GAME PLY   [same measure, split by progress "
          f"instead of material]")
    print(f"      {'ply':>10} {'n_pairs':>11} {'lambda':>8} {'vs current':>11}")
    ply = {}
    for pb in range(8):
        lam, n = acc_lambda(lag_ply, (pb,), LAGS)
        if lam == lam:
            ply[pb] = lam
            lab = f"{pb*20}-{pb*20+19}" if pb < 7 else "140+"
            print(f"      {lab:>10} {n:>11,} {lam:>8.5f} "
                  f"{lam - args.lambda_current:>+11.5f}")

    if mat and ply:
        sm = max(mat.values()) - min(mat.values())
        sp = max(ply.values()) - min(ply.values())
        print(f"\n      spread across material: {sm:.5f}")
        print(f"      spread across ply:      {sp:.5f}")

    print(f"\n    (B4) lambda CROSS-TAB: material stack (rows) x game ply "
          f"(cols).  Which one does lambda actually track?")
    print("      " + f"{'stack':>7}" + "".join(f"{pb*20:>9}+" for pb in range(8)))
    for st in range(8):
        cells = []
        for pb in range(8):
            lam, n = acc_lambda(lag_cross, (st, pb), LAGS, minn=3000)
            cells.append(f"{lam:>10.4f}" if lam == lam else f"{'-':>10}")
        if any(c.strip() != "-" for c in cells):
            print(f"      {st*4+1:>3}-{st*4+4:<3}" + "".join(cells))

    print(f"\n(3) OUTCOME RELIABILITY  [position-only: no N, no trajectory]")
    print(f"    Var(outcome-ev) = sigma^2_outcome + sigma^2_eval.  The sum is")
    print(f"    measured; 'rel weight' assumes sigma^2_eval is uniform across")
    print(f"    buckets, so w_b is proportional to 1/Var.")

    print(f"\n    by MATERIAL STACK:")
    print(f"      {'stack':>5} {'pieces':>9} {'n':>11} {'Var(o-ev)':>10} "
          f"{'bias':>8} {'corr':>7} {'rel weight':>11}")
    mv = {}
    for st in range(8):
        v, m, n = acc_var_diff(rel_mat.get(st, [0.0] * 6))
        r, _ = acc_corr(rel_mat.get(st, [0.0] * 6))
        if n >= 5000:
            mv[st] = v
    inv = {k: 1.0 / v for k, v in mv.items() if v > 0}
    tot_n = sum(rel_mat[k][0] for k in inv)
    norm = sum(rel_mat[k][0] * inv[k] for k in inv) / max(tot_n, 1)
    for st in range(8):
        if st not in mv:
            continue
        v, m, n = acc_var_diff(rel_mat[st])
        r, _ = acc_corr(rel_mat[st])
        print(f"      {st:>5} {st*4+1:>4}-{st*4+4:<4} {n:>11,} {v:>10.4f} "
              f"{m:>+8.4f} {r:>7.3f} {inv[st]/norm:>11.3f}")

    print(f"\n    by |cp| (the competing explanation):")
    print(f"      {'|cp|':>10} {'n':>11} {'Var(o-ev)':>10} {'bias':>8} "
          f"{'corr':>7}")
    for cb, lab in enumerate(['0-24', '25-49', '50-99', '100-199', '200-399', '400+']):
        v, m, n = acc_var_diff(rel_cp.get(cb, [0.0] * 6))
        r, _ = acc_corr(rel_cp.get(cb, [0.0] * 6))
        if n >= 5000:
            print(f"      {lab:>10} {n:>11,} {v:>10.4f} {m:>+8.4f} {r:>7.3f}")

    print(f"\n    Var(outcome-ev) CROSS-TAB: material (rows) x |cp| (cols)")
    print("      " + f"{'stack':>7}" + "".join(
        f"{l:>10}" for l in ['0-24', '25-49', '50-99', '100-199', '200-399', '400+']))
    for st in range(8):
        cells = []
        for cb in range(6):
            v, m, n = acc_var_diff(rel_cross.get((st, cb), [0.0] * 6))
            cells.append(f"{v:>10.4f}" if n >= 3000 else f"{'-':>10}")
        if any(c.strip() != "-" for c in cells):
            print(f"      {st*4+1:>3}-{st*4+4:<3}" + "".join(cells))

    print(f"\n    corr(ev_t, outcome) by STAGE x plies-to-end:")
    hdr = "      " + f"{'pieces':>8}" + "".join(
        f"{b*20:>7}+" for b in range(6))
    print(hdr)
    for st in range(8):
        cells = []
        for b in range(6):
            ev, out = toend_stage.get((st, b), ([], []))
            cells.append(f"{corr(ev, out):>8.3f}" if len(ev) >= 1000 else
                         f"{'-':>8}")
        if any(c.strip() != "-" for c in cells):
            print(f"      {st*4:>3}-{st*4+3:<4}" + "".join(cells))


if __name__ == "__main__":
    main()
