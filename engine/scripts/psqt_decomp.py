#!/usr/bin/env python3
# Leaf chess engine — training and analysis tooling.
# Copyright (C) 2026 Daniel C. Homan
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.  See the LICENSE file at the root of this repository.
"""Decompose a net's PSQT into MATERIAL and POSITIONAL parts, on real positions.

Every PSQT entry is (plane, psqt bucket) x (king bucket, square).  Within each
(plane, bucket) group the usage-weighted mean over (king bucket, square) is that
piece's MATERIAL value in that bucket; the deviation from it is the POSITIONAL
part, which splits further into a king-independent part (the per-square mean
over king buckets) and a king-dependent remainder.  Entries are weighted by how
often they are actually active in the supplied positions, so unreachable or
never-trained entries do not distort anything.

    python3 scripts/psqt_decomp.py learn/m260921-2.5e6g_final.nnue \\
        --positions 'learn/tderr_noise/td_s0/dump.*.root.tsv' --fc
    python3 scripts/psqt_decomp.py net_a.nnue net_b.nnue net_c.nnue \\
        --positions 'learn/<tag>_work/*root.tsv*'           # composition over time

POSITIONS are root-row TSVs (the TDLEAF_DUMP_TSV / train.py corpus format:
fen, cp, result, ply, depth, gid, endply, gate).  Consecutive plies of a game
are used for the per-move numbers, so whole games are sampled (--games).

--fc      Also split the WHOLE static eval into PSQT and FC parts.  Column 8
          ("gate") of a root row is the root STATIC eval of the net that
          dumped it, so this is only valid when the TSV was dumped by the
          (single) net given -- e.g. a frozen dump of that net, or the corpus
          of the leg that produced it (approximately: the learner's weights
          move during a leg).  Without --fc only PSQT quantities are reported,
          which are valid for any net on any positions.
--write-ref FILE  Write the usage-weighted material reference the engine's
          --psqt-noise needs (--psqt-noise-ref FILE).
--detail  Print the per-(plane, bucket) table for every net (default: only
          when a single net is given).

Reading it (m260921-2.5e6g, 2026-09-24): PSQT positional sd 88 cp across
positions and 35 cp per quiet move, FC 78 cp per quiet move, the two nearly
uncorrelated -- so on quiet moves ~17% of the positional variance is PSQT and
~83% FC.  Also reported: how far a PLAIN mean over reachable entries sits from
the usage-weighted material, which is the error an in-engine perturbation that
scales deviations from a plain mean would leak into material.
"""
import argparse, glob, os, random, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare_nnue_learning as C

PS = 704
KB = [28, 29, 30, 31, 31, 30, 29, 28, 24, 25, 26, 27, 27, 26, 25, 24,
      20, 21, 22, 23, 23, 22, 21, 20, 16, 17, 18, 19, 19, 18, 17, 16,
      12, 13, 14, 15, 15, 14, 13, 12, 8, 9, 10, 11, 11, 10, 9, 8,
      4, 5, 6, 7, 7, 6, 5, 4, 0, 1, 2, 3, 3, 2, 1, 0]
PT = {'p': 1, 'n': 2, 'b': 3, 'r': 4, 'q': 5, 'k': 6}
PLANES = ['own P', 'enm P', 'own N', 'enm N', 'own B', 'enm B',
          'own R', 'enm R', 'own Q', 'enm Q', 'kings']
CP = 100.0 / 5776.0          # nnue.cpp: score = (psqt_diff/2 + pos) * 100/5776
NF = 32 * PS


def features(fen):
    """HalfKAv2_hm feature lists per perspective (nnue.cpp halfkav2_feature)."""
    board, stm = fen.split()[:2]
    pcs = []
    for i, row in enumerate(board.split('/')):
        r, f = 7 - i, 0
        for ch in row:
            if ch.isdigit():
                f += int(ch); continue
            pcs.append((PT[ch.lower()], 1 if ch.isupper() else 0, r * 8 + f)); f += 1
    ks = {s: q for t, s, q in pcs if t == 6}
    out = []
    for persp in (0, 1):                       # BLACK=0, WHITE=1
        flip = 56 if persp == 0 else 0
        ksf = ks[persp] ^ flip
        orient = 7 if (ksf & 7) < 4 else 0
        out.append([KB[ksf] * PS + (640 if t == 6 else (t - 1) * 128 + (0 if s == persp else 64))
                    + ((q ^ flip) ^ orient) for t, s, q in pcs])
    return out, (1 if stm == 'w' else 0), (len(pcs) - 1) // 4


def _rows(path):
    import gzip
    op = gzip.open if path.endswith('.gz') else open
    with op(path, 'rt', errors='replace') as fh:
        for line in fh:
            if line[0] in '#f':
                continue
            c = line.rstrip('\n').split('\t')
            if len(c) >= 8:
                yield c


def load_positions(pattern, n_games, seed):
    """Whole-game sample.  Two passes -- game ids first, then only the sampled
    games' rows -- so a full leg corpus (tens of millions of rows) never has
    to fit in memory."""
    files = sorted({f for x in pattern for f in glob.glob(x)
                    if f.endswith('.tsv') or f.endswith('.tsv.gz')})
    gids = sorted({f + ':' + c[5] for f in files for c in _rows(f)})
    if not gids:
        sys.exit("no positions found")
    random.seed(seed)
    keep = set(random.sample(gids, n_games)) if n_games and n_games < len(gids) else set(gids)
    rows = [(f + ':' + c[5], int(c[3]), c[0], int(c[7]))
            for f in files for c in _rows(f) if f + ':' + c[5] in keep]
    rows.sort()
    return rows, len(keep)


class Positions:
    """Features precomputed once, then every net is a vectorised gather."""
    def __init__(self, rows):
        own, opp, bkt, sgn, static, key = [], [], [], [], [], []
        self.use = np.zeros((NF, 8))
        for gid, ply, fen, st in rows:
            fs, stm, b = features(fen)
            own.append(fs[stm]); opp.append(fs[1 - stm])
            bkt.append(b); sgn.append(1 if stm == 1 else -1)
            static.append(st); key.append((gid, ply))
            for f in fs[0] + fs[1]:
                self.use[f, b] += 1
        self.bkt = np.array(bkt); self.sgn = np.array(sgn); self.static = np.array(static, float)
        self._pack(own, 'own'); self._pack(opp, 'opp')
        self.n = len(rows)
        pairs = [i for i in range(1, self.n)
                 if key[i][0] == key[i - 1][0] and key[i][1] == key[i - 1][1] + 1]
        self.pairs = np.array(pairs, dtype=int)

    def _pack(self, lists, name):
        lens = np.array([len(x) for x in lists])
        setattr(self, name + '_f', np.concatenate([np.array(x) for x in lists]))
        setattr(self, name + '_row', np.repeat(np.arange(len(lists)), lens))
        setattr(self, name + '_b', np.repeat(self.bkt, lens))

    def white_cp(self, T):
        """PSQT eval (cp, white POV) of every position under table T[NF,8]."""
        s = np.zeros(self.n)
        np.add.at(s, self.own_row, T[self.own_f, self.own_b])
        np.add.at(s, self.opp_row, -T[self.opp_f, self.opp_b])
        return s / 2 * CP * self.sgn


def decompose(W, use):
    """Material (usage-weighted group mean) and king-independent tables."""
    plane = (np.arange(NF) % PS) // 64
    sq = np.arange(NF) % 64
    mat = np.zeros_like(W); kind = np.zeros_like(W); plain = np.zeros_like(W)
    table = []; ref = {}
    for pl in range(11):
        idx = np.where(plane == pl)[0]
        reach = idx[~((pl <= 1) & ((sq[idx] < 8) | (sq[idx] >= 56)))]   # no pawns on ranks 1/8
        for b in range(8):
            u, w = use[idx, b], W[idx, b]
            pm = W[reach, b].mean()
            plain[idx, b] = pm
            ref[(pl, b)] = pm                  # unused group: any reference is exact
            if u.sum() == 0:
                continue
            m = (u * w).sum() / u.sum()
            mat[idx, b] = m
            ref[(pl, b)] = m
            num = np.bincount(sq[idx], weights=u * w, minlength=64)
            den = np.bincount(sq[idx], weights=u, minlength=64)
            ki = np.where(den > 0, num / np.maximum(den, 1e-12), m)
            kind[idx, b] = ki[sq[idx]]
            sd = np.sqrt((u * (w - m) ** 2).sum() / u.sum())
            sdk = np.sqrt((u * (ki[sq[idx]] - m) ** 2).sum() / u.sum())
            table.append((pl, b, int(u.sum()), m * CP, sd * CP, sdk * CP, (pm - m) * CP))
    return mat, kind, plain, table, ref


def report(name, W, P, fc, detail):
    mat, kind, plain, table, _ = decompose(W, P.use)
    print(f"\n=== {name} ===")
    if detail:
        print(f"{'plane':>6} {'bkt':>3} {'uses':>9} {'mat cp':>8} {'pos sd':>7} "
              f"{'k-indep':>8} {'plain-mat':>9}")
        for pl, b, u, m, sd, sdk, pd in table:
            print(f"{PLANES[pl]:>6} {b:>3} {u:>9} {m:>8.1f} {sd:>7.1f} {sdk:>8.1f} {pd:>+9.1f}")
    # bucket-averaged material per piece: (own - enemy)/2, usage weighted
    print("material (cp, usage-weighted over buckets, (own-enemy)/2):  " + "  ".join(
        f"{p} {np.average([(t[3] - e[3]) / 2 for t, e in zip([x for x in table if x[0] == 2*i], [x for x in table if x[0] == 2*i+1])], weights=[t[2] for t in table if t[0] == 2*i]):.0f}"
        for i, p in enumerate("PNBRQ")))
    leak = np.sqrt(np.average([t[6] ** 2 for t in table], weights=[t[2] for t in table]))
    print(f"plain-mean vs usage-weighted material: rms {leak:.1f} cp over groups "
          f"(max |{max(abs(t[6]) for t in table):.1f}|)")
    tot = P.white_cp(W); matp = P.white_cp(mat); kip = P.white_cp(kind)
    pos = tot - matp
    parts = [("PSQT total", tot), ("PSQT material", matp), ("PSQT positional", pos),
             ("  king-indep", kip - matp), ("  king-dep", tot - kip)]
    if fc:
        parts = [("static (whole eval)", P.static)] + parts + [("FC (static-PSQT)", P.static - tot)]
    d = P.pairs
    quiet = np.abs(matp[d] - matp[d - 1]) < 1
    print(f"{'':22s} {'sd pos':>7} {'sd move':>8} {'sd quiet':>9} {'med |quiet|':>11}")
    for n, v in parts:
        dv = v[d] - v[d - 1]
        print(f"{n:22s} {v.std():7.1f} {dv.std():8.1f} {dv[quiet].std():9.1f} "
              f"{np.median(np.abs(dv[quiet])):11.1f}")
    if fc:
        a = (pos[d] - pos[d - 1])[quiet]; b = ((P.static - tot)[d] - (P.static - tot)[d - 1])[quiet]
        va, vb = a.var(), b.var()
        print(f"quiet-move positional variance: PSQT {100*va/(va+vb):.0f}% / FC {100*vb/(va+vb):.0f}%"
              f"   corr(PSQT pos, FC) {np.corrcoef(a, b)[0,1]:+.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("nets", nargs="+", help=".nnue file(s)")
    ap.add_argument("--positions", nargs="+", required=True,
                    help="root-row TSV glob(s) (.tsv or .tsv.gz)")
    ap.add_argument("--games", type=int, default=2500,
                    help="whole games to sample (default 2500, ~330k positions)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--fc", action="store_true",
                    help="split the static into PSQT and FC (TSV must be dumped by the single net given)")
    ap.add_argument("--detail", action="store_true")
    ap.add_argument("--write-ref", metavar="FILE",
                    help="write the usage-weighted material reference (88 lines: "
                         "plane bucket mean, raw PSQT units) that the engine's "
                         "--psqt-noise-ref needs.  Needs exactly one net.")
    a = ap.parse_args()
    if a.fc and len(a.nets) > 1:
        sys.exit("--fc needs exactly one net: the TSV's static column belongs to one net")
    if a.write_ref and len(a.nets) > 1:
        sys.exit("--write-ref needs exactly one net")
    rows, ng = load_positions(a.positions, a.games, a.seed)
    P = Positions(rows)
    print(f"positions {P.n} from {ng} games; {len(P.pairs)} consecutive-ply pairs")
    if a.write_ref:
        W = C.read_nnue_ft(a.nets[0])["psqt_w"].astype(np.float64)
        ref = decompose(W, P.use)[4]
        with open(a.write_ref, "w") as fh:
            fh.write(f"# psqt material reference (usage-weighted), net {os.path.basename(a.nets[0])}, "
                     f"{P.n} positions from {ng} games\n# plane bucket mean_raw\n")
            for (pl, b), m in sorted(ref.items()):
                fh.write(f"{pl} {b} {m:.3f}\n")
        print(f"wrote {a.write_ref}")
    for n in a.nets:
        W = C.read_nnue_ft(n)["psqt_w"].astype(np.float64)
        report(os.path.basename(n), W, P, a.fc, a.detail or len(a.nets) == 1)


if __name__ == "__main__":
    main()
