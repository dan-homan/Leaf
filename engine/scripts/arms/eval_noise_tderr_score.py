#!/usr/bin/env python3
"""Score eval_noise_tderr.sh: reconstruct the learner's TD errors per sigma.

Input: the frozen learner's wide leaf dump (every leaf_ok record, clean
refreshed labels) and root dump (every record) per arm.  The TD recursion is
tdleaf_accumulate_game() exactly -- white-POV sigmoid d = 1/(1+exp(-cp/K)),
score-change clip at 100 cp on each bootstrap delta, trace decay
lambda^dply, terminal e = result - d_T.  Omitted: the per-record ID-variance
weight (not dumped), which scales the gradient, not e.

Columns
  games / rec/g      games consumed, trainable (leaf_ok) records per game
  draw%              draw rate of the arm's games
  rms_e              rms of e_t, the multiplier on every record's gradient
  rms_g              rms of e_t * 4 d(1-d): e weighted by the sigmoid slope,
                     i.e. what actually reaches the weights (saturated
                     positions contribute little)
  rms_d              rms of the one-step clipped bootstrap delta
  d_pawn / d_other   rms_d split by whether the move(s) between the two
                     records changed the pawn structure.  The noise acts only
                     through pawn structure, so extra signal from it should
                     concentrate in d_pawn.
  slope              regression of e_t on (d_t - 0.5).  0 = calibrated to
                     the arm's own play; NEGATIVE = the net is overconfident
                     relative to how these games actually go, i.e. the trace
                     is pulling values toward what the NOISY player achieves
                     (the bias side of the trade)

Errors are 1 sigma from 20 blocks of games.
"""
import glob, math, os, sys
from collections import defaultdict

K, LAM, CLIP = 220.0, 0.985, 100.0
NBLK = 20


def pawn_sig(fen):
    board = fen.split(" ", 1)[0]
    out, r = [], 0
    for rank in board.split("/"):
        f = 0
        for ch in rank:
            if ch.isdigit():
                f += int(ch)
            else:
                if ch in "pP":
                    out.append((ch, r, f))
                f += 1
        r += 1
    return tuple(out)


def load(arm):
    games = defaultdict(list)            # gid -> [(ply, cp, result)]
    for p in glob.glob(os.path.join(arm, "dump.*.leaf.tsv")):
        for line in open(p):
            if line[0] in "#f":
                continue
            c = line.rstrip("\n").split("\t")
            games[c[5]].append((int(c[3]), int(c[1]), float(c[2])))
    pawns = {}                            # (gid, ply) -> pawn signature
    for p in glob.glob(os.path.join(arm, "dump.*.root.tsv")):
        for line in open(p):
            if line[0] in "#f":
                continue
            c = line.rstrip("\n").split("\t")
            pawns[(c[5], int(c[3]))] = pawn_sig(c[0])
    return games, pawns


def game_stats(gid, recs, pawns):
    recs.sort()
    T = len(recs)
    cp = [r[1] for r in recs]
    d = [1.0 / (1.0 + math.exp(-c / K)) for c in cp]
    res = recs[0][2]
    e = [0.0] * T
    e[-1] = res - d[-1]
    deltas = []
    for t in range(T - 2, -1, -1):
        dd = d[t + 1] - d[t]
        dc = abs(cp[t + 1] - cp[t])
        if dc > CLIP:
            dd *= CLIP / dc
        dply = max(1, recs[t + 1][0] - recs[t][0])
        e[t] = dd + (LAM ** dply) * e[t + 1]
        a, b = pawns.get((gid, recs[t][0])), pawns.get((gid, recs[t + 1][0]))
        deltas.append((dd, None if a is None or b is None else a != b))
    s = defaultdict(float)
    s["n"] = T
    s["draw"] = 1.0 if res == 0.5 else 0.0
    for t in range(T):
        x = d[t] - 0.5
        s["e2"] += e[t] ** 2
        s["g2"] += (e[t] * 4 * d[t] * (1 - d[t])) ** 2
        s["xe"] += x * e[t]
        s["xx"] += x * x
    for dd, pw in deltas:
        s["dn"] += 1; s["d2"] += dd * dd
        if pw is True:
            s["pn"] += 1; s["p2"] += dd * dd
        elif pw is False:
            s["on"] += 1; s["o2"] += dd * dd
    return s


def metrics(S):
    n = S["n"] or 1
    f = lambda a, b: math.sqrt(S[a] / S[b]) if S[b] else float("nan")
    return {
        "rec/g": S["n"] / S["games"],
        "draw%": 100 * S["draw"] / S["games"],
        "rms_e": f("e2", "n"), "rms_g": f("g2", "n"), "rms_d": f("d2", "dn"),
        "d_pawn": f("p2", "pn"), "d_other": f("o2", "on"),
        "pawn%": 100 * S["pn"] / (S["pn"] + S["on"]) if S["pn"] + S["on"] else float("nan"),
        "slope": S["xe"] / S["xx"] if S["xx"] else float("nan"),
    }


def score(arm):
    games, pawns = load(arm)
    blocks = [defaultdict(float) for _ in range(NBLK)]
    tot = defaultdict(float)
    for i, gid in enumerate(sorted(games)):
        s = game_stats(gid, games[gid], pawns)
        s["games"] = 1
        for k, v in s.items():
            tot[k] += v
            blocks[i % NBLK][k] += v
    if not tot["games"]:
        return None
    m = metrics(tot)
    per = [metrics(b) for b in blocks if b["games"]]
    err = {}
    for k in m:
        vals = [p[k] for p in per if not math.isnan(p[k])]
        if len(vals) > 1:
            mu = sum(vals) / len(vals)
            err[k] = math.sqrt(sum((v - mu) ** 2 for v in vals) / (len(vals) - 1) / len(vals))
    return int(tot["games"]), m, err


def main():
    work, sigmas = sys.argv[1], sys.argv[2:]
    cols = ["rec/g", "draw%", "rms_e", "rms_g", "rms_d", "d_pawn", "d_other", "pawn%", "slope"]
    print(f"\n{'sigma':>5} {'games':>6} " + " ".join(f"{c:>15}" for c in cols))
    base = None
    for s in sigmas:
        r = score(os.path.join(work, f"td_s{s}"))
        if r is None:
            print(f"{s:>5}  no data"); continue
        n, m, err = r
        cells = []
        for c in cols:
            p = 1 if c in ("rec/g", "draw%", "pawn%") else 4
            cells.append(f"{m[c]:>8.{p}f}±{err.get(c, float('nan')):<6.{p}f}"[:15].rjust(15))
        print(f"{s:>5} {n:>6} " + " ".join(cells))
        if base is None:
            base = m
        else:
            rel = "  ".join(f"{c} {100 * (m[c] / base[c] - 1):+.1f}%"
                            for c in ("rms_e", "rms_g", "rms_d", "d_pawn", "d_other"))
            print(f"{'':>13}vs sigma={sigmas[0]}: {rel}")
    print()


if __name__ == "__main__":
    main()
