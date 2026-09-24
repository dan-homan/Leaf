#!/usr/bin/env python3
"""Score psqt_noise_tderr.sh: is the TD gradient more COHERENT under a shared,
directional perturbation than under clean play?

The perturbation scales the positional part of each (piece type, PSQT bucket)
group by k = 1 + eps.  The derivative of the white-POV eval with respect to
that k is EXACTLY the piece's positional PSQT contribution in the leaf's
bucket, sum over its active entries of (w - m) / 2 * 100/5776, own-perspective
entries minus enemy-perspective ones -- so the learner's TD gradient along all
48 pattern directions is computable exactly from the leaf dump:

    g_t = e_t * d_t (1 - d_t) / K * c_t        (c_t = 48 pattern contributions)
    G   = sum_t g_t over a game                  (ascent direction on k)

e_t is the learner's TD(lambda) error, reconstructed as in
eval_noise_tderr_score.py (clip, lambda^dply, terminal).  Labels are the
learner's clean refreshed ones, and c_t uses the CLEAN weights -- the
learner's view.

Columns
  rec/g     trainable records per game
  rms|G|    per-game gradient magnitude along the 48 patterns
  R50       batch coherence |sum G|^2 / sum |G|^2 over consecutive 50-game
            batches (the learner's batch size).  1 = the games' pattern
            gradients are mutually independent; n = perfectly aligned.
  chi2/48   mean z^2 of the whole-arm mean gradient over the 48 patterns: how
            much SYSTEMATIC direction the arm's games agree on.  ~1 = none.
  align     z of the mean projection of G onto the arm's own eps direction
            (for the clean arm: onto each perturbed arm's eps, a null).
            Nonzero = the learner is pushed along (+) or against (-) the
            hypothesis the actors played.
Errors: R50 1 sigma across batches.
"""
import glob, math, os, sys
from collections import defaultdict
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import compare_nnue_learning as C
import psqt_decomp as D

K, LAM, CLIP = 220.0, 0.985, 100.0
BATCH = 50


def load_ref(path):
    m = np.zeros((11, 8))
    for l in open(path):
        if l[0] == '#':
            continue
        a, b, v = l.split()
        m[int(a), int(b)] = float(v)
    return m


def load_eps(arm):
    p = os.path.join(arm, "eps.txt")
    if not os.path.exists(p):
        return None
    e = np.zeros(48)
    for l in open(p):
        if l.startswith("PSQT noise eps"):
            pc = "PNBRQK".index(l.split()[3][0])
            e[pc * 8:(pc + 1) * 8] = [float(x) for x in l.split()[4:]]
    return e


def side_a_elo(arm):
    """Side A's score from the arm's PGNs (two-sided arms only): Elo and a
    pentanomial 1-sigma error over the opening pairs (consecutive Rounds of
    one actor file play the same opening with colours swapped)."""
    import re
    pairs, flat = [], []
    for p in sorted(glob.glob(os.path.join(arm, "pgn", "*.pgn"))):
        txt = open(p, errors="replace").read()
        sc = []
        for w, b, r in re.findall(r'\[White "([^"]+)"\]\n\[Black "([^"]+)"\]\n\[Result "([^"]+)"\]', txt):
            if r == "*" or "hypA" not in (w, b):
                sc.append(None); continue
            v = {"1-0": 1.0, "0-1": 0.0}.get(r, 0.5)
            sc.append(v if w == "hypA" else 1.0 - v)
        for i in range(0, len(sc) - 1, 2):
            if sc[i] is not None and sc[i + 1] is not None:
                pairs.append(sc[i] + sc[i + 1])
        flat += [x for x in sc if x is not None]
    if not pairs:
        return None
    x = np.array(pairs) / 2.0
    mu, sd = x.mean(), x.std(ddof=1) / math.sqrt(len(x))
    elo = lambda q: -400 * math.log10(1 / min(max(q, 1e-6), 1 - 1e-6) - 1)
    return len(flat), 100 * np.mean(flat), elo(mu), (elo(mu + sd) - elo(mu - sd)) / 2


def games_of(arm):
    g = defaultdict(list)
    for p in glob.glob(os.path.join(arm, "dump.*.leaf.tsv")):
        for line in open(p):
            if line[0] in "#f":
                continue
            c = line.rstrip("\n").split("\t")
            g[int(c[5])].append((int(c[3]), int(c[1]), float(c[2]), c[0]))
    return [g[k] for k in sorted(g)]          # gid order = consumption order


def contributions(fens, Wd):
    """c[i, p*8+b]: white-POV positional PSQT contribution of piece type p."""
    rows, cols, vals = [], [], []
    for i, fen in enumerate(fens):
        fs, stm, b = D.features(fen)
        for persp, sign in ((1, 1.0), (0, -1.0)):      # white persp minus black
            for f in fs[persp]:
                pl = (f % 704) // 64
                p = 5 if pl == 10 else pl // 2
                rows.append(i); cols.append(p * 8 + b); vals.append(sign * Wd[f, b])
    c = np.zeros((len(fens), 48))
    np.add.at(c, (np.array(rows), np.array(cols)), np.array(vals))
    return c / 2 * D.CP


def game_gradient(recs, Wd):
    recs.sort()
    cp = np.array([r[1] for r in recs], float)
    d = 1 / (1 + np.exp(-cp / K))
    T = len(recs)
    e = np.zeros(T)
    e[-1] = recs[0][2] - d[-1]
    for t in range(T - 2, -1, -1):
        dd = d[t + 1] - d[t]
        dc = abs(cp[t + 1] - cp[t])
        if dc > CLIP:
            dd *= CLIP / dc
        dply = max(1, recs[t + 1][0] - recs[t][0])
        e[t] = dd + LAM ** dply * e[t + 1]
    c = contributions([r[3] for r in recs], Wd)
    return ((e * d * (1 - d) / K)[:, None] * c).sum(0), T


def main():
    work, ref, net, arms = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
    if not os.path.isabs(ref):
        ref = os.path.join(work, ref)
    if not os.path.isabs(net):
        net = os.path.join(work, net)
    W = C.read_nnue_ft(net)["psqt_w"].astype(np.float64)
    m = load_ref(ref)
    plane = (np.arange(D.NF) % 704) // 64
    Wd = W - m[plane]                                    # deviation from material
    eps = {a: load_eps(os.path.join(work, a)) for a in arms}
    dirs = [(a, e / np.linalg.norm(e)) for a, e in eps.items() if e is not None]

    print(f"\n{'arm':>14} {'games':>6} {'rec/g':>6} {'rms|G|':>10} {'R50':>13} "
          f"{'chi2/48':>8}  align (z) with eps of: " + " ".join(a for a, _ in dirs))
    for a in arms:
        gs = games_of(os.path.join(work, a))
        if not gs:
            print(f"{a:>14}  no data"); continue
        G = np.zeros((len(gs), 48)); n = 0
        for i, recs in enumerate(gs):
            G[i], t = game_gradient(recs, Wd); n += t
        nb = len(gs) // BATCH
        R = [np.sum(G[j*BATCH:(j+1)*BATCH].sum(0) ** 2) /
             np.sum(G[j*BATCH:(j+1)*BATCH] ** 2) for j in range(nb)]
        mu, se = G.mean(0), G.std(0, ddof=1) / math.sqrt(len(gs))
        chi = np.mean((mu / se) ** 2)
        al = []
        for _, u in dirs:
            s = G @ u
            al.append(s.mean() / (s.std(ddof=1) / math.sqrt(len(s))))
        print(f"{a:>14} {len(gs):>6} {n/len(gs):>6.1f} {np.sqrt((G**2).sum(1).mean()):>10.3e} "
              f"{np.mean(R):>6.3f}±{np.std(R, ddof=1)/math.sqrt(nb):<6.3f} {chi:>8.2f}  "
              + " ".join(f"{z:+6.2f}" for z in al))
        if eps[a] is not None:
            print(f"{'':>14} eps sd {eps[a].std():.3f}; corr(mean G, eps) over 48 patterns "
                  f"{np.corrcoef(mu, eps[a])[0, 1]:+.3f}")
        sa = side_a_elo(os.path.join(work, a))
        if sa:
            print(f"{'':>14} side A (+eps): {sa[0]} games, {sa[1]:.2f}%, "
                  f"Elo {sa[2]:+.1f} +- {sa[3]:.1f} (1 sigma, pentanomial over opening pairs)")
    print()


if __name__ == "__main__":
    main()
