#!/usr/bin/env python3
# Leaf chess engine — training and analysis tooling.
# Copyright (C) 2026 Daniel C. Homan
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.  See the LICENSE file at the root of this repository.
"""Per-leg summary for a training chain: Elo, decomposition, and canaries.

One table row per leg, combining the numbers that are scattered across the
sidecar JSON, the actor logs and the generation PGN.  Run it after each leg.

    python3 scripts/leg_summary.py m260921
    python3 scripts/leg_summary.py m260921 m260916 m260720     # compare chains
    python3 scripts/leg_summary.py m260921 --sample 20000      # quick, but see --sample

COLUMNS

  anc/Mg      ⭐ Leg gain on the FOREIGN ANCHOR, per million games.  THIS is
              the handover trigger, because §5 designates the foreign anchor as
              the figure of merit and says family matches are non-transitive in
              BOTH directions.  m260720 switched to d8 when this fell to **73**
              (2M cumulative) and its next two legs returned 122 and 177.
              Single readings are noisy (±10 per anchor match, so ±28/Mg on a
              500k leg) -- read the TREND over three or more legs.

  par/Mg      The same thing measured against the PARENT net.  Kept because it
              is tighter per leg, but it is NOT the trigger: at m260921's
              1.5e6 leg it read 166 while the anchor read 33.  A family match
              one leg apart shares blind spots and rewards style differences
              that do not transfer.  When the two disagree, the anchor wins.

  on / off    Online (tdleaf) and offline contributions to the leg, both
              measured against the parent.  ⚠️ These are two matches against a
              common opponent, which §5 says do not subtract; `off` is
              `total - on` and is indicative only.  Read the SHARE, not the
              value: at 2e5 the 4x-LR chain m260720 ran +19 online / +53
              offline while the reduced-LR chains ran ~+70 / ~+24.  A large
              offline share means the online phase is excursion-dominated,
              which is statements B and C.

  draw%       Self-play draw rate over the whole leg, from the actor logs
              (exact, free, and verified to agree with the generation PGN to
              0.01%).  The health canary: healthy is 35-40% at d8, and the
              level a chain settles at is set by search depth (§1 M).

  ply         Mean game length, from the generation PGN.

  depth       Mean RECORDED search depth, and the fraction below the --depth
              floor.  This is a CORRECTNESS canary, not a tuning one: below
              floor must read 0.0%.  Anything else means results are being
              recorded from an iteration other than the one that was searched
              — the defect that cost m260916 ~298 Elo of generation strength
              and made 45-56% of its labels shallower than requested
              (change_log 2026_09_21a).

  quiet       Gated root rows per game divided by mean ply: the fraction of
              plies that survive the quiet gate and become training rows.
              Fell 0.562 -> 0.499 across m260916 at constant depth.
"""
import argparse, glob, gzip, json, os, re, subprocess, sys

LEARN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "learn")


def sidecars(chain):
    out = []
    for f in glob.glob(os.path.join(LEARN, f"{chain}-*_final.json")):
        try:
            out.append(json.load(open(f)))
        except Exception:
            pass
    out.sort(key=lambda j: (j.get("cumulative_games", 0), j.get("tag", "")))
    return out


def vs(lst, pat):
    for e in lst or []:
        if pat in e.get("opponent", ""):
            return e["elo"], e["err"]
    return None, None


def draw_rate(tag):
    """Whole-leg W/D/L from the actor logs — exact and free."""
    pat = re.compile(r"selfplay: (\d+)/(\d+) games\s+\+(\d+) =(\d+) -(\d+)")
    W = D = L = 0
    for lg in glob.glob(os.path.join(LEARN, f"{tag}_work", "traj", "actor_*.log")):
        try:
            txt = open(lg, errors="replace").read()
        except Exception:
            continue
        for m in pat.finditer(txt):
            a, b, w, d, l = (int(x) for x in m.groups())
            if a == b:                      # completed generation block only
                W += w; D += d; L += l
    n = W + D + L
    return (100.0 * D / n, n) if n else (None, 0)


def pgn_stats(tag, sample):
    """Mean ply and recorded depth from the generation PGN (archive or loose)."""
    gz = glob.glob(os.path.join(LEARN, f"{tag}_work", "*gen.pgn.gz"))
    loose = sorted(glob.glob(os.path.join(LEARN, f"{tag}_work", "pgn", "*.pgn")))
    plies, depths, below = [], 0, 0
    nd = 0
    floor = None
    games = 0

    def feed(fh):
        nonlocal depths, below, nd, games, floor
        for line in fh:
            if line.startswith('[SearchBudget '):
                m = re.search(r'depth[ >=]*(\d+)', line)
                if m and floor is None:
                    floor = int(m.group(1))
            elif line.startswith('[PlyCount '):
                plies.append(int(line.split('"')[1])); games += 1
                if sample and games >= sample:
                    return True
            elif line and line[0] not in '[\n':
                for d in re.findall(r'\{[+-][0-9.]+/(\d+)', line):
                    d = int(d); depths += d; nd += 1
                    if floor is not None and d < floor:
                        below += 1
        return False

    try:
        if gz:
            with gzip.open(gz[0], "rt", errors="replace") as fh:
                feed(fh)
        else:
            for p in loose:
                with open(p, errors="replace") as fh:
                    if feed(fh):
                        break
    except Exception:
        pass
    if not plies or not nd:
        return None, None, None
    return sum(plies) / len(plies), depths / nd, 100.0 * below / nd


def root_rows(j):
    """Gated root rows for THIS leg — never the windowed corpus."""
    tag = j["tag"]
    for f in glob.glob(os.path.join(LEARN, f"{tag}_work", "*root.tsv*.rows.g60")):
        parts = open(f).read().split()
        if len(parts) >= 2 and int(parts[1]) > 0:
            return int(parts[1])
    if len(j.get("corpus_window") or []) <= 1 and j.get("corpus_rows"):
        return j["corpus_rows"]
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("chains", nargs="+")
    ap.add_argument("--sample", type=int, default=0, metavar="N",
                    help="read only the first N games of each generation PGN. "
                         "OFF by default because the head of a leg is NOT "
                         "representative: games shorten as the net improves, so "
                         "a 20k-game head read 175.6 mean ply against the leg's "
                         "true 161.1 (+9%%), and that bias propagates straight "
                         "into `quiet`.  Use it only for a quick look.  draw%% "
                         "always comes whole-leg from the actor logs and is "
                         "never affected.")
    args = ap.parse_args()
    sample = args.sample

    for chain in args.chains:
        legs = sidecars(chain)
        if not legs:
            print(f"\n{chain}: no sidecars found"); continue
        print(f"\n=== {chain} ===")
        print(f"{'leg':>7} {'cum':>9} {'d/nod':>7} {'anchor':>9} {'anc/Mg':>7} "
              f"{'par/Mg':>7} {'on':>7} {'off':>7} {'draw%':>7} {'ply':>6} "
              f"{'depth':>6} {'<floor':>7} {'quiet':>6}")
        prev_tag = None; prev_anc = None
        for j in legs:
            tag, cum = j["tag"], j.get("cumulative_games", 0)
            gi = j.get("games_this_iter", 0) or 0
            anc, _ = vs(j.get("final_gauntlet"), "classic")
            tot, _ = vs(j.get("final_gauntlet"), prev_tag) if prev_tag else (None, None)
            onl, _ = vs(j.get("tdleaf_gauntlet"), prev_tag) if prev_tag else (None, None)
            perM = (tot / (gi / 1e6)) if (tot is not None and gi) else None
            dan = (anc - prev_anc) if (anc is not None and prev_anc is not None) else None
            ancM = (dan / (gi / 1e6)) if (dan is not None and gi) else None
            off = (tot - onl) if (tot is not None and onl is not None) else None
            dr, _ = draw_rate(tag)
            ply, dep, blw = pgn_stats(tag, sample)
            rr = root_rows(j)
            quiet = (rr / gi / ply) if (rr and gi and ply) else None
            f = lambda v, w, p=1, s="": f"{v:>{w}.{p}f}{s}" if v is not None else f"{'--':>{w}}"
            print(f"{tag.split('-')[-1]:>7} {cum:>9,} "
                  f"{str(j.get('depth'))+'/'+str(j.get('nodes')):>7} "
                  f"{f(anc,9)} {f(ancM,7,0)} {f(perM,7,0)} {f(onl,7)} {f(off,7)} "
                  f"{f(dr,7,2)} {f(ply,6,1)} {f(dep,6,2)} {f(blw,6,1,'%')} {f(quiet,6,3)}")
            prev_tag = tag
            if anc is not None: prev_anc = anc
        print("  TRIGGER is anc/Mg, not par/Mg: m260720 switched d6->d8 at 73 "
              "and its next two legs gave 122 and 177.")
        print("  <floor MUST be 0.0% — anything else means results are recorded "
              "from an iteration other than the one searched.")


if __name__ == "__main__":
    main()
