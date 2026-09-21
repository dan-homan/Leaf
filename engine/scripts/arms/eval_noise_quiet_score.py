#!/usr/bin/env python3
"""Score the eval_noise quiet-fraction arms.

Reads the WIDE root-row dumps produced by eval_noise_quiet.sh and reports, per
sigma, the fraction of root positions that would pass the production 60cp quiet
gate -- plus the same figure at other gate widths, which the wide dump gives
for free.

Root TSV columns (tdleaf.cpp, root row):
    1 fen  2 cp  3 result  4 game_ply  5 depth  6 gid  7 final_game_ply  8 gate
cp and gate are both WHITE POV, and the gate test is |cp - gate| <= QUIET_CP.

The headline number is quiet rows per PLY, not per game: rows-per-game alone
confounds "fewer positions are quiet" with "games got shorter", and separating
those two is the whole point (the production drift was 0.562 -> 0.499 per ply
while mean game length moved only 138.7 -> 135.6).
"""
import sys, os, glob

GATES = (30, 60, 100, 200)


def score(arm_dir):
    rows = 0
    passes = {g: 0 for g in GATES}
    plies_by_game = {}          # (file, gid) -> final_game_ply
    for path in sorted(glob.glob(os.path.join(arm_dir, "*root.tsv"))):
        with open(path, errors="replace") as fh:
            for line in fh:
                if line.startswith("#"):
                    continue
                f = line.rstrip("\n").split("\t")
                if len(f) < 8:
                    continue
                try:
                    cp, gid, final_ply, gate = int(f[1]), f[5], int(f[6]), int(f[7])
                except ValueError:
                    continue
                rows += 1
                d = abs(cp - gate)
                for g in GATES:
                    if d <= g:
                        passes[g] += 1
                plies_by_game[(path, gid)] = final_ply
    games = len(plies_by_game)
    total_plies = sum(plies_by_game.values())
    return rows, passes, games, total_plies


def main():
    work = sys.argv[1]
    sigmas = [int(s) for s in sys.argv[2:]]
    print()
    print("quiet fraction = rows passing the gate, per game PLY")
    print("production drift for reference: 0.562 (2e6) -> 0.499 (6e6) at gate 60")
    print()
    hdr = f"{'sigma':>5} {'games':>7} {'meanply':>8} {'wide/game':>10}"
    for g in GATES:
        hdr += f" {'q@'+str(g):>8}"
    print(hdr)
    base = None
    for s in sigmas:
        d = os.path.join(work, f"q_s{s}")
        if not os.path.isdir(d):
            continue
        rows, passes, games, total_plies = score(d)
        if not games or not total_plies:
            print(f"{s:>5}   no data")
            continue
        mean_ply = total_plies / games
        line = f"{s:>5} {games:>7} {mean_ply:>8.1f} {rows/games:>10.1f}"
        for g in GATES:
            line += f" {passes[g]/total_plies:>8.3f}"
        if base is None:
            base = passes[60] / total_plies
        else:
            line += f"   ({passes[60]/total_plies - base:+.3f} vs sigma=0 at gate 60)"
        print(line)
    print()


if __name__ == "__main__":
    main()
