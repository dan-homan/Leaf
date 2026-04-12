#!/usr/bin/env python3
"""
bayeselo_ratings.py -- compute a relative Elo rating list from one or more PGN files.

Usage:
    python3 scripts/bayeselo_ratings.py <pgn_file> [<pgn_file> ...] [options]

Multiple PGN files are combined before analysis (bayeselo reads them sequentially).

Options:
    --bayeselo PATH   path to the bayeselo binary
                      (default: tools/BayesElo/bayeselo relative to this script)
    --min N           exclude players with fewer than N games (default: 0)
    --advantage       also optimise first-move advantage
    --drawelo         also optimise draw-Elo

Output:
    A ranked table of players and their Bayesian Elo ratings.
"""

import argparse
import os
import re
import subprocess
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DEFAULT_BAYESELO = os.path.join(REPO_ROOT, "tools", "BayesElo", "bayeselo")


def parse_args():
    p = argparse.ArgumentParser(
        description="Compute a Bayesian Elo rating list from a PGN file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("pgn", nargs="+", help="PGN file(s) to analyse (combined before analysis)")
    p.add_argument("--bayeselo", default=DEFAULT_BAYESELO,
                   help="path to the bayeselo binary")
    p.add_argument("--min", type=int, default=0, dest="min_games",
                   help="minimum games to include a player (default: 0)")
    p.add_argument("--advantage", action="store_true",
                   help="optimise first-move advantage")
    p.add_argument("--drawelo", action="store_true",
                   help="optimise draw-Elo")
    return p.parse_args()


def run_bayeselo(bayeselo_bin, pgn_paths, min_games, advantage, drawelo):
    mm_flags = ""
    if advantage:
        mm_flags += " 1"
    if drawelo:
        mm_flags = (mm_flags or " 0") + " 1"

    read_cmds = [f"readpgn {os.path.abspath(p)}" for p in pgn_paths]
    commands = "\n".join(read_cmds + [
        "elo",
        f"mm{mm_flags}",
        f"ratings {min_games}",
        "x",
        "x",
        "",
    ])

    result = subprocess.run(
        [bayeselo_bin],
        input=commands,
        capture_output=True,
        text=True,
    )

    output = result.stdout + result.stderr
    return output


def parse_ratings(raw_output):
    """
    Extract the ratings table from bayeselo output.
    Returns (games_loaded, rows) where rows is a list of dicts.
    """
    games_loaded = 0
    m = re.findall(r'(\d+) game\(s\) loaded', raw_output)
    if m:
        games_loaded = int(m[-1])

    # Header line looks like (may be prefixed by a prompt string):
    # ResultSet-EloRating>Rank Name                 Elo    +    - games score oppo. draws
    header_pattern = re.compile(
        r'^.*Rank\s+Name\s+Elo\s+\+\s+-\s+games\s+score\s+oppo\.\s+draws\s*$'
    )
    row_pattern = re.compile(
        r'^\s*(\d+)\s+(\S.*?)\s{2,}(-?\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\S+)\s+(-?\d+)\s+(\S+)\s*$'
    )

    rows = []
    in_table = False
    for line in raw_output.splitlines():
        if header_pattern.match(line):
            in_table = True
            continue
        if in_table:
            m = row_pattern.match(line)
            if m:
                rows.append({
                    "rank":   int(m.group(1)),
                    "name":   m.group(2).strip(),
                    "elo":    int(m.group(3)),
                    "plus":   int(m.group(4)),
                    "minus":  int(m.group(5)),
                    "games":  int(m.group(6)),
                    "score":  m.group(7),
                    "oppo":   int(m.group(8)),
                    "draws":  m.group(9),
                })
            elif line.strip() == "":
                # blank line ends table
                if rows:
                    break

    return games_loaded, rows


def print_ratings(games_loaded, rows, pgn_paths):
    if len(pgn_paths) == 1:
        label = os.path.basename(pgn_paths[0])
    else:
        label = f"{len(pgn_paths)} PGN files combined"
    print(f"\nBayesian Elo ratings — {label}")
    print(f"{games_loaded} games loaded, {len(rows)} players rated\n")

    if not rows:
        print("No ratings found in bayeselo output.")
        return

    # Column widths
    name_w = max(len(r["name"]) for r in rows)
    name_w = max(name_w, 4)  # "Name"

    header = (f"{'Rank':>4}  {'Name':<{name_w}}  {'Elo':>5}  "
              f"{'±':>4}  {'Games':>5}  {'Score':>6}  {'Oppo':>5}  {'Draws':>5}")
    sep = "-" * len(header)

    print(header)
    print(sep)
    for r in rows:
        ci = max(r["plus"], r["minus"])
        print(f"{r['rank']:>4}  {r['name']:<{name_w}}  {r['elo']:>+5}  "
              f"{ci:>4}  {r['games']:>5}  {r['score']:>6}  {r['oppo']:>+5}  {r['draws']:>5}")
    print()


def main():
    args = parse_args()

    for pgn in args.pgn:
        if not os.path.isfile(pgn):
            sys.exit(f"Error: PGN file not found: {pgn}")
    if not os.path.isfile(args.bayeselo):
        sys.exit(f"Error: bayeselo binary not found: {args.bayeselo}")

    raw = run_bayeselo(
        args.bayeselo, args.pgn,
        args.min_games, args.advantage, args.drawelo,
    )

    games_loaded, rows = parse_ratings(raw)

    if not rows:
        print("Could not parse ratings from bayeselo output. Raw output:\n")
        print(raw)
        sys.exit(1)

    print_ratings(games_loaded, rows, args.pgn)


if __name__ == "__main__":
    main()
