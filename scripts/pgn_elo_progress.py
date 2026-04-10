#!/usr/bin/env python3
"""
pgn_elo_progress.py -- track Elo progress across training by splitting a PGN
into fixed-size windows and running bayeselo on each window.

Usage:
    python3 scripts/pgn_elo_progress.py <pgn_file> [options]

Options:
    --window N        games per window (default: 10000)
    --player NAME     player name to track (default: first non-reference player found)
    --bayeselo PATH   path to the bayeselo binary
    --cumulative      use cumulative games instead of sliding windows
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.dirname(SCRIPT_DIR)
DEFAULT_BAYESELO = os.path.join(REPO_ROOT, "tools", "BayesElo", "bayeselo")


def parse_args():
    p = argparse.ArgumentParser(
        description="Track Elo progress over a training PGN by windowed bayeselo analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("pgn", help="PGN file to analyse")
    p.add_argument("--window", type=int, default=10000,
                   help="games per window (default: 10000)")
    p.add_argument("--player", default=None,
                   help="player name to track (default: auto-detect training engine)")
    p.add_argument("--bayeselo", default=DEFAULT_BAYESELO,
                   help="path to the bayeselo binary")
    p.add_argument("--cumulative", action="store_true",
                   help="cumulative windows instead of sliding")
    return p.parse_args()


def split_pgn_into_games(pgn_path):
    """Split a PGN file into a list of game strings."""
    games = []
    current = []
    with open(pgn_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            current.append(line)
            # A result line ends a game
            stripped = line.strip()
            if stripped in ("1-0", "0-1", "1/2-1/2", "*") or re.match(
                r'^(1-0|0-1|1/2-1/2|\*)\s*$', stripped
            ):
                # Check if this is a standalone result line or end of moves
                games.append("".join(current))
                current = []
            # Also detect result embedded at end of move line
            elif re.search(r'\s+(1-0|0-1|1/2-1/2|\*)\s*$', stripped):
                games.append("".join(current))
                current = []
    if current:
        # flush any trailing content
        text = "".join(current).strip()
        if text:
            games.append(text + "\n")
    return games


def run_bayeselo_on_text(bayeselo_bin, pgn_text):
    """Write pgn_text to a temp file and run bayeselo on it."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".pgn", delete=False,
                                     encoding="utf-8") as tf:
        tf.write(pgn_text)
        tmp_path = tf.name
    try:
        commands = f"readpgn {tmp_path}\nelo\nmm\nratings 0\nx\nx\n\n"
        result = subprocess.run(
            [bayeselo_bin],
            input=commands,
            capture_output=True,
            text=True,
        )
        return result.stdout + result.stderr
    finally:
        os.unlink(tmp_path)


def parse_ratings(raw_output):
    """Extract ratings table; return list of dicts."""
    row_pattern = re.compile(
        r'^\s*(\d+)\s+(\S.*?)\s{2,}(-?\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\S+)\s+(-?\d+)\s+(\S+)\s*$'
    )
    header_pattern = re.compile(
        r'^.*Rank\s+Name\s+Elo\s+\+\s+-\s+games\s+score\s+oppo\.\s+draws\s*$'
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
                    "name":  m.group(2).strip(),
                    "elo":   int(m.group(3)),
                    "plus":  int(m.group(4)),
                    "minus": int(m.group(5)),
                    "games": int(m.group(6)),
                    "score": m.group(7),
                    "oppo":  int(m.group(8)),
                })
            elif line.strip() == "" and rows:
                break
    return rows


def detect_training_player(games):
    """Pick the player that appears most often in White/Black headers."""
    counts = {}
    for g in games[:200]:  # sample first 200 games
        for line in g.splitlines():
            m = re.match(r'^\[(White|Black)\s+"(.+)"\]', line)
            if m:
                name = m.group(2)
                counts[name] = counts.get(name, 0) + 1
    if not counts:
        return None
    # prefer the training engine (heuristic: name contains "train")
    train = [n for n in counts if "train" in n.lower()]
    if train:
        return max(train, key=lambda n: counts[n])
    return max(counts, key=lambda n: counts[n])


def main():
    args = parse_args()

    if not os.path.isfile(args.pgn):
        sys.exit(f"Error: PGN file not found: {args.pgn}")
    if not os.path.isfile(args.bayeselo):
        sys.exit(f"Error: bayeselo binary not found: {args.bayeselo}")

    print(f"Reading {args.pgn} ...", flush=True)
    games = split_pgn_into_games(args.pgn)
    total = len(games)
    print(f"  {total} games found", flush=True)

    player = args.player or detect_training_player(games)
    if player is None:
        sys.exit("Could not detect training player. Use --player NAME.")
    print(f"  Tracking: {player}")
    print(f"  Window:   {args.window} games {'(cumulative)' if args.cumulative else '(sliding)'}")
    print()

    results = []
    window = args.window
    starts = range(0, total, window)

    for i, start in enumerate(starts):
        end = start + window
        if args.cumulative:
            chunk = games[:end]
        else:
            chunk = games[start:end]
        if not chunk:
            break

        label_game = end if args.cumulative else (start + len(chunk))
        label_game = min(label_game, total)

        pgn_text = "".join(chunk)
        raw = run_bayeselo_on_text(args.bayeselo, pgn_text)
        rows = parse_ratings(raw)

        elo_val = None
        for r in rows:
            if r["name"] == player:
                elo_val = r["elo"]
                break

        results.append((label_game, elo_val, rows))
        if elo_val is not None:
            print(f"  game {label_game:>7,}: Elo {elo_val:>+5}", flush=True)
        else:
            print(f"  game {label_game:>7,}: player not found in ratings", flush=True)

    # Summary table
    print()
    print(f"Elo progress — {os.path.basename(args.pgn)}")
    print(f"Player: {player}")
    print()
    col = max(len(f"{total:,}"), 8)
    hdr = f"{'Games':>{col}}  {'Elo':>5}  {'Opponents':>9}"
    print(hdr)
    print("-" * len(hdr))
    for label_game, elo_val, rows in results:
        oppo_str = ""
        if elo_val is not None:
            for r in rows:
                if r["name"] == player:
                    oppo_str = f"{r['oppo']:>+5}"
                    break
        elo_str = f"{elo_val:>+5}" if elo_val is not None else "  N/A"
        print(f"{label_game:>{col},}  {elo_str}  {oppo_str:>9}")
    print()


if __name__ == "__main__":
    main()
