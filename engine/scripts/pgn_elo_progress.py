#!/usr/bin/env python3
"""
pgn_elo_progress.py -- track Elo progress across training by splitting a PGN
into fixed-size windows and running bayeselo on each window.

Usage:
    python3 scripts/pgn_elo_progress.py <pgn_file> [options]

Options:
    --window N        games per window (default: 10000)
    --player NAME     player name to track (default: auto-detect training engine)
    --bayeselo PATH   path to the bayeselo binary
    --cumulative      use cumulative games instead of sliding windows
"""

import argparse
import collections
import os
import re
import subprocess
import sys
import tempfile


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.dirname(os.path.dirname(SCRIPT_DIR))
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


def iter_game_records(pgn_path):
    """Stream (white, black, result) tuples from a PGN without storing move text.

    Yields one tuple per game as it is encountered; never holds more than one
    game's header fields in memory at a time.
    """
    white = black = result = None
    seen_moves = False

    with open(pgn_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("["):
                if seen_moves:
                    # Entering a new game's headers — emit the completed game.
                    if white and black and result:
                        yield (white, black, result)
                    white = black = result = None
                    seen_moves = False
                m = re.match(r'^\[(\w+)\s+"(.*)"\]', line)
                if m:
                    tag, val = m.group(1), m.group(2)
                    if tag == "White":
                        white = val
                    elif tag == "Black":
                        black = val
                    elif tag == "Result":
                        result = val
            elif line.strip():
                seen_moves = True

    # Flush the last game.
    if white and black and result:
        yield (white, black, result)


def detect_training_player(records):
    """Return the most likely training engine name from a sample of records."""
    counts = collections.Counter()
    for white, black, _ in records:
        counts[white] += 1
        counts[black] += 1
    if not counts:
        return None
    train = [n for n in counts if "train" in n.lower()]
    if train:
        return max(train, key=lambda n: counts[n])
    return counts.most_common(1)[0][0]


def build_minimal_pgn(records):
    """Produce the minimal PGN bayeselo needs: headers + result token only."""
    parts = []
    for white, black, result in records:
        parts.append(
            f'[White "{white}"]\n'
            f'[Black "{black}"]\n'
            f'[Result "{result}"]\n'
            f'\n{result}\n\n'
        )
    return "".join(parts)


def run_bayeselo(bayeselo_bin, records):
    """Run bayeselo on a list of (white, black, result) records."""
    pgn_text = build_minimal_pgn(records)
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
    header_re = re.compile(
        r'^.*Rank\s+Name\s+Elo\s+\+\s+-\s+games\s+score\s+oppo\.\s+draws\s*$'
    )
    row_re = re.compile(
        r'^\s*(\d+)\s+(\S.*?)\s{2,}(-?\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\S+)\s+(-?\d+)\s+(\S+)\s*$'
    )
    rows = []
    in_table = False
    for line in raw_output.splitlines():
        if header_re.match(line):
            in_table = True
            continue
        if in_table:
            m = row_re.match(line)
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


def find_player(rows, player):
    for r in rows:
        if r["name"] == player:
            return r
    return None


def main():
    args = parse_args()

    if not os.path.isfile(args.pgn):
        sys.exit(f"Error: PGN file not found: {args.pgn}")
    if not os.path.isfile(args.bayeselo):
        sys.exit(f"Error: bayeselo binary not found: {args.bayeselo}")

    window   = args.window
    cum_mode = args.cumulative

    # --- streaming pass ---
    # In sliding mode we only keep `window` records in memory at a time.
    # In cumulative mode we keep all records (they are tiny: 3 strings each).

    results       = []          # [(game_number, elo, oppo), ...]
    buf           = []          # current window buffer (sliding) or full history (cumulative)
    player        = args.player
    game_count    = 0

    print(f"Analysing {args.pgn} ...", flush=True)

    for white, black, result in iter_game_records(args.pgn):
        game_count += 1
        record = (white, black, result)

        if cum_mode:
            buf.append(record)
        else:
            buf.append(record)

        # Auto-detect player from the first window's games.
        if player is None and game_count == window:
            player = detect_training_player(buf)
            if player is None:
                sys.exit("Could not detect training player. Use --player NAME.")
            print(f"  Tracking: {player}")
            print(f"  Window:   {window} games {'(cumulative)' if cum_mode else '(sliding)'}")
            print()

        if game_count % window == 0:
            raw  = run_bayeselo(args.bayeselo, buf)
            rows = parse_ratings(raw)

            if player is None:
                # file has fewer than `window` games — shouldn't reach here normally
                player = detect_training_player(buf)

            pr = find_player(rows, player)
            elo_val  = pr["elo"]  if pr else None
            oppo_val = pr["oppo"] if pr else None
            results.append((game_count, elo_val, oppo_val))

            if elo_val is not None:
                print(f"  game {game_count:>7,}: Elo {elo_val:>+5}", flush=True)
            else:
                print(f"  game {game_count:>7,}: player not found in ratings", flush=True)

            if not cum_mode:
                buf.clear()

    # Handle a trailing partial window.
    if buf and (game_count % window != 0):
        if player is None:
            player = detect_training_player(buf)
            if player is None:
                sys.exit("Could not detect training player. Use --player NAME.")
            print(f"  Tracking: {player}")
            print(f"  Window:   {window} games {'(cumulative)' if cum_mode else '(sliding)'}")
            print()

        raw  = run_bayeselo(args.bayeselo, buf)
        rows = parse_ratings(raw)
        pr   = find_player(rows, player)
        elo_val  = pr["elo"]  if pr else None
        oppo_val = pr["oppo"] if pr else None
        results.append((game_count, elo_val, oppo_val))

        if elo_val is not None:
            print(f"  game {game_count:>7,}: Elo {elo_val:>+5}", flush=True)
        else:
            print(f"  game {game_count:>7,}: player not found in ratings", flush=True)

    # --- summary table ---
    print()
    print(f"Elo progress — {os.path.basename(args.pgn)}")
    print(f"Player: {player}  |  {game_count:,} games total")
    print()
    col = max(len(f"{game_count:,}"), 8)
    hdr = f"{'Games':>{col}}  {'Elo':>5}  {'Oppo':>5}"
    print(hdr)
    print("-" * len(hdr))
    for g, elo_val, oppo_val in results:
        elo_str  = f"{elo_val:>+5}"  if elo_val  is not None else "  N/A"
        oppo_str = f"{oppo_val:>+5}" if oppo_val is not None else "  N/A"
        print(f"{g:>{col},}  {elo_str}  {oppo_str}")
    print()


if __name__ == "__main__":
    main()
