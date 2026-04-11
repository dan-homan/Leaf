#!/usr/bin/env python3
"""
pgn_winrate.py — Analyse win/draw/loss rates per N-game window for one player in a PGN file.

Usage:
    python3 scripts/pgn_winrate.py <pgn_file> [options]

Options:
    --player <name>     Player name to track (default: auto-detect first non-opponent player)
    --opponent <name>   Opponent name (used for auto-detection if --player omitted)
    --window <N>        Window size in games (default: 100)
    --csv               Output CSV instead of formatted table

Examples:
    # Auto-detect the training engine (non-material-eval side)
    python3 scripts/pgn_winrate.py learn/pgn/run1/match_run1_0g.pgn

    # Explicit player name and window
    python3 scripts/pgn_winrate.py learn/pgn/run1/match_run1_0g.pgn \
        --player Leaf_vtrain_nn-fresh-260409_a --window 200

    # CSV output for plotting
    python3 scripts/pgn_winrate.py learn/pgn/run1/match_run1_0g.pgn --csv
"""

import re
import argparse
import sys


def parse_pgn(pgn_path):
    """Return list of (white, black, result) tuples."""
    games = []
    white = black = result = None
    with open(pgn_path, 'r') as f:
        for line in f:
            line = line.strip()
            m = re.match(r'\[White "(.+?)"\]', line)
            if m:
                white = m.group(1)
            m = re.match(r'\[Black "(.+?)"\]', line)
            if m:
                black = m.group(1)
            m = re.match(r'\[Result "(.+?)"\]', line)
            if m:
                result = m.group(1)
            if white and black and result:
                games.append((white, black, result))
                white = black = result = None
    return games


def get_outcome(games, target):
    """Return list of 'W'/'D'/'L' strings from target's perspective."""
    outcomes = []
    for w, b, r in games:
        if w == target:
            if r == '1-0':
                outcomes.append('W')
            elif r == '1/2-1/2':
                outcomes.append('D')
            else:
                outcomes.append('L')
        elif b == target:
            if r == '0-1':
                outcomes.append('W')
            elif r == '1/2-1/2':
                outcomes.append('D')
            else:
                outcomes.append('L')
        # games not involving target are skipped
    return outcomes


def analyse(outcomes, window):
    """Return list of dicts with per-window stats."""
    rows = []
    for i in range(0, len(outcomes), window):
        chunk = outcomes[i:i + window]
        n = len(chunk)
        w = chunk.count('W')
        d = chunk.count('D')
        l = chunk.count('L')
        rows.append({
            'start': i + 1,
            'end': i + n,
            'n': n,
            'W': w,
            'D': d,
            'L': l,
            'win_pct': 100 * w / n,
            'draw_pct': 100 * d / n,
            'loss_pct': 100 * l / n,
            'score_pct': 100 * (w + 0.5 * d) / n,
        })
    return rows


def print_table(rows, target, window):
    print(f"Player : {target}")
    print(f"Window : {window} games")
    print()
    hdr = f"{'Window':>12}  {'W':>5}  {'D':>5}  {'L':>5}  {'Win%':>6}  {'Draw%':>6}  {'Loss%':>6}  {'Score%':>7}"
    sep = '-' * len(hdr)
    print(hdr)
    print(sep)
    for r in rows:
        label = f"{r['start']}-{r['end']}"
        print(f"{label:>12}  {r['W']:>5}  {r['D']:>5}  {r['L']:>5}"
              f"  {r['win_pct']:>5.1f}%  {r['draw_pct']:>5.1f}%"
              f"  {r['loss_pct']:>5.1f}%  {r['score_pct']:>6.1f}%")
    print(sep)
    total_w = sum(r['W'] for r in rows)
    total_d = sum(r['D'] for r in rows)
    total_l = sum(r['L'] for r in rows)
    n_total = total_w + total_d + total_l
    print(f"{'TOTAL':>12}  {total_w:>5}  {total_d:>5}  {total_l:>5}"
          f"  {100*total_w/n_total:>5.1f}%  {100*total_d/n_total:>5.1f}%"
          f"  {100*total_l/n_total:>5.1f}%  {100*(total_w+0.5*total_d)/n_total:>6.1f}%")

    # Summary
    print()
    peak = max(rows, key=lambda r: r['win_pct'])
    print(f"Starting win rate (games {rows[0]['start']}-{rows[0]['end']}): {rows[0]['win_pct']:.1f}%")
    print(f"Peak win rate: {peak['win_pct']:.1f}% in window {peak['start']}-{peak['end']}")
    crashes = [r for r in rows if r['win_pct'] < 5.0]
    if crashes:
        first_crash = crashes[0]
        print(f"First crash below 5% win rate: window {first_crash['start']}-{first_crash['end']}")
        # Check for recovery: any window after the crash with win% >= 5%
        crash_idx = rows.index(first_crash)
        recoveries = [r for r in rows[crash_idx+1:] if r['win_pct'] >= 5.0]
        if recoveries:
            print(f"Recovery above 5%: window {recoveries[0]['start']}-{recoveries[0]['end']}"
                  f" ({recoveries[0]['win_pct']:.1f}%)")
        else:
            print("No recovery above 5% after the crash.")
    else:
        print("Win rate never crashed below 5%.")


def print_csv(rows, target):
    print("player,start,end,n,W,D,L,win_pct,draw_pct,loss_pct,score_pct")
    for r in rows:
        print(f"{target},{r['start']},{r['end']},{r['n']},"
              f"{r['W']},{r['D']},{r['L']},"
              f"{r['win_pct']:.2f},{r['draw_pct']:.2f},{r['loss_pct']:.2f},{r['score_pct']:.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyse per-window win/draw/loss rates for one player in a PGN file.")
    parser.add_argument('pgn_file', help="Path to PGN file")
    parser.add_argument('--player', default=None, help="Player name to track")
    parser.add_argument('--opponent', default=None, help="Opponent name (used for auto-detection)")
    parser.add_argument('--window', type=int, default=100, help="Window size in games (default: 100)")
    parser.add_argument('--csv', action='store_true', help="Output CSV instead of table")
    args = parser.parse_args()

    games = parse_pgn(args.pgn_file)
    if not games:
        print("No games found in PGN file.", file=sys.stderr)
        sys.exit(1)

    all_players = list({p for g in games for p in (g[0], g[1])})

    if args.player:
        target = args.player
    elif args.opponent:
        others = [p for p in all_players if p != args.opponent]
        if not others:
            print(f"Cannot find a player other than '{args.opponent}'.", file=sys.stderr)
            sys.exit(1)
        target = others[0]
    else:
        # Auto-detect: pick the player that is not the "material_eval" or similar baseline
        # Heuristic: pick the player whose name comes last alphabetically among the two
        # (training engines typically have longer/later names than baselines)
        # More robust: pick non-"material_eval" if present
        non_baseline = [p for p in all_players if 'material_eval' not in p]
        if len(non_baseline) == 1:
            target = non_baseline[0]
        else:
            target = sorted(all_players)[-1]
        print(f"Auto-detected player: {target}", file=sys.stderr)

    outcomes = get_outcome(games, target)
    if not outcomes:
        print(f"Player '{target}' not found in any game.", file=sys.stderr)
        sys.exit(1)

    rows = analyse(outcomes, args.window)

    if args.csv:
        print_csv(rows, target)
    else:
        print_table(rows, target, args.window)


if __name__ == '__main__':
    main()
