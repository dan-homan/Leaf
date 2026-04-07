#!/usr/bin/env python3
#
# Generate a combined opening EPD file for Leaf TDLeaf(λ) training:
#   - All 960 Chess960 starting positions (FRC)
#   - ~N positions sampled from a Polyglot opening book at a given ply depth
#
# The output EPD file can be used with:
#   cutechess-cli -openings file=training_openings.epd format=epd order=random \
#                 -variant fischerandom
#
# Run from learn/ (or scripts/) after placing normbk02.bin in learn/:
#   python3 make_training_epd.py
#   python3 make_training_epd.py --book normbk02.bin --book-positions 2000 --ply 8
#

import argparse
import os
import random
import sys

try:
    import chess
    import chess.polyglot
except ImportError:
    print("Error: python-chess is required.  pip install chess", file=sys.stderr)
    sys.exit(1)


def all_frc_epds():
    """Return EPD strings for all 960 Chess960 starting positions (indices 0–959)."""
    epds = []
    for idx in range(960):
        board = chess.Board.from_chess960_pos(idx)
        epds.append(board.epd())
    return epds


def sample_book_positions(book_path, n_target, ply, seed):
    """
    Sample up to n_target unique EPD strings from a Polyglot opening book by
    doing weighted random walks of `ply` half-moves from the start position.

    Returns a list of EPD strings (may be shorter than n_target if the book
    has insufficient branching to produce enough unique positions).
    """
    rng = random.Random(seed)
    seen = set()
    positions = []
    max_attempts = n_target * 25

    try:
        reader = chess.polyglot.open_reader(book_path)
    except Exception as e:
        print(f"Error: cannot open book {book_path}: {e}", file=sys.stderr)
        return []

    attempts = 0
    while len(positions) < n_target and attempts < max_attempts:
        attempts += 1
        board = chess.Board()
        ok = True
        for _ in range(ply):
            entries = list(reader.find_all(board))
            if not entries:
                ok = False
                break
            total_w = sum(e.weight for e in entries)
            if total_w == 0:
                entry = rng.choice(entries)
            else:
                r = rng.randint(0, total_w - 1)
                cum = 0
                entry = entries[-1]
                for e in entries:
                    cum += e.weight
                    if r < cum:
                        entry = e
                        break
            board.push(entry.move)

        if not ok:
            continue
        if not list(board.legal_moves):
            continue

        epd = board.epd()
        if epd not in seen:
            seen.add(epd)
            positions.append(epd)

    reader.close()

    if len(positions) < n_target:
        print(f"  Note: only {len(positions)} unique book positions found "
              f"(requested {n_target}; {attempts} attempts).")
    return positions


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(
        description="Generate training_openings.epd: 960 FRC positions + Polyglot book positions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--book", default=None, metavar="FILE",
                        help="Polyglot book .bin file "
                             "(default: normbk02.bin in the same directory as this script)")
    parser.add_argument("--book-positions", type=int, default=2000, metavar="N",
                        help="Approximate number of book positions to include (default: 2000)")
    parser.add_argument("--ply", type=int, default=8,
                        help="Ply depth for book random walks (default: 8)")
    parser.add_argument("--output", default=None, metavar="FILE",
                        help="Output EPD file "
                             "(default: training_openings.epd in the same directory as this script)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    args = parser.parse_args()

    book_path   = args.book   or os.path.join(script_dir, "normbk02.bin")
    output_path = args.output or os.path.join(script_dir, "training_openings.epd")

    # --- FRC positions ---
    print("Generating all 960 Chess960 starting positions ...")
    frc_epds = all_frc_epds()
    print(f"  Done: {len(frc_epds)} FRC positions.")

    # --- Book positions ---
    book_epds = []
    if os.path.isfile(book_path):
        print(f"Sampling {args.book_positions} positions from "
              f"{os.path.basename(book_path)} at ply {args.ply} ...")
        book_epds = sample_book_positions(
            book_path, args.book_positions, args.ply, args.seed)
        print(f"  Done: {len(book_epds)} unique book positions.")
    else:
        print(f"  Book not found: {book_path} — skipping book positions.")

    # --- Combine, deduplicate, shuffle ---
    combined = list(dict.fromkeys(frc_epds + book_epds))   # order-preserving dedup
    n_dupes  = len(frc_epds) + len(book_epds) - len(combined)
    rng = random.Random(args.seed + 1)
    rng.shuffle(combined)

    # --- Write ---
    with open(output_path, "w") as f:
        for epd in combined:
            f.write(epd + "\n")

    print(f"\nWritten {len(combined)} positions → {output_path}")
    print(f"  FRC: {len(frc_epds)}   Book: {len(book_epds)}   "
          f"Duplicates removed: {n_dupes}")
    print(f"\nUse with cutechess-cli:")
    print(f"  -openings file={os.path.basename(output_path)} format=epd order=random "
          f"-variant fischerandom -noswap")


if __name__ == "__main__":
    main()
