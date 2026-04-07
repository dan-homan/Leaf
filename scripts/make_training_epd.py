#!/usr/bin/env python3
#
# Generate a combined opening EPD file for Leaf TDLeaf(λ) training:
#   - All 960 Chess960 starting positions (FRC), optionally replicated K times
#   - ~N positions sampled from a Polyglot opening book at a given ply depth
#
# The output EPD file can be used with:
#   cutechess-cli -openings file=training_openings.epd format=epd order=random \
#                 -variant fischerandom -noswap
#
# Sizing modes
# ------------
# Explicit (default):
#   python3 make_training_epd.py --frc-replicates 1 --book-positions 2000
#   Produces: 960 × 1 = 960 FRC entries + 2000 book entries = 2960 total.
#
# Fraction-based (recommended for large EPDs):
#   python3 make_training_epd.py --total 100000 --frc-fraction 0.20
#   Computes frc_replicates and book_positions so that FRC entries are ~20% of
#   the total.  Example: frc_replicates=21 → 20160 FRC + 79840 book = 100000.
#
# Run from learn/ (or scripts/) after placing normbk02.bin in learn/.
#

import argparse
import os
import random
import sys

FRC_COUNT = 960   # total unique Chess960 starting positions

try:
    import chess
    import chess.polyglot
except ImportError:
    print("Error: python-chess is required.  pip install chess", file=sys.stderr)
    sys.exit(1)


def all_frc_epds(replicates=1):
    """Return EPD strings for all 960 FRC starting positions, repeated `replicates` times.

    Each of the 960 positions appears exactly `replicates` times in the returned list
    (total length = 960 × replicates).  The order within each replica pass is
    position index 0–959; the caller shuffles the combined list.
    """
    epds = []
    for _ in range(replicates):
        for idx in range(FRC_COUNT):
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
        description="Generate training_openings.epd: FRC positions (replicated) + Polyglot book positions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Sizing examples:
  # Small default: 960 FRC + 2000 book = 2960 total
  python3 make_training_epd.py

  # Explicit replication: 960×5 FRC + 2000 book = 6800 total
  python3 make_training_epd.py --frc-replicates 5 --book-positions 2000

  # Fraction-based: 100k total, ~20% FRC (960×21=20160 FRC + 79840 book)
  python3 make_training_epd.py --total 100000 --frc-fraction 0.20
""",
    )
    parser.add_argument("--book", default=None, metavar="FILE",
                        help="Polyglot book .bin file "
                             "(default: normbk02.bin in the same directory as this script)")
    parser.add_argument("--book-positions", type=int, default=None, metavar="N",
                        help="Number of book positions to sample (default: 2000, or computed "
                             "from --total/--frc-fraction)")
    parser.add_argument("--frc-replicates", type=int, default=None, metavar="K",
                        help="Include each of the 960 FRC positions K times "
                             "(default: 1, or computed from --total/--frc-fraction)")
    parser.add_argument("--total", type=int, default=None, metavar="N",
                        help="Target total output size; use with --frc-fraction")
    parser.add_argument("--frc-fraction", type=float, default=None, metavar="F",
                        help="Fraction of --total to fill with FRC positions (0.0–1.0); "
                             "computes --frc-replicates and --book-positions automatically")
    parser.add_argument("--ply", type=int, default=8,
                        help="Ply depth for book random walks (default: 8)")
    parser.add_argument("--output", default=None, metavar="FILE",
                        help="Output EPD file "
                             "(default: training_openings.epd in the same directory as this script)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    args = parser.parse_args()

    # --- Validate and resolve sizing ---
    use_fraction = args.total is not None or args.frc_fraction is not None
    use_explicit = args.frc_replicates is not None or args.book_positions is not None

    if use_fraction and use_explicit:
        parser.error("--total/--frc-fraction and --frc-replicates/--book-positions "
                     "are mutually exclusive sizing modes.")

    if use_fraction:
        if args.total is None or args.frc_fraction is None:
            parser.error("--total and --frc-fraction must be used together.")
        if not 0.0 < args.frc_fraction < 1.0:
            parser.error("--frc-fraction must be between 0.0 and 1.0 (exclusive).")
        frc_target      = round(args.total * args.frc_fraction)
        frc_replicates  = max(1, round(frc_target / FRC_COUNT))
        actual_frc      = FRC_COUNT * frc_replicates
        book_positions  = max(0, args.total - actual_frc)
    else:
        frc_replicates = args.frc_replicates if args.frc_replicates is not None else 1
        book_positions = args.book_positions if args.book_positions is not None else 2000
        actual_frc     = FRC_COUNT * frc_replicates

    if frc_replicates < 1:
        parser.error("--frc-replicates must be at least 1.")

    book_path   = args.book   or os.path.join(script_dir, "normbk02.bin")
    output_path = args.output or os.path.join(script_dir, "training_openings.epd")

    # --- FRC positions ---
    print(f"Generating FRC positions: 960 × {frc_replicates} replicate(s) = {actual_frc:,} entries ...")
    frc_epds = all_frc_epds(replicates=frc_replicates)
    print(f"  Done.")

    # --- Book positions ---
    book_epds = []
    if book_positions > 0:
        if os.path.isfile(book_path):
            print(f"Sampling {book_positions:,} positions from "
                  f"{os.path.basename(book_path)} at ply {args.ply} ...")
            book_epds = sample_book_positions(
                book_path, book_positions, args.ply, args.seed)
            print(f"  Done: {len(book_epds):,} unique book positions.")
        else:
            print(f"  Book not found: {book_path} — skipping book positions.")

    # --- Combine and shuffle ---
    # No global dedup: FRC replication is intentional.
    # Book positions are already deduplicated within sample_book_positions().
    combined = frc_epds + book_epds
    rng = random.Random(args.seed + 1)
    rng.shuffle(combined)

    # --- Write ---
    with open(output_path, "w") as f:
        for epd in combined:
            f.write(epd + "\n")

    total      = len(combined)
    frc_pct    = len(frc_epds)  / total * 100 if total else 0
    book_pct   = len(book_epds) / total * 100 if total else 0
    replicate_str = f"960 × {frc_replicates}" if frc_replicates > 1 else "960"

    print(f"\nWritten {total:,} positions → {output_path}")
    print(f"  FRC:  {len(frc_epds):>8,}  ({replicate_str} replicates, {frc_pct:.1f}%)")
    print(f"  Book: {len(book_epds):>8,}  ({book_pct:.1f}%)")
    print(f"\nUse with cutechess-cli:")
    print(f"  -openings file={os.path.basename(output_path)} format=epd order=random "
          f"-variant fischerandom -noswap")


if __name__ == "__main__":
    main()
