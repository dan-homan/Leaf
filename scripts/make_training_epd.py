#!/usr/bin/env python3
#
# Generate a combined opening EPD file for Leaf TDLeaf(λ) training:
#   - All 960 Chess960 starting positions (FRC), optionally with K random suffix moves
#   - ~N positions sampled from a Polyglot opening book at a given ply depth,
#     optionally with K random suffix moves
#
# Random suffix moves dramatically increase unique position counts, preventing
# game replication in training.  Use --quiet-only to restrict suffix moves to
# non-captures (keeps material balanced).
#
# Sizing modes (mutually exclusive):
#   Explicit:        --frc-replicates K --book-positions N
#   Fraction-based:  --total N --frc-fraction F
#
# Examples:
#   # Default: 960 FRC + 2000 book, no suffix (2,960 total)
#   python3 make_training_epd.py
#
#   # 2 quiet suffix moves → many more unique positions per source
#   python3 make_training_epd.py --frc-replicates 21 --book-positions 79840 \
#       --random-suffix 2 --quiet-only
#
#   # Fraction-based: 100k total, ~20% FRC-derived, 2 quiet suffix moves
#   python3 make_training_epd.py --total 100000 --frc-fraction 0.20 \
#       --random-suffix 2 --quiet-only
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


def apply_random_suffix(board, k, quiet_only, rng):
    """Play k random legal moves on board in-place.

    If quiet_only, restricts to non-captures; falls back to any legal move if
    no quiet moves are available.  Stops early if the position is terminal.
    """
    for _ in range(k):
        if quiet_only:
            moves = [m for m in board.legal_moves if not board.is_capture(m)]
            if not moves:
                moves = list(board.legal_moves)   # fallback: accept any move
        else:
            moves = list(board.legal_moves)
        if not moves:
            break
        board.push(rng.choice(moves))


def all_frc_epds(replicates, random_suffix, quiet_only, seed):
    """Return EPD strings derived from all 960 FRC starting positions.

    When random_suffix == 0:
      Each position is included exactly `replicates` times (intentional
      weighting; no deduplication).

    When random_suffix > 0:
      For each replicate pass, each FRC position gets a fresh random suffix walk
      (the RNG advances sequentially, so each replicate produces different moves).
      Duplicate EPDs across replicates are silently skipped; the returned list
      may be slightly shorter than 960 × replicates if collisions occur (rare).
    """
    rng = random.Random(seed)
    if random_suffix > 0:
        seen = set()
        epds = []
        for _ in range(replicates):
            for idx in range(FRC_COUNT):
                board = chess.Board.from_chess960_pos(idx)
                apply_random_suffix(board, random_suffix, quiet_only, rng)
                epd = board.epd()
                if epd not in seen:
                    seen.add(epd)
                    epds.append(epd)
    else:
        epds = []
        for _ in range(replicates):
            for idx in range(FRC_COUNT):
                board = chess.Board.from_chess960_pos(idx)
                epds.append(board.epd())
    return epds


def sample_book_positions(book_path, n_target, ply, random_suffix, quiet_only, seed):
    """Sample up to n_target unique EPD strings from a Polyglot opening book.

    Each sample is a weighted random walk of `ply` half-moves from the start
    position, optionally followed by `random_suffix` random (or quiet) moves.
    Positions are deduplicated by EPD string.

    Returns a list of EPD strings (may be shorter than n_target if the book
    has insufficient branching even after suffix moves).
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

        if random_suffix > 0:
            apply_random_suffix(board, random_suffix, quiet_only, rng)

        if not list(board.legal_moves):
            continue

        epd = board.epd()
        if epd not in seen:
            seen.add(epd)
            positions.append(epd)

    reader.close()

    if len(positions) < n_target:
        print(f"  Note: only {len(positions):,} unique positions found "
              f"(requested {n_target:,}; {attempts:,} attempts).")
    return positions


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(
        description="Generate training_openings.epd: FRC positions + Polyglot book positions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Without --random-suffix, normbk02.bin at ply 8 yields ~2500 unique positions.
Adding suffix moves explodes the unique count:

  --random-suffix 1  →  ~60k unique book + ~19k unique FRC positions
  --random-suffix 2  →  ~1M+ unique book + ~300k+ unique FRC positions

Use --quiet-only to restrict suffix moves to non-captures (recommended).

Sizing modes (mutually exclusive):

  Explicit (direct control):
    python3 make_training_epd.py --frc-replicates 21 --book-positions 80000 \\
        --random-suffix 2 --quiet-only

  Fraction-based (auto-compute frc_replicates and book_positions):
    python3 make_training_epd.py --total 100000 --frc-fraction 0.20 \\
        --random-suffix 2 --quiet-only
    → frc_replicates=21, 20160 FRC-derived + 79840 book = 100000 total
""",
    )
    parser.add_argument("--book", default=None, metavar="FILE",
                        help="Polyglot book .bin file "
                             "(default: normbk02.bin alongside this script)")
    parser.add_argument("--book-positions", type=int, default=None, metavar="N",
                        help="Book positions to sample (default: 2000, or computed from "
                             "--total/--frc-fraction)")
    parser.add_argument("--frc-replicates", type=int, default=None, metavar="K",
                        help="Samples per FRC position (default: 1).  Without "
                             "--random-suffix: K identical copies (for weighting).  "
                             "With --random-suffix: K unique suffix-varied samples "
                             "per FRC position (up to 960×K unique EPDs).")
    parser.add_argument("--total", type=int, default=None, metavar="N",
                        help="Target total output size; use with --frc-fraction")
    parser.add_argument("--frc-fraction", type=float, default=None, metavar="F",
                        help="Desired FRC fraction of --total (0.0–1.0); "
                             "auto-computes --frc-replicates and --book-positions")
    parser.add_argument("--random-suffix", type=int, default=0, metavar="K",
                        help="Random moves to play after each book/FRC position "
                             "(default: 0).  Greatly increases unique position count.  "
                             "Pair with --quiet-only for material-balanced positions.")
    parser.add_argument("--quiet-only", action="store_true", default=False,
                        help="Restrict random suffix moves to non-captures "
                             "(recommended with --random-suffix)")
    parser.add_argument("--ply", type=int, default=8,
                        help="Ply depth for book random walks (default: 8)")
    parser.add_argument("--output", default=None, metavar="FILE",
                        help="Output EPD file "
                             "(default: training_openings.epd alongside this script)")
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
        frc_target     = round(args.total * args.frc_fraction)
        frc_replicates = max(1, round(frc_target / FRC_COUNT))
        book_positions = max(0, args.total - FRC_COUNT * frc_replicates)
    else:
        frc_replicates = args.frc_replicates if args.frc_replicates is not None else 1
        book_positions = args.book_positions if args.book_positions is not None else 2000

    if frc_replicates < 1:
        parser.error("--frc-replicates must be at least 1.")
    if args.random_suffix < 0:
        parser.error("--random-suffix must be >= 0.")

    book_path   = args.book   or os.path.join(script_dir, "normbk02.bin")
    output_path = args.output or os.path.join(script_dir, "training_openings.epd")

    suffix_desc = (f"{args.random_suffix} {'quiet ' if args.quiet_only else ''}random move(s)"
                   if args.random_suffix > 0 else "none")

    # --- FRC positions ---
    frc_label = (f"960 × {frc_replicates} replicates"
                 if frc_replicates > 1 else "960 × 1")
    print(f"Generating FRC positions: {frc_label}, suffix: {suffix_desc} ...")
    frc_epds = all_frc_epds(
        replicates=frc_replicates,
        random_suffix=args.random_suffix,
        quiet_only=args.quiet_only,
        seed=args.seed,
    )
    print(f"  Done: {len(frc_epds):,} FRC-derived positions.")

    # --- Book positions ---
    book_epds = []
    if book_positions > 0:
        if os.path.isfile(book_path):
            print(f"Sampling {book_positions:,} positions from "
                  f"{os.path.basename(book_path)} "
                  f"at ply {args.ply}, suffix: {suffix_desc} ...")
            book_epds = sample_book_positions(
                book_path, book_positions, args.ply,
                args.random_suffix, args.quiet_only,
                seed=args.seed + 1,
            )
            print(f"  Done: {len(book_epds):,} unique book-derived positions.")
        else:
            print(f"  Book not found: {book_path} — skipping book positions.")

    # --- Combine and shuffle ---
    # FRC without suffix: intentional duplicates preserved (for weighting).
    # FRC with suffix: already deduplicated in all_frc_epds().
    # Book positions: always deduplicated within sample_book_positions().
    combined = frc_epds + book_epds
    rng = random.Random(args.seed + 2)
    rng.shuffle(combined)

    # --- Write ---
    with open(output_path, "w") as f:
        for epd in combined:
            f.write(epd + "\n")

    total    = len(combined)
    frc_pct  = len(frc_epds)  / total * 100 if total else 0
    book_pct = len(book_epds) / total * 100 if total else 0

    print(f"\nWritten {total:,} positions → {output_path}")
    print(f"  FRC-derived:  {len(frc_epds):>8,}  ({frc_pct:.1f}%)")
    print(f"  Book-derived: {len(book_epds):>8,}  ({book_pct:.1f}%)")
    if args.random_suffix > 0:
        q = "quiet " if args.quiet_only else ""
        print(f"  Suffix: {args.random_suffix} {q}random move(s) applied to every position")
    print(f"\nUse with cutechess-cli:")
    print(f"  -openings file={os.path.basename(output_path)} format=epd order=random "
          f"-variant fischerandom -noswap")


if __name__ == "__main__":
    main()
