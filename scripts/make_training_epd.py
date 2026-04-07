#!/usr/bin/env python3
#
# Generate a combined opening EPD file for Leaf TDLeaf(λ) training:
#   - All 960 Chess960 starting positions (FRC), optionally with K random suffix moves
#   - ~N positions sampled from a Polyglot opening book at a given ply depth,
#     optionally with K random suffix moves
#
# --quiet-only restricts random suffix moves to non-captures AND (when a Leaf eval
# binary is available) filters the resulting positions to those scoring within
# --eval-limit centipawns of even.  The default eval binary is Leaf_vclassic_eval
# in the same directory as this script.  Compile it with:
#   perl src/comp.pl classic_eval OVERWRITE
#
# Sizing modes (mutually exclusive):
#   Explicit:        --frc-replicates K --book-positions N
#   Fraction-based:  --total N --frc-fraction F
#
# Examples:
#   # Default: 960 FRC + 2000 book, no suffix (2,960 total)
#   python3 make_training_epd.py
#
#   # 2 quiet suffix moves + eval filter → balanced, unique positions
#   python3 make_training_epd.py --frc-replicates 21 --book-positions 80000 \
#       --random-suffix 2 --quiet-only
#
#   # Fraction-based: 100k total, ~20% FRC, 2 quiet suffix moves, eval filtered
#   python3 make_training_epd.py --total 100000 --frc-fraction 0.20 \
#       --random-suffix 2 --quiet-only
#
# Run from learn/ (or scripts/) after placing normbk02.bin in learn/.
#

import argparse
import os
import random
import re
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

FRC_COUNT = 960   # total unique Chess960 starting positions

try:
    import chess
    import chess.polyglot
except ImportError:
    print("Error: python-chess is required.  pip install chess", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Position generation
# ---------------------------------------------------------------------------

def apply_random_suffix(board, k, quiet_only, rng):
    """Play k random legal moves on board in-place.

    If quiet_only, restricts to non-captures; falls back to any legal move
    if no quiet moves are available.  Stops early if the position is terminal.
    """
    for _ in range(k):
        if quiet_only:
            moves = [m for m in board.legal_moves if not board.is_capture(m)]
            if not moves:
                moves = list(board.legal_moves)   # fallback
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
      (the RNG advances sequentially so each replicate produces different moves).
      Duplicate EPDs across replicates are silently skipped (rare); the returned
      list may be slightly shorter than 960 × replicates.
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


# ---------------------------------------------------------------------------
# Eval filtering via UCI engine
# ---------------------------------------------------------------------------

def fen_for_eval(epd):
    """Convert an EPD string to a FEN suitable for a standard (non-FRC) engine.

    Strips castling rights (sets to '-') to avoid Shredder/Chess960 notation
    being sent to an engine that may not parse it.  For a balance filter at
    shallow depth this has negligible effect on the score.
    """
    parts = epd.split()
    ep = parts[3] if len(parts) > 3 else "-"
    # EPD fields: placement active castling en_passant [opcodes...]
    # FEN fields: placement active castling en_passant halfmove fullmove
    return f"{parts[0]} {parts[1]} - {ep} 0 1"


def _engine_worker(engine_path, epd_batch, score_limit, depth):
    """Evaluate a batch of EPDs using a single persistent UCI engine process.

    Returns a list of (epd, passes) tuples where passes is True when
    |score| <= score_limit cp.  Positions with a forced mate score are
    considered unbalanced and filtered out.  Positions where no score
    could be read are kept (should not happen in normal operation).
    """
    proc = subprocess.Popen(
        [engine_path],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL, text=True, bufsize=1,
    )

    def send(s):
        proc.stdin.write(s + "\n")
        proc.stdin.flush()

    # UCI handshake
    send("uci")
    for line in proc.stdout:
        if "uciok" in line:
            break
    send("isready")
    for line in proc.stdout:
        if "readyok" in line:
            break

    cp_re   = re.compile(r"score cp (-?\d+)")
    mate_re = re.compile(r"score mate -?\d+")
    results = []

    for epd in epd_batch:
        fen = fen_for_eval(epd)
        send(f"position fen {fen}")
        send(f"go depth {depth}")

        score   = None
        is_mate = False
        for line in proc.stdout:
            m = cp_re.search(line)
            if m:
                score   = int(m.group(1))
                is_mate = False          # cp score overrides any earlier mate
            elif mate_re.search(line):
                is_mate = True
            if line.startswith("bestmove"):
                break

        if is_mate:
            passes = False               # clearly unbalanced → discard
        elif score is None:
            passes = True                # no score read → keep (shouldn't happen)
        else:
            passes = abs(score) <= score_limit

        results.append((epd, passes))

    send("quit")
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()

    return results


def filter_by_eval(epds, engine_path, score_limit, depth, n_workers):
    """Filter EPDs, keeping those with |score| <= score_limit cp.

    Spawns n_workers parallel UCI engine processes for speed.
    Returns a filtered list preserving original order.
    """
    n = len(epds)
    if n == 0:
        return []

    batch_size  = max(1, (n + n_workers - 1) // n_workers)
    batches     = [epds[i : i + batch_size] for i in range(0, n, batch_size)]
    lock        = threading.Lock()
    evaluated   = [0]
    all_results = [None] * len(batches)

    def run_batch(batch_idx, batch):
        results = _engine_worker(engine_path, batch, score_limit, depth)
        with lock:
            evaluated[0] += len(batch)
            print(f"\r  Evaluated {evaluated[0]:,}/{n:,} ...", end="", flush=True)
        return batch_idx, results

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(run_batch, i, b) for i, b in enumerate(batches)]
        for future in as_completed(futures):
            idx, results = future.result()
            all_results[idx] = results

    print()   # terminate \r progress line

    # Flatten in original order
    return [epd
            for batch_results in all_results
            for epd, passes in batch_results
            if passes]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cpu_count  = os.cpu_count() or 1

    parser = argparse.ArgumentParser(
        description="Generate training_openings.epd: FRC positions + Polyglot book positions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Without --random-suffix, normbk02.bin at ply 8 yields ~2500 unique positions.
Adding suffix moves explodes the unique count:

  --random-suffix 1  →  ~60k unique book + ~19k unique FRC positions
  --random-suffix 2  →  ~1M+ unique book + ~300k+ unique FRC positions

--quiet-only: restricts suffix moves to non-captures AND filters the output
by a Leaf eval binary (default: Leaf_vclassic_eval), keeping only positions
within --eval-limit cp of even (scored at depth 10 by default).
Compile the eval binary with:
  perl src/comp.pl classic_eval OVERWRITE

Sizing modes (mutually exclusive):
  Explicit:
    python3 make_training_epd.py --frc-replicates 21 --book-positions 80000 \\
        --random-suffix 2 --quiet-only

  Fraction-based (auto-compute frc_replicates and book_positions):
    python3 make_training_epd.py --total 100000 --frc-fraction 0.20 \\
        --random-suffix 2 --quiet-only
    → frc_replicates=21, 20160 FRC-derived + 79840 book = ~100000 before filter
""",
    )

    # Sizing
    parser.add_argument("--book", default=None, metavar="FILE",
                        help="Polyglot book .bin file "
                             "(default: normbk02.bin alongside this script)")
    parser.add_argument("--book-positions", type=int, default=None, metavar="N",
                        help="Book positions to sample "
                             "(default: 2000, or computed from --total/--frc-fraction)")
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

    # Suffix / quality
    parser.add_argument("--random-suffix", type=int, default=0, metavar="K",
                        help="Random moves to play after each book/FRC position "
                             "(default: 0).  Greatly increases unique position count.")
    parser.add_argument("--quiet-only", action="store_true", default=False,
                        help="Restrict random suffix moves to non-captures AND "
                             "filter output positions to those within --eval-limit cp "
                             "of even using a Leaf eval binary.")

    # Eval filter (active when --quiet-only)
    parser.add_argument("--eval-binary", default=None, metavar="FILE",
                        help="Leaf binary for eval filtering (default: "
                             "Leaf_vclassic_eval alongside this script).  "
                             "Only used with --quiet-only.")
    parser.add_argument("--eval-limit", type=int, default=50, metavar="CP",
                        help="Discard positions where |score| > this many centipawns "
                             "(default: 50 = 0.5 pawns).  Only used with --quiet-only.")
    parser.add_argument("--eval-depth", type=int, default=10, metavar="N",
                        help="Search depth for eval filtering (default: 10).  "
                             "Only used with --quiet-only.")
    parser.add_argument("--eval-workers", type=int,
                        default=max(1, cpu_count // 2), metavar="N",
                        help=f"Parallel eval engine processes "
                             f"(default: {max(1, cpu_count // 2)}; "
                             f"only used with --quiet-only)")

    # Book walk / output
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
    if args.eval_limit < 0:
        parser.error("--eval-limit must be >= 0.")
    if args.eval_depth < 1:
        parser.error("--eval-depth must be >= 1.")

    book_path   = args.book   or os.path.join(script_dir, "normbk02.bin")
    output_path = args.output or os.path.join(script_dir, "training_openings.epd")

    # --- Resolve eval binary (only matters when --quiet-only) ---
    eval_binary = None
    if args.quiet_only:
        default_eval = os.path.join(script_dir, "Leaf_vclassic_eval")
        candidate    = args.eval_binary or default_eval
        if os.path.isfile(candidate):
            eval_binary = candidate
            print(f"Eval binary:  {os.path.basename(eval_binary)}"
                  f"  (limit ±{args.eval_limit} cp, depth {args.eval_depth},"
                  f" {args.eval_workers} worker(s))")
        else:
            print(f"Warning: eval binary not found: {candidate}")
            print(f"  To enable eval filtering, compile with:")
            print(f"    perl src/comp.pl classic_eval OVERWRITE")
            print(f"  Proceeding without eval filter.")

    suffix_desc = (f"{args.random_suffix} {'quiet ' if args.quiet_only else ''}random move(s)"
                   if args.random_suffix > 0 else "none")

    # --- FRC positions ---
    frc_label = f"960 × {frc_replicates} replicates" if frc_replicates > 1 else "960 × 1"
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
    combined = frc_epds + book_epds
    rng = random.Random(args.seed + 2)
    rng.shuffle(combined)

    # --- Eval filter (--quiet-only + eval binary present) ---
    if eval_binary:
        n_before = len(combined)
        print(f"\nFiltering {n_before:,} positions by eval "
              f"(|score| ≤ {args.eval_limit} cp, depth {args.eval_depth}) ...")
        combined = filter_by_eval(
            combined, eval_binary,
            args.eval_limit, args.eval_depth, args.eval_workers,
        )
        n_after   = len(combined)
        n_removed = n_before - n_after
        print(f"  Kept {n_after:,}  ({n_removed:,} rejected,"
              f" {n_removed / n_before * 100:.1f}%)")
        # Re-shuffle after filtering (filter_by_eval preserves generation order)
        rng2 = random.Random(args.seed + 3)
        rng2.shuffle(combined)

    # --- Write ---
    with open(output_path, "w") as f:
        for epd in combined:
            f.write(epd + "\n")

    total    = len(combined)
    frc_kept = sum(1 for e in combined if e in set(frc_epds))   # approximate
    book_kept = total - frc_kept

    print(f"\nWritten {total:,} positions → {output_path}")
    if args.random_suffix > 0:
        q = "quiet " if args.quiet_only else ""
        print(f"  Suffix: {args.random_suffix} {q}random move(s) per position")
    if eval_binary:
        print(f"  Eval filter: ±{args.eval_limit} cp  (depth {args.eval_depth})")
    print(f"\nUse with cutechess-cli:")
    print(f"  -openings file={os.path.basename(output_path)} format=epd order=random "
          f"-variant fischerandom -noswap")


if __name__ == "__main__":
    main()
