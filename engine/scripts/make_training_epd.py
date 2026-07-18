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
# Book sampling (two-phase):
#   Phase 1 — enumerate_book_leaves() BFS-walks the polyglot book up to --ply depth,
#     collecting every unique position reachable (including shallower lines where the
#     book runs out), with a probability weight equal to the sum over all paths that
#     lead there.  This is fast and done once.
#   Phase 2 — generate_from_pool() weighted-samples from the leaf pool, applies random
#     suffix moves, and deduplicates by EPD.  Progress is reported every 10 k positions.
#     A saturation guard stops early if a full pass through the pool yields too few
#     new positions.
#

import argparse
import bisect
import os
import random
import re
import subprocess
import sys
import threading
import time
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


def enumerate_book_leaves(reader, ply):
    """BFS through a Polyglot book up to `ply` depth.

    Returns (boards, weights) where:
      boards   — list of chess.Board objects (one per unique position)
      weights  — parallel list of floats; weight[i] = sum of path probabilities
                 over all routes through the book that reach boards[i]

    Positions where the book has no moves before `ply` depth is reached are
    included as leaves (shallower lines), giving a larger and more diverse pool.
    Transpositions (same position reached via different move orders) are merged:
    their weights are summed, and the board object from the first visit is kept.

    The BFS deduplicates by EPD at each depth level to avoid exponential blowup
    from transpositions.
    """
    # level: dict  epd -> (board, accumulated_path_weight)
    start_board = chess.Board()
    level = {start_board.epd(): (start_board, 1.0)}

    # all_leaves: positions where the book ran out before reaching `ply`
    all_leaves = {}   # epd -> (board, weight)

    for _depth in range(ply):
        next_level = {}
        for epd, (board, weight) in level.items():
            entries = list(reader.find_all(board))
            if not entries:
                # Book has no moves here — collect as a leaf
                if epd in all_leaves:
                    all_leaves[epd] = (all_leaves[epd][0], all_leaves[epd][1] + weight)
                else:
                    all_leaves[epd] = (board.copy(), weight)
                continue

            total_w = sum(e.weight for e in entries)
            if total_w == 0:
                total_w = len(entries)   # treat zero-weight entries as uniform

            for e in entries:
                prob = (e.weight if e.weight > 0 else 1) / total_w
                child = board.copy()
                child.push(e.move)
                child_epd = child.epd()
                child_weight = weight * prob
                if child_epd in next_level:
                    prev_board, prev_w = next_level[child_epd]
                    next_level[child_epd] = (prev_board, prev_w + child_weight)
                else:
                    next_level[child_epd] = (child, child_weight)

        level = next_level
        if not level:
            break   # book exhausted at this depth

    # Positions remaining in `level` at the target depth are also leaves
    for epd, (board, weight) in level.items():
        if epd in all_leaves:
            all_leaves[epd] = (all_leaves[epd][0], all_leaves[epd][1] + weight)
        else:
            all_leaves[epd] = (board.copy(), weight)

    boards  = [b for b, _w in all_leaves.values()]
    weights = [w for _b, w in all_leaves.values()]
    return boards, weights


def generate_from_pool(pool_boards, pool_weights, n_target,
                       random_suffix, quiet_only, seed,
                       saturation_fraction=0.002):
    """Generate n_target unique EPD strings from a pool of starting boards.

    Weighted-samples from pool_boards (using pool_weights), applies random
    suffix moves, and deduplicates by EPD.  Progress is printed every 10 k
    positions.  Stops early when a full pass through the pool yields fewer
    than pool_size * saturation_fraction new positions, indicating the
    position space is nearly exhausted.

    When random_suffix == 0, returns up to min(n_target, pool_size) EPDs
    directly (no suffix generation needed).
    """
    rng = random.Random(seed)
    pool_size = len(pool_boards)

    if pool_size == 0:
        return []

    # No suffix: return pool EPDs directly (weighted order)
    if random_suffix == 0:
        order = list(range(pool_size))
        rng.shuffle(order)
        return [pool_boards[i].epd() for i in order[:n_target]]

    # Build cumulative weight array for O(log n) weighted sampling
    total_w = sum(pool_weights)
    if total_w > 0:
        cumulative = []
        cum = 0.0
        for w in pool_weights:
            cum += w / total_w
            cumulative.append(cum)
    else:
        step = 1.0 / pool_size
        cumulative = [(i + 1) * step for i in range(pool_size)]

    seen      = set()
    positions = []
    attempts  = 0
    cycle     = 0
    last_prog = 0
    min_yield = max(1, int(pool_size * saturation_fraction))

    while len(positions) < n_target:
        cycle_start = len(positions)

        for _ in range(pool_size):
            if len(positions) >= n_target:
                break
            attempts += 1

            # Weighted sample
            r = rng.random()
            idx = bisect.bisect_left(cumulative, r)
            idx = min(idx, pool_size - 1)

            board = pool_boards[idx].copy()
            apply_random_suffix(board, random_suffix, quiet_only, rng)

            if not list(board.legal_moves):
                continue

            epd = board.epd()
            if epd not in seen:
                seen.add(epd)
                positions.append(epd)

                n_found = len(positions)
                if n_found - last_prog >= 10_000:
                    pct = n_found / n_target * 100
                    print(f"\r  Found {n_found:,} / {n_target:,} "
                          f"({pct:.1f}%)  cycle {cycle + 1}, "
                          f"{attempts:,} attempts",
                          end="", flush=True)
                    last_prog = n_found

        cycle_new = len(positions) - cycle_start
        cycle += 1

        if cycle_new < min_yield:
            print(f"\n  Saturation: {cycle_new} new positions in cycle {cycle} "
                  f"(threshold {min_yield}). Stopping.")
            break

    # Final progress line
    n_found = len(positions)
    pct = n_found / n_target * 100 if n_target else 100
    print(f"\r  Found {n_found:,} / {n_target:,} ({pct:.1f}%)  "
          f"{cycle} cycle(s), {attempts:,} attempts       ")

    if n_found < n_target:
        print(f"  Note: only {n_found:,} unique positions found "
              f"(requested {n_target:,}).")

    return positions


def sample_book_positions(book_path, n_target, ply, random_suffix, quiet_only, seed):
    """Sample up to n_target unique EPD strings from a Polyglot opening book.

    Two-phase approach:
      1. Enumerate all unique positions reachable through the book up to `ply`
         depth (including shallower lines where the book runs out), weighted by
         the sum of path probabilities — fast BFS, done once.
      2. Weighted-sample from that leaf pool, apply random suffix moves, and
         deduplicate.  Progress is reported every 10 k positions.

    Returns a list of EPD strings (may be shorter than n_target if the position
    space is exhausted before reaching the target).
    """
    try:
        reader = chess.polyglot.open_reader(book_path)
    except Exception as e:
        print(f"Error: cannot open book {book_path}: {e}", file=sys.stderr)
        return []

    print(f"  Enumerating book leaves at ply {ply} ...", end="", flush=True)
    boards, weights = enumerate_book_leaves(reader, ply)
    reader.close()
    print(f" {len(boards):,} unique positions found.")

    return generate_from_pool(
        boards, weights, n_target,
        random_suffix, quiet_only,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Eval filtering via UCI engine
# ---------------------------------------------------------------------------

def _epd_for_setboard(epd):
    """Return a 4-field EPD with castling stripped, safe for Leaf's setboard command.

    Strips castling rights (sets to '-') to avoid Shredder/Chess960 file-letter
    notation (e.g. 'HAha') that Leaf's xboard parser does not accept.  The en
    passant square is preserved.  Omitting castling rights has negligible effect
    on a depth-10 balance eval.
    """
    parts = epd.split()
    ep = parts[3] if len(parts) > 3 else "-"
    return f"{parts[0]} {parts[1]} - {ep}"


def _engine_worker(engine_path, epd_batch, score_limit, depth):
    """Evaluate a batch of EPDs using a single persistent Leaf process (xboard mode).

    Uses xboard protocol with 'post' (thinking output) and 'sd N' (depth limit).
    The score line format emitted by Leaf post output is:
        <depth>  <score_cp>  <time_cs>  <nodes>  <pv>
    where score_cp is centipawns from the side-to-move's perspective.

    Returns a list of (epd, passes) tuples.  passes is True when a score was
    read and |score| <= score_limit.  Positions that produce no score line
    (e.g. immediate checkmate) are conservatively discarded (passes=False).
    """
    engine_dir = os.path.dirname(os.path.abspath(engine_path))

    proc = subprocess.Popen(
        [engine_path],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True, bufsize=1,
        cwd=engine_dir,
    )

    def send(s):
        proc.stdin.write(s + "\n")
        proc.stdin.flush()

    # xboard setup: force mode (no auto-reply), thinking output, depth cap
    send("xboard")
    send("force")
    send("post")
    send(f"sd {depth}")

    # Post output format: leading whitespace + depth + score_cp + time + nodes + pv
    score_re = re.compile(r"^\s+(\d+)\s+(-?\d+)\s+\d+\s+\d+")
    results  = []

    for epd in epd_batch:
        send(f"setboard {_epd_for_setboard(epd)}")
        send("go")

        last_score = None
        deadline   = time.time() + 60   # safety timeout per position
        while time.time() < deadline:
            line = proc.stdout.readline()
            if not line:
                break
            m = score_re.match(line)
            if m:
                last_score = int(m.group(2))
            if line.startswith("move"):
                break

        # Conservatively discard if no score was produced
        passes = (last_score is not None) and (abs(last_score) <= score_limit)
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

Book sampling uses a two-phase approach for speed at large targets:
  1. BFS enumerates all unique book positions up to --ply depth (including
     shallower lines where the book runs out) — fast, done once.
  2. Weighted-samples from that leaf pool, applying suffix moves, with
     progress printed every 10k positions and early-stop on saturation.

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
                        help="Ply depth for book walks (default: 8)")
    parser.add_argument("--saturation", type=float, default=0.002, metavar="F",
                        help="Stop when a full pool pass yields fewer than "
                             "pool_size × F new positions (default: 0.002).  "
                             "Lower values allow more exhaustive search at the cost "
                             "of more attempts near saturation.")
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
