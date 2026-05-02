#!/usr/bin/env python3
"""
extract_positions.py  —  Stream Leaf self-play PGN files into a per-position
                         parquet dataset for calibration analysis.

Columns
-------
  game_id          int32    sequential game counter (across all files)
  training_stage   int8     0=untrained … 6=1.5M-game-trained  (see STAGE_MAP)
  ply              int16    0-based ply index (0 = White's first move)
  white_score_cp   int16    eval from White's POV, centipawns, capped ±MATE_CP
  result           float32  White's result: 1.0 / 0.5 / 0.0
  n_plies          int16    total plies in the game

Scores are read from the inline PGN comment on each half-move, e.g.:
  "1. e4 {+0.31/8 0.12s} e5 {-0.30/8 0.09s}"
Score units in the PGN are pawns (e.g. +0.31); the script converts to
centipawns.  Mate scores ("+M3") are mapped to ±MATE_CP.

Usage
-----
  python3 extract_positions.py                         # defaults below
  python3 extract_positions.py \\
      --pgn-dir  ../learn/pgn/nn-fresh-260410 \\
      --out      ../learn/positions.parquet \\
      --max-games 200000 \\
      --min-plies 8 \\
      --seed 42
"""

import re
import sys
import random
import argparse
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MATE_CP = 2000          # centipawn cap for mate / very large scores
BATCH_GAMES = 20_000    # flush a parquet row-group every N games

RESULT_MAP = {'1-0': 1.0, '1/2-1/2': 0.5, '0-1': 0.0}

# Map filename substrings → training-stage index (earliest → latest).
# Stage reflects the number of self-play games the network was trained on
# *before* this match was recorded.
STAGE_MAP = [
    ('_0g',      0),   # untrained network
    ('_1e4g',    1),   # after   10 000 games
    ('_5e4g',    2),   # after   50 000 games
    ('_2e5g',    3),   # after  200 000 games
    ('_5e5g',    4),   # after  500 000 games
    ('_8e5g',    5),   # after  800 000 games
    ('_1.5e6g',  6),   # after 1 500 000 games
]

SCHEMA = pa.schema([
    pa.field('game_id',        pa.int32()),
    pa.field('training_stage', pa.int8()),
    pa.field('ply',            pa.int16()),
    pa.field('white_score_cp', pa.int16()),
    pa.field('result',         pa.float32()),
    pa.field('n_plies',        pa.int16()),
])

# Matches the score portion of a Leaf PGN comment, e.g.:
#   {+2.74/6 0.002s}   →  group(1) = "+2.74"
#   {-M4/3 0s}         →  group(1) = "-M4"
#   {0.00/6 0.01s}     →  group(1) = "0.00"
SCORE_RE = re.compile(r'\{([+-]?M?\d+(?:\.\d+)?)/\d+')
RESULT_HDR_RE = re.compile(r'\[Result\s+"([^"]+)"\]')

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def stage_for_file(fname: str) -> int:
    for tag, stage in STAGE_MAP:
        if tag in fname:
            return stage
    return -1  # unknown


def parse_score_cp(s: str) -> int:
    """Convert a PGN comment score token to centipawns (capped ±MATE_CP)."""
    if 'M' in s:
        return MATE_CP if not s.startswith('-') else -MATE_CP
    cp = int(round(float(s) * 100))
    return max(-MATE_CP, min(MATE_CP, cp))


def mover_to_white_pov(scores_mover: list) -> list:
    """
    PGN scores are from the moving side's perspective.
    Flip sign on odd plies (Black's moves) to get White-POV scores.
    """
    return [s if i % 2 == 0 else -s for i, s in enumerate(scores_mover)]


# ---------------------------------------------------------------------------
# PGN streaming parser
# ---------------------------------------------------------------------------

def iter_games(path: Path):
    """
    Yield (result_float, white_pov_scores_cp) for each complete game in a
    PGN file.  Uses a simple state machine; no external PGN library needed.

    States: WAITING → HEADERS → MOVES → (back to WAITING)
    """
    state = 'WAITING'
    result = None
    scores_mover = []

    def _emit():
        nonlocal result, scores_mover
        wp = mover_to_white_pov(scores_mover)
        result_val = result
        scores_mover = []
        result = None
        return result_val, wp

    with open(path, 'r', encoding='utf-8', errors='replace') as fh:
        for raw in fh:
            line = raw.rstrip('\n')

            # ---- header line ------------------------------------------------
            if line.startswith('['):
                if state == 'MOVES' and scores_mover and result is not None:
                    yield _emit()
                state = 'HEADERS'
                m = RESULT_HDR_RE.match(line)
                if m:
                    result = RESULT_MAP.get(m.group(1))
                continue

            # ---- blank line -------------------------------------------------
            if not line.strip():
                if state == 'HEADERS':
                    state = 'MOVES'
                elif state == 'MOVES':
                    if scores_mover and result is not None:
                        yield _emit()
                    state = 'WAITING'
                continue

            # ---- move text --------------------------------------------------
            if state == 'HEADERS':
                state = 'MOVES'   # headers→moves without blank line (rare)
            if state in ('MOVES', 'WAITING'):
                state = 'MOVES'
                for m in SCORE_RE.finditer(line):
                    scores_mover.append(parse_score_cp(m.group(1)))

    # flush final game at EOF
    if state == 'MOVES' and scores_mover and result is not None:
        yield _emit()


# ---------------------------------------------------------------------------
# Batch writer
# ---------------------------------------------------------------------------

class ParquetBatchWriter:
    """Accumulates rows in numpy arrays and flushes to a ParquetWriter."""

    def __init__(self, path: Path, schema: pa.Schema):
        self._writer = pq.ParquetWriter(str(path), schema,
                                        compression='snappy')
        self._schema = schema
        self._reset()

    def _reset(self):
        self._game_ids   = []
        self._stages     = []
        self._plies      = []
        self._scores     = []
        self._results    = []
        self._n_plies    = []

    def add_game(self, game_id: int, stage: int, result: float,
                 scores_wp: list):
        n = len(scores_wp)
        self._game_ids.extend([game_id] * n)
        self._stages.extend([stage] * n)
        self._plies.extend(range(n))
        self._scores.extend(scores_wp)
        self._results.extend([result] * n)
        self._n_plies.extend([n] * n)

    def flush(self):
        if not self._game_ids:
            return
        batch = pa.table({
            'game_id':        pa.array(self._game_ids,  type=pa.int32()),
            'training_stage': pa.array(self._stages,    type=pa.int8()),
            'ply':            pa.array(self._plies,     type=pa.int16()),
            'white_score_cp': pa.array(self._scores,    type=pa.int16()),
            'result':         pa.array(self._results,   type=pa.float32()),
            'n_plies':        pa.array(self._n_plies,   type=pa.int16()),
        }, schema=self._schema)
        self._writer.write_table(batch)
        self._reset()

    def close(self):
        self.flush()
        self._writer.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DEFAULT_PGN_DIR = Path(__file__).parent.parent / 'learn' / 'pgn' / 'nn-fresh-260410'
DEFAULT_OUT     = Path(__file__).parent.parent / 'learn' / 'positions.parquet'


def main():
    ap = argparse.ArgumentParser(
        description='Extract per-position data from Leaf self-play PGN files.')
    ap.add_argument('--pgn-dir',   type=Path, default=DEFAULT_PGN_DIR,
                    help='Directory containing .pgn files')
    ap.add_argument('--out',       type=Path, default=DEFAULT_OUT,
                    help='Output parquet file path')
    ap.add_argument('--max-games', type=int,  default=200_000,
                    help='Approximate max games to sample (0 = all ~1.6M)')
    ap.add_argument('--min-plies', type=int,  default=8,
                    help='Skip games shorter than this many plies')
    ap.add_argument('--seed',      type=int,  default=42,
                    help='Random seed for sampling')
    args = ap.parse_args()

    if not args.pgn_dir.is_dir():
        sys.exit(f'ERROR: PGN directory not found: {args.pgn_dir}')

    pgn_files = sorted(args.pgn_dir.glob('*.pgn'))
    if not pgn_files:
        sys.exit(f'ERROR: No .pgn files found in {args.pgn_dir}')

    # Estimate total games to compute sample probability.
    # Approximate: count [Result lines in first file, scale by file size ratio.
    APPROX_TOTAL = 1_600_000
    sample_prob = 1.0 if args.max_games == 0 else min(1.0, args.max_games / APPROX_TOTAL)

    random.seed(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = ParquetBatchWriter(args.out, SCHEMA)

    game_id   = 0   # global counter over all seen games (before sampling)
    kept      = 0   # games written
    batch_buf = 0   # games in current batch

    print(f'PGN dir : {args.pgn_dir}')
    print(f'Output  : {args.out}')
    print(f'Files   : {len(pgn_files)}')
    print(f'Sample  : {sample_prob:.3f}  (target ≈ {args.max_games:,} games)')
    print(f'Min plies: {args.min_plies}')
    print()

    try:
        for fpath in pgn_files:
            stage = stage_for_file(fpath.name)
            file_kept = 0

            for result, scores_wp in iter_games(fpath):
                game_id += 1

                if len(scores_wp) < args.min_plies:
                    continue
                if result is None:
                    continue
                if sample_prob < 1.0 and random.random() >= sample_prob:
                    continue

                writer.add_game(game_id, stage, result, scores_wp)
                kept += 1
                file_kept += 1
                batch_buf += 1

                if batch_buf >= BATCH_GAMES:
                    writer.flush()
                    batch_buf = 0

                if kept % 10_000 == 0:
                    print(f'\r  {kept:>7,} games kept  ({game_id:>9,} seen)  '
                          f'current: {fpath.name}          ',
                          end='', file=sys.stderr)

            print(f'  {fpath.name:<60}  stage={stage}  kept={file_kept:,}',
                  flush=True)

    except KeyboardInterrupt:
        print('\nInterrupted — flushing partial data...', file=sys.stderr)

    writer.close()
    print(f'\nDone.  {kept:,} games written to {args.out}')

    # Quick sanity check
    import pyarrow.parquet as pq2
    meta = pq2.read_metadata(str(args.out))
    print(f'Parquet: {meta.num_row_groups} row groups, '
          f'{meta.num_rows:,} rows, '
          f'{args.out.stat().st_size / 1e6:.1f} MB')


if __name__ == '__main__':
    main()
