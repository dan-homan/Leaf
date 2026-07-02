#!/usr/bin/env python3
"""
extract_quiet_positions.py — build an offline-training position set from Leaf PGNs.

Replays each game (python-chess, Chess960-aware) and emits one TSV record per
QUIET position:

    fen \t cp \t result \t ply \t depth \t gid

    fen     Shredder-FEN of the position BEFORE the recorded move
    cp      search eval of that position, WHITE POV, centipawns (from the PGN
            move comment "{+0.25/6 0.007s}"; mover POV in the file, negated
            here for Black to move)
    result  game outcome, WHITE POV: 1 / 0.5 / 0
    ply     1-based ply index from the game's start position
    depth   search depth of the eval (from the "/6" comment field)
    gid     stable game id (for train/validation splits BY GAME)

cp and result are stored separately so the trainer's lambda-blend
    p_target = lambda * result + (1 - lambda) * sigmoid(cp / K)
keeps lambda and K as training-time hyperparameters, not baked into the data.

Quiet filters (a position is skipped if ANY holds):
  - side to move is in check
  - the played move is a capture, promotion, or gives check
  - no parsable eval comment, or the eval is a mate score
  - |eval| > --max-eval centipawns
  - ply <= --min-ply (opening-book starts repeat massively)
  - halfmove clock >= --max-fifty (shuffle endgames)

Duplicate control: positions are keyed by polyglot Zobrist hash; at most
--max-dups records are kept per unique position (0 = unlimited).  NOTE: the
dedup table lives in RAM (~100 B per unique position).

Usage (from learn/):
    # depth-8 self-play file, both sides are the learner
    python3 extract_quiet_positions.py \
        --pgn-file pgn/nn-fresh-260628/match_nn-fresh-260628_2e6g.pgn \
        --out quiet_d8.tsv

    # rotation segments, keep only positions where the learner moved
    python3 extract_quiet_positions.py \
        --pgn-dir rotation_segs --player Leaf_vtrain_nn-fresh-260628_a \
        --out quiet_rotation.tsv

    # sampled subset for a fast pilot
    python3 extract_quiet_positions.py --pgn-file big.pgn --max-games 50000 \
        --seed 42 --out quiet_sample.tsv
"""

import argparse
import gzip
import io
import multiprocessing as mp
import os
import random
import re
import sys
import time
from pathlib import Path

import chess
import chess.pgn
import chess.polyglot

EVAL_RE   = re.compile(r'([+-]?)(M?)(\d+(?:\.\d+)?)/(\d+)')
RESULT_MAP = {'1-0': '1', '0-1': '0', '1/2-1/2': '0.5'}

BATCH_GAMES = 200          # games per worker task
REPORT_EVERY = 10_000      # progress cadence (games)


# ---------------------------------------------------------------------------
# Game splitting — stream a PGN file and yield complete game texts
# ---------------------------------------------------------------------------

def iter_game_texts(path):
    """Yield one string per game.  A game starts at a line beginning with
    '[Event ' and runs until the next such line."""
    buf = []
    with open(path, 'r', encoding='utf-8', errors='replace') as fh:
        for line in fh:
            if line.startswith('[Event ') and buf:
                yield ''.join(buf)
                buf = [line]
            else:
                buf.append(line)
    if buf and any(l.startswith('[Event ') for l in buf[:3]):
        yield ''.join(buf)


# ---------------------------------------------------------------------------
# Worker: parse one batch of game texts → list of per-game record lists
# ---------------------------------------------------------------------------

_CFG = None  # per-worker config set by _init_worker


def _init_worker(cfg):
    global _CFG
    _CFG = cfg


def _parse_eval(comment):
    """Return (cp_mover_pov, depth) or None for missing/mate evals."""
    m = EVAL_RE.search(comment)
    if not m:
        return None
    sign, mate, val, depth = m.groups()
    if mate:
        return None
    cp = float(val) * 100.0
    if sign == '-':
        cp = -cp
    return int(round(cp)), int(depth)


def _extract_batch(game_texts):
    """Returns list (one entry per game) of lists of
    (zobrist, fen, cp_white, ply, depth) plus the game's result char.
    Also returns per-filter counters."""
    cfg = _CFG
    out = []
    stats = dict(games=0, no_result=0, parse_fail=0, plies=0, kept=0,
                 in_check=0, tactical=0, no_eval=0, eval_cap=0,
                 min_ply=0, fifty=0, not_player=0)
    for text in game_texts:
        stats['games'] += 1
        try:
            game = chess.pgn.read_game(io.StringIO(text))
            if game is None:
                stats['parse_fail'] += 1
                continue
            result = RESULT_MAP.get(game.headers.get('Result', '*'))
            if result is None:
                stats['no_result'] += 1
                continue
            white_name = game.headers.get('White', '')
            black_name = game.headers.get('Black', '')

            board = game.board()
            records = []
            ply = 0
            for node in game.mainline():
                move = node.move
                ply += 1
                stats['plies'] += 1
                keep = True
                if cfg['player'] is not None:
                    mover = white_name if board.turn == chess.WHITE else black_name
                    if cfg['player'] not in mover:
                        stats['not_player'] += 1
                        keep = False
                if keep and ply <= cfg['min_ply']:
                    stats['min_ply'] += 1
                    keep = False
                if keep and board.is_check():
                    stats['in_check'] += 1
                    keep = False
                if keep and (board.is_capture(move) or move.promotion
                             or board.gives_check(move)):
                    stats['tactical'] += 1
                    keep = False
                if keep and board.halfmove_clock >= cfg['max_fifty']:
                    stats['fifty'] += 1
                    keep = False
                if keep:
                    ev = _parse_eval(node.comment)
                    if ev is None:
                        stats['no_eval'] += 1
                        keep = False
                    elif abs(ev[0]) > cfg['max_eval']:
                        stats['eval_cap'] += 1
                        keep = False
                if keep:
                    cp_mover, depth = ev
                    cp_white = cp_mover if board.turn == chess.WHITE else -cp_mover
                    records.append((chess.polyglot.zobrist_hash(board),
                                    board.fen(shredder=True),
                                    cp_white, ply, depth))
                    stats['kept'] += 1
                board.push(move)
            out.append((result, records))
        except Exception:
            stats['parse_fail'] += 1
            continue
    return out, stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def count_games(path):
    n = 0
    with open(path, 'rb') as fh:
        while True:
            chunk = fh.read(1 << 24)
            if not chunk:
                break
            n += chunk.count(b'[Event ')
    return n


def main():
    ap = argparse.ArgumentParser(
        description='Extract quiet positions with eval + outcome labels from PGNs.')
    ap.add_argument('--pgn-file', action='append', default=[],
                    help='PGN file to extract from (repeatable)')
    ap.add_argument('--pgn-dir', default=None,
                    help='Directory of .pgn files (all are processed, sorted)')
    ap.add_argument('--out', required=True,
                    help='Output TSV path (.gz suffix → gzip-compressed)')
    ap.add_argument('--player', default=None,
                    help='Keep only positions where this engine (substring of the '
                         'White/Black header) is to move')
    ap.add_argument('--max-games', type=int, default=0,
                    help='Random game sample size per the whole input (0 = all)')
    ap.add_argument('--min-ply', type=int, default=8,
                    help='Skip the first N plies of each game (default 8)')
    ap.add_argument('--max-eval', type=int, default=1500,
                    help='Skip positions with |eval| above this, cp (default 1500)')
    ap.add_argument('--max-fifty', type=int, default=80,
                    help='Skip positions with halfmove clock >= N (default 80)')
    ap.add_argument('--max-dups', type=int, default=4,
                    help='Max records kept per unique position (default 4, 0 = off)')
    ap.add_argument('--workers', type=int, default=max(1, os.cpu_count() - 2),
                    help='Parallel parser processes')
    ap.add_argument('--seed', type=int, default=42, help='Sampling seed')
    args = ap.parse_args()

    files = [Path(p) for p in args.pgn_file]
    if args.pgn_dir:
        files += sorted(Path(args.pgn_dir).glob('*.pgn'))
    if not files:
        sys.exit('No input PGN files (use --pgn-file / --pgn-dir).')
    for p in files:
        if not p.is_file():
            sys.exit(f'Not a file: {p}')

    # Game sampling probability from a fast count pass.
    keep_prob = 1.0
    if args.max_games > 0:
        total = sum(count_games(p) for p in files)
        keep_prob = min(1.0, args.max_games / max(total, 1))
        print(f'Sampling: {total:,} games total, keep_prob={keep_prob:.4f}')

    cfg = dict(player=args.player, min_ply=args.min_ply,
               max_eval=args.max_eval, max_fifty=args.max_fifty)

    opener = gzip.open if args.out.endswith('.gz') else open
    out = opener(args.out, 'wt')
    out.write('# extract_quiet_positions v1\n')
    out.write(f'# files: {", ".join(str(p) for p in files)}\n')
    out.write(f'# filters: min_ply={args.min_ply} max_eval={args.max_eval} '
              f'max_fifty={args.max_fifty} max_dups={args.max_dups} '
              f'player={args.player} max_games={args.max_games} seed={args.seed}\n')
    out.write('fen\tcp\tresult\tply\tdepth\tgid\n')

    rng = random.Random(args.seed)
    dup_count = {} if args.max_dups > 0 else None
    totals = {}
    gid = 0
    written = 0
    dup_skipped = 0
    t0 = time.time()

    def batches():
        batch = []
        for path in files:
            print(f'  reading {path.name} ...', flush=True)
            for text in iter_game_texts(path):
                if keep_prob < 1.0 and rng.random() > keep_prob:
                    continue
                batch.append(text)
                if len(batch) >= BATCH_GAMES:
                    yield batch
                    batch = []
        if batch:
            yield batch

    with mp.Pool(args.workers, initializer=_init_worker, initargs=(cfg,)) as pool:
        for games, stats in pool.imap(_extract_batch, batches()):
            for k, v in stats.items():
                totals[k] = totals.get(k, 0) + v
            for result, records in games:
                gid += 1
                for zob, fen, cp_white, ply, depth in records:
                    if dup_count is not None:
                        c = dup_count.get(zob, 0)
                        if c >= args.max_dups:
                            dup_skipped += 1
                            continue
                        dup_count[zob] = c + 1
                    out.write(f'{fen}\t{cp_white}\t{result}\t{ply}\t{depth}\t{gid}\n')
                    written += 1
            if totals.get('games', 0) % REPORT_EVERY < BATCH_GAMES:
                el = time.time() - t0
                print(f'  {totals.get("games", 0):>9,} games  '
                      f'{written:>11,} positions  ({el:6.0f}s)', flush=True)

    out.close()
    el = time.time() - t0

    print(f'\nDone in {el:.0f}s.  {totals.get("games",0):,} games parsed, '
          f'{written:,} positions written to {args.out}')
    plies = max(totals.get('plies', 1), 1)
    print(f'  plies seen        {plies:>12,}')
    for key, label in [('kept', 'kept (pre-dedup)'), ('in_check', 'in check'),
                       ('tactical', 'tactical move'), ('no_eval', 'no/mate eval'),
                       ('eval_cap', '|eval| cap'), ('min_ply', 'min-ply'),
                       ('fifty', 'fifty-move'), ('not_player', 'not player')]:
        v = totals.get(key, 0)
        print(f'  {label:<18}{v:>12,}  ({100.0*v/plies:5.1f}%)')
    if dup_count is not None:
        print(f'  dup-capped        {dup_skipped:>12,}')
        print(f'  unique positions  {len(dup_count):>12,}')
    if totals.get('parse_fail', 0) or totals.get('no_result', 0):
        print(f'  parse failures    {totals.get("parse_fail",0):>12,}   '
              f'no-result games {totals.get("no_result",0):,}')


if __name__ == '__main__':
    main()
