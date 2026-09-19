#!/usr/bin/env python3
# Leaf — GPL v3-or-later.  Copyright (C) 2026 Daniel C. Homan.
"""Build a game-stratified composite corpus from several legs' raw TSV dumps.

Why this exists
---------------
A leg's assembled ``corpus.tsv`` is fine for that leg's own consolidation, but
it is useless for a multi-leg experiment.  It carries only the row type and
gate that leg happened to use, so a wider gate or the leaf rows cannot be
recovered from it.  And its ``gid`` column is RENUMBERED per leg from 0, so
game 5 of one leg and game 5 of another are indistinguishable -- which breaks
both game-stratified sampling across legs and root-to-leaf pairing.  The raw
per-leg dumps (``<tag>.<pid>.root.tsv.gz`` / ``...leaf.tsv.gz``) keep the
original game ids, so this script builds from those.

The sampling rule
-----------------
Two properties are wanted at once, and the naive approach gets neither:

  (1) every arm trains on the SAME number of rows, and
  (2) every arm draws on approximately the same number of GAMES, with games
      weighted equally.

Uniform sampling over rows gives (1) and destroys (2) -- rows per game has
mean 77 and sd 47 at ``--quiet-cp 60`` (min 2, max 389), so a long game would
outweigh a short one 195:1.  Uniform sampling over games gives (2) and
destroys (1).

So: draw a fixed QUOTA of rows per game, uniformly within the game.  At quota
19 about 98.8% of games can fill it, so game weights are equal to within ~1%
and the row total is quota x games to the same tolerance.  The residual is
removed exactly (see ``--target-rows``), leaving both properties intact.

Equal contribution per SOURCE is enforced on top: each source is sampled down
to the same game count and the same row target, so a leg with a longer dump
cannot dominate the composite.

Passes
------
Two, because an exact row total cannot be known until the eligible rows have
been counted.  Pass 1 counts eligible rows per game (cached in a JSON sidecar
keyed by file+gate, so arms sharing a gate pay for it once).  Pass 2 emits.

Usage
-----
    python3 sample_corpus.py --source m260916-{2,3,4,5}e6g_work \\
        --rows root --quiet-cp 60 --quota 19 \\
        --out arms/composite_root.tsv --seed 1000

    # matched-dose single-leg null: same row total, one leg, big quota
    python3 sample_corpus.py --source m260916-5e6g_work \\
        --rows root --quiet-cp 60 --quota 100 \\
        --target-rows <the composite's total> \\
        --out arms/single_root.tsv --seed 1000
"""

import argparse
import gzip
import hashlib
import json
import os
import random
import sys
from pathlib import Path

HEADER = "# tdleaf-corpus axis=game-ply\n"
COLS = "fen\tcp\tresult\tply\tdepth\tgid\tendply\tgate\n"


def log(msg):
    print(f"[sample_corpus] {msg}", file=sys.stderr, flush=True)


def die(msg):
    log(f"ERROR: {msg}")
    sys.exit(1)


def find_dump(src, kind):
    """Locate the raw <kind> dump inside a work directory (or accept a file)."""
    p = Path(src)
    if p.is_file():
        return p
    if not p.is_dir():
        die(f"{src} is neither a file nor a directory")
    hits = sorted(p.glob(f"*.{kind}.tsv.gz")) + sorted(p.glob(f"*.{kind}.tsv"))
    if not hits:
        die(f"no *.{kind}.tsv[.gz] in {p} -- the raw dumps are what carry gid; "
            f"the assembled corpus.tsv does not")
    if len(hits) > 1:
        die(f"{len(hits)} {kind} dumps in {p}: {[h.name for h in hits]} -- "
            f"pass the one you want explicitly")
    return hits[0]


def opener(path):
    return gzip.open(path, "rt", newline="") if str(path).endswith(".gz") \
        else open(path, "r", newline="")


def eligible(parts, gate_cp):
    """The trainer's load-time quiet gate, reproduced exactly.

    Rows without the 8th 'gate' column are kept: their gate is unknowable and
    the trainer keeps them too (nnue_batch_train.cpp, bt_load_file)."""
    if gate_cp <= 0 or len(parts) < 8:
        return True
    try:
        return abs(int(parts[1]) - int(parts[7])) <= gate_cp
    except ValueError:
        return True


def load_gids(path):
    """Game ids to restrict to, one per line (or a corpus TSV to read them
    from).  Used to hold the GAME SET fixed across arms that change the row
    type: leaf rows survive in games whose root rows the quiet gate removed, so
    an unrestricted leaf sample draws more games at fewer rows each, and the
    leaf-vs-root contrast would carry a game-count difference inside it."""
    gids = set()
    with opener(path) as fh:
        for line in fh:
            if line.startswith("#") or line.startswith("fen\t"):
                continue
            parts = line.rstrip("\n").split("\t")
            gids.add(parts[5] if len(parts) >= 7 else parts[0])
    return gids


def count_games(path, gate_cp, cache_dir):
    """Pass 1: eligible rows per game, in file order.  Returns a list of
    (gid, n_eligible).  Cached -- the pass costs a full decompress."""
    # Keyed on the RESOLVED PATH, not the basename: two legs can carry dumps
    # with the same file name, and a collision here makes pass 2 stream one
    # file against another's game list -- which emits zero rows and says
    # nothing.  That silent failure is why the digest is in the key.
    dig = hashlib.sha1(str(Path(path).resolve()).encode()).hexdigest()[:10]
    key = f"{Path(path).name}.g{gate_cp}.{dig}.counts.json"
    cache = Path(cache_dir) / key if cache_dir else None
    if cache and cache.exists():
        log(f"pass 1: reusing {cache.name}")
        return json.loads(cache.read_text())

    log(f"pass 1: counting {Path(path).name} at gate {gate_cp}")
    counts, cur, n, rows = [], None, 0, 0
    with opener(path) as fh:
        for line in fh:
            if line.startswith("#") or line.startswith("fen\t"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 7:
                continue
            rows += 1
            gid = parts[5]
            if gid != cur:
                if cur is not None:
                    counts.append((cur, n))
                cur, n = gid, 0
            if eligible(parts, gate_cp):
                n += 1
    if cur is not None:
        counts.append((cur, n))
    log(f"pass 1: {rows:,} rows, {len(counts):,} games, "
        f"{sum(c for _, c in counts):,} eligible")
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(counts))
    return counts


def plan(counts, quota, n_games, target_rows, rng, keep_gids=None):
    """Decide how many rows to take from each game.

    Games are chosen uniformly (n_games of them), each contributing
    min(eligible, quota).  If that overshoots target_rows the surplus is shed
    one row at a time from randomly chosen games, which keeps game weights
    within one row of each other -- far gentler than dropping whole games.

    ⚠️ This is for trimming a SMALL surplus (a few percent).  Shedding a large
    one flattens the quota down past the short games and leaves only the long
    ones -- at quota 100000 against a 3.9% target it kept 121 games of 1244,
    destroying property (2).  To take a small fraction of a corpus, lower the
    QUOTA (--count-only sizes it) instead of leaning on --target-rows."""
    idx = list(range(len(counts)))
    if keep_gids is not None:
        idx = [i for i in idx if counts[i][0] in keep_gids]
    if n_games < len(idx):
        idx = rng.sample(idx, n_games)
    idx.sort()

    take = {i: min(counts[i][1], quota) for i in idx}
    take = {i: t for i, t in take.items() if t > 0}
    total = sum(take.values())
    if target_rows is None or total <= target_rows:
        return take, total

    surplus = total - target_rows
    # Shed uniformly first (O(n)), then distribute what is left one row at a
    # time (O(remainder) < n).  Shedding one row at a time from the start would
    # be O(surplus), which is minutes when the overshoot is millions of rows.
    while surplus > 0:
        donors = [i for i, t in take.items() if t > 0]
        if not donors:
            die("cannot reach --target-rows: every sampled game is exhausted; "
                "raise --quota or --games-per-source")
        flat = surplus // len(donors)
        if flat:
            for i in donors:
                d = min(flat, take[i])
                take[i] -= d
                surplus -= d
        else:
            for i in rng.sample(donors, surplus):
                take[i] -= 1
                surplus -= 1
    take = {i: t for i, t in take.items() if t > 0}
    return take, target_rows


def emit(path, gate_cp, counts, take, rng, out):
    """Pass 2: stream the file, reservoir-sample take[g] rows from game g."""
    wanted = {counts[i][0]: t for i, t in take.items()}
    written = 0
    cur, buf, need = None, None, 0
    done = set()

    def flush():
        nonlocal written
        if buf:
            for line in buf:
                out.write(line)
            written += len(buf)

    with opener(path) as fh:
        seen = 0
        for line in fh:
            if line.startswith("#") or line.startswith("fen\t"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 7:
                continue
            gid = parts[5]
            if gid != cur:
                flush()
                cur = gid
                # A game is written contiguously by the dumper, so a gid seen
                # again is a different game that collided in the 32-bit hash
                # (~1800 pairs per 4M games).  Taking its quota a second time
                # would double-weight that slot, so skip it.
                if gid in done:
                    need, buf = 0, None
                else:
                    done.add(gid)
                    need = wanted.get(gid, 0)
                    buf = [] if need else None
                seen = 0
            if buf is None or not eligible(parts, gate_cp):
                continue
            # Reservoir sampling of `need` rows, uniform over the game's
            # eligible rows and streaming in one pass.
            if len(buf) < need:
                buf.append(line)
            else:
                j = rng.randrange(seen + 1)
                if j < need:
                    buf[j] = line
            seen += 1
        flush()
    return written


def main():
    ap = argparse.ArgumentParser(
        description="Game-stratified composite corpus sampler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    ap.add_argument("--source", nargs="+", required=True,
                    help="work directories (or explicit .tsv[.gz] files)")
    ap.add_argument("--rows", choices=["root", "leaf"], default="root",
                    help="which raw dump to sample (default root)")
    ap.add_argument("--quiet-cp", type=int, default=60,
                    help="load-time quiet gate |cp - gate|, matching the "
                         "trainer's --bt-quiet-cp (default 60; 0 = keep all)")
    ap.add_argument("--quota", type=int, default=19,
                    help="rows drawn per game (default 19)")
    ap.add_argument("--games-per-source", type=int, default=None,
                    help="games sampled from each source (default: the "
                         "smallest source's game count, so every leg "
                         "contributes equally)")
    ap.add_argument("--target-rows", type=int, default=None,
                    help="exact total row count; the surplus is shed one row "
                         "per game at a time.  Use it to match an earlier "
                         "run's total exactly")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", default=None,
                    help="JSON accounting sidecar (default: <out>.json)")
    ap.add_argument("--restrict-gids", default=None,
                    help="file of game ids (or a corpus TSV to read them "
                         "from): sample only these games.  Use it to hold the "
                         "game set fixed when changing --rows, so the row-type "
                         "contrast is not also a game-set contrast")
    ap.add_argument("--count-only", action="store_true",
                    help="run pass 1 only and print per-source "
                         "{games, eligible_rows} as JSON on stdout.  Use it to "
                         "size --quota for a target row count before "
                         "committing to the emit pass")
    ap.add_argument("--count-cache", default=None,
                    help="directory for pass-1 count caches "
                         "(default: alongside --out)")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = args.count_cache or str(out.parent / "counts")

    dumps = [find_dump(s, args.rows) for s in args.source]
    for s, d in zip(args.source, dumps):
        log(f"source {s} -> {d.name}")

    counts = [count_games(d, args.quiet_cp, cache_dir) for d in dumps]

    if args.count_only:
        print(json.dumps({s: {"games": len(c),
                              "eligible_rows": sum(n for _, n in c)}
                          for s, c in zip(args.source, counts)}, indent=2))
        return

    keep = load_gids(args.restrict_gids) if args.restrict_gids else None
    if keep is not None:
        log(f"restricting to {len(keep):,} game ids from "
            f"{Path(args.restrict_gids).name}")
        counts = [[(g, n) for g, n in c if g in keep] for c in counts]
        for s_, c in zip(args.source, counts):
            if not c:
                die(f"{s_} has no games in --restrict-gids")

    ng = min(len(c) for c in counts)
    if args.games_per_source is not None:
        if args.games_per_source > ng:
            die(f"--games-per-source {args.games_per_source} exceeds the "
                f"smallest source ({ng} games)")
        ng = args.games_per_source
    log(f"games per source: {ng:,} (of {[len(c) for c in counts]})")

    per_src_target = None
    if args.target_rows is not None:
        if args.target_rows % len(dumps):
            log(f"note: --target-rows {args.target_rows} is not divisible by "
                f"{len(dumps)} sources; the remainder goes to the first")
        per_src_target = [args.target_rows // len(dumps)] * len(dumps)
        per_src_target[0] += args.target_rows - sum(per_src_target)

    plans = []
    for i, c in enumerate(counts):
        rng = random.Random(args.seed + 7919 * i)
        tgt = per_src_target[i] if per_src_target else None
        take, total = plan(c, args.quota, ng, tgt, rng)
        if tgt is not None and total < tgt:
            die(f"source {args.source[i]} can supply only {total:,} rows at "
                f"quota {args.quota} over {ng:,} games, short of its "
                f"{tgt:,} share -- raise --quota or lower --target-rows")
        plans.append((take, total))
        log(f"{args.source[i]}: {len(take):,} games contribute {total:,} rows "
            f"(mean {total / max(len(take), 1):.2f}/game)")

    written_per = []
    with open(out, "w", newline="") as fh:
        fh.write(HEADER)
        fh.write(COLS)
        for i, (d, c) in enumerate(zip(dumps, counts)):
            rng = random.Random(args.seed + 104729 * i)
            n = emit(d, args.quiet_cp, c, plans[i][0], rng, fh)
            if n != plans[i][1]:
                die(f"pass 2 wrote {n:,} rows from {d.name} but pass 1 "
                    f"planned {plans[i][1]:,} -- the count cache does not "
                    f"describe this file.  Delete {cache_dir} and retry")
            written_per.append(n)
            log(f"pass 2: {d.name} -> {n:,} rows")

    total = sum(written_per)
    manifest = {
        "out": str(out),
        "rows_total": total,
        "rows_per_source": dict(zip(args.source, written_per)),
        "games_per_source": {s: len(p[0]) for s, p in zip(args.source, plans)},
        "dumps": {s: str(d) for s, d in zip(args.source, dumps)},
        "row_type": args.rows,
        "quiet_cp": args.quiet_cp,
        "quota": args.quota,
        "games_per_source_cap": ng,
        "target_rows": args.target_rows,
        "seed": args.seed,
        "restrict_gids": args.restrict_gids,
    }
    mpath = Path(args.manifest) if args.manifest else out.with_suffix(
        out.suffix + ".json")
    mpath.write_text(json.dumps(manifest, indent=2) + "\n")
    log(f"wrote {total:,} rows -> {out}  ({os.path.getsize(out) / 1e9:.2f} GB)")
    log(f"manifest -> {mpath}")
    # The trainer wants the exact row count for --bt-max (it reserves once
    # instead of growing the record vector by doubling).
    print(total)


if __name__ == "__main__":
    main()
