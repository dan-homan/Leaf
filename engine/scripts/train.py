#!/usr/bin/env python3
# Leaf chess engine — training and analysis tooling.
# Copyright (C) 2026 Daniel C. Homan
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.  See the LICENSE file at the root of this repository.
"""
train.py — one command per hybrid-loop iteration:

    promote state -> online self-play generation (TDLeaf learning + leaf/root
    corpus dumping) -> checkpoint -> assemble corpus -> threaded offline
    consolidation -> promote best-epoch net -> gauntlet.

Implements the playbook in the Hybrid_Loop_Runbook note.  Non-interactive;
every phase is skippable, so it also covers the general "start from a given
point and continue learning" cases:

  Full iteration (generate + consolidate + rate; settled gen-3 recipe —
  pure lambda-return: --bt-lambda defaults to 1.0, the --bt-td-lambda
  distance decay (default TDLEAF_LAMBDA = 0.98) supplies all the outcome
  moderation and is the single knob of record):
    python3 train.py --tag iter3 --games 400000 --depth 8 \
        --state tdL10F10x6_ep4.tdleaf.bin --recompile \
        --bt-K 220 --bt-threads 8 \
        --gauntlet-epochs --gauntlet Leaf_vtdL10F10x6-ep4 Leaf_vclassic_eval

  Chained iteration — --continue reads <prev_tag>_final.json and defaults
  --net/--state/--gauntlet-anchors from it, tracks cumulative_games across
  the chain, and auto-adds Leaf_v<prev_tag>-final to the final gauntlet:
    python3 train.py --tag iter4 --continue iter3 --games 1000000 --depth 8 \
        --bt-K 220 --bt-threads 8 --gauntlet-epochs

  --gauntlet-anchors <binary...> is the fixed opponent list that carries
  forward automatically across a --continue chain (e.g. Leaf_vclassic_eval);
  one-off opponents still go in --gauntlet.  --keep-epoch-states keeps every
  epoch's .tdleaf.bin in <tag>_work/train/ (default: only the promoted epoch
  survives).  --keep-work disables all end-of-run pruning inside
  <tag>_work/ for one run (for postmortems on a run that looks suspicious).

  Consolidation is a single process with within-batch thread parallelism
  (--bt-threads, default 8) — synchronous data parallelism, mathematically
  identical to single-threaded training up to float summation order (the
  removed multi-process --bt-sync sharding suffered gradient staleness that
  destroyed the subtle gen-2+ signal; see docs/TRAINING.md).

  Leaf rows (depth 0) default to the same lambda as roots; give them their own
  outcome weight with --bt-leaf-lambda.  Both lambda ceilings are dormant
  scale knobs in the settled recipe (kept to renormalize across corpora with
  different ply-gap distributions, and for reproducing past runs).

  --gauntlet-epochs rates every epoch snapshot vs the net as it stood BEFORE
  offline training as soon as that epoch finishes training (default 1000 games
  at 1+0.01), and prints an epoch ladder table at the end; the best epoch is
  promoted as the final net.  That baseline is the post-online checkpoint when
  this run generated games, or the incoming live state under --skip-online — so
  the ladder always isolates what offline consolidation added, and --gauntlet is
  reserved for opponents of the final promoted net.  The trainer is SIGSTOPped
  while each ladder match runs so training never contends with the games for
  cores, then SIGCONTed to resume.

  Consolidate-only (offline training on existing corpora):
    python3 train.py --tag redo --skip-online \
        --corpus quiet_d8_260702.tsv --corpus quiet_arch_260628d6.tsv \
        --gauntlet Leaf_v260628-2.4e6g

  Generate-only (online games + corpus dump, no offline training):
    python3 train.py --tag gen3 --games 200000 --depth 8 --skip-train

  Start-to-finish from scratch (--init-nnue creates --net + its companion
  .tdleaf.bin, then the same generate -> consolidate -> gauntlet pipeline runs;
  prior = material|classical|noprior, bare flag = material):
    python3 train.py --tag scratch --net nn-scratch.nnue \
        --init-nnue --games 400000 --depth 8 --bt-threads 8 \
        --gauntlet Leaf_vclassic_eval

Run from engine/learn/.  Artifacts land in learn/ and <tag>_work/:
    <netbase>.tdleaf.bin-pre<tag>      backup of the live state (if --state;
                                       deleted on a successful run)
    <netbase>.tdleaf.bin-<tag>-online  post-generation online checkpoint
                                       (deleted on a successful run)
    <tag>_final.nnue                   consolidated net (piece_val baked;
                                       compile rating binaries from this)
    <tag>_final.tdleaf.bin             seeds the next iteration (pairs with
                                       the ORIGINAL base .nnue)
    <tag>_final.json                   run metadata (cumulative games,
                                       gauntlet anchors, epoch-ladder and
                                       final-gauntlet results) — read by
                                       --continue for the next iteration
    Leaf_v<tag>-final                  rating binary (when gauntlet runs);
                                       never resident in run/
    <tag>_work/                        permanent per-run archive, never
                                       deleted on success: corpus.tsv.gz,
                                       the online-generation PGN (gzipped)
                                       and final-gauntlet PGNs, train/
                                       (train.log with the epoch-ladder and
                                       final-gauntlet tables appended, plus
                                       per-epoch .tdleaf.bin if
                                       --keep-epoch-states).  A failed run's
                                       <tag>_work/ is never pruned.
"""

import argparse
import gzip
import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import zlib
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ENGINE_DIR = SCRIPT_DIR.parent
RUN_DIR    = ENGINE_DIR / "run"
LEARN_DIR  = ENGINE_DIR / "learn"
COMP_PL    = "../src/comp.pl"
DEFAULT_NET = "nn-fresh-260628.nnue"


def log(msg):
    print(f"[train {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def die(msg):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def sh(cmd, cwd, env=None, check=True):
    log("$ " + " ".join(str(c) for c in cmd))
    r = subprocess.run([str(c) for c in cmd], cwd=str(cwd), env=env)
    if check and r.returncode != 0:
        die(f"command failed (rc={r.returncode}): {' '.join(str(c) for c in cmd)}")
    return r.returncode


def open_corpus(path):
    """Open a corpus TSV for reading, transparently handling .gz (archived
    <tag>_work/corpus.tsv.gz files are gzipped; live dumps are not)."""
    return (gzip.open(path, "rt", errors="replace") if str(path).endswith(".gz")
            else open(path, errors="replace"))


# --bt-rows -> index into the (total, root, leaf) count triple.
ROW_KIND = {"both": 0, "root": 1, "leaf": 2}


def corpus_row_counts(path, gate_cp=0):
    """(total, root, leaf) data rows, cached alongside the corpus as
    <path>.rows.  Counting a multi-GB .gz costs a full decompress and the
    window re-reads the same archives every iteration, so it is paid once.
    A one-field cache from before the root/leaf split is ignored and rewritten.

    GATE_CP > 0 counts only rows that survive the quiet re-cut (|cp - gate| <=
    GATE_CP), so a budget built from these numbers means "rows actually trained
    on".  Rows with no `gate` column (corpora dumped before 2026-09-03) were
    already gated at dump time and pass through.  Each gate width gets its own
    cache file so the unfiltered <path>.rows stays valid."""
    cache = Path(str(path) + (f".rows.g{gate_cp}" if gate_cp > 0 else ".rows"))
    if cache.is_file():
        parts = cache.read_text().split()
        if len(parts) >= 3:
            try:
                return tuple(int(x) for x in parts[:3])
            except ValueError:
                pass
    total = root = leaf = 0
    with open_corpus(path) as f:
        for line in f:
            if line.startswith("#") or line.startswith("fen\t"):
                continue
            col = line.rstrip("\n").split("\t")
            if len(col) < 5:
                continue
            if gate_cp > 0 and len(col) >= 8 and \
               abs(int(col[1]) - int(col[7])) > gate_cp:
                continue
            total += 1
            if col[4] == "0":
                leaf += 1
            else:
                root += 1
    try:
        cache.write_text(f"{total} {root} {leaf}\n")
    except OSError:
        pass
    return total, root, leaf


def final_anchor_elo(tag, anchors):
    """Elo of <tag>'s promoted net against the first gauntlet anchor it shares
    with ANCHORS (falling back to its first final-gauntlet entry), or None."""
    if not tag:
        return None
    p = LEARN_DIR / f"{tag}_final.json"
    if not p.is_file():
        return None
    try:
        with open(p) as f:
            fg = json.load(f).get("final_gauntlet") or []
    except (OSError, ValueError):
        return None
    for a in anchors or []:
        for e in fg:
            if e.get("opponent") == a:
                return e.get("elo")
    return fg[0].get("elo") if fg else None


def chain_corpora(continue_tag, window, anchors):
    """Walk parent_tag back from CONTINUE_TAG and return up to WINDOW entries
    (tag, corpus_path, generator_elo) for chain iterations that still have an
    archived corpus.

    A corpus is labelled by its GENERATOR — the net that played those games,
    which is that iteration's PARENT's promoted net, not its own.  (Getting
    this backwards is what made the A1 arm silently include a corpus 75 Elo
    staler than the rest; see docs/Offline_Learning_Investigation.md 2.1.)"""
    out, tag, seen = [], continue_tag, set()
    while tag and len(out) < window and tag not in seen:
        seen.add(tag)
        parent = None
        sc = LEARN_DIR / f"{tag}_final.json"
        if sc.is_file():
            try:
                with open(sc) as f:
                    parent = json.load(f).get("parent_tag")
            except (OSError, ValueError):
                parent = None
        corpus = LEARN_DIR / f"{tag}_work" / "corpus.tsv.gz"
        if corpus.is_file():
            out.append((tag, corpus, final_anchor_elo(parent, anchors)))
        tag = parent
    return out


def share_quotas(sizes, total):
    """Split TOTAL rows as evenly as possible across sources, capped at each
    source's own size, redistributing whatever a small source cannot take."""
    quota = [0] * len(sizes)
    active = [i for i in range(len(sizes)) if sizes[i] > 0]
    remaining = min(total, sum(sizes))
    while active and remaining > 0:
        share = max(1, remaining // len(active))
        for i in list(active):
            take = min(share, sizes[i] - quota[i], remaining)
            quota[i] += take
            remaining -= take
            if quota[i] >= sizes[i]:
                active.remove(i)
            if remaining == 0:
                break
    return quota


def write_corpus(corpus_path, sources, sizes, quota, game_ply_axis, row_kind=0,
                 gate_cp=0):
    """Sample, renumber and dedup SOURCES into CORPUS_PATH.

    SOURCES is [(label, [files], generator_elo)] and QUOTA[i] rows are taken
    from SOURCES[i] (which holds SIZES[i] rows), spread evenly by a Bresenham
    accumulator so every game contributes rather than taking a prefix of the
    shards.

    Two things happen per row beyond the sampling:

    * **gid renumbering, per source.**  Raw dump gids are (pid & 0xFFF) << 20
      plus a counter (tdleaf.cpp), so the same gid recurs across iterations;
      concatenating corpora unchanged would fuse two unrelated games into one
      endply / validation-split unit.
    * **Dedup**, unconditional.  Duplicate rows (identical in every field but
      gid) come from replayed games — worst case a frozen deterministic pair,
      one unique game per opening — and straddle the trainer's by-game split,
      both overfitting and leaking validation.  Keys are 8-byte blake2b of the
      gid-stripped row stored as ints (~5 GB at 134M rows vs ~9 GB for full
      digests); a truncation collision costs one falsely-dropped row with
      probability ~5e-4 per 134M-row corpus.  Dedup runs AFTER sampling, so the
      set is sized by the budget, not by the union of every window corpus.

    Returns (rows_written, distinct_games, duplicates_dropped, rows_per_source).
    """
    rows = dropped = gid_next = 0
    seen = set()
    per_source = []
    with open(corpus_path, "w") as out:
        if game_ply_axis:
            out.write("# tdleaf-corpus axis=game-ply\n")
        for (_tag, files, _elo), size, want in zip(sources, sizes, quota):
            if want <= 0 or size <= 0:
                per_source.append(0)
                continue
            gmap = {}
            acc = 0
            taken = 0
            for src in files:
                with open_corpus(src) as f:
                    for line in f:
                        if line.startswith("#") or line.startswith("fen\t"):
                            continue
                        p = line.rstrip("\n").split("\t")
                        if len(p) < 6:
                            continue
                        # Row-type filter FIRST, so the Bresenham accumulator
                        # counts only eligible rows and `want` means "rows
                        # actually trained on".  Filtering after sampling would
                        # silently deliver a fraction of the requested budget.
                        if row_kind == 1 and p[4] == "0":
                            continue
                        if row_kind == 2 and p[4] != "0":
                            continue
                        # Quiet re-cut, for the same reason and at the same
                        # point: generation now dumps wide (--quiet-cp 1000) and
                        # every row carries its `gate`, so the width that
                        # actually trains is chosen HERE.  Pre-2026-09-03
                        # corpora have no gate column and were gated at dump.
                        if gate_cp > 0 and len(p) >= 8 and \
                           abs(int(p[1]) - int(p[7])) > gate_cp:
                            continue
                        acc += want
                        if acc < size:
                            continue
                        acc -= size
                        # fen cp result ply depth gid endply — drop gid (5)
                        key = int.from_bytes(hashlib.blake2b(
                            "\t".join(p[:5] + p[6:]).encode(),
                            digest_size=8).digest(), "little")
                        if key in seen:
                            dropped += 1
                            continue
                        seen.add(key)
                        g = gmap.get(p[5])
                        if g is None:
                            g = gid_next
                            gmap[p[5]] = g
                            gid_next += 1
                        p[5] = str(g)
                        out.write("\t".join(p) + "\n")
                        rows += 1
                        taken += 1
            per_source.append(taken)
    return rows, gid_next, dropped, per_source


def _corpus_is_game_ply(path):
    """True if a corpus TSV carries the '# tdleaf-corpus axis=game-ply' marker
    (Phase C dumps, ply column = game-ply).  Absence means legacy record-index
    axis.  Only the leading comment/header block is scanned, so this is cheap
    even on multi-GB dumps."""
    with open_corpus(path) as f:
        for line in f:
            if line.startswith("#"):
                if "axis=game-ply" in line:
                    return True
                continue
            if line.startswith("fen\t"):
                continue
            break
    return False


def binary_baked_net_matches(binary, net_name):
    """True if BINARY was compiled with NNUE_NET=net_name.  The default net
    path is embedded as a literal string, so a stale binary built for a
    different net (common when the Leaf_v<version> name is reused across runs
    with different --net) is detected by a simple string search."""
    try:
        blob = binary.read_bytes()
    except OSError:
        return False
    return net_name.encode() in blob


def compile_binary(version, net_name, tdleaf, force=False):
    """Compile Leaf_v<version> in run/; returns the binary path."""
    binary = RUN_DIR / f"Leaf_v{version}"
    if binary.exists() and not force:
        if binary_baked_net_matches(binary, net_name):
            log(f"using existing binary {binary.name}")
            return binary
        # The name was reused for a different net — reusing it would load the
        # wrong (or a missing) .nnue at runtime.  Recompile against net_name.
        log(f"{binary.name} was built for a different net — recompiling "
            f"against {net_name}")
    flags = ["NNUE=1", f"NNUE_NET={net_name}"]
    if tdleaf:
        flags.append("TDLEAF=1")
    sh(["perl", COMP_PL, version] + flags + ["OVERWRITE"], cwd=RUN_DIR)
    if not binary.exists():
        die(f"compile did not produce {binary}")
    return binary


def tdleaf_content_hash(path):
    """Return the v10+ source-.nnue content hash stored in a .tdleaf.bin,
    or None for pre-v10 files / unreadable headers."""
    try:
        with open(path, "rb") as f:
            hdr = f.read(12)
        magic = int.from_bytes(hdr[0:4], "little")
        version = int.from_bytes(hdr[4:8], "little")
        if magic != 0x544D4C46 or version < 10:
            return None
        return int.from_bytes(hdr[8:12], "little")
    except OSError:
        return None


def piece_value_canary(binary, cwd, label):
    """Drift canary for pure-PSQT: run BINARY (which loads its .nnue + .tdleaf.bin
    companion) and log the extracted PSQT piece values.  Warns loudly if the
    extracted pawn value leaves [85, 130] cp — the pure-PSQT material scale is
    only loss-anchored, so slow drift is expected but a large excursion signals
    the outcome-imbalance pathology (see docs/TRAINING.md).  Report-only under
    NNUE_FIXED_PIECE_VALUES; never constrains training."""
    try:
        # netinfo prints the piece-value banner on demand (startup went silent
        # when the unconditional load dump was removed).
        out = subprocess.run([f"./{binary}"], cwd=str(cwd), input="netinfo\nquit\n",
                             capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError) as e:
        log(f"piece-value canary ({label}): could not run {binary}: {e}")
        return
    m = re.search(r"piece values from PSQT[^:]*: "
                  r"P=(-?\d+) N=(-?\d+) B=(-?\d+) R=(-?\d+) Q=(-?\d+) cp", out)
    if not m:
        log(f"piece-value canary ({label}): banner not found")
        return
    p, n, b, r, q = (int(x) for x in m.groups())
    log(f"piece-value canary ({label}): P={p} N={n} B={b} R={r} Q={q} cp")
    if not (85 <= p <= 130):
        log(f"  *** WARNING: extracted pawn {p} cp is outside [85, 130] — "
            f"possible material-scale drift; investigate before trusting this net ***")


def pgn_score(pgn_path, name_substr):
    """W/L/D and Elo for the engine whose name contains name_substr.

    The error is the **PENTANOMIAL one sigma**.  match.py plays each opening
    twice with colours reversed and `[Round]` is the pair key, so games arrive
    in correlated pairs; the per-game binomial variance this used to assume is
    simply the wrong model for that design.  Pairing cuts the variance, so the
    old figure ran ~10% conservative (1000-game sample: 11.72 binomial against
    10.74 pentanomial).

    ⚠️ **UNITS.**  This returns ONE SIGMA.  fastchess's own `Elo: x +/- y` line
    is a **95% confidence interval** over the same pentanomial model — 1.96x
    this number, verified to 0.01 Elo on three separate 1000-game matches.  The
    two must never be mixed in one table: doing so is what made a set of
    Part 7 arms look far less significant than they were, and made the chain
    history look more precise than it was by comparison.
    """
    W = L = D = 0
    rounds = {}
    text = Path(pgn_path).read_text(errors="replace")
    for rnd, w, b, r in re.findall(
            r'\[Round "([^"]+)"\][\s\S]*?\[White "([^"]+)"\]\s*'
            r'\[Black "([^"]+)"\]\s*\[Result "([^"]+)"\]', text):
        is_white = name_substr in w
        if r == "1/2-1/2":
            pt = 0.5; D += 1
        elif (r == "1-0") == is_white:
            pt = 1.0; W += 1
        elif r in ("1-0", "0-1"):
            pt = 0.0; L += 1
        else:
            continue
        rounds.setdefault(rnd, []).append(pt)
    n = W + L + D
    if n == 0:
        return 0, 0, 0, float("nan"), float("nan")
    s = (W + 0.5 * D) / n
    if not 0 < s < 1:
        return W, L, D, (float("inf") if s == 1 else float("-inf")), float("nan")
    elo = -400 * math.log10(1 / s - 1)

    pairs = [sum(v) for v in rounds.values() if len(v) == 2]
    if len(pairs) > 1 and 2 * len(pairs) >= 0.9 * n:
        # Pentanomial: variance of the PAIR score, over pairs.
        m = sum(pairs) / len(pairs)
        var = sum((x - m) ** 2 for x in pairs) / (len(pairs) - 1)
        se_s = math.sqrt(var / len(pairs)) / 2.0
    else:
        # Unpaired (or mostly unpaired) PGN: trinomial per-game variance.  Still
        # correct for W/D/L outcomes, unlike the binomial s(1-s) it replaces.
        ex2 = (W * 1.0 + D * 0.25) / n
        se_s = math.sqrt(max(ex2 - s * s, 0.0) / n)
    err = 400 / math.log(10) * se_s / (s * (1 - s))
    return W, L, D, elo, err


def render_epoch_ladder(tag, opp, games, tc, results):
    """Format the epoch-ladder results table as printable lines (used for
    both stdout and the persisted train.log)."""
    lines = [f"=== Epoch ladder: {tag} vs {opp} ({games} games at {tc}) ==="]
    for ep, (W, L, D, elo, err) in results:
        n = W + L + D
        s = 100 * (W + 0.5 * D) / max(n, 1)
        lines.append(f"  epoch {ep}:  W/L/D {W}/{L}/{D}  score {s:5.1f}%  "
                     f"Elo {elo:+.0f} ± {err:.0f}")
    return lines


def render_gauntlet(label, results):
    """Format the final-gauntlet results table (stdout + train.log)."""
    lines = [f"=== Gauntlet results: {label} ==="]
    for opp, (W, L, D, elo, err) in results:
        n = W + L + D
        s = 100 * (W + 0.5 * D) / max(n, 1)
        lines.append(f"  vs {opp:<28} n={n:<5} W/L/D {W}/{L}/{D}  "
                     f"score {s:5.1f}%  Elo {elo:+.0f} ± {err:.0f}")
    return lines


def gzip_and_remove(path):
    """Gzip PATH to PATH.gz in place and remove the original."""
    with open(path, "rb") as fin, gzip.open(f"{path}.gz", "wb") as fout:
        shutil.copyfileobj(fin, fout)
    path.unlink()


def prune_work_dir(work, tdir, epoch_bin_dir, tag, pick_ep, keep_epoch_states):
    """End-of-run pruning inside <tag>_work/ on a successful run.  The work
    dir itself is never deleted — it's the permanent per-run archive — but
    genuinely single-use/regenerable contents are pruned: raw per-shard
    dumps (superseded by corpus.tsv.gz), epoch-ladder PGNs (their Elo is
    already captured in the log/sidecar), non-winning epoch .nnue files
    (regenerable via Leaf_vbt --write-nnue), and per-epoch .tdleaf.bin
    unless --keep-epoch-states.  corpus.tsv and the online-generation PGN
    are gzip'd in place, not deleted."""
    for dump in work.glob(f"{tag}.*.tsv"):
        dump.unlink()

    corpus = work / "corpus.tsv"
    if corpus.is_file():
        gzip_and_remove(corpus)

    for pgn in work.glob(f"match_{tag}_d*.pgn"):
        gzip_and_remove(pgn)

    for pgn in work.glob(f"match_{tag}-ep*_vs_*.pgn"):
        pgn.unlink()

    if tdir.is_dir():
        for nnue in tdir.glob(f"{tag}_ep*.nnue"):
            if nnue.name != f"{tag}_ep{pick_ep}.nnue":
                nnue.unlink()
        if not keep_epoch_states:
            for td in tdir.glob(f"{tag}_ep*.tdleaf.bin"):
                td.unlink()

    # Epoch rating binaries are deleted right after each match already —
    # sweep any leftovers defensively (e.g. a ladder that was interrupted).
    if epoch_bin_dir.is_dir():
        for p in epoch_bin_dir.iterdir():
            p.unlink()
        try:
            epoch_bin_dir.rmdir()
        except OSError:
            pass


def main():
    ap = argparse.ArgumentParser(
        description="One hybrid-loop iteration: generate -> consolidate -> gauntlet.")
    ap.add_argument("--tag", required=True,
                    help="Iteration name (prefixes all artifacts)")
    ap.add_argument("--net", default=None,
                    help=f"Base .nnue in learn/ (default: {DEFAULT_NET}, or "
                         "the net recorded in --continue's sidecar)")
    ap.add_argument("--continue", dest="continue_tag", default=None, metavar="PREV_TAG",
                    help="Chain from a prior run: read learn/PREV_TAG_final.json "
                         "and default --net/--state/--gauntlet-anchors from it, "
                         "track cumulative_games across the chain, and "
                         "auto-add Leaf_v<PREV_TAG>-final to the final gauntlet")
    ap.add_argument("--init-nnue", nargs="?", const="material", default=None,
                    choices=["material", "classical", "noprior"],
                    help="Initialise a fresh --net (and its companion .tdleaf.bin) "
                         "before generating — turns this into a start-to-finish run. "
                         "'material' (bare flag default) = classical material-only "
                         "PSQT prior; 'classical' = material + phase-interpolated "
                         "piece-square tables; 'noprior' = uniform 100 cp PSQT "
                         "(materially blind). Fails if --net already exists.")
    ap.add_argument("--state", default=None,
                    help="Promote this .tdleaf.bin to the live training state "
                         "before generating (default: keep the live state)")
    # online generation
    ap.add_argument("--skip-online", action="store_true",
                    help="Skip generation; train on --corpus files only")
    ap.add_argument("--games", type=int, default=400000)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--concurrency", type=int, default=9)
    ap.add_argument("--hash", type=int, default=128,
                    help="Per-actor hash size in MB passed to generation "
                         "(default 128; 16 is ~25%% faster but measured +8.9 +- "
                         "11.4 Elo weaker at fixed depth -- see "
                         "selfplay_run.py --hash and "
                         "docs/Online_Learning_Investigation.md 7.5)")
    ap.add_argument("--openings", default="training_openings.epd")
    # Online generation is always the actor/learner split (scripts/selfplay_run.py):
    # concurrency-1 FROZEN actors play internal self-play and emit .tdg trajectories;
    # ONE learner consumes them with a single optimizer (sole .tdleaf.bin writer, no
    # multi-writer merge) and dumps the corpus TSVs.  Learner runs --refresh-scores
    # and actors --no-adjudication (both mandatory for online stability, see
    # docs/TRAINING.md).  There is no other generation mode.
    ap.add_argument("--games-per-actor", type=int, default=1000,
                    help="Actor respawn cadence / weight-refresh interval "
                         "(default 1000)")
    ap.add_argument("--no-repeat", action="store_true",
                    help="DEPRECATED no-op (kept for backward compatibility; the "
                         "actor/learner split plays each opening once, striped "
                         "across actors — there is no fastchess -games 2 -repeat "
                         "pairing to suppress).")
    ap.add_argument("--dedup-corpus", action="store_true",
                    help="DEPRECATED no-op: corpus assembly now always drops "
                         "duplicate rows (identical in every field except "
                         "gid).  Duplicate games straddle the trainer's "
                         "by-game train/val split (different gids), so "
                         "training on them both overfits and leaks "
                         "validation; frozen deterministic pairs are the "
                         "worst case (one unique game per opening).")
    ap.add_argument("--quiet-cp", type=int, default=1000,
                    help="TDLEAF_DUMP_QUIET_CP for generation: a row is dumped "
                         "when |cp - gate| <= this.  Default 1000 (effectively "
                         "open) since 2026-09-03 — every row now carries the "
                         "`gate` column, so the gate is re-cut at TRAINING time "
                         "with the trainer's --bt-quiet-cp and dumping narrow "
                         "only destroys information.  Costs the online phase "
                         "nothing: the gate is consulted in the dump path only, "
                         "never in the TD update.  Pass 60 to reproduce the "
                         "historical corpora")
    ap.add_argument("--bt-quiet-cp", type=int, default=60, metavar="CP",
                    help="Quiet gate applied at CORPUS ASSEMBLY: keep a row only "
                         "when |cp - gate| <= CP.  This is the width that "
                         "actually trains, and it is the counterpart of the wide "
                         "--quiet-cp dump — every row carries its `gate`, so the "
                         "decision is made here rather than being burned in at "
                         "generation.  Default 60: A2 found 60/120/200 flat "
                         "(+103.7/+100.7/+100.3) but ungated much worse (+67.5), "
                         "i.e. training on the raw wide dump costs ~28 Elo.  "
                         "Budget and quotas count only surviving rows.  0 "
                         "disables the re-cut and trains on the dump as-is.  "
                         "Corpora dumped before 2026-09-03 have no gate column "
                         "and pass through — they were gated at dump time")
    ap.add_argument("--skip-train", action="store_true",
                    help="Skip offline training (generate-only)")
    ap.add_argument("--corpus", action="append", default=[],
                    help="Extra corpus TSV(s) to include in training (repeatable)")
    ap.add_argument("--corpus-window", type=int, default=4, metavar="N",
                    help="Dilute this run's dump with the archived corpora of "
                         "up to N prior iterations from the --continue chain, "
                         "holding the TOTAL row count fixed (see --corpus-rows) "
                         "so epoch cost is unchanged.  Measured worth ~45 Elo: "
                         "row-matched arms differing only in game diversity "
                         "scored +112.9 (500k games) vs +148.7 (2.5M games) "
                         "against the classical anchor.  0 disables the window "
                         "and trains on this run's corpus alone (the pre-A1 "
                         "behaviour).  Needs --continue to find the chain. "
                         "Default: 4")
    ap.add_argument("--corpus-rows", type=int, default=0, metavar="N",
                    help="Total row budget for the assembled corpus, split "
                         "evenly across this run's dump and each window corpus. "
                         "0 (default) = auto: match this run's own dump row "
                         "count, so the window changes WHICH games the rows "
                         "come from without changing how many there are")
    ap.add_argument("--corpus-window-max-stale", type=float, default=0.0,
                    metavar="ELO",
                    help="Drop window corpora whose GENERATOR rates more than "
                         "ELO below the freshest generator in the window "
                         "(corpus labels distil their generator, so a stale "
                         "corpus pulls the net backwards through the eval "
                         "bootstrap).  0 (default) = no filter; the generator "
                         "Elo of every corpus is logged either way")
    ap.add_argument("--bt-threads", type=int, default=8,
                    help="Worker threads for within-batch gradient compute "
                         "(single-process; default 8)")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--bt-lr", type=float, default=0.25)
    ap.add_argument("--bt-lambda", type=float, default=1.0)
    ap.add_argument("--bt-K", type=float, default=220.0)
    ap.add_argument("--bt-batch", type=int, default=512)
    ap.add_argument("--bt-leaf-lambda", type=float, default=None,
                    help="Outcome-weight ceiling for depth-0 (leaf) rows "
                         "(default: same as --bt-lambda)")
    ap.add_argument("--bt-td-lambda", type=float, default=None,
                    help="Result decay per ply from the game end "
                         "(default: trainer's TDLEAF_LAMBDA; 1.0 = flat blend)")
    ap.add_argument("--bt-loss-gamma", type=float, default=None,
                    help="Focal-gamma loss exponent (d(1-d))^gamma: "
                         "1.0=MSE (default), 0.0=cross-entropy, 0.5=between")
    ap.add_argument("--bt-rows", choices=["leaf", "root", "both"],
                    default="root",
                    help="Which corpus rows to train on, by the depth column: "
                         "'root' (depth > 0, search-score labels), 'leaf' "
                         "(depth 0, the generator's own static eval), or "
                         "'both'.  Default 'root' since 2026-09-03: at a fixed "
                         "row budget root-only beat leaf-only by +35.6 +- 11.0 "
                         "head-to-head and the natural mix by +46 on the "
                         "foreign anchor, so the ~54%% of every corpus that is "
                         "leaf rows was worse than useless at the margin "
                         "(docs/Offline_Learning_Investigation.md Part 3).  "
                         "Budget and quotas count only the selected row type, "
                         "and the archived corpus.tsv.gz holds only that type "
                         "— use 'both' to keep the full mix.  "
                         "NOTE the outcome/eval blend (--bt-lambda / "
                         "--bt-leaf-lambda / --bt-td-lambda) was calibrated on "
                         "the MIXTURE and has never been retuned per row type; "
                         "see Part 3.5")
    ap.add_argument("--gauntlet", nargs="*", default=[],
                    help="Opponent binaries in learn/ for the FINAL gauntlet "
                         "(rates the promoted best-epoch net; empty = skip). "
                         "The --gauntlet-epochs ladder opponent is independent "
                         "(always the pre-offline-training net).")
    ap.add_argument("--gauntlet-anchors", nargs="*", default=None,
                    help="Fixed opponent binaries in learn/ for the final "
                         "gauntlet, carried forward automatically across a "
                         "--continue chain (default: inherited from "
                         "--continue's sidecar, or empty; pass with no "
                         "arguments to explicitly clear the inherited list). "
                         "Combined with --gauntlet and (under --continue) "
                         "Leaf_v<PREV_TAG>-final.")
    ap.add_argument("--gauntlet-games", type=int, default=1000)
    ap.add_argument("--tc", default="3+0.05")
    ap.add_argument("--gauntlet-epochs", action="store_true",
                    help="Fast per-epoch ladder: after each epoch's training, "
                         "rate that snapshot vs the net as it stood BEFORE "
                         "offline training (post-online checkpoint, or the "
                         "incoming state under --skip-online) — measures what "
                         "consolidation added.  Independent of --gauntlet.")
    ap.add_argument("--gauntlet-tdleaf", action="store_true",
                    help="Also rate the net as it ENTERED offline training "
                         "(post-online checkpoint, or the incoming state "
                         "under --skip-online): save it permanently to "
                         "learn/ as <tag>-tdleaf.nnue + Leaf_v<tag>-tdleaf "
                         "and run it through the same final gauntlet "
                         "(opponents, --gauntlet-games, --tc).  Gives every "
                         "run a same-conditions baseline so per-iteration "
                         "deltas read directly from one sidecar.  Under "
                         "TDLEAF_FREEZE this net is identical to the "
                         "incoming seed — the gauntlet still measures the "
                         "baseline under THIS run's conditions.")
    ap.add_argument("--epoch-games", type=int, default=1000,
                    help="Games per epoch-ladder match (default 1000)")
    ap.add_argument("--epoch-tc", default="1+0.01",
                    help="Epoch-ladder time control (default 1+0.01)")
    ap.add_argument("--no-final-gauntlet", action="store_true",
                    help="Skip the final full gauntlet (with --gauntlet-epochs, "
                         "for ladder-only runs — no --gauntlet opponents needed)")
    ap.add_argument("--force", action="store_true",
                    help="Reuse an existing <tag>_work directory")
    ap.add_argument("--recompile", action="store_true",
                    help="Force recompile of helper binaries")
    ap.add_argument("--keep-epoch-states", action="store_true",
                    help="Keep every epoch's .tdleaf.bin in <tag>_work/train/ "
                         "(default: only the promoted epoch's state survives; "
                         "the corresponding .nnue is always regenerable via "
                         "Leaf_vbt --write-nnue and is never kept)")
    ap.add_argument("--keep-work", action="store_true",
                    help="Skip all end-of-run pruning inside <tag>_work/ "
                         "(raw dumps, non-winning epoch .nnue, epoch-ladder "
                         "PGNs, epoch rating binaries all stay; corpus.tsv "
                         "stays uncompressed) — <tag>_work/ is never deleted "
                         "either way, this only controls pruning aggressiveness")
    args = ap.parse_args()

    if args.no_repeat:
        log("note: --no-repeat is now always on — the flag is a no-op")
    if args.dedup_corpus:
        log("note: --dedup-corpus is now always on — the flag is a no-op")

    if Path.cwd().resolve() != LEARN_DIR.resolve():
        die(f"run from {LEARN_DIR} (cwd is {Path.cwd()})")

    # ---- --continue: chain --net/--state/--gauntlet-anchors from a prior run
    continue_json = None
    if args.continue_tag:
        sidecar = LEARN_DIR / f"{args.continue_tag}_final.json"
        if not sidecar.is_file():
            die(f"--continue {args.continue_tag}: sidecar not found: {sidecar} "
                f"(pre-migration or wrong tag — pass --net/--state explicitly "
                f"instead)")
        with open(sidecar) as f:
            continue_json = json.load(f)
        log(f"--continue {args.continue_tag}: net={continue_json['net']} "
            f"cumulative_games so far={continue_json['cumulative_games']:,}")

    if args.net is None:
        args.net = continue_json["net"] if continue_json else DEFAULT_NET
    if args.state is None and args.continue_tag:
        args.state = str(LEARN_DIR / f"{args.continue_tag}_final.tdleaf.bin")
    if args.gauntlet_anchors is None:
        args.gauntlet_anchors = (continue_json["gauntlet_anchors"]
                                  if continue_json else [])

    net_path = LEARN_DIR / args.net
    netbase = args.net[:-5] if args.net.endswith(".nnue") else args.net
    live_td = LEARN_DIR / f"{netbase}.tdleaf.bin"

    work = LEARN_DIR / f"{args.tag}_work"
    if work.exists() and not args.force:
        die(f"{work} exists — use --force to reuse it")
    work.mkdir(exist_ok=True)

    # ---- Phase 0: initialise a fresh network -----------------------------
    # Turns the loop into a start-to-finish run: create --net (+ its companion
    # .tdleaf.bin, which becomes the live training state) from scratch, then fall
    # through into online generation exactly as if the net had been provided.
    if args.init_nnue:
        if args.state:
            die("--init-nnue creates a fresh training state; don't also pass --state")
        # Refuse to clobber existing artifacts.  The engine's --init-nnue also
        # refuses to merge-save over an existing companion, but a pre-existing
        # .nnue would be silently overwritten, so guard both here.
        if net_path.is_file() or live_td.is_file():
            die(f"--init-nnue would overwrite existing {net_path.name} or "
                f"{live_td.name} — remove them first")
        # Any TDLeaf binary can init; reuse the trainer binary (compiled against
        # --net, but --init-nnue writes a fresh net rather than reading one, so
        # the not-yet-existing net file is fine at compile time without EMBED).
        # Compiled in run/ (build-system requirement) but never executed there —
        # run/ holds files (e.g. main_bk.dat) that must never affect a training
        # binary's behavior, so run() only ever happens from a copy in learn/.
        init_bin = compile_binary("bt", args.net, tdleaf=True, force=args.recompile)
        init_bin_learn = LEARN_DIR / init_bin.name
        shutil.copy2(init_bin, init_bin_learn)
        flag = {"material":  "--init-nnue",
                "classical": "--init-nnue-classical",
                "noprior":   "--init-nnue-noprior"}[args.init_nnue]
        log(f"initialising fresh net {net_path.name} (prior={args.init_nnue})")
        sh([f"./{init_bin_learn.name}", flag, "--write-nnue", net_path], cwd=LEARN_DIR)
        init_bin_learn.unlink()
        if not net_path.is_file() or not live_td.is_file():
            die(f"--init-nnue did not produce {net_path.name} + {live_td.name}")

    if not net_path.is_file():
        die(f"base net not found: {net_path}")

    # ---- Binaries --------------------------------------------------------
    # Only train_hl_a is needed: the actor/learner split runs frozen actors and
    # one learner, all from the same binary.
    bt_bin = compile_binary("bt", args.net, tdleaf=True, force=args.recompile)
    if not args.skip_online:
        tr_a = compile_binary("train_hl_a", args.net, tdleaf=True, force=args.recompile)
        shutil.copy2(tr_a, LEARN_DIR / tr_a.name)

    # ---- Phase 1: promote state -----------------------------------------
    if args.state:
        state = Path(args.state)
        if not state.is_file():
            die(f"--state not found: {state}")
        # If --state already IS the live companion, it's already promoted — skip
        # the copy (shutil.copy2 raises SameFileError on a self-copy).
        if state.resolve() == live_td.resolve():
            log(f"--state is already the live state ({live_td.name}) — "
                f"starting from it as-is")
        else:
            # Pairing pre-flight: the state must have been trained against the
            # SAME base .nnue (--net).  The engine refuses a mismatched pair at
            # load but then falls back to the raw .nnue and keeps running — this
            # check fails loudly up front instead.  Compared via the v10 content
            # hash stored in the live (known-paired) .tdleaf.bin.
            sh_state = tdleaf_content_hash(state)
            sh_live  = tdleaf_content_hash(live_td) if live_td.is_file() else None
            if sh_state is not None and sh_live is not None and sh_state != sh_live:
                die(f"--state {state} was trained against a different base .nnue "
                    f"(state hash 0x{sh_state:08X} != live 0x{sh_live:08X} for "
                    f"{args.net}).  Wrong --net or wrong --state.")
            if live_td.is_file():
                backup = LEARN_DIR / f"{netbase}.tdleaf.bin-pre{args.tag}"
                log(f"backing up live state -> {backup.name}")
                shutil.copy2(live_td, backup)
            log(f"promoting {state} -> {live_td.name}")
            shutil.copy2(state, live_td)

    dump_files = []

    # ---- Phase 2: online generation with dumping -------------------------
    if not args.skip_online:
        env = dict(os.environ)
        env["TDLEAF_DUMP_TSV"] = str(work / args.tag)
        env["TDLEAF_DUMP_QUIET_CP"] = str(args.quiet_cp)
        log(f"online generation: {args.games} games at depth {args.depth} "
            f"(dump -> {work}/{args.tag}.*)")
        # Actors stripe the shuffled opening book; openings recycle once --games
        # exceeds the book size (the extra games repeat openings) — warn.
        opening_file = LEARN_DIR / args.openings
        if opening_file.is_file():
            n_openings = sum(1 for _ in open(opening_file))
            if args.games > n_openings:
                log(f"WARNING: --games {args.games} exceeds the opening book "
                    f"({n_openings} lines in {args.openings}) — openings will "
                    f"recycle past one pass")
        # Stage-1 actor/learner: selfplay_run.py drives concurrency-1 frozen
        # actors + one learner (single optimizer, sole state writer).  The
        # learner inherits this env, so it produces the corpus dump; actors are
        # forced frozen by the driver.
        blob = (LEARN_DIR / "Leaf_vtrain_hl_a").read_bytes()
        if b"--learn-stream" not in blob:
            die("Leaf_vtrain_hl_a predates the --learn-stream driver — "
                "rerun with --recompile")
        seed = zlib.crc32(args.tag.encode()) & 0x7FFFFFFF
        n_actors = max(1, int(args.concurrency) - 1)
        traj_dir = work / "traj"
        traj_dir.mkdir(exist_ok=True)
        log(f"actor/learner generation: {n_actors} actors + 1 learner "
            f"(refresh every {args.games_per_actor} games/actor, "
            f"seed {seed}); logs -> {traj_dir}/")
        sh(["python3", SCRIPT_DIR / "selfplay_run.py",
            "--binary", "Leaf_vtrain_hl_a",
            "--epd", args.openings,
            "--actors", n_actors,
            "--hash", args.hash,
            "--depth", args.depth,
            "--games-per-actor", args.games_per_actor,
            "--total-games", args.games,
            "--traj-dir", traj_dir,
            "--tdleaf-out", f"{netbase}.tdleaf.bin",
            "--delete-consumed", "--refresh-scores",
            "--seed", seed],
           cwd=LEARN_DIR, env=env)

        # Phase 3: checkpoint post-generation state
        ckpt = LEARN_DIR / f"{netbase}.tdleaf.bin-{args.tag}-online"
        log(f"checkpointing post-generation state -> {ckpt.name}")
        shutil.copy2(live_td, ckpt)

        # Drift canary: Leaf_vtrain_hl_a loads the just-checkpointed live state.
        piece_value_canary("Leaf_vtrain_hl_a", LEARN_DIR, f"{args.tag}-online")

        dump_files = sorted(work.glob(f"{args.tag}.*.tsv"))
        if not dump_files:
            die("online phase produced no dump files")
        log(f"{len(dump_files)} dump files")

    else:
        # Resume path: --skip-online over an existing work dir (--force) reuses
        # the dumps already generated there, so a crashed run continues from the
        # consolidation phase without re-running generation or re-passing every
        # dump as --corpus.
        existing = sorted(work.glob(f"{args.tag}.*.tsv"))
        if existing:
            dump_files = existing
            log(f"reusing {len(dump_files)} existing dump file(s) from {work.name}")

    if args.skip_train:
        log("generate-only mode: done.")
        return

    # ---- Phase 4: assemble corpus ----------------------------------------
    primary = dump_files + [Path(c) for c in args.corpus]
    for c in primary:
        if not c.is_file():
            die(f"corpus not found: {c}")

    # Multi-iteration corpus window (arm A1).  Consolidating one iteration's own
    # dump leaves ~45 Elo on the table: two arms with IDENTICAL row counts,
    # epochs, optimizer steps and wall clock, differing only in how many
    # distinct games the rows came from, scored +112.9 (500k games) and +148.7
    # (2.5M games) against the classical anchor, and the diverse arm won the
    # head-to-head by +45.4 +- 11.1.  So dilute the fresh dump with archived
    # corpora from the --continue chain at a FIXED total row count — the window
    # changes which games the rows come from, not how many.
    # See docs/Offline_Learning_Investigation.md Part 2.
    window = []
    if args.corpus_window > 0:
        if args.continue_tag:
            window = chain_corpora(args.continue_tag, args.corpus_window,
                                   args.gauntlet_anchors)
            if not window:
                log(f"--corpus-window {args.corpus_window}: no archived corpora "
                    f"found on the {args.continue_tag} chain")
        else:
            log(f"--corpus-window {args.corpus_window} needs --continue to walk "
                f"the chain — training on this run's corpus alone")

    # Staleness filter.  Corpus labels distil their generator, so a corpus made
    # by a much weaker net drags the student back through the (1-w) eval
    # bootstrap term that carries ~70% of the target weight.
    if window and args.corpus_window_max_stale > 0:
        known = [e for _, _, e in window if e is not None]
        if known:
            best = max(known)
            kept = []
            for tag, path, elo in window:
                if elo is not None and best - elo > args.corpus_window_max_stale:
                    log(f"  window: dropping {tag} — generator {elo:+.1f} is "
                        f"{best - elo:.1f} Elo below the freshest "
                        f"({best:+.1f}), over --corpus-window-max-stale "
                        f"{args.corpus_window_max_stale:.1f}")
                else:
                    kept.append((tag, path, elo))
            window = kept

    # Sources: this run's dump(s) + --corpus as one unit, then one per window
    # corpus.  Row quotas are split evenly across sources (capped at each
    # source's size) so the budget buys as many distinct games as possible.
    # A window-only run (--skip-online --continue, no --corpus) is legitimate:
    # it re-consolidates the chain's archived games without generating any.
    sources = ([("this run", primary, None)] if primary else []) \
              + [(t, [p], e) for t, p, e in window]
    if not sources:
        die("nothing to train on (no dumps, no --corpus, and no window corpora "
            "— pass --corpus, or --continue with --corpus-window)")

    axes = {_corpus_is_game_ply(c) for _, files, _ in sources for c in files}
    if len(axes) > 1:
        die("cannot mix game-ply-axis and legacy record-index corpora in one "
            "run — the ply column means different things; train them separately")
    game_ply_axis = axes.pop()

    # Quotas are computed over the rows that will actually be TRAINED ON, so a
    # row-type filter shrinks the corpus rather than silently shrinking the
    # budget: with --bt-rows root over a natural corpus, filtering after
    # sampling would have delivered ~45% of the requested rows.
    row_kind = ROW_KIND[args.bt_rows]
    gate_cp = args.bt_quiet_cp
    log("counting corpus rows (cached as <corpus>.rows) ...")
    sizes = [sum(corpus_row_counts(c, gate_cp)[row_kind] for c in files)
             for _, files, _ in sources]
    if gate_cp > 0:
        log(f"--bt-quiet-cp {gate_cp}: budget and quotas count only rows within "
            f"the gate (dump gate was --quiet-cp {args.quiet_cp})")
    if row_kind:
        log(f"--bt-rows {args.bt_rows}: budget and quotas count {args.bt_rows} "
            f"rows only")
        log(f"NOTE: {work.name}/corpus.tsv.gz will archive {args.bt_rows} rows "
            f"only — the other row type is dropped at assembly and this run's "
            f"raw dumps are pruned at end of run.  Use --bt-rows both (or "
            f"--keep-work) to retain the full mix for later re-analysis.")
    if args.corpus_rows > 0:
        budget = args.corpus_rows
    elif len(sources) > 1:
        budget = sizes[0]      # auto: hold the total at this run's own dump size
    else:
        budget = sum(sizes)    # no window and no explicit budget -> no thinning
    quota = share_quotas(sizes, budget)

    log(f"corpus window: {len(sources)} source(s), budget {budget:,} rows")
    for (tag, _, elo), size, want in zip(sources, sizes, quota):
        gen = f"generator {elo:+.1f} Elo" if elo is not None else "generator n/a"
        log(f"  {tag:<24} {want:>12,} of {size:>12,} rows "
            f"({100.0 * want / max(size, 1):5.1f}%)  {gen}")

    corpus_path = work / "corpus.tsv"
    log(f"assembling -> {corpus_path.name} "
        f"(axis={'game-ply' if game_ply_axis else 'legacy record-index'}, "
        f"dedup) ...")
    rows, gid_next, dropped, per_source = write_corpus(
        corpus_path, sources, sizes, quota, game_ply_axis, row_kind, gate_cp)
    log(f"{rows:,} positions assembled from {gid_next:,} distinct games "
        f"({dropped:,} duplicate rows dropped)")

    # ---- Phase 5: offline consolidation (single threaded process) ---------
    tdir = work / "train"
    tdir.mkdir(exist_ok=True)
    shutil.copy2(bt_bin, tdir / "Leaf_vbt")
    shutil.copy2(net_path, tdir / args.net)
    shutil.copy2(live_td, tdir / f"{netbase}.tdleaf.bin")

    # Per-epoch rating binaries are single-use (needed only for their one
    # ladder match) and relocate here rather than living flat in learn/ —
    # run/ is a build-output step only, never an execution location (main_bk.dat
    # and other run/-only files must never reach a training/rating binary).
    epoch_bin_dir = work / "epoch_binaries"
    epoch_bin_dir.mkdir(exist_ok=True)

    def build_rating_binary(snap, ver):
        """Compile a TDLEAF-off inference binary Leaf_v<ver> for the net SNAP
        (a .nnue in tdir), then relocate both binary and net out of run/ into
        <tag>_work/epoch_binaries/ so the net resolves next to the binary and
        nothing lingers in run/.  Returns (binary_path, net_path, display_name):
        binary_path is the absolute path match.py should be given (it resolves
        absolute paths directly and derives each engine's own execution
        directory from os.path.dirname(exe), so this guarantees the engine
        never runs with run/ as its directory); display_name is the bare
        Leaf_v<ver> for PGN filenames and log messages."""
        shutil.copy2(snap, RUN_DIR / snap.name)
        b = compile_binary(ver, snap.name, tdleaf=False, force=True)
        dest_bin = epoch_bin_dir / b.name
        dest_net = epoch_bin_dir / snap.name
        shutil.move(str(b), str(dest_bin))
        shutil.copy2(snap, dest_net)   # net resolves next to binary
        (RUN_DIR / snap.name).unlink()
        return dest_bin, dest_net, b.name

    # Pre-training baseline net = the net exactly as it enters offline
    # training: the post-online checkpoint when we generated games this run,
    # or the incoming live state under --skip-online.  Bake base .nnue + live
    # .tdleaf.bin (the trainer's own starting state, freshly copied into tdir
    # above) into a standalone net via the trainer binary's --write-nnue.
    # Used as the epoch-ladder opponent (--gauntlet-epochs) and/or saved to
    # learn/ as the permanent <tag>-tdleaf net (--gauntlet-tdleaf).  Done
    # before the trainer launches so the bake sees the untouched state and
    # doesn't contend with training for cores.
    ladder_opp_bin = ladder_opp_net = ladder_opp = None
    tdleaf_rate_bin = None
    if args.gauntlet_epochs or args.gauntlet_tdleaf:
        pre_nnue = tdir / f"{args.tag}_pretrain.nnue"
        log(f"baking pre-offline-training baseline net -> {pre_nnue.name}")
        sh(["./Leaf_vbt", "--write-nnue", pre_nnue.name], cwd=tdir)
        if not pre_nnue.is_file():
            die(f"failed to bake pre-training baseline {pre_nnue}")
    if args.gauntlet_epochs:
        ladder_opp_bin, ladder_opp_net, ladder_opp = build_rating_binary(
            pre_nnue, f"{args.tag}-pretrain")
        log(f"epoch-ladder opponent: {ladder_opp} (net before offline training)")
    if args.gauntlet_tdleaf:
        # Permanent copy in learn/, mirroring the -final pair and the chain's
        # historical -tdleaf naming.  Compiled against its own .nnue name (the
        # -pretrain rating binary bakes the _pretrain.nnue filename, so it
        # can't simply be renamed).
        tdleaf_nnue = LEARN_DIR / f"{args.tag}-tdleaf.nnue"
        shutil.copy2(pre_nnue, tdleaf_nnue)
        shutil.copy2(tdleaf_nnue, RUN_DIR / tdleaf_nnue.name)
        b = compile_binary(f"{args.tag}-tdleaf", tdleaf_nnue.name,
                           tdleaf=False, force=True)
        tdleaf_rate_bin = LEARN_DIR / b.name
        shutil.move(str(b), str(tdleaf_rate_bin))
        (RUN_DIR / tdleaf_nnue.name).unlink()
        log(f"tdleaf-phase net saved: {tdleaf_nnue.name} + {tdleaf_rate_bin.name}")

    log(f"training: {args.bt_threads} threads x {args.epochs} epochs "
        f"(lr {args.bt_lr}, lambda {args.bt_lambda}, K {args.bt_K})")
    cmd = ["./Leaf_vbt", "--batch-train", "../corpus.tsv",
           "--bt-epochs", str(args.epochs), "--bt-out", args.tag,
           "--bt-threads", str(args.bt_threads),
           "--bt-lr", str(args.bt_lr), "--bt-lambda", str(args.bt_lambda),
           "--bt-K", str(args.bt_K), "--bt-batch", str(args.bt_batch),
           # Exact row count -> the trainer reserves once instead of growing the
           # record vector by doubling.  The doubling peak (old + new buffer
           # live simultaneously) is ~1.5x the final size and is what actually
           # caps corpus size on a 30 GB box: ~16 GB of transient just to reach
           # 190M rows.  Assembly counted the rows, so hand them over.
           "--bt-max", str(rows),
           "--bt-seed", "1000"]
    if args.bt_leaf_lambda is not None:
        cmd += ["--bt-leaf-lambda", str(args.bt_leaf_lambda)]
    if args.bt_td_lambda is not None:
        cmd += ["--bt-td-lambda", str(args.bt_td_lambda)]
    if args.bt_loss_gamma is not None:
        cmd += ["--bt-loss-gamma", str(args.bt_loss_gamma)]
    if args.bt_rows != "both":
        cmd += ["--bt-rows", args.bt_rows]
    logf = open(tdir / "train.log", "w")
    proc = subprocess.Popen(cmd, cwd=str(tdir),
                            stdout=subprocess.DEVNULL, stderr=logf)

    # Per-epoch ladder: rate each epoch snapshot as soon as its epoch's
    # training finishes, while the trainer keeps running.  The trainer writes
    # _epN.nnue then _epN.tdleaf.bin, so the .tdleaf.bin appearing means the
    # .nnue is complete.
    def rate_epoch(ep, opp_bin, opp_name):
        ver = f"{args.tag}-ep{ep}"
        bpath, npath, bname = build_rating_binary(
            tdir / f"{args.tag}_ep{ep}.nnue", ver)
        pgn = work / f"match_{ver}_vs_{opp_name.replace('Leaf_v', '')}.pgn"
        log(f"epoch ladder: epoch {ep} vs {opp_name} "
            f"({args.epoch_games} games at {args.epoch_tc})")
        sh(["python3", SCRIPT_DIR / "match.py", str(bpath), str(opp_bin),
            "-n", args.epoch_games, "-c", 8, "-tc", args.epoch_tc,
            "--openings", args.openings, "--fischer-random",
            "--pgn-out", pgn], cwd=LEARN_DIR)
        W, L, D, elo, err = pgn_score(pgn, ver)
        log(f"epoch ladder: epoch {ep}  W/L/D {W}/{L}/{D}  Elo {elo:+.0f} ± {err:.0f}")
        # single-use: prune this epoch's rating binary + net right after its match
        bpath.unlink(missing_ok=True)
        npath.unlink(missing_ok=True)
        return (W, L, D, elo, err)

    epoch_results = []
    if args.gauntlet_epochs:
        rated = 0
        while True:
            if rated < args.epochs and \
                    (tdir / f"{args.tag}_ep{rated + 1}.tdleaf.bin").exists():
                rated += 1
                # Freeze the trainer while the ladder match runs: the next
                # epoch's training must not contend for cores with the games.
                if proc.poll() is None:
                    proc.send_signal(signal.SIGSTOP)
                try:
                    epoch_results.append(
                        (rated, rate_epoch(rated, ladder_opp_bin, ladder_opp)))
                finally:
                    if proc.poll() is None:
                        proc.send_signal(signal.SIGCONT)
                # Auto-decider hook: to stop a run that is going poorly,
                # decide on epoch_results here, then proc.terminate() + break.
                continue
            if proc.poll() is not None:
                break
            time.sleep(10)
        rc = proc.wait()
        logf.close()
        if rc != 0:
            die(f"trainer process failed (rc={rc}) — see {tdir}/train.log")
        # Sweep snapshots that landed between the last check and process exit.
        while rated < args.epochs and \
                (tdir / f"{args.tag}_ep{rated + 1}.tdleaf.bin").exists():
            rated += 1
            epoch_results.append(
                (rated, rate_epoch(rated, ladder_opp_bin, ladder_opp)))
        # Pretrain baseline is only needed as the fixed ladder opponent —
        # prune it once the whole ladder is done with it.
        if ladder_opp_bin is not None:
            ladder_opp_bin.unlink(missing_ok=True)
            ladder_opp_net.unlink(missing_ok=True)
    else:
        rc = proc.wait()
        logf.close()
        if rc != 0:
            die(f"trainer process failed (rc={rc}) — see {tdir}/train.log")

    if epoch_results:
        print()
        for line in render_epoch_ladder(args.tag, ladder_opp, args.epoch_games,
                                        args.epoch_tc, epoch_results):
            print(line)
        print()

    # Choose the final net.  With the epoch ladder, promote the best epoch (max
    # Elo; ties → later epoch) — both gen-2 runs peaked at epoch 4 of 6, so the
    # last epoch is not automatically best.  Without the ladder, use the last
    # epoch.  Each snapshot is a complete net (single process, no merge needed).
    if epoch_results:
        best_ep, (_, _, _, best_elo, _) = max(
            epoch_results, key=lambda r: (r[1][3], r[0]))
        log(f"final = epoch {best_ep} of {args.epochs} "
            f"(ladder best, Elo {best_elo:+.0f})")
        pick_ep = best_ep
    else:
        pick_ep = args.epochs
    src_nnue = tdir / f"{args.tag}_ep{pick_ep}.nnue"
    src_td   = tdir / f"{args.tag}_ep{pick_ep}.tdleaf.bin"
    if not src_nnue.is_file() or not src_td.is_file():
        die(f"epoch {pick_ep} snapshot missing in {tdir} — see train.log")
    out_nnue = LEARN_DIR / f"{args.tag}_final.nnue"
    out_td   = LEARN_DIR / f"{args.tag}_final.tdleaf.bin"
    shutil.copy2(src_nnue, out_nnue)
    shutil.copy2(src_td, out_td)
    log(f"consolidated net: {out_nnue.name}  (seed for next iteration: {out_td.name})")

    # ---- Phase 6: gauntlet -------------------------------------------------
    # Resolved opponent list = --gauntlet-anchors (explicit, or inherited via
    # --continue) + Leaf_v<prev_tag>-final (auto, when chaining) + --gauntlet.
    gauntlet_list = list(args.gauntlet_anchors)
    if args.continue_tag:
        prev_final = f"Leaf_v{args.continue_tag}-final"
        if prev_final not in gauntlet_list:
            gauntlet_list.append(prev_final)
    for opp in args.gauntlet:
        if opp not in gauntlet_list:
            gauntlet_list.append(opp)

    # Always build Leaf_v<tag>-final (needed as the anchor opponent for any
    # future --continue chain, even if this run has nothing to gauntlet
    # against yet) — compiled in run/ (build-system requirement) but moved
    # into learn/ before anything executes it, never left resident in run/.
    shutil.copy2(out_nnue, RUN_DIR / out_nnue.name)
    compiled_final = compile_binary(f"{args.tag}-final", out_nnue.name,
                                    tdleaf=False, force=True)
    rate_bin = LEARN_DIR / compiled_final.name
    shutil.move(str(compiled_final), str(rate_bin))
    (RUN_DIR / out_nnue.name).unlink()

    def run_gauntlet(bin_name, label):
        """Rate Leaf_v<label> (binary bin_name in learn/) against every
        opponent in gauntlet_list under the shared conditions
        (--gauntlet-games, --tc).  Returns [(opp, (W,L,D,elo,err)), ...]."""
        res = []
        for opp in gauntlet_list:
            if not (LEARN_DIR / opp).is_file():
                log(f"WARNING: opponent {opp} not found in learn/ — skipping")
                continue
            pgn = work / f"match_{label}_vs_{opp.replace('Leaf_v','')}.pgn"
            log(f"gauntlet: {label} vs {opp} ({args.gauntlet_games} games)")
            sh(["python3", SCRIPT_DIR / "match.py", bin_name, opp,
                "-n", args.gauntlet_games, "-c", 8, "-tc", args.tc,
                "--openings", args.openings, "--fischer-random",
                "--pgn-out", pgn], cwd=LEARN_DIR)
            res.append((opp, pgn_score(pgn, label)))
        return res

    results = []
    tdleaf_results = []
    if not gauntlet_list:
        log("no gauntlet opponents (--gauntlet/--gauntlet-anchors): "
            "skipping final gauntlet matches.")
    elif args.no_final_gauntlet:
        log("--no-final-gauntlet: skipping final gauntlet matches.")
    else:
        # The -tdleaf baseline first (when requested), then the final — same
        # opponents and conditions, so per-iteration deltas read directly
        # from one run.
        if tdleaf_rate_bin is not None:
            tdleaf_results = run_gauntlet(tdleaf_rate_bin.name,
                                          f"{args.tag}-tdleaf")
        results = run_gauntlet(rate_bin.name, f"{args.tag}-final")

        print()
        if tdleaf_results:
            for line in render_gauntlet(tdleaf_rate_bin.name, tdleaf_results):
                print(line)
            print()
        for line in render_gauntlet(rate_bin.name, results):
            print(line)
        print()

    # ---- persisted log: append the tables that were only ever on stdout ---
    with open(tdir / "train.log", "a") as f:
        if epoch_results:
            f.write("\n" + "\n".join(render_epoch_ladder(
                args.tag, ladder_opp, args.epoch_games, args.epoch_tc,
                epoch_results)) + "\n")
        if tdleaf_results:
            f.write("\n" + "\n".join(render_gauntlet(
                tdleaf_rate_bin.name, tdleaf_results)) + "\n")
        if results:
            f.write("\n" + "\n".join(render_gauntlet(rate_bin.name, results)) + "\n")

    # ---- sidecar: the self-describing handoff unit for --continue ---------
    games_this_iter = 0 if args.skip_online else args.games
    cumulative_games = games_this_iter + (
        continue_json["cumulative_games"] if continue_json else 0)
    sidecar = {
        "tag": args.tag,
        "net": args.net,
        "parent_tag": args.continue_tag,
        "date": time.strftime("%Y-%m-%d"),
        "games_this_iter": games_this_iter,
        "cumulative_games": cumulative_games,
        "gen_mode": ("skip-online" if args.skip_online else "actor-learner"),
        "depth": args.depth,
        "epochs": args.epochs,
        "picked_epoch": pick_ep,
        "bt_lr": args.bt_lr,
        "bt_lambda": args.bt_lambda,
        "bt_K": args.bt_K,
        "bt_td_lambda": args.bt_td_lambda,
        "bt_rows": args.bt_rows,
        "bt_quiet_cp": args.bt_quiet_cp,
        "corpus_rows": rows,
        "corpus_games": gid_next,
        "corpus_window": [
            {"tag": tag, "rows_used": used, "rows_avail": size,
             "generator_elo": elo}
            for (tag, _, elo), size, used in zip(sources, sizes, per_source)
        ],
        "gauntlet_anchors": args.gauntlet_anchors,
        "epoch_ladder": [
            {"epoch": ep, "W": W, "L": L, "D": D, "elo": elo, "err": err}
            for ep, (W, L, D, elo, err) in epoch_results
        ],
        "tdleaf_gauntlet": [
            {"opponent": opp, "W": W, "L": L, "D": D, "elo": elo, "err": err}
            for opp, (W, L, D, elo, err) in tdleaf_results
        ],
        "final_gauntlet": [
            {"opponent": opp, "W": W, "L": L, "D": D, "elo": elo, "err": err}
            for opp, (W, L, D, elo, err) in results
        ],
    }
    sidecar_path = LEARN_DIR / f"{args.tag}_final.json"
    with open(sidecar_path, "w") as f:
        json.dump(sidecar, f, indent=2)
    log(f"cumulative games for this net: {cumulative_games:,} "
        f"(this iteration: {games_this_iter:,}) -> {sidecar_path.name}")

    # ---- end-of-run archive pruning (only on success, unless --keep-work) -
    if not args.keep_work:
        prune_work_dir(work, tdir, epoch_bin_dir, args.tag, pick_ep,
                       args.keep_epoch_states)
        for stray in (LEARN_DIR / f"{netbase}.tdleaf.bin-pre{args.tag}",
                      LEARN_DIR / f"{netbase}.tdleaf.bin-{args.tag}-online"):
            if stray.is_file():
                stray.unlink()

    log("iteration complete.")


if __name__ == "__main__":
    main()
