#!/usr/bin/env python3
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


def _corpus_is_game_ply(path):
    """True if a corpus TSV carries the '# tdleaf-corpus axis=game-ply' marker
    (Phase C dumps, ply column = game-ply).  Absence means legacy record-index
    axis.  Only the leading comment/header block is scanned, so this is cheap
    even on multi-GB dumps."""
    with open(path) as f:
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
        out = subprocess.run([f"./{binary}"], cwd=str(cwd), input="quit\n",
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
    """W/L/D and Elo for the engine whose name contains name_substr."""
    W = L = D = 0
    text = Path(pgn_path).read_text(errors="replace")
    for w, b, r in re.findall(
            r'\[White "([^"]+)"\]\s*\[Black "([^"]+)"\]\s*\[Result "([^"]+)"\]', text):
        is_white = name_substr in w
        if r == "1/2-1/2":
            D += 1
        elif (r == "1-0") == is_white:
            W += 1
        elif r in ("1-0", "0-1"):
            L += 1
    n = W + L + D
    if n == 0:
        return 0, 0, 0, float("nan"), float("nan")
    s = (W + 0.5 * D) / n
    if 0 < s < 1:
        elo = -400 * math.log10(1 / s - 1)
        err = 400 / math.log(10) * math.sqrt(s * (1 - s) / n) / (s * (1 - s))
    else:
        elo, err = float("inf") if s == 1 else float("-inf"), float("nan")
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
    ap.add_argument("--openings", default="training_openings.epd")
    ap.add_argument("--no-repeat", action="store_true",
                    help="Play each opening once during generation (match.py "
                         "--no-repeat).  Essential for frozen pairs "
                         "(TDLEAF_FREEZE=1): two identical deterministic "
                         "engines replay the exact same game from an opening "
                         "every time it comes up — including the color-swapped "
                         "repeat — so anything beyond one game per opening is "
                         "duplicated compute.  Cap --games at the opening "
                         "count.")
    ap.add_argument("--dedup-corpus", action="store_true",
                    help="Drop duplicate corpus rows (identical in every "
                         "field except gid) during corpus assembly.  "
                         "Auto-enabled when TDLEAF_FREEZE is set in the "
                         "environment — frozen-pair generation duplicates "
                         "whole games, and duplicates straddle the by-game "
                         "train/val split (they carry different gids), so "
                         "training on them both overfits and leaks validation.")
    ap.add_argument("--quiet-cp", type=int, default=60,
                    help="TDLEAF_DUMP_QUIET_CP for the dump (default 60)")
    # offline consolidation
    ap.add_argument("--skip-train", action="store_true",
                    help="Skip offline training (generate-only)")
    ap.add_argument("--corpus", action="append", default=[],
                    help="Extra corpus TSV(s) to include in training (repeatable)")
    ap.add_argument("--bt-threads", type=int, default=8,
                    help="Worker threads for within-batch gradient compute "
                         "(single-process; default 8)")
    ap.add_argument("--epochs", type=int, default=6)
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
    # gauntlet
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
    ap.add_argument("--gauntlet-games", type=int, default=400)
    ap.add_argument("--tc", default="3+0.05")
    ap.add_argument("--gauntlet-epochs", action="store_true",
                    help="Fast per-epoch ladder: after each epoch's training, "
                         "rate that snapshot vs the net as it stood BEFORE "
                         "offline training (post-online checkpoint, or the "
                         "incoming state under --skip-online) — measures what "
                         "consolidation added.  Independent of --gauntlet.")
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
    bt_bin = compile_binary("bt", args.net, tdleaf=True, force=args.recompile)
    if not args.skip_online:
        tr_a = compile_binary("train_hl_a", args.net, tdleaf=True, force=args.recompile)
        tr_b = compile_binary("train_hl_b", args.net, tdleaf=True, force=args.recompile)
        shutil.copy2(tr_a, LEARN_DIR / tr_a.name)
        shutil.copy2(tr_b, LEARN_DIR / tr_b.name)

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
        # PGN lives in <tag>_work/ (kept, gzip'd at end of run) rather than a
        # flat learn/pgn/ tree — the work dir is the permanent per-run archive.
        pgn_out = work / f"match_{args.tag}_d{args.depth}.pgn"
        log(f"online generation: {args.games} games at depth {args.depth} "
            f"(dump -> {work}/{args.tag}.*)")
        gen_cmd = ["python3", SCRIPT_DIR / "match.py",
                   "Leaf_vtrain_hl_a", "Leaf_vtrain_hl_b",
                   "-n", args.games, "-c", args.concurrency,
                   "--depth1", args.depth, "--depth2", args.depth,
                   "--openings", args.openings, "--fischer-random",
                   "--no-adjudication", "--pgn-out", pgn_out]
        if args.no_repeat:
            gen_cmd.append("--no-repeat")
        sh(gen_cmd, cwd=LEARN_DIR, env=env)

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
    inputs = dump_files + [Path(c) for c in args.corpus]
    if not inputs:
        die("nothing to train on (no dumps and no --corpus)")
    for c in inputs:
        if not c.is_file():
            die(f"corpus not found: {c}")
    # Detect corpus axis (game-ply vs legacy record-index).  Concatenation
    # strips '#' comment lines, which would erase the "# tdleaf-corpus
    # axis=game-ply" marker the trainer keys on — so we detect it here and
    # re-emit it at the top of the combined file.  Mixing axes changes the
    # meaning of the ply column (and hence the td_lambda decay), so refuse a
    # mix, mirroring the trainer.
    axes = {_corpus_is_game_ply(c) for c in inputs}
    if len(axes) > 1:
        die("cannot mix game-ply-axis and legacy record-index corpora in one "
            "run — the ply column means different things; train them separately")
    game_ply_axis = axes.pop()
    corpus_path = work / "corpus.tsv"
    # Frozen-pair generation duplicates whole games (deterministic engines
    # replay each opening identically), and the duplicates carry different
    # gids — so they land on both sides of the trainer's by-game train/val
    # split.  Dedup on every field except gid.  Auto-enabled under
    # TDLEAF_FREEZE; a no-op-with-overhead on learning corpora (~1.01x dup).
    dedup = args.dedup_corpus or bool(os.environ.get("TDLEAF_FREEZE"))
    if dedup and not args.dedup_corpus:
        log("TDLEAF_FREEZE set — enabling corpus row dedup")
    log(f"assembling {len(inputs)} corpus file(s) -> {corpus_path.name} "
        f"(axis={'game-ply' if game_ply_axis else 'legacy record-index'}"
        f"{', dedup' if dedup else ''}) ...")
    rows = 0
    dropped = 0
    seen = set()
    with open(corpus_path, "w") as out:
        if game_ply_axis:
            out.write("# tdleaf-corpus axis=game-ply\n")
        for src in inputs:
            with open(src) as f:
                for line in f:
                    if line.startswith("#") or line.startswith("fen\t"):
                        continue
                    if dedup:
                        p = line.split("\t")
                        # fen cp result ply depth gid endply — drop gid (5)
                        key = hashlib.md5(
                            "\t".join(p[:5] + p[6:]).encode()).digest()
                        if key in seen:
                            dropped += 1
                            continue
                        seen.add(key)
                    out.write(line)
                    rows += 1
    del seen
    if dedup:
        log(f"{rows:,} positions assembled ({dropped:,} duplicate rows dropped)")
    else:
        log(f"{rows:,} positions assembled")

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

    # Epoch-ladder opponent = the net exactly as it enters offline training: the
    # post-online checkpoint when we generated games this run, or the incoming
    # live state under --skip-online.  Bake base .nnue + live .tdleaf.bin (the
    # trainer's own starting state, freshly copied into tdir above) into a
    # standalone net via the trainer binary's --write-nnue, then compile an
    # inference binary from it.  Done before the trainer launches so the bake
    # sees the untouched state and doesn't contend with training for cores.
    ladder_opp_bin = ladder_opp_net = ladder_opp = None
    if args.gauntlet_epochs:
        pre_nnue = tdir / f"{args.tag}_pretrain.nnue"
        log(f"baking pre-offline-training baseline net -> {pre_nnue.name}")
        sh(["./Leaf_vbt", "--write-nnue", pre_nnue.name], cwd=tdir)
        if not pre_nnue.is_file():
            die(f"failed to bake pre-training baseline {pre_nnue}")
        ladder_opp_bin, ladder_opp_net, ladder_opp = build_rating_binary(
            pre_nnue, f"{args.tag}-pretrain")
        log(f"epoch-ladder opponent: {ladder_opp} (net before offline training)")

    log(f"training: {args.bt_threads} threads x {args.epochs} epochs "
        f"(lr {args.bt_lr}, lambda {args.bt_lambda}, K {args.bt_K})")
    cmd = ["./Leaf_vbt", "--batch-train", "../corpus.tsv",
           "--bt-epochs", str(args.epochs), "--bt-out", args.tag,
           "--bt-threads", str(args.bt_threads),
           "--bt-lr", str(args.bt_lr), "--bt-lambda", str(args.bt_lambda),
           "--bt-K", str(args.bt_K), "--bt-batch", str(args.bt_batch),
           "--bt-seed", "1000"]
    if args.bt_leaf_lambda is not None:
        cmd += ["--bt-leaf-lambda", str(args.bt_leaf_lambda)]
    if args.bt_td_lambda is not None:
        cmd += ["--bt-td-lambda", str(args.bt_td_lambda)]
    if args.bt_loss_gamma is not None:
        cmd += ["--bt-loss-gamma", str(args.bt_loss_gamma)]
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

    results = []
    if not gauntlet_list:
        log("no gauntlet opponents (--gauntlet/--gauntlet-anchors): "
            "skipping final gauntlet matches.")
    elif args.no_final_gauntlet:
        log("--no-final-gauntlet: skipping final gauntlet matches.")
    else:
        for opp in gauntlet_list:
            if not (LEARN_DIR / opp).is_file():
                log(f"WARNING: opponent {opp} not found in learn/ — skipping")
                continue
            pgn = work / f"match_{args.tag}-final_vs_{opp.replace('Leaf_v','')}.pgn"
            log(f"gauntlet: vs {opp} ({args.gauntlet_games} games)")
            sh(["python3", SCRIPT_DIR / "match.py", rate_bin.name, opp,
                "-n", args.gauntlet_games, "-c", 8, "-tc", args.tc,
                "--openings", args.openings, "--fischer-random",
                "--pgn-out", pgn], cwd=LEARN_DIR)
            results.append((opp, pgn_score(pgn, f"{args.tag}-final")))

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
        "depth": args.depth,
        "epochs": args.epochs,
        "picked_epoch": pick_ep,
        "bt_lr": args.bt_lr,
        "bt_lambda": args.bt_lambda,
        "bt_K": args.bt_K,
        "bt_td_lambda": args.bt_td_lambda,
        "gauntlet_anchors": args.gauntlet_anchors,
        "epoch_ladder": [
            {"epoch": ep, "W": W, "L": L, "D": D, "elo": elo, "err": err}
            for ep, (W, L, D, elo, err) in epoch_results
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
