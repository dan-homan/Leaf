#!/usr/bin/env python3
"""
hybrid_loop.py — one command per hybrid-loop iteration:

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
    python3 hybrid_loop.py --tag iter3 --games 400000 --depth 8 \
        --state tdL10F10x6_ep4.tdleaf.bin --recompile \
        --bt-K 220 --bt-threads 8 \
        --gauntlet-epochs --gauntlet Leaf_vtdL10F10x6-ep4 Leaf_vclassic_eval

  Consolidation is a single process with within-batch thread parallelism
  (--bt-threads, default 8) — synchronous data parallelism, mathematically
  identical to single-threaded training up to float summation order (the
  removed multi-process --bt-sync sharding suffered gradient staleness that
  destroyed the subtle gen-2+ signal; see docs/OFFLINE_TRAINING.md).

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
    python3 hybrid_loop.py --tag redo --skip-online \
        --corpus quiet_d8_260702.tsv --corpus quiet_arch_260628d6.tsv \
        --gauntlet Leaf_v260628-2.4e6g

  Generate-only (online games + corpus dump, no offline training):
    python3 hybrid_loop.py --tag gen3 --games 200000 --depth 8 --skip-train

  Start-to-finish from scratch (--init-nnue creates --net + its companion
  .tdleaf.bin, then the same generate -> consolidate -> gauntlet pipeline runs;
  prior = material|classical|noprior, bare flag = material):
    python3 hybrid_loop.py --tag scratch --net nn-scratch.nnue \
        --init-nnue --games 400000 --depth 8 --bt-threads 8 \
        --gauntlet Leaf_vclassic_eval

Run from engine/learn/.  Artifacts land in learn/:
    <netbase>.tdleaf.bin-pre<tag>      backup of the live state (if --state)
    <netbase>.tdleaf.bin-<tag>-online  post-generation online checkpoint
    <tag>_work/                        dumps, corpus.tsv, training dir, logs
    <tag>_final.nnue                   consolidated net (piece_val baked;
                                       compile rating binaries from this)
    <tag>_final.tdleaf.bin             seeds the next iteration (pairs with
                                       the ORIGINAL base .nnue)
    Leaf_v<tag>-final                  rating binary (when gauntlet runs)
"""

import argparse
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

# Set from --wdl-head: compile TDLEAF binaries with WDL_HEAD=1 so the auxiliary
# WDL head is trained (online + offline) and persisted in the .tdleaf.bin.
# Only applied to TDLEAF builds (WDL_HEAD requires TDLEAF=1); TDLEAF-off gauntlet
# rating binaries are unaffected — the head is a read-out and doesn't alter play.
WDL_HEAD_FLAG = False


def log(msg):
    print(f"[hybrid_loop {time.strftime('%H:%M:%S')}] {msg}", flush=True)


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
        if WDL_HEAD_FLAG:
            flags.append("WDL_HEAD=1")
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
    the outcome-imbalance pathology (see docs/TDLEAF.md).  Report-only under
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


def main():
    ap = argparse.ArgumentParser(
        description="One hybrid-loop iteration: generate -> consolidate -> gauntlet.")
    ap.add_argument("--tag", required=True,
                    help="Iteration name (prefixes all artifacts)")
    ap.add_argument("--net", default="nn-fresh-260628.nnue",
                    help="Base .nnue in learn/ (default nn-fresh-260628.nnue)")
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
    ap.add_argument("--wdl-head", action="store_true",
                    help="Compile TDLEAF binaries with WDL_HEAD=1 so the auxiliary "
                         "win/draw/loss head is trained (online TD(lambda) + offline "
                         "outcome target) and persisted in the .tdleaf.bin. The head "
                         "is a read-out (does not change play or the promoted .nnue); "
                         "watch offline WDL_Brier in the batch-train log for calibration. "
                         "Pair with --recompile so existing same-named binaries rebuild.")
    args = ap.parse_args()
    global WDL_HEAD_FLAG
    WDL_HEAD_FLAG = args.wdl_head

    if Path.cwd().resolve() != LEARN_DIR.resolve():
        die(f"run from {LEARN_DIR} (cwd is {Path.cwd()})")

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
        init_bin = compile_binary("bt", args.net, tdleaf=True, force=args.recompile)
        flag = {"material":  "--init-nnue",
                "classical": "--init-nnue-classical",
                "noprior":   "--init-nnue-noprior"}[args.init_nnue]
        log(f"initialising fresh net {net_path.name} (prior={args.init_nnue})")
        sh([f"./{init_bin.name}", flag, "--write-nnue", net_path], cwd=RUN_DIR)
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
        pgn_dir = LEARN_DIR / "pgn" / netbase
        pgn_dir.mkdir(parents=True, exist_ok=True)
        pgn_out = pgn_dir / f"match_{args.tag}_d{args.depth}.pgn"
        log(f"online generation: {args.games} games at depth {args.depth} "
            f"(dump -> {work}/{args.tag}.*)")
        sh(["python3", SCRIPT_DIR / "match.py",
            "Leaf_vtrain_hl_a", "Leaf_vtrain_hl_b",
            "-n", args.games, "-c", args.concurrency,
            "--depth1", args.depth, "--depth2", args.depth,
            "--openings", args.openings, "--fischer-random",
            "--no-adjudication", "--pgn-out", pgn_out],
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
    log(f"assembling {len(inputs)} corpus file(s) -> {corpus_path.name} "
        f"(axis={'game-ply' if game_ply_axis else 'legacy record-index'}) ...")
    rows = 0
    with open(corpus_path, "w") as out:
        if game_ply_axis:
            out.write("# tdleaf-corpus axis=game-ply\n")
        for src in inputs:
            with open(src) as f:
                for line in f:
                    if line.startswith("#") or line.startswith("fen\t"):
                        continue
                    out.write(line)
                    rows += 1
    log(f"{rows:,} positions assembled")

    # ---- Phase 5: offline consolidation (single threaded process) ---------
    tdir = work / "train"
    tdir.mkdir(exist_ok=True)
    shutil.copy2(bt_bin, tdir / "Leaf_vbt")
    shutil.copy2(net_path, tdir / args.net)
    shutil.copy2(live_td, tdir / f"{netbase}.tdleaf.bin")

    def build_rating_binary(snap, ver):
        """Compile a TDLEAF-off inference binary Leaf_v<ver> for the net SNAP
        (a .nnue in tdir), staging both binary and net into learn/ so the net
        resolves next to the binary.  Returns the binary name."""
        shutil.copy2(snap, RUN_DIR / snap.name)
        b = compile_binary(ver, snap.name, tdleaf=False, force=True)
        shutil.copy2(b, LEARN_DIR / b.name)
        shutil.copy2(snap, LEARN_DIR / snap.name)   # net resolves next to binary
        (RUN_DIR / snap.name).unlink()
        return b.name

    # Epoch-ladder opponent = the net exactly as it enters offline training: the
    # post-online checkpoint when we generated games this run, or the incoming
    # live state under --skip-online.  Bake base .nnue + live .tdleaf.bin (the
    # trainer's own starting state, freshly copied into tdir above) into a
    # standalone net via the trainer binary's --write-nnue, then compile an
    # inference binary from it.  Done before the trainer launches so the bake
    # sees the untouched state and doesn't contend with training for cores.
    ladder_opp = None
    if args.gauntlet_epochs:
        pre_nnue = tdir / f"{args.tag}_pretrain.nnue"
        log(f"baking pre-offline-training baseline net -> {pre_nnue.name}")
        sh(["./Leaf_vbt", "--write-nnue", pre_nnue.name], cwd=tdir)
        if not pre_nnue.is_file():
            die(f"failed to bake pre-training baseline {pre_nnue}")
        ladder_opp = build_rating_binary(pre_nnue, f"{args.tag}-pretrain")
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
    def rate_epoch(ep, opp):
        ver = f"{args.tag}-ep{ep}"
        bname = build_rating_binary(tdir / f"{args.tag}_ep{ep}.nnue", ver)
        pgn = LEARN_DIR / f"match_{ver}_vs_{opp.replace('Leaf_v', '')}.pgn"
        log(f"epoch ladder: epoch {ep} vs {opp} "
            f"({args.epoch_games} games at {args.epoch_tc})")
        sh(["python3", SCRIPT_DIR / "match.py", bname, opp,
            "-n", args.epoch_games, "-c", 8, "-tc", args.epoch_tc,
            "--openings", args.openings, "--fischer-random",
            "--pgn-out", pgn], cwd=LEARN_DIR)
        W, L, D, elo, err = pgn_score(pgn, ver)
        log(f"epoch ladder: epoch {ep}  W/L/D {W}/{L}/{D}  Elo {elo:+.0f} ± {err:.0f}")
        return (W, L, D, elo, err)

    epoch_results = []
    if args.gauntlet_epochs:
        opp = ladder_opp
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
                    epoch_results.append((rated, rate_epoch(rated, opp)))
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
            epoch_results.append((rated, rate_epoch(rated, opp)))
    else:
        rc = proc.wait()
        logf.close()
        if rc != 0:
            die(f"trainer process failed (rc={rc}) — see {tdir}/train.log")

    if epoch_results:
        print(f"\n=== Epoch ladder: {args.tag} vs {ladder_opp} "
              f"({args.epoch_games} games at {args.epoch_tc}) ===")
        for ep, (W, L, D, elo, err) in epoch_results:
            n = W + L + D
            s = 100 * (W + 0.5 * D) / max(n, 1)
            print(f"  epoch {ep}:  W/L/D {W}/{L}/{D}  score {s:5.1f}%  "
                  f"Elo {elo:+.0f} ± {err:.0f}")
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
    if not args.gauntlet:
        log("no --gauntlet opponents given: done.")
        return
    if args.no_final_gauntlet:
        log("--no-final-gauntlet: done.")
        return
    shutil.copy2(out_nnue, RUN_DIR / out_nnue.name)
    rate_bin = compile_binary(f"{args.tag}-final", out_nnue.name,
                              tdleaf=False, force=True)
    shutil.copy2(rate_bin, LEARN_DIR / rate_bin.name)
    (RUN_DIR / out_nnue.name).unlink()

    results = []
    for opp in args.gauntlet:
        if not (LEARN_DIR / opp).is_file():
            log(f"WARNING: opponent {opp} not found in learn/ — skipping")
            continue
        pgn = LEARN_DIR / f"match_{args.tag}-final_vs_{opp.replace('Leaf_v','')}.pgn"
        log(f"gauntlet: vs {opp} ({args.gauntlet_games} games)")
        sh(["python3", SCRIPT_DIR / "match.py", rate_bin.name, opp,
            "-n", args.gauntlet_games, "-c", 8, "-tc", args.tc,
            "--openings", args.openings, "--fischer-random",
            "--pgn-out", pgn], cwd=LEARN_DIR)
        results.append((opp, pgn_score(pgn, f"{args.tag}-final")))

    print("\n=== Gauntlet results:", rate_bin.name, "===")
    for opp, (W, L, D, elo, err) in results:
        n = W + L + D
        s = 100 * (W + 0.5 * D) / max(n, 1)
        print(f"  vs {opp:<28} n={n:<5} W/L/D {W}/{L}/{D}  "
              f"score {s:5.1f}%  Elo {elo:+.0f} ± {err:.0f}")
    print()
    log("iteration complete.")


if __name__ == "__main__":
    main()
