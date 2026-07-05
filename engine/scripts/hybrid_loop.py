#!/usr/bin/env python3
"""
hybrid_loop.py — one command per hybrid-loop iteration:

    promote state -> online self-play generation (TDLeaf learning + leaf/root
    corpus dumping) -> checkpoint -> shard -> sharded offline consolidation ->
    export merged net -> gauntlet.

Implements the playbook in the Hybrid_Loop_Runbook note.  Non-interactive;
every phase is skippable, so it also covers the general "start from a given
point and continue learning" cases:

  Full iteration (generate + consolidate + rate; settled gen-3 recipe —
  pure lambda-return: --bt-lambda defaults to 1.0, the --bt-td-lambda
  distance decay (default TDLEAF_LAMBDA = 0.98) supplies all the outcome
  moderation and is the single knob of record):
    python3 hybrid_loop.py --tag iter3 --games 400000 --depth 8 \
        --state tdL10F10x6_p0_ep4.tdleaf.bin --recompile \
        --shards 1 --bt-K 220 \
        --gauntlet-epochs --gauntlet Leaf_vtdL10F10x6-ep4 Leaf_vclassic_eval

  Leaf rows (depth 0) default to the same lambda as roots; give them their own
  outcome weight with --bt-leaf-lambda.  Both lambda ceilings are dormant
  scale knobs in the settled recipe (kept to renormalize across corpora with
  different ply-gap distributions, and for reproducing past runs).

  --gauntlet-epochs (requires --shards 1) rates every epoch snapshot vs the
  first --gauntlet opponent as soon as that epoch finishes training (default
  1000 games at 1+0.01), and prints an epoch ladder table at the end.  The
  trainer is SIGSTOPped while each ladder match runs so training never
  contends with the games for cores, then SIGCONTed to resume.

  Consolidate-only (offline training on existing corpora):
    python3 hybrid_loop.py --tag redo --skip-online \
        --corpus quiet_d8_260702.tsv --corpus quiet_arch_260628d6.tsv \
        --gauntlet Leaf_v260628-2.4e6g

  Generate-only (online games + corpus dump, no offline training):
    python3 hybrid_loop.py --tag gen3 --games 200000 --depth 8 --skip-train

Run from engine/learn/.  Artifacts land in learn/:
    <netbase>.tdleaf.bin-pre<tag>      backup of the live state (if --state)
    <netbase>.tdleaf.bin-<tag>-online  post-generation online checkpoint
    <tag>_work/                        dumps, shards, training dir, logs
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


def compile_binary(version, net_name, tdleaf, force=False):
    """Compile Leaf_v<version> in run/; returns the binary path."""
    binary = RUN_DIR / f"Leaf_v{version}"
    if binary.exists() and not force:
        log(f"using existing binary {binary.name}")
        return binary
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
    ap.add_argument("--shards", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--bt-lr", type=float, default=0.25)
    ap.add_argument("--bt-lambda", type=float, default=1.0)
    ap.add_argument("--bt-K", type=float, default=220.0)
    ap.add_argument("--bt-batch", type=int, default=512)
    ap.add_argument("--sync-every", type=int, default=256)
    ap.add_argument("--bt-leaf-lambda", type=float, default=None,
                    help="Outcome-weight ceiling for depth-0 (leaf) rows "
                         "(default: same as --bt-lambda)")
    ap.add_argument("--bt-td-lambda", type=float, default=None,
                    help="Result decay per ply from the game end "
                         "(default: trainer's TDLEAF_LAMBDA; 1.0 = flat blend)")
    # gauntlet
    ap.add_argument("--gauntlet", nargs="*", default=[],
                    help="Opponent binaries in learn/ (empty = skip gauntlet)")
    ap.add_argument("--gauntlet-games", type=int, default=400)
    ap.add_argument("--tc", default="3+0.05")
    ap.add_argument("--gauntlet-epochs", action="store_true",
                    help="Fast per-epoch ladder: after each epoch's training, "
                         "rate that snapshot vs the first --gauntlet opponent "
                         "(requires --shards 1)")
    ap.add_argument("--epoch-games", type=int, default=1000,
                    help="Games per epoch-ladder match (default 1000)")
    ap.add_argument("--epoch-tc", default="1+0.01",
                    help="Epoch-ladder time control (default 1+0.01)")
    ap.add_argument("--no-final-gauntlet", action="store_true",
                    help="Skip the final full gauntlet (with --gauntlet-epochs: "
                         "ladder-only runs, --gauntlet names just the ladder "
                         "opponent)")
    ap.add_argument("--force", action="store_true",
                    help="Reuse an existing <tag>_work directory")
    ap.add_argument("--recompile", action="store_true",
                    help="Force recompile of helper binaries")
    args = ap.parse_args()

    if Path.cwd().resolve() != LEARN_DIR.resolve():
        die(f"run from {LEARN_DIR} (cwd is {Path.cwd()})")

    if args.gauntlet_epochs:
        if args.shards != 1:
            die("--gauntlet-epochs requires --shards 1 (per-shard snapshots "
                "are partial views of the merged state)")
        if not args.gauntlet:
            die("--gauntlet-epochs needs at least one --gauntlet opponent "
                "(the first is used for the epoch ladder)")

    net_path = LEARN_DIR / args.net
    if not net_path.is_file():
        die(f"base net not found: {net_path}")
    netbase = args.net[:-5] if args.net.endswith(".nnue") else args.net
    live_td = LEARN_DIR / f"{netbase}.tdleaf.bin"

    work = LEARN_DIR / f"{args.tag}_work"
    if work.exists() and not args.force:
        die(f"{work} exists — use --force to reuse it")
    work.mkdir(exist_ok=True)

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

        dump_files = sorted(work.glob(f"{args.tag}.*.tsv"))
        if not dump_files:
            die("online phase produced no dump files")
        log(f"{len(dump_files)} dump files")

    if args.skip_train:
        log("generate-only mode: done.")
        return

    # ---- Phase 4: assemble corpus and shard -------------------------------
    inputs = dump_files + [Path(c) for c in args.corpus]
    if not inputs:
        die("nothing to train on (no dumps and no --corpus)")
    for c in inputs:
        if not c.is_file():
            die(f"corpus not found: {c}")
    log(f"sharding {len(inputs)} corpus file(s) into {args.shards} shards ...")
    shard_fh = [open(work / f"shard_{n}.tsv", "w") for n in range(args.shards)]
    rows = 0
    for src in inputs:
        with open(src) as f:
            for line in f:
                if line.startswith("#") or line.startswith("fen\t"):
                    continue
                shard_fh[rows % args.shards].write(line)
                rows += 1
    for f in shard_fh:
        f.close()
    log(f"{rows:,} positions sharded")

    # ---- Phase 5: sharded offline consolidation ---------------------------
    tdir = work / "train"
    tdir.mkdir(exist_ok=True)
    shutil.copy2(bt_bin, tdir / "Leaf_vbt")
    shutil.copy2(net_path, tdir / args.net)
    shutil.copy2(live_td, tdir / f"{netbase}.tdleaf.bin")

    log(f"training: {args.shards} workers x {args.epochs} epochs "
        f"(lr {args.bt_lr}, lambda {args.bt_lambda}, sync every {args.sync_every})")
    procs = []
    for n in range(args.shards):
        logf = open(tdir / f"p{n}.log", "w")
        cmd = ["./Leaf_vbt", "--batch-train", f"../shard_{n}.tsv",
               "--bt-epochs", str(args.epochs), "--bt-out", f"{args.tag}_p{n}",
               "--bt-sync", f"{args.tag}_sync.tdleaf.bin",
               "--bt-sync-every", str(args.sync_every),
               "--bt-lr", str(args.bt_lr), "--bt-lambda", str(args.bt_lambda),
               "--bt-K", str(args.bt_K), "--bt-batch", str(args.bt_batch),
               "--bt-seed", str(1000 + n)]
        if args.bt_leaf_lambda is not None:
            cmd += ["--bt-leaf-lambda", str(args.bt_leaf_lambda)]
        if args.bt_td_lambda is not None:
            cmd += ["--bt-td-lambda", str(args.bt_td_lambda)]
        p = subprocess.Popen(cmd, cwd=str(tdir),
                             stdout=subprocess.DEVNULL, stderr=logf)
        procs.append((p, logf))

    # Per-epoch ladder: rate each epoch snapshot as soon as its epoch's
    # training finishes, while the trainer keeps running.  The trainer writes
    # _epN.nnue then _epN.tdleaf.bin, so the .tdleaf.bin appearing means the
    # .nnue is complete.
    def rate_epoch(ep, opp):
        snap = tdir / f"{args.tag}_p0_ep{ep}.nnue"
        ver = f"{args.tag}-ep{ep}"
        shutil.copy2(snap, RUN_DIR / snap.name)
        b = compile_binary(ver, snap.name, tdleaf=False, force=True)
        shutil.copy2(b, LEARN_DIR / b.name)
        shutil.copy2(snap, LEARN_DIR / snap.name)   # net resolves next to binary
        (RUN_DIR / snap.name).unlink()
        pgn = LEARN_DIR / f"match_{ver}_vs_{opp.replace('Leaf_v', '')}.pgn"
        log(f"epoch ladder: epoch {ep} vs {opp} "
            f"({args.epoch_games} games at {args.epoch_tc})")
        sh(["python3", SCRIPT_DIR / "match.py", b.name, opp,
            "-n", args.epoch_games, "-c", 8, "-tc", args.epoch_tc,
            "--openings", args.openings, "--fischer-random",
            "--pgn-out", pgn], cwd=LEARN_DIR)
        W, L, D, elo, err = pgn_score(pgn, ver)
        log(f"epoch ladder: epoch {ep}  W/L/D {W}/{L}/{D}  Elo {elo:+.0f} ± {err:.0f}")
        return (W, L, D, elo, err)

    epoch_results = []
    if args.gauntlet_epochs:
        opp = args.gauntlet[0]
        if not (LEARN_DIR / opp).is_file():
            die(f"epoch-ladder opponent {opp} not found in learn/")
        proc, logf = procs[0]
        rated = 0
        while True:
            if rated < args.epochs and \
                    (tdir / f"{args.tag}_p0_ep{rated + 1}.tdleaf.bin").exists():
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
            die(f"trainer process failed (rc={rc}) — see {tdir}/p0.log")
        # Sweep snapshots that landed between the last check and process exit.
        while rated < args.epochs and \
                (tdir / f"{args.tag}_p0_ep{rated + 1}.tdleaf.bin").exists():
            rated += 1
            epoch_results.append((rated, rate_epoch(rated, args.gauntlet[0])))
    else:
        fail = 0
        for p, logf in procs:
            rc = p.wait()
            logf.close()
            if rc != 0:
                fail += 1
        if fail:
            die(f"{fail} trainer process(es) failed — see {tdir}/p*.log")

    if epoch_results:
        print(f"\n=== Epoch ladder: {args.tag} vs {args.gauntlet[0]} "
              f"({args.epoch_games} games at {args.epoch_tc}) ===")
        for ep, (W, L, D, elo, err) in epoch_results:
            n = W + L + D
            s = 100 * (W + 0.5 * D) / max(n, 1)
            print(f"  epoch {ep}:  W/L/D {W}/{L}/{D}  score {s:5.1f}%  "
                  f"Elo {elo:+.0f} ± {err:.0f}")
        print()

    # export merged final via a zero-LR pass over the sync file
    log("exporting merged final net ...")
    exp = tdir / "export"
    exp.mkdir(exist_ok=True)
    shutil.copy2(bt_bin, exp / "Leaf_vbt")
    shutil.copy2(net_path, exp / args.net)
    shutil.copy2(tdir / f"{args.tag}_sync.tdleaf.bin", exp / f"{netbase}.tdleaf.bin")
    with open(work / "shard_0.tsv") as f, open(exp / "tiny.tsv", "w") as g:
        for i, line in enumerate(f):
            if i >= 3000:
                break
            g.write(line)
    sh(["./Leaf_vbt", "--batch-train", "tiny.tsv", "--bt-epochs", "1",
        "--bt-lr", "0.0", "--bt-out", f"../{args.tag}_final"], cwd=exp)

    final_nnue = tdir / f"{args.tag}_final_ep1.nnue"
    final_td   = tdir / f"{args.tag}_final_ep1.tdleaf.bin"
    out_nnue = LEARN_DIR / f"{args.tag}_final.nnue"
    out_td   = LEARN_DIR / f"{args.tag}_final.tdleaf.bin"
    shutil.copy2(final_nnue, out_nnue)
    shutil.copy2(final_td, out_td)
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
