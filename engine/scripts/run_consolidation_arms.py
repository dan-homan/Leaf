#!/usr/bin/env python3
# Leaf — GPL v3-or-later.  Copyright (C) 2026 Daniel C. Homan.
"""Composite-corpus consolidation arms: one epoch each, matched dose, rated.

The question
------------
A leg's online phase generates ~1M games; its consolidation has been seeing
only those games (``--corpus-window 0``).  Four legs' dumps are now on disk.
Does a consolidation round over the COMBINED corpus beat another epoch over
the newest one alone, at matched dose -- and if so, what should its target and
row type be?

The arms
--------
Every arm starts from the same base ``.nnue`` + ``.tdleaf.bin`` (so identical
weights AND Adam moments), trains ONE epoch, and is rated over the same
opponents.  ⚠️ The base net is the chain's CONSTANT one (``m260916.nnue``); the
``<tag>_final.nnue`` files are baked exports for rating, and handing one to the
trainer makes it reject the state and quietly train from the 5M-game-old base.  Row counts are
matched exactly by ``sample_corpus.py``; see that script for how (1) equal rows
and (2) equal game weight are held simultaneously.

  null   the newest leg's ENTIRE eligible corpus, unsampled.  This is exactly
         "another epoch on the same corpus", the control the composite must
         beat, and it sets the dose every other arm matches.  It is the one arm
         with no sampling question hanging over it.
  base   four legs, quota-sampled, same row total.  base - null = the value of
         game diversity and label age at matched dose and matched steps.
  bout   base's EXACT row file, --bt-td-lambda raised: more outcome, less cp.
  bcp    base's EXACT row file, --bt-td-lambda lowered: more cp, less outcome.
  leaf   four legs, leaf rows instead of root, same total AND the same games
         as base.  Pure leaf against pure root -- not a blend, and not a
         different game set, so the answer is attributable.

bout/bcp reuse base's row file byte for byte, so those three differ ONLY in the
target and carry no sampling noise between them.  The default td_lambda is
TDLEAF_LAMBDA = 0.985, which puts the outcome's weight at 0.08 at ply 0 of a
167-ply game and ~0.37 averaged over the game.

Why the b arms come before rescoring
------------------------------------
Rescoring (``--bt-rescore``) is the mechanism-matched fix for stale cp labels
and the only expensive arm here -- a search over every row.  bout/bcp bound its
payoff for the price of a flag: if the target curve slopes toward the outcome,
stale cp labels are hurting and rescoring is worth its cost; if it peaks at or
below the default, cp labels are not the binding constraint and rescoring
cannot pay.

Usage
-----
    cd engine/learn
    python3 run_consolidation_arms.py --tag cons1 \\
        --seed-net m260916.nnue \\
        --seed-state m260916-5e6g_final.tdleaf.bin \\
        --sources m260916-2e6g_work m260916-3e6g_work \\
                  m260916-4e6g_work m260916-5e6g_work

Everything is resumable: an arm whose ``_ep1.nnue`` exists is not retrained,
and a match whose PGN exists is not replayed.  ``--only`` runs a subset.
"""

import argparse
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from train import pgn_score          # noqa: E402  (pure function, no globals)

LEARN = Path.cwd()
T0 = time.time()

# arm -> (corpus key, --bt-rows value, extra trainer flags[, epochs=1])
ARMS = {
    # Phase 1 -- is a wide consolidation window worth running?
    "null": ("null", "root", []),
    "base": ("base", "root", []),
    "bout": ("base", "root", ["--bt-td-lambda", "0.9925"]),
    "bcp":  ("base", "root", ["--bt-td-lambda", "0.97"]),
    "leaf": ("leaf", "leaf", []),
    # Phase 2 -- target and row type on the SHIPPING corpus (one leg).  Added
    # once phase 1 found base below null: tuning the target on a corpus we
    # would not use answers the wrong question, and these reuse corpus_null
    # byte for byte, so they carry no sampling noise against the null.
    "nout":  ("null", "root", ["--bt-td-lambda", "0.9925"]),
    "ncp":   ("null", "root", ["--bt-td-lambda", "0.97"]),
    "nleaf": ("nullleaf", "leaf", []),
    # Interior point: nout (0.9925) is clearly worse and ncp (0.97) is level
    # with the default (0.985), so the optimum, if the curve has one, lies
    # between 0.97 and 0.985.
    "ncp2":  ("null", "root", ["--bt-td-lambda", "0.9775"]),
    # DOSE, not composition: the newest leg's root rows AND its leaf rows over
    # the same games, ~2.2x the null's rows with the game count unchanged.
    # This is the one arm that deliberately breaks dose-matching -- the question
    # is whether a bigger corpus off the same games helps at all.  Note it also
    # takes 2.2x the Adam steps, so "more data" and "more steps" are confounded
    # here; `null2` (2 epochs on the null corpus) is the control that separates
    # them, at matched steps on repeated rather than fresh rows.
    "nboth": ("nullboth", "both", []),
    "null2": ("null", "root", [], 2),
    # Elo-test the K calibration: 60k games fit K = 188-192 against the
    # configured 220.  It LOST (-16.5 paired / -7.7 anchor), which is one of the
    # seven results behind Learning_Investigation.md SS1 O.
    "nK":    ("null", "root", ["--bt-K", "190"]),
}
# Arms that need compile flags living only on the `k-by-material` branch
# (TDLEAF_K_SHAPE, TDLEAF_LAMBDA_SHAPE, TDLEAF_W_RELIABILITY) are NOT listed
# here.  On main those flags are undefined, so the build would succeed and the
# arm would silently be a duplicate of `null` -- a result that looks like an arm
# and is not one.  See the guard in compile_binary().
ORDER = ["null", "base", "bout", "bcp", "leaf",
         "nout", "ncp", "nleaf", "ncp2", "nboth", "null2", "nK"]


def log(msg):
    el = time.time() - T0
    print(f"[{int(el) // 3600:d}:{int(el) // 60 % 60:02d}:{int(el) % 60:02d}] "
          f"{msg}", flush=True)


def die(msg):
    log(f"ERROR: {msg}")
    sys.exit(1)


def sh(cmd, **kw):
    cmd = [str(c) for c in cmd]
    log("$ " + " ".join(cmd))
    r = subprocess.run(cmd, **kw)
    if r.returncode:
        die(f"command failed ({r.returncode}): {' '.join(cmd)}")
    return r


def guard_run_dir():
    """Nothing here may execute with run/ as its cwd: main_bk.dat lives there
    and would silently feed book moves into training and rating alike.  See
    docs/TRAINING.md, 'The run/ invariant'."""
    if LEARN.name == "run" or (LEARN / "main_bk.dat").exists():
        die(f"refusing to run in {LEARN}: main_bk.dat is present (or this is "
            f"run/).  Training and rating binaries must run from learn/")


def compile_binary(version, net, tdleaf, dest, extra_flags=()):
    """Build Leaf_v<version> against NET, and land it in DEST beside a copy.

    comp.pl compiles into the current directory AND resolves ../src/Leaf.cc
    against it, so it only works from run/ or learn/ -- a per-arm subdirectory
    has no ../src.  So: compile in learn/ (never run/, which holds
    main_bk.dat), then move the binary into DEST with its .nnue, so the net
    resolves next to the executable and nothing is left lying in learn/."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    binary = dest / f"Leaf_v{version}"
    if binary.exists():
        binary.unlink()
    built = LEARN / f"Leaf_v{version}"
    if built.exists():
        built.unlink()
    staged = LEARN / net.name
    staged_here = not staged.exists()
    if staged_here:
        shutil.copy2(net, staged)
    flags = ["NNUE=1", f"NNUE_NET={net.name}"]
    if tdleaf:
        flags.append("TDLEAF=1")
        # Guard: comp.pl happily passes -D for a macro the source never reads,
        # so an arm keyed on a flag from another branch would build clean and
        # run as a duplicate of its control.  Refuse rather than measure that.
        src = SCRIPT_DIR.parent / "src"
        for f in extra_flags:
            name = str(f).split("=", 1)[0]
            hit = subprocess.run(["grep", "-rlq", name, str(src)],
                                 capture_output=True)
            if hit.returncode != 0:
                die(f"--compile-flags {f}: {name} appears nowhere in {src}. "
                    f"The build would succeed and the arm would be a silent "
                    f"duplicate of its control. Wrong branch?")
        flags += list(extra_flags)
    try:
        sh(["perl", "comp.pl", version] + flags + ["OVERWRITE"], cwd=str(LEARN))
    finally:
        if staged_here:
            staged.unlink(missing_ok=True)
    if not built.exists():
        die(f"compile did not produce {built}")
    shutil.move(str(built), str(binary))
    if not (dest / net.name).exists():
        shutil.copy2(net, dest / net.name)
    return binary


def check_engine_state(binary, cwd, expect_state=False):
    """Refuse to proceed unless the binary really loaded what it should.

    Two silent failures live here, both of which have cost real runs:

      * a missing .nnue does NOT error -- the engine falls back to classical
        eval, and the arm rates an entirely different program;
      * a .tdleaf.bin whose recorded source-.nnue hash does not match the
        loaded net is REFUSED with a warning and a zero exit, and training
        then proceeds from the base net with no learned weights and no Adam
        moments.  Every arm would "work" and none would mean anything.

    The second is why the trainer must be given the chain's BASE .nnue (the
    constant one, e.g. m260916.nnue) and not a baked <tag>_final.nnue export:
    the state's hash refers to the base."""
    r = subprocess.run([str(binary)], input="quit\n", capture_output=True,
                       text=True, cwd=str(cwd), timeout=180)
    out = r.stdout + r.stderr
    if "NNUE: not found" in out or "not found" in out:
        die(f"{binary.name} cannot load its .nnue (would run classical eval):\n"
            f"{out.strip()[:400]}")
    if expect_state:
        for bad in ("Refusing to load", "could NOT be loaded",
                    "NOT in memory"):
            if bad in out:
                die(f"{binary.name} did not load its .tdleaf.bin -- it would "
                    f"train from the base net with no learned weights:\n"
                    f"{out.strip()[:700]}")


def build_corpora(args, arms_dir, needed):
    """The DOSE is one leg's full eligible corpus, and every arm matches it.

    The null must be "another epoch on the newest leg alone", so the cleanest
    dose is that leg's entire eligible row set -- the null is then the corpus
    itself, untouched, with no sampling question hanging over the control.  The
    composite is quota-sampled and trimmed down to the same total.

    The order matters: count the null leg FIRST, because the composite at any
    useful quota is larger than one leg (four legs at quota 19 is 75.8M rows
    against 68.9M eligible in 5e6g), so the dose has to be set by the smaller
    side or the null cannot reach it."""
    corpora, rows = {}, {}

    null_src = args.null_source or args.sources[-1]
    cnt = sh(["python3", SCRIPT_DIR / "sample_corpus.py",
              "--source", null_src, "--rows", "root",
              "--quiet-cp", args.quiet_cp, "--count-only",
              "--out", arms_dir / "corpus_null.tsv"],
             capture_output=True, text=True)
    sys.stderr.write(cnt.stderr)
    info = json.loads(cnt.stdout)[str(null_src)]
    n = info["eligible_rows"]
    log(f"dose = {n:,} rows: every eligible row of {null_src} "
        f"({info['games']:,} games, {n / info['games']:.1f}/game)")

    have = args.quota * len(args.sources) * info["games"]
    if have < n:
        die(f"--quota {args.quota} over {len(args.sources)} legs yields about "
            f"{have:,} rows, short of the {n:,} dose -- raise --quota")

    base = arms_dir / "corpus_base.tsv"
    if "base" in needed and not base.exists():
        r = sh(["python3", SCRIPT_DIR / "sample_corpus.py",
                "--source", *args.sources, "--rows", "root",
                "--quiet-cp", args.quiet_cp, "--quota", args.quota,
                "--target-rows", n, "--seed", args.seed, "--out", base],
               capture_output=True, text=True)
        sys.stderr.write(r.stderr)
    corpora["base"], rows["base"] = base, n

    null = arms_dir / "corpus_null.tsv"
    if "null" in needed and not null.exists():
        # No per-game cap: the null IS the leg's corpus.  A quota here would
        # cost rows to the right tail of a skewed length distribution and the
        # leg could not reach its own row count.
        r = sh(["python3", SCRIPT_DIR / "sample_corpus.py",
                "--source", null_src, "--rows", "root",
                "--quiet-cp", args.quiet_cp, "--quota", 1 << 30,
                "--seed", args.seed, "--out", null],
               capture_output=True, text=True)
        sys.stderr.write(r.stderr)
    corpora["null"], rows["null"] = null, n

    leaf = arms_dir / "corpus_leaf.tsv"
    if "leaf" in needed and not leaf.exists():
        # Same GAMES as base, not merely the same row count.  Leaf rows survive
        # in games whose root rows the quiet gate removed entirely, so an
        # unrestricted leaf sample draws ~45% more games at correspondingly
        # fewer rows each -- and "leaf vs root" would then have a game-set
        # difference folded into it.
        r = sh(["python3", SCRIPT_DIR / "sample_corpus.py",
                "--source", *args.sources, "--rows", "leaf",
                "--quiet-cp", args.quiet_cp, "--quota", args.quota,
                "--restrict-gids", base,
                "--target-rows", n, "--seed", args.seed, "--out", leaf],
               capture_output=True, text=True)
        sys.stderr.write(r.stderr)
    corpora["leaf"], rows["leaf"] = leaf, n

    # Single-leg leaf: the same GAMES as the null, so nleaf - null is a pure
    # row-type contrast on the corpus we would actually ship.
    nleaf = arms_dir / "corpus_nullleaf.tsv"
    if "nullleaf" in needed and not nleaf.exists():
        r = sh(["python3", SCRIPT_DIR / "sample_corpus.py",
                "--source", null_src, "--rows", "leaf",
                "--quiet-cp", args.quiet_cp, "--quota", 1 << 30,
                "--restrict-gids", null,
                "--target-rows", n, "--seed", args.seed, "--out", nleaf],
               capture_output=True, text=True)
        sys.stderr.write(r.stderr)
    corpora["nullleaf"], rows["nullleaf"] = nleaf, n

    # Root AND leaf over the same games, both untrimmed: a DOSE arm.  Built by
    # concatenating the null (all eligible root rows) with every eligible leaf
    # row of the same games, so the game set is identical to the null's and only
    # the row count changes.
    nboth = arms_dir / "corpus_nullboth.tsv"
    if "nullboth" in needed and not nboth.exists():
        leaf_all = arms_dir / "corpus_nullleaf_all.tsv"
        if not leaf_all.exists():
            r = sh(["python3", SCRIPT_DIR / "sample_corpus.py",
                    "--source", null_src, "--rows", "leaf",
                    "--quiet-cp", args.quiet_cp, "--quota", 1 << 30,
                    "--restrict-gids", null,
                    "--seed", args.seed, "--out", leaf_all],
                   capture_output=True, text=True)
            sys.stderr.write(r.stderr)
        log(f"concatenating {null.name} + {leaf_all.name} -> {nboth.name}")
        tmp = nboth.with_suffix(".tmp")
        with open(tmp, "w") as out:
            with open(null) as fh:
                shutil.copyfileobj(fh, out)
            with open(leaf_all) as fh:
                for line in fh:            # drop the second file's 2 header lines
                    if line.startswith("#") or line.startswith("fen\t"):
                        continue
                    out.write(line)
        tmp.rename(nboth)
        lm = json.loads(leaf_all.with_suffix(".tsv.json").read_text())
        nboth.with_suffix(".tsv.json").write_text(json.dumps({
            "out": str(nboth), "rows_total": n + lm["rows_total"],
            "rows_per_source": {"root": n, "leaf": lm["rows_total"]},
            "games_per_source": lm["games_per_source"],
            "row_type": "both", "quiet_cp": args.quiet_cp,
            "note": "null root rows + all eligible leaf rows, same games",
        }, indent=2) + "\n")
    if nboth.exists():
        m = json.loads(nboth.with_suffix(".tsv.json").read_text())
        corpora["nullboth"], rows["nullboth"] = nboth, m["rows_total"]

    for k in needed:
        m = json.loads(corpora[k].with_suffix(".tsv.json").read_text())
        if k == "nullboth":
            log(f"{k}: {m['rows_total']:,} rows "
                f"({m['rows_total'] / n:.2f}x the dose -- this arm tests DOSE)")
            continue
        if m["rows_total"] != n:
            die(f"{k} corpus has {m['rows_total']:,} rows, dose is {n:,} -- "
                f"the arms would not be dose-matched")
        log(f"{k}: {m['rows_total']:,} rows over "
            f"{sum(m['games_per_source'].values()):,} games")
    return corpora, rows


def train_arm(arm, corpus, n_rows, seed_nnue, seed_state, args, arms_dir):
    adir = arms_dir / arm
    ep = ARMS[arm][3] if len(ARMS[arm]) > 3 else 1
    out_nnue = adir / f"{arm}_ep{ep}.nnue"
    if out_nnue.exists():
        log(f"{arm}: already trained ({out_nnue.name}) -- skipping")
        return out_nnue
    adir.mkdir(parents=True, exist_ok=True)
    # Every arm starts from the identical net AND optimizer state.  The state
    # must sit beside the binary under the net's own name, or the trainer
    # starts from scratch Adam moments and the arms are not comparable.
    bt = compile_binary(f"bt_{arm}", seed_nnue, tdleaf=True, dest=adir,
                        extra_flags=args.compile_flags)
    # The engine looks for <loaded-net-basename>.tdleaf.bin, so the state is
    # staged under the BASE net's name regardless of what it is called in
    # learn/.  Its FP32 shadow weights and Adam moments are the real starting
    # point -- the base .nnue alone is the chain's origin, 5M games ago.
    shutil.copy2(seed_state, adir / f"{seed_nnue.stem}.tdleaf.bin")
    check_engine_state(bt, adir, expect_state=True)

    spec = ARMS[arm]
    corpus_key, rows_mode, extra = spec[0], spec[1], spec[2]
    epochs = spec[3] if len(spec) > 3 else 1
    cmd = [f"./{bt.name}", "--batch-train", str(Path(corpus).resolve()),
           "--bt-epochs", str(epochs), "--bt-out", arm,
           "--bt-threads", str(args.threads),
           "--bt-lr", str(args.bt_lr), "--bt-lambda", str(args.bt_lambda),
           "--bt-K", str(args.bt_K), "--bt-batch", str(args.bt_batch),
           "--bt-quiet-cp", str(args.quiet_cp),
           "--bt-rows", rows_mode,
           "--bt-max", str(n_rows), "--bt-seed", "1000"] + extra
    with open(adir / "train.log", "w") as lf:
        log(f"{arm}: training {epochs} epoch(s) on {Path(corpus).name} "
            f"({n_rows:,} rows){' ' + ' '.join(extra) if extra else ''}")
        sh(cmd, cwd=str(adir), stdout=subprocess.DEVNULL, stderr=lf)
    if not out_nnue.exists():
        die(f"{arm}: trainer produced no {out_nnue.name} "
            f"(see {adir / 'train.log'})")
    return out_nnue


def rate(arm, net, opponents, args, arms_dir):
    """Rate ARM's net against every opponent.  Binary and net live together in
    the arm directory so the .nnue resolves next to the executable."""
    adir = arms_dir / arm
    ver = f"{args.tag}-{arm}"
    binary = adir / f"Leaf_v{ver}"
    if not binary.exists():
        compile_binary(ver, net, tdleaf=False, dest=adir)
        check_engine_state(binary, adir)

    out = {}
    for opp_bin, opp_name in opponents:
        pgn = arms_dir / f"match_{ver}_vs_{opp_name}.pgn"
        if not pgn.exists():
            log(f"{arm}: {args.games} games at {args.tc} vs {opp_name}")
            sh(["python3", SCRIPT_DIR / "match.py",
                str(binary.resolve()), str(Path(opp_bin).resolve()),
                "-n", args.games, "-c", args.cores, "-tc", args.tc,
                "--openings", args.openings, "--fischer-random",
                "--pgn-out", str(pgn)], cwd=str(LEARN))
        W, L, D, elo, err = pgn_score(pgn, f"Leaf_v{ver}")
        out[opp_name] = dict(W=W, L=L, D=D, elo=elo, err=err)
        log(f"{arm} vs {opp_name}: W/L/D {W}/{L}/{D}  Elo {elo:+.1f} ± {err:.1f}")
    return out


def render(results, opponents, args):
    names = [n for _, n in opponents]
    w = max(len(n) for n in names) + 14
    lines = [f"=== {args.tag}: composite-corpus consolidation arms "
             f"({args.games} games at {args.tc}, 1 epoch, one sigma) ===", ""]
    lines.append("  arm   " + "".join(f"{n:>{w}}" for n in names))
    for arm in ORDER:
        if arm not in results:
            continue
        row = f"  {arm:<6}"
        for n in names:
            r = results[arm].get(n)
            row += f"{r['elo']:+9.1f} ± {r['err']:4.1f}".rjust(w) if r \
                else "—".rjust(w)
        lines.append(row)
    if "base" in results and "null" in results:
        lines += ["", "  Reading (paired opponent is the sensitive one; the "
                      "foreign anchor is the tiebreak — §5):"]
        for n in names:
            b, nl = results["base"].get(n), results["null"].get(n)
            if b and nl:
                d = b["elo"] - nl["elo"]
                e = math.hypot(b["err"], nl["err"])
                lines.append(f"    base - null on {n}: {d:+.1f} ± {e:.1f} "
                             f"({abs(d / e) if e else 0:.1f} sigma) "
                             f"= diversity + label age at matched dose")
        for arm, what in (("bout", "more outcome weight"),
                          ("bcp", "more cp weight"),
                          ("leaf", "leaf rows instead of root")):
            for n in names[:1]:
                a, b = results.get(arm, {}).get(n), results["base"].get(n)
                if a and b:
                    d = a["elo"] - b["elo"]
                    e = math.hypot(a["err"], b["err"])
                    lines.append(f"    {arm} - base on {n}: {d:+.1f} ± {e:.1f} "
                                 f"({abs(d / e) if e else 0:.1f} sigma) "
                                 f"= {what}")
    return lines


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seed-net", required=True,
                    help="the chain's BASE .nnue (constant across the whole "
                         "chain, e.g. m260916.nnue) -- NOT a baked "
                         "<tag>_final.nnue export.  The state's recorded "
                         "source hash refers to the base, and a mismatch makes "
                         "the trainer silently discard every learned weight")
    ap.add_argument("--seed-state", required=True,
                    help="the .tdleaf.bin every arm starts from (e.g. "
                         "m260916-5e6g_final.tdleaf.bin).  This carries the "
                         "FP32 weights and Adam moments -- it, not --seed-net, "
                         "is where 5M games of learning live")
    ap.add_argument("--seed-export", default=None,
                    help="baked .nnue of the seed, used as the PAIRED rating "
                         "opponent (default: --seed-state with .tdleaf.bin "
                         "replaced by .nnue)")
    ap.add_argument("--sources", nargs="+", required=True,
                    help="work directories holding the raw per-leg dumps")
    ap.add_argument("--null-source", default=None,
                    help="single leg for the null arm (default: last --sources)")
    ap.add_argument("--quota", type=int, default=19,
                    help="rows per game in the composite (default 19)")
    ap.add_argument("--quiet-cp", type=int, default=60)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--bt-lr", type=float, default=1.0)
    ap.add_argument("--bt-lambda", type=float, default=1.0)
    ap.add_argument("--bt-K", type=float, default=220.0)
    ap.add_argument("--bt-batch", type=int, default=512)
    ap.add_argument("--games", type=int, default=2000)
    ap.add_argument("--tc", default="1+0.01")
    ap.add_argument("--cores", type=int, default=8)
    ap.add_argument("--openings", default="holdout_openings.epd",
                    help="holdout by default: the corpus games came from "
                         "training_openings.epd")
    ap.add_argument("--anchor", default="Leaf_vclassic_eval",
                    help="foreign anchor binary in learn/ (§5: family matches "
                         "are non-transitive, so never rate on paired alone)")
    ap.add_argument("--compile-flags", nargs="*", default=[], metavar="F=V",
                    help="extra comp.pl flags for the TRAINER build only (the "
                         "rating binary stays stock).  Refused if the macro "
                         "appears nowhere in src/")
    ap.add_argument("--only", nargs="+", choices=ORDER, default=None)
    ap.add_argument("--corpus-only", action="store_true",
                    help="build the corpora and stop")
    args = ap.parse_args()

    guard_run_dir()
    seed_nnue = (LEARN / args.seed_net).resolve()
    if not seed_nnue.exists():
        die(f"no {seed_nnue}")
    seed_state = (LEARN / args.seed_state).resolve()
    if not seed_state.exists():
        die(f"no {seed_state}")
    export = args.seed_export or str(seed_state).replace(".tdleaf.bin", ".nnue")
    seed_export = Path(export).resolve()
    if not seed_export.exists():
        die(f"no baked seed export {seed_export} -- pass --seed-export")

    if not (LEARN / "comp.pl").exists():
        die(f"no comp.pl in {LEARN} -- run this from engine/learn/")
    arms_dir = LEARN / f"{args.tag}_arms"
    arms_dir.mkdir(exist_ok=True)

    selected = args.only or ORDER
    needed = {ARMS[a][0] for a in selected}
    if args.corpus_only:
        needed = {ARMS[a][0] for a in ORDER}
    corpora, rows = build_corpora(args, arms_dir, needed)
    if args.corpus_only:
        log("--corpus-only: stopping after corpus assembly")
        return

    # The paired opponent is the seed itself: every arm is "the seed plus one
    # epoch", so the seed is the natural, most sensitive reference.
    paired_dir = arms_dir / "opponents"
    paired_dir.mkdir(exist_ok=True)
    paired_name = f"{args.tag}-seed"
    paired_bin = paired_dir / f"Leaf_v{paired_name}"
    if not paired_bin.exists():
        compile_binary(paired_name, seed_export, tdleaf=False, dest=paired_dir)
        check_engine_state(paired_bin, paired_dir)
    anchor_bin = LEARN / args.anchor
    if not anchor_bin.exists():
        die(f"no foreign anchor {anchor_bin}")
    opponents = [(paired_bin, "seed"), (anchor_bin, args.anchor[6:])]

    results = {}
    rpath = arms_dir / "results.json"
    if rpath.exists():
        results = json.loads(rpath.read_text())
    for arm in (args.only or ORDER):
        key = ARMS[arm][0]
        net = train_arm(arm, corpora[key], rows[key], seed_nnue, seed_state,
                        args, arms_dir)
        results[arm] = rate(arm, net, opponents, args, arms_dir)
        rpath.write_text(json.dumps(results, indent=2) + "\n")

    lines = render(results, opponents, args)
    print()
    print("\n".join(lines))
    (arms_dir / "results.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
