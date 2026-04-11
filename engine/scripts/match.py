#!/usr/bin/env python3
#
# Run a head-to-head match or gauntlet between chess engines using cutechess-cli.
# Run from the engine/run directory:
#
#   python3 match.py                              # fully interactive
#   python3 match.py Leaf_vA                      # interactive for opponent/options
#   python3 match.py Leaf_vA Leaf_vB [options]    # CLI mode (no prompts)
#   python3 match.py Leaf_vA Leaf_vB Leaf_vC ...  # gauntlet
#
# Examples:
#   python3 match.py Leaf_v2026_03_01 Leaf_vtest
#   python3 match.py Leaf_v2026_03_01 Leaf_vtest -n 200 -c 4 -tc 5+0.05
#   python3 match.py Leaf_v2026_03_01 Leaf_vtest -n 400 --openings ../testing/testsuites/wac.epd
#   python3 match.py Leaf_vtrain Leaf_vtrain_ro -n 500 -i 10 --wait 500
#     (10 sequential 500-game matches; engines restart between iterations so the
#      read-only engine picks up the latest .tdleaf.bin weights each time)
#   python3 match.py Leaf_vnew Leaf_v1 Leaf_v2 Leaf_v3 -n 100 --pgn all.pgn
#     (gauntlet: probe engine vs three opponents; all games appended to all.pgn)
#

import argparse
import math
import os
import re
import stat
import subprocess
import sys

script_dir    = os.path.dirname(os.path.abspath(__file__))
run_dir       = os.path.normpath(os.path.join(script_dir, "../run"))
tools_dir     = os.path.normpath(os.path.join(script_dir, "../tools"))
cutechess_cli = os.path.normpath(os.path.join(tools_dir, "cutechess-1.4.0/build/cutechess-cli"))


def resolve_exe(name):
    """Return absolute path: join with run_dir unless already absolute."""
    return name if os.path.isabs(name) else os.path.join(run_dir, name)


def discover_engines():
    """Scan for available engines.

    Returns (leaf_engines, external_engines) where each is a list of
    (display_name, absolute_path) tuples.
    """
    leaf_engines = []
    if os.path.isdir(run_dir):
        for f in sorted(os.listdir(run_dir)):
            if not f.startswith("Leaf_v"):
                continue
            p = os.path.join(run_dir, f)
            if os.path.isfile(p) and not f.endswith((".lock", ".py", ".pl", ".txt")):
                # Check executable
                try:
                    if os.stat(p).st_mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH):
                        leaf_engines.append((f, p))
                except OSError:
                    pass

    external_engines = []
    engines_dir = os.path.join(tools_dir, "engines")
    if os.path.isdir(engines_dir):
        for d in sorted(os.listdir(engines_dir)):
            dp = os.path.join(engines_dir, d)
            if not os.path.isdir(dp):
                continue
            # Find the main executable: prefer a file whose name contains
            # the directory name, otherwise pick the largest executable.
            candidates = []
            for f in os.listdir(dp):
                fp = os.path.join(dp, f)
                if not os.path.isfile(fp):
                    continue
                try:
                    st = os.stat(fp)
                    if st.st_mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH):
                        candidates.append((f, fp, st.st_size))
                except OSError:
                    pass
            if not candidates:
                continue
            # Prefer name match, then largest
            match = [c for c in candidates if d.lower() in c[0].lower()]
            if match:
                best = max(match, key=lambda c: c[2])
            else:
                best = max(candidates, key=lambda c: c[2])
            external_engines.append((d, best[1]))

    return leaf_engines, external_engines


def pick_engine(prompt, leaf_engines, external_engines):
    """Interactive engine selection. Returns (name, absolute_path) or exits."""
    print(f"\n{prompt}")
    idx = 1
    entries = []  # (display, name, path)
    if leaf_engines:
        print("  Leaf binaries (engine/run/):")
        for name, path in leaf_engines:
            print(f"    [{idx}] {name}")
            entries.append((name, name, path))
            idx += 1
    if external_engines:
        print("  External engines (tools/engines/):")
        for name, path in external_engines:
            print(f"    [{idx}] {name}")
            entries.append((name, name, path))
            idx += 1
    print(f"  [c] Custom path")
    print()

    while True:
        choice = input("  Select: ").strip()
        if not choice:
            continue
        if choice.lower() == "c":
            while True:
                p = input("  Path to engine: ").strip()
                if os.path.isfile(p):
                    return (os.path.basename(p), os.path.abspath(p))
                # Try relative to run_dir
                rp = os.path.join(run_dir, p)
                if os.path.isfile(rp):
                    return (os.path.basename(p), os.path.abspath(rp))
                print(f"  File not found: {p}")
        try:
            n = int(choice)
            if 1 <= n <= len(entries):
                _, name, path = entries[n - 1]
                return (name, path)
        except ValueError:
            # Maybe they typed a name directly
            p = resolve_exe(choice)
            if os.path.isfile(p):
                return (choice, p)
        print(f"  Invalid selection: {choice}")


def ask(prompt, default=None):
    suffix = f" [{default}]" if default is not None else ""
    val = input(f"{prompt}{suffix}: ").strip()
    return val if val else (str(default) if default is not None else "")


def ask_yes_no(prompt, default="n"):
    hint = "Y/n" if default == "y" else "y/N"
    val = input(f"{prompt} [{hint}]: ").strip().lower()
    if not val:
        return default == "y"
    return val.startswith("y")


def elo_from_wdl(w, d, l):
    """
    Compute Elo difference and ±95% CI from W/D/L counts (engine1 perspective).
    Returns (elo, elo_err) or (None, None) if score is degenerate.
    """
    n = w + d + l
    if n == 0:
        return None, None
    score = (w + 0.5 * d) / n
    if score <= 0.0 or score >= 1.0:
        return None, None
    elo = -400.0 * math.log10(1.0 / score - 1.0)
    # Propagate Wald std on score → Elo space
    std   = math.sqrt(score * (1.0 - score) / n)
    denom = math.log(10) * score * (1.0 - score)
    elo_err = 400.0 * std / denom if denom != 0 else float("inf")
    return elo, elo_err


def run_match(cmd):
    """
    Run cutechess-cli, stream its output to stdout line-by-line, and capture the
    final Score and Elo difference lines.

    cutechess-cli output format (engine1 perspective):
      Score of E1 vs E2: W - L - D  [pct]  N
      Elo difference: ELO +/- ERR, LOS: ...

    Returns (w, d, l, elo, elo_err, returncode).
    w/d/l are engine1 wins/draws/losses; elo/elo_err may be None.
    """
    score_re = re.compile(
        r"Score of .+? vs .+?:\s+(\d+)\s+-\s+(\d+)\s+-\s+(\d+)"
    )
    elo_re = re.compile(
        r"Elo difference:\s*([+-]?\d+(?:\.\d+)?)\s*\+/-\s*(\d+(?:\.\d+)?)"
    )

    w = d = l = 0
    elo = elo_err = None

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    for line in proc.stdout:
        print(line, end="", flush=True)
        m = score_re.search(line)
        if m:
            # cutechess: "W - L - D" from engine1 perspective
            w, l, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        m = elo_re.search(line)
        if m:
            elo     = float(m.group(1))
            elo_err = float(m.group(2))

    proc.wait()
    return w, d, l, elo, elo_err, proc.returncode


def main():
    cpu_count           = os.cpu_count() or 1
    default_concurrency = max(1, cpu_count // 2)

    parser = argparse.ArgumentParser(
        description="Run a match or gauntlet between chess engines via cutechess-cli.\n"
                    "Run with no arguments for interactive mode.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("engine1", nargs="?", default=None,
                        help="Probe engine (name in run/ or absolute path). "
                             "Omit for interactive selection.")
    parser.add_argument("opponents", nargs="*",
                        help="Opponent engine(s). More than one → gauntlet mode.")
    parser.add_argument("-n", "--games", type=int, default=100,
                        help="Games per iteration per opponent (default: 100)")
    parser.add_argument("-i", "--iterations", type=int, default=1,
                        help="Sequential iterations per opponent (default: 1); engines "
                             "restart between iterations so a read-only TDLeaf engine "
                             "picks up the latest weights each time")
    parser.add_argument("-c", "--concurrency", type=int, default=default_concurrency,
                        help=f"Simultaneous games (default: {default_concurrency})")
    parser.add_argument("-tc", "--time-control", default="10+0.1",
                        help="Time control for both engines: 'moves/time+inc' or 'time+inc' "
                             "in seconds (default: 10+0.1)")
    parser.add_argument("--tc1", default=None, metavar="TC",
                        help="Override time control for engine1 only (default: same as -tc)")
    parser.add_argument("--tc2", default=None, metavar="TC",
                        help="Override time control for engine2 only (default: same as -tc)")
    parser.add_argument("--proto", default="uci", choices=["uci", "xboard"],
                        help="Protocol for both engines (default: uci)")
    parser.add_argument("--proto1", default=None, choices=["uci", "xboard"],
                        help="Override protocol for engine1 only")
    parser.add_argument("--proto2", default=None, choices=["uci", "xboard"],
                        help="Override protocol for engine2 only")
    parser.add_argument("--pgn", default=None, metavar="FILE",
                        help="Persistent PGN: all games from all opponents/iterations are "
                             "appended to this file (cutechess -pgnout FILE append)")
    parser.add_argument("--pgn-out", default=None, metavar="FILE",
                        help="Per-iteration PGN base name "
                             "(default: match_<engine1>_vs_<engine2>.pgn); "
                             "with -i > 1 an iteration number is appended before the extension")
    parser.add_argument("--openings", default=None, metavar="FILE",
                        help="Openings file (.epd or .pgn); randomly ordered")
    parser.add_argument("--noswap", action="store_true", default=False,
                        help="Pass -noswap to cutechess-cli: don't swap colors between "
                             "paired games.  Off by default (both engines play both sides "
                             "from each opening position, which is correct for training).")
    parser.add_argument("--no-repeat", action="store_true", default=False,
                        help="Play each opening once (-rounds N, no -games 2 -repeat). "
                             "Increases opening diversity at the cost of color-balance "
                             "per opening.  Recommended for symmetric self-play training.")
    parser.add_argument("--fischer-random", action="store_true", default=False,
                        help="Use Chess960 / Fischer Random starting positions")
    parser.add_argument("--depth1", type=int, default=None, metavar="N",
                        help="Limit engine1 search to depth N (default: no limit)")
    parser.add_argument("--depth2", type=int, default=None, metavar="N",
                        help="Limit engine2 search to depth N (default: no limit)")
    parser.add_argument("--ponder", action="store_true", default=False,
                        help="Enable pondering (default: off)")
    parser.add_argument("--wait", type=int, default=0, metavar="MS",
                        help="Milliseconds to wait between games (default: 0)")
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Interactive mode: fill in missing engines and options
    # -----------------------------------------------------------------------
    interactive = args.engine1 is None or not args.opponents
    leaf_engines = ext_engines = None

    if args.engine1 is None:
        leaf_engines, ext_engines = discover_engines()
        name1, exe1 = pick_engine("Select engine 1:", leaf_engines, ext_engines)
        args.engine1 = name1
    else:
        exe1 = resolve_exe(args.engine1)

    if not args.opponents:
        if leaf_engines is None:
            leaf_engines, ext_engines = discover_engines()
        opponents_list = []
        print("\nAdd opponents (press Enter when done):")
        while True:
            n_opp = len(opponents_list)
            label = f"opponent #{n_opp + 1}" if n_opp > 0 else "opponent"
            name, path = pick_engine(f"Select {label}:", leaf_engines, ext_engines)
            opponents_list.append((name, path))
            print(f"  Added: {name}")
            if not ask_yes_no("  Add another opponent?", "n"):
                break
        args.opponents = [name for name, _ in opponents_list]
        opponent_exes = [path for _, path in opponents_list]
    else:
        opponent_exes = None  # resolved below

    # Interactive options (only when we entered interactive engine selection)
    if interactive and opponent_exes is not None:
        print()
        val = ask("Games per opponent", args.games)
        args.games = int(val)
        val = ask("Time control", args.time_control)
        args.time_control = val
        val = ask("Concurrency", args.concurrency)
        args.concurrency = int(val)
        val = ask("Iterations", args.iterations)
        args.iterations = int(val)
        args.fischer_random = ask_yes_no("Fischer Random (Chess960)?", "n")
        openings = ask("Openings file (empty for none)", "")
        if openings:
            args.openings = openings

    # -----------------------------------------------------------------------
    # Validate
    # -----------------------------------------------------------------------
    if args.concurrency < 1:
        parser.error("--concurrency must be at least 1")
    if args.concurrency > cpu_count:
        print(f"Warning: concurrency {args.concurrency} exceeds CPU count {cpu_count}, clamping.",
              file=sys.stderr)
        args.concurrency = cpu_count
    if args.games < 1:
        parser.error("--games must be at least 1")
    if args.iterations < 1:
        parser.error("--iterations must be at least 1")

    # Resolve executables
    if not os.path.isabs(exe1):
        exe1 = resolve_exe(args.engine1)
    if not os.path.isfile(exe1):
        print(f"Error: executable not found: {exe1}", file=sys.stderr)
        sys.exit(1)

    if opponent_exes is None:
        opponent_exes = []
        for opp in args.opponents:
            exe = resolve_exe(opp)
            if not os.path.isfile(exe):
                print(f"Error: executable not found: {exe}", file=sys.stderr)
                sys.exit(1)
            opponent_exes.append(exe)

    # Protocols
    proto1 = args.proto1 or args.proto
    proto2 = args.proto2 or args.proto

    # Openings
    openings_args = []
    polyglot_book = None   # .bin Polyglot book path (added per-engine, not via -openings)
    if args.openings:
        if not os.path.isfile(args.openings):
            print(f"Error: openings file not found: {args.openings}", file=sys.stderr)
            sys.exit(1)
        if args.openings.lower().endswith(".bin"):
            polyglot_book = args.openings  # injected into each -engine spec as book=FILE
        else:
            fmt = "epd" if args.openings.lower().endswith(".epd") else "pgn"
            openings_args = ["-openings", f"file={args.openings}", f"format={fmt}", "order=random"]

    name1    = os.path.basename(args.engine1)
    gauntlet = len(args.opponents) > 1
    multi    = args.iterations > 1

    # Header
    print(f"\nProbe engine: {name1}")
    if gauntlet:
        print(f"Gauntlet vs:  {', '.join(os.path.basename(o) for o in args.opponents)}")
    depth_str = ""
    if args.depth1 is not None or args.depth2 is not None:
        d1 = str(args.depth1) if args.depth1 is not None else "unlimited"
        d2 = str(args.depth2) if args.depth2 is not None else "unlimited"
        depth_str = f"   Depth: {name1}={d1} / opponent={d2}"
    tc_display = (args.time_control if not args.tc1 and not args.tc2
                  else f"{args.tc1 or args.time_control} / {args.tc2 or args.time_control}")
    print(f"Games: {args.games}   Iterations: {args.iterations}   "
          f"Concurrency: {args.concurrency}   TC: {tc_display}   "
          f"Protocol: {proto1}/{proto2}   "
          f"Fischer Random: {'on' if args.fischer_random else 'off'}{depth_str}")
    if args.pgn:
        print(f"Persistent PGN: {args.pgn}")
    if args.openings:
        print(f"Openings:     {args.openings}")
    print()

    gauntlet_results = []   # (name2, total_w, total_d, total_l, display_elo, display_elo_err)
    grand_w = grand_d = grand_l = 0

    for opp_exe, opp_arg in zip(opponent_exes, args.opponents):
        name2    = os.path.basename(opp_arg)
        pgn_base = args.pgn_out or f"match_{name1}_vs_{name2}.pgn"

        # rounds/games setup
        if args.no_repeat:
            rounds_arg = str(args.games)
            games_arg  = []
        elif args.games % 2 == 0:
            rounds_arg = str(args.games // 2)
            games_arg  = ["-games", "2", "-repeat"]
        else:
            rounds_arg = str(args.games)
            games_arg  = []

        tc1 = args.tc1 or args.time_control
        tc2 = args.tc2 or args.time_control

        # Per-engine dir: use the directory containing each binary so engines
        # can find their data files (books, NNUE nets, etc.)
        dir1 = os.path.dirname(exe1)
        dir2 = os.path.dirname(opp_exe)

        eng1_spec = [f"cmd={exe1}",    f"name={name1}", f"proto={proto1}", f"dir={dir1}"]
        eng2_spec = [f"cmd={opp_exe}", f"name={name2}", f"proto={proto2}", f"dir={dir2}"]
        if polyglot_book:
            eng1_spec.append(f"book={polyglot_book}")
            eng2_spec.append(f"book={polyglot_book}")
        if args.depth1 is not None:
            eng1_spec.append(f"depth={args.depth1}")
        if args.depth2 is not None:
            eng2_spec.append(f"depth={args.depth2}")

        # Per-engine TC: use -each when identical, per-engine spec when different.
        if tc1 == tc2:
            each_tc = [f"tc={tc1}"]
        else:
            eng1_spec.append(f"tc={tc1}")
            eng2_spec.append(f"tc={tc2}")
            each_tc = []

        base_cmd = [
            cutechess_cli,
            "-engine", *eng1_spec,
            "-engine", *eng2_spec,
            "-each",   *each_tc, *(["ponder"] if args.ponder else []),
            *([ "-variant", "fischerandom"] if args.fischer_random else []),
            "-concurrency", str(args.concurrency),
            "-rounds", rounds_arg,
            "-recover",
            "-draw",   "movenumber=40", "movecount=8", "score=10",
            "-resign", "movecount=6",   "score=600",
            "-ratinginterval", "10",
        ] + games_arg + (["-wait", str(args.wait)] if args.wait > 0 else []) \
          + (["-noswap"] if args.noswap else [])

        if gauntlet:
            print("=" * 60)
            print(f"  vs {name2}")
            print("=" * 60)

        opp_w = opp_d = opp_l = 0
        last_elo = last_elo_err = None

        for it in range(1, args.iterations + 1):
            if args.pgn:
                pgnout_args = ["-pgnout", args.pgn]
                if multi:
                    print(f"--- Iteration {it} / {args.iterations} ---")
                elif not gauntlet:
                    print(f"PGN output:  {args.pgn}  (append)")
            else:
                if multi:
                    root, ext = os.path.splitext(pgn_base)
                    pgn_out   = f"{root}_iter{it:02d}{ext}"
                    print(f"--- Iteration {it} / {args.iterations}   PGN: {pgn_out} ---")
                else:
                    pgn_out = pgn_base
                    if not gauntlet:
                        print(f"PGN output:  {pgn_out}")
                pgnout_args = ["-pgnout", pgn_out]

            cmd = base_cmd + pgnout_args + openings_args

            w, d, l, elo, elo_err, rc = run_match(cmd)

            if rc != 0:
                print(f"\nError: cutechess-cli exited with code {rc} on iteration {it}.",
                      file=sys.stderr)
                sys.exit(rc)

            opp_w += w
            opp_d += d
            opp_l += l
            if elo is not None:
                last_elo, last_elo_err = elo, elo_err

            if multi and it < args.iterations:
                print()

        # Elo for this opponent: aggregate W/D/L when multi-iteration, else cutechess's value
        if multi or last_elo is None:
            display_elo, display_elo_err = elo_from_wdl(opp_w, opp_d, opp_l)
        else:
            display_elo, display_elo_err = last_elo, last_elo_err

        gauntlet_results.append((name2, opp_w, opp_d, opp_l, display_elo, display_elo_err))
        grand_w += opp_w
        grand_d += opp_d
        grand_l += opp_l

        if multi:
            n   = opp_w + opp_d + opp_l
            pct = (opp_w + 0.5 * opp_d) / n * 100 if n else 0
            elo_s = (f"{display_elo:+.0f} ±{display_elo_err:.0f}"
                     if display_elo is not None else "n/a")
            print(f"\nIterations done: {args.iterations} × {args.games} = {n} games  "
                  f"W={opp_w} D={opp_d} L={opp_l} [{pct:.1f}%]  Elo {elo_s}")
        print()

    # Gauntlet summary table
    if gauntlet:
        col_w = 32
        sep   = "=" * 76
        dash  = "-" * 74
        print(sep)
        print(f"  GAUNTLET SUMMARY — {name1}")
        print(sep)
        print(f"  {'Opponent':<{col_w}} {'Games':>5} {'W':>4} {'D':>4} {'L':>4}  "
              f"{'Score%':>7}  {'Elo diff':>10}")
        print("  " + dash)
        for name2, w, d, l, elo, elo_err in gauntlet_results:
            n   = w + d + l
            pct = (w + 0.5 * d) / n * 100 if n else 0
            elo_s = (f"{elo:+.0f} ±{elo_err:.0f}" if elo is not None else "n/a")
            print(f"  {name2:<{col_w}} {n:>5} {w:>4} {d:>4} {l:>4}  {pct:>6.1f}%  {elo_s:>10}")
        print("  " + dash)
        n_tot   = grand_w + grand_d + grand_l
        pct_tot = (grand_w + 0.5 * grand_d) / n_tot * 100 if n_tot else 0
        print(f"  {'TOTAL':<{col_w}} {n_tot:>5} {grand_w:>4} {grand_d:>4} {grand_l:>4}  "
              f"{pct_tot:>6.1f}%  {'':>10}")
        print(sep)

    elif multi:
        print(f"All {args.iterations} iterations complete "
              f"({args.iterations * args.games} games total).")


if __name__ == "__main__":
    main()
