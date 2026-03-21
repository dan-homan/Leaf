#!/usr/bin/env python3
#
# Leaf TDLeaf(λ) training run manager.
# Run from any directory; paths are resolved relative to this script.
#
#   python3 training_run.py
#
# What this script does:
#   1. Asks for a starting .nnue file or offers to initialise a fresh random one.
#   2. Builds two training binaries for the chosen network (symmetric self-play;
#      both write to the shared .tdleaf.bin via the flock+delta-merge mechanism
#      in nnue.cpp), then moves both binaries to learn/ so all training files
#      live together.
#   3. Checks for an existing .tdleaf.bin (continue or start fresh).
#   4. Asks for match parameters, then runs match.py for N games.
#      PGN files land in  learn/pgn/<net_base>/  so all PGNs for a given net
#      accumulate in one place across multiple training runs.
#   5. After matches complete, exports the trained weights to a .nnue file whose
#      name encodes the total number of training games to that point:
#        <net_base>-<total_games>g.nnue
#      Game counts are tracked in a sidecar file (learn/<net_base>.games) that
#      persists across multiple runs on the same network.
#
# Train-validate loop (optional):
#   Each cycle trains for X games, then runs a validation match between the
#   candidate (new weights) and the current best.  The candidate is accepted
#   if its LOS (likelihood of superiority) meets the threshold; otherwise the
#   .tdleaf.bin is reverted to the pre-cycle checkpoint.
#
# All working files (.nnue, .tdleaf.bin, .games, binaries, output .nnue) are
# kept in learn/ for easy access.  run/ is used for match.py; src/comp.pl is
# invoked directly for builds.
#

import datetime
import fcntl
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import time

learn_dir = os.path.dirname(os.path.abspath(__file__))
run_dir   = os.path.normpath(os.path.join(learn_dir, "../run"))
src_dir   = os.path.normpath(os.path.join(learn_dir, "../src"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ask(prompt, default=None):
    """Prompt the user; return input stripped, or default if empty."""
    suffix = f" [{default}]" if default is not None else ""
    val = input(f"{prompt}{suffix}: ").strip()
    return val if val else (str(default) if default is not None else "")


def ask_yes_no(prompt, default="y"):
    hint = "Y/n" if default == "y" else "y/N"
    val = input(f"{prompt} [{hint}]: ").strip().lower()
    if not val:
        return default == "y"
    return val.startswith("y")


def read_game_count(sidecar_path):
    try:
        with open(sidecar_path) as f:
            return int(f.read().strip())
    except Exception:
        return 0


def write_game_count(sidecar_path, count):
    with open(sidecar_path, "w") as f:
        f.write(f"{count}\n")


def _prompt_init_cnt(is_fresh_random):
    """Ask the user for an initial update count to prime Adam LR decay.

    Returns an int (0 = no priming, i.e. full LR0 from game 1).

    The Adam LR decay formula is:  lr(cnt) = LR0 / (1 + cnt / C)  where C=5000.
    Suggested values:
      0    — untrained / fresh random network  → full LR0 from game 1
      5000  — lightly pre-trained               → start at 50% of LR0  (lr × 0.50)
      10000 — moderately pre-trained            → start at 33% of LR0  (lr × 0.33)
      20000 — well-trained (e.g. SF15.1 fine-tuning) → start at 20% of LR0  (lr × 0.20)
    """
    default = 0 if is_fresh_random else 10000
    print()
    print("Initial update count (cnt) for Adam LR decay:")
    print("  lr(cnt) = LR0 × (0.01 + 0.99 / (1 + cnt / 5000))")
    print("    0     — untrained / fresh random network  (start at 100% of LR0, floor  1%)")
    print("    5000  — lightly pre-trained               (start at  51% of LR0, floor  1%)")
    print("    10000 — moderately pre-trained            (start at  34% of LR0, floor  1%)")
    print("    20000 — well-trained network              (start at  21% of LR0, floor  1%)")
    val = int(ask("  Initial cnt", default))
    return val


def wait_until_stable(path, stable_secs=3, timeout=120):
    """Wait until all training engines have finished with the .tdleaf.bin file.

    Two-phase check:
      1. Poll mtime/size until the file has not changed for stable_secs seconds
         (confirms no active writes to the data file).
      2. Acquire and immediately release a blocking exclusive flock on the
         companion <path>.lock file (the same lock the engines use).  This
         returns only when every engine process has released its lock, which
         guarantees no engine is still in a write or read-init critical section.

    Returns True if both phases completed, False on timeout or missing file.
    """
    if not os.path.isfile(path):
        return False

    label = os.path.basename(path)
    print(f"  Waiting for {label} to stabilise ...", end="", flush=True)

    # Phase 1 — file content stability
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            st0 = os.stat(path)
            time.sleep(stable_secs)
            st1 = os.stat(path)
            if st0.st_mtime_ns == st1.st_mtime_ns and st0.st_size == st1.st_size:
                break
        except OSError:
            print(" timed out — proceeding anyway.", flush=True)
            return False
    else:
        print(" timed out — proceeding anyway.", flush=True)
        return False

    # Phase 2 — lock clearance
    # The engines use a companion "<path>.lock" file with POSIX flock.
    # Acquiring LOCK_EX blocks until every engine releases its lock,
    # confirming no engine is still in a write or startup-read critical section.
    lock_path = path + ".lock"
    if os.path.isfile(lock_path):
        try:
            with open(lock_path, "r+b") as lf:
                fcntl.flock(lf, fcntl.LOCK_EX)   # blocks until all locks released
                fcntl.flock(lf, fcntl.LOCK_UN)
        except OSError:
            pass   # lock file disappeared or not lockable — proceed

    print(" done.", flush=True)
    return True


def build_binary(version, flags):
    """Invoke src/comp.pl (cwd=run_dir), then move the resulting binary to learn_dir."""
    comp_pl = os.path.join(src_dir, "comp.pl")
    cmd = ["perl", comp_pl, version] + flags + ["OVERWRITE"]
    print(f"  $ perl src/comp.pl {' '.join([version] + flags)} OVERWRITE")
    result = subprocess.run(cmd, cwd=run_dir)
    if result.returncode != 0:
        return False
    built = os.path.join(run_dir, f"Leaf_v{version}")
    dest  = os.path.join(learn_dir, f"Leaf_v{version}")
    if os.path.isfile(built):
        shutil.move(built, dest)
    return os.path.isfile(dest)


def compute_los(w, d, l):
    """Likelihood of superiority for engine1 (candidate) from W/D/L counts.

    Uses a normal approximation on the per-game score distribution.
    Returns a value in [0, 1]; 0.5 means indistinguishable from chance.
    """
    n = w + d + l
    if n == 0:
        return 0.5
    score = (w + 0.5 * d) / n
    if score <= 0.0:
        return 0.0
    if score >= 1.0:
        return 1.0
    var = (w * (1.0 - score) ** 2 +
           d * (0.5 - score) ** 2 +
           l * (0.0 - score) ** 2) / n
    if var == 0.0:
        return 1.0 if score > 0.5 else 0.0
    z = (score - 0.5) / math.sqrt(var / n)
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


def run_match_streaming(cmd, los_stop_hi=None, los_stop_lo=None):
    """Run a subprocess, stream its stdout to the console, and capture the
    final Score line (cutechess-cli format).

    If los_stop_hi or los_stop_lo are given (fractions, e.g. 0.90 / 0.10),
    the match is terminated early once the running LOS crosses either bound
    after at least 20 games.  Early termination is treated as success.

    Returns (w, d, l, returncode) from engine1's perspective.
    """
    score_re = re.compile(
        r"Score of .+? vs .+?:\s+(\d+)\s+-\s+(\d+)\s+-\s+(\d+)"
    )
    w = d = l = 0
    early_stop = False
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        start_new_session=True   # isolate in its own process group for clean teardown
    )
    for line in proc.stdout:
        print(line, end="", flush=True)
        m = score_re.search(line)
        if m:
            # cutechess: "W - L - D" from engine1 perspective
            w, l, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
            n = w + d + l
            if n >= 20 and n % 10 == 0 and (los_stop_hi is not None or los_stop_lo is not None):
                los = compute_los(w, d, l)
                if los_stop_hi is not None and los >= los_stop_hi:
                    print(f"\n  [Early stop: LOS={los*100:.1f}% ≥ {los_stop_hi*100:.0f}%"
                          f" after {n} games]", flush=True)
                    early_stop = True
                    break
                if los_stop_lo is not None and los <= los_stop_lo:
                    print(f"\n  [Early stop: LOS={los*100:.1f}% ≤ {los_stop_lo*100:.0f}%"
                          f" after {n} games]", flush=True)
                    early_stop = True
                    break
    if early_stop:
        # Kill the entire process group (match.py + cutechess-cli + all engine processes).
        # Without this, orphaned engine processes keep consuming CPU and starve
        # cycle-2+ training engine startups.
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, OSError):
            pass
        for line in proc.stdout:   # drain to unblock the child
            print(line, end="", flush=True)
    proc.wait()
    return w, d, l, 0 if early_stop else proc.returncode


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cpu_count           = os.cpu_count() or 1
    default_concurrency = max(1, cpu_count // 2)
    date_str            = datetime.datetime.now().strftime("%y%m%d")   # YYMMDD

    print()
    print("=" * 62)
    print("  Leaf TDLeaf(λ) Training Run")
    print("=" * 62)

    # -----------------------------------------------------------------------
    # Step 1 — Net selection
    # -----------------------------------------------------------------------
    print()
    print("Starting network:")
    print("  [1] Use existing .nnue file")
    print("  [2] Initialise a fresh random network")
    choice = ask("Choice", "1")

    if choice.strip() == "2":
        default_fresh = f"nn-fresh-{date_str}.nnue"
        fresh_name    = ask("Output filename for fresh network", default_fresh)
        if not fresh_name.endswith(".nnue"):
            fresh_name += ".nnue"
        net_filename   = fresh_name
        net_base       = os.path.splitext(net_filename)[0]
        net_file       = os.path.join(learn_dir, net_filename)
        do_random_init = True
    else:
        # Show .nnue files available in learn/ to help catch typos.
        nnue_files = sorted(f for f in os.listdir(learn_dir) if f.endswith(".nnue"))
        if nnue_files:
            print("  Available in learn/:")
            for f in nnue_files:
                print(f"    {f}")
        else:
            print("  (no .nnue files found in learn/)")
        while True:
            net_path = ask("Path to .nnue file").strip()
            if os.path.isfile(net_path):
                break
            # Also accept bare filenames relative to learn/.
            if os.path.isfile(os.path.join(learn_dir, net_path)):
                net_path = os.path.join(learn_dir, net_path)
                break
            print(f"  File not found: {net_path}")
        net_filename   = os.path.basename(net_path)
        net_base       = os.path.splitext(net_filename)[0]
        net_file       = os.path.join(learn_dir, net_filename)
        do_random_init = False
        if os.path.abspath(net_path) != os.path.abspath(net_file):
            print(f"  Copying {net_path} → learn/{net_filename}")
            shutil.copy2(net_path, net_file)

    tdleaf_bin   = os.path.join(learn_dir, net_base + ".tdleaf.bin")
    sidecar_path = os.path.join(learn_dir, net_base + ".games")

    # -----------------------------------------------------------------------
    # Step 1b — Training partner selection
    # -----------------------------------------------------------------------
    print()
    print("Training partner:")
    print("  [1] Self-play           — both instances learn (symmetric)")
    print("  [2] Read-only opponent  — learner vs. fixed-weight copy of same net")
    print("  [3] External opponent   — learner vs. user-supplied executable")
    partner_choice = ask("Choice", "1").strip()

    opponent_exe = None   # set for choices 2 and 3
    if partner_choice == "3":
        while True:
            opp_path = ask("  Path to opponent executable").strip()
            if os.path.isfile(opp_path):
                opponent_exe = os.path.abspath(opp_path)
                break
            print(f"  File not found: {opp_path}")

    # -----------------------------------------------------------------------
    # Step 1c — Train-validate loop (optional)
    # -----------------------------------------------------------------------
    print()
    use_loop       = ask_yes_no("Enable train-validate loop?", default="n")
    n_cycles       = 0
    val_games      = 200
    los_thresh_pct = 70.0
    los_stop_hi    = 0.90   # early-stop if LOS rises above this
    los_stop_lo    = 0.10   # early-stop if LOS falls below this
    val_tc         = None   # set in Step 4 after tc1 is known
    # Eval binary names (set in Step 2 if use_loop)
    best_nnue_name = None
    cand_nnue_name = None
    eval_best_exe  = None
    eval_cand_exe  = None

    if use_loop:
        n_cycles       = int(ask("  Cycles (0 = run forever until Ctrl-C)", "0"))
        val_games      = int(ask("  Validation games per cycle           ", "200"))
        los_thresh_pct = float(ask("  LOS acceptance threshold (%)        ", "70"))
        los_stop_hi    = float(ask("  Early-stop if LOS ≥ (%)            ", "90")) / 100.0
        los_stop_lo    = float(ask("  Early-stop if LOS ≤ (%)            ", "10")) / 100.0
        # val_tc is asked in Step 4 once tc1 is known
        best_nnue_name = f"{net_base}-best.nnue"
        cand_nnue_name = f"{net_base}-cand.nnue"

    # -----------------------------------------------------------------------
    # Step 2 — Build executables  (comp.pl runs in run/, binaries moved to learn/)
    # -----------------------------------------------------------------------
    print()
    print("Building executables:")
    train_ver   = f"train_{net_base}_a"
    train_exe   = os.path.join(learn_dir, f"Leaf_v{train_ver}")
    nnue_flag   = f"NNUE_NET={net_filename}"
    tdleaf_flags = ["NNUE=1", "TDLEAF=1", nnue_flag]

    if partner_choice == "1":
        # Symmetric self-play: two learning instances.
        train_ver2 = f"train_{net_base}_b"
        train_exe2 = os.path.join(learn_dir, f"Leaf_v{train_ver2}")
        need_exe2  = True
        ro_build   = False
    elif partner_choice == "2":
        # Read-only opponent: build a second binary with TDLEAF_READONLY=1.
        train_ver2 = f"train_{net_base}_ro"
        train_exe2 = os.path.join(learn_dir, f"Leaf_v{train_ver2}")
        need_exe2  = True
        ro_build   = True
    else:
        # External opponent: no second binary to build.
        train_ver2 = None
        train_exe2 = opponent_exe
        need_exe2  = False
        ro_build   = False

    exes_exist = os.path.isfile(train_exe) and (not need_exe2 or os.path.isfile(train_exe2))
    rebuild = True
    if exes_exist:
        rebuild = ask_yes_no(
            "  Executable(s) already exist.  Rebuild?", default="n")

    if rebuild:
        print(f"  Building learning binary (Leaf_v{train_ver}) ...")
        if not build_binary(train_ver, tdleaf_flags):
            print("  Build failed.", file=sys.stderr)
            sys.exit(1)
        if need_exe2:
            label = "read-only" if ro_build else "B"
            print(f"  Building {label} binary (Leaf_v{train_ver2}) ...")
            ro_flags = tdleaf_flags + ["TDLEAF_READONLY=1"] if ro_build else tdleaf_flags
            if not build_binary(train_ver2, ro_flags):
                print("  Build failed.", file=sys.stderr)
                sys.exit(1)
        print("  Build complete.")
    else:
        print("  Using existing binaries.")

    # Eval-only binaries for train-validate loop
    if use_loop:
        eval_best_ver = f"train_{net_base}_eval_best"
        eval_cand_ver = f"train_{net_base}_eval_cand"
        eval_best_exe = os.path.join(learn_dir, f"Leaf_v{eval_best_ver}")
        eval_cand_exe = os.path.join(learn_dir, f"Leaf_v{eval_cand_ver}")

        eval_exist = os.path.isfile(eval_best_exe) and os.path.isfile(eval_cand_exe)
        rebuild_eval = True
        if eval_exist:
            rebuild_eval = ask_yes_no(
                "  Eval binaries already exist.  Rebuild?", default="n")
        if rebuild_eval:
            print(f"  Building eval-best binary (loads {best_nnue_name}) ...")
            if not build_binary(eval_best_ver, ["NNUE=1", f"NNUE_NET={best_nnue_name}"]):
                print("  Eval-best build failed.", file=sys.stderr)
                sys.exit(1)
            print(f"  Building eval-cand binary (loads {cand_nnue_name}) ...")
            if not build_binary(eval_cand_ver, ["NNUE=1", f"NNUE_NET={cand_nnue_name}"]):
                print("  Eval-cand build failed.", file=sys.stderr)
                sys.exit(1)
        else:
            print("  Using existing eval binaries.")

    # -----------------------------------------------------------------------
    # Step 2b — Random init (after training binary exists in learn/)
    # -----------------------------------------------------------------------
    if do_random_init:
        print()
        if os.path.isfile(net_file):
            print(f"  Fresh net already exists: {net_filename}")
            overwrite = ask_yes_no("  Overwrite?", default="n")
        else:
            overwrite = True

        if overwrite:
            print(f"  Initialising fresh network → {net_filename}")
            result = subprocess.run(
                [train_exe, "--init-nnue", "--write-nnue", net_file],
                cwd=learn_dir
            )
            if result.returncode != 0:
                print("  --init-nnue failed.", file=sys.stderr)
                sys.exit(1)
        else:
            print("  Using existing fresh net.")

    # -----------------------------------------------------------------------
    # Step 3 — .tdleaf.bin continuity check
    # -----------------------------------------------------------------------
    prior_games = read_game_count(sidecar_path)
    init_cnt    = None   # None = no --set-cnt step needed
    print()
    if os.path.isfile(tdleaf_bin):
        mtime = datetime.datetime.fromtimestamp(os.path.getmtime(tdleaf_bin))
        print(f"Found existing {net_base}.tdleaf.bin")
        print(f"  Last modified:   {mtime.strftime('%Y-%m-%d %H:%M')}")
        print(f"  Games on record: {prior_games:,}")
        print("  [1] Continue training from this file")
        print("  [2] Start fresh (rename existing to .tdleaf.bin.bak)")
        cont = ask("Choice", "1")
        if cont.strip() == "2":
            bak = tdleaf_bin + ".bak"
            print(f"  Renaming to {os.path.basename(bak)}")
            os.rename(tdleaf_bin, bak)
            prior_games = 0
            write_game_count(sidecar_path, 0)
            init_cnt = _prompt_init_cnt(do_random_init)
    else:
        print(f"No existing .tdleaf.bin for {net_filename} — starting fresh.")
        init_cnt = _prompt_init_cnt(do_random_init)

    # -----------------------------------------------------------------------
    # Step 4 — Match parameters
    # -----------------------------------------------------------------------
    print()
    print("Match parameters:")
    if use_loop:
        n_games = int(ask("  Games per cycle     [-n]        ", 5000))
    else:
        n_games = int(ask("  Games                [-n]        ", 5000))
    tc1         = ask(    "  Learner time control   [--tc1]  ", "0:03+0.05")
    tc2_raw     = ask(    "  Opponent time control  [--tc2]  ", tc1)
    tc2         = tc2_raw if tc2_raw.strip() else tc1
    concurrency = int(ask( "  Concurrency         [-c]        ", default_concurrency))
    wait_ms     = int(ask("  Wait between games  [--wait ms] ", 500))
    fischer     = ask_yes_no("  Fischer Random? [--fischer-random]", default="y")
    depth1_str  = ask("  Learner depth limit (0=none) [--depth1]", "0")
    depth1      = int(depth1_str) if depth1_str.strip() else 0
    depth2_str  = ask(f"  Opponent depth limit (0=none) [--depth2]", str(depth1))
    depth2      = int(depth2_str) if depth2_str.strip() else 0
    if use_loop:
        val_tc = ask("  Validation time control  [--val-tc]  ", tc1)

    games_per_cycle = n_games

    if use_loop:
        max_new = games_per_cycle * n_cycles if n_cycles > 0 else None
        total_after_str = (f"{prior_games + max_new:,} (if all accepted)"
                           if max_new is not None else "open-ended")
    else:
        total_after_str = f"{prior_games + games_per_cycle:,}"

    # -----------------------------------------------------------------------
    # PGN directory  —  learn/pgn/<net_base>/
    # -----------------------------------------------------------------------
    pgn_dir       = os.path.join(learn_dir, "pgn", net_base)
    os.makedirs(pgn_dir, exist_ok=True)
    pgn_base_path = os.path.join(pgn_dir, f"match_{net_base}.pgn")

    # -----------------------------------------------------------------------
    # Confirm
    # -----------------------------------------------------------------------
    print()
    print("=" * 62)
    print(f"  Net in:           {net_filename}")
    print(f"  Learner:          Leaf_v{train_ver}")
    if partner_choice == "1":
        print(f"  Opponent:         Leaf_v{train_ver2}  (symmetric self-play)")
    elif partner_choice == "2":
        print(f"  Opponent:         Leaf_v{train_ver2}  (read-only, same net)")
    else:
        print(f"  Opponent:         {os.path.basename(train_exe2)}  (external)")
    if use_loop:
        cycle_label = str(n_cycles) if n_cycles > 0 else "∞"
        print(f"  Loop:             {cycle_label} cycles × {games_per_cycle:,} games/cycle")
        print(f"  Validation:       {val_games} games @ {val_tc}  "
              f" accept≥{los_thresh_pct:.0f}%  "
              f"early-stop ≥{los_stop_hi*100:.0f}% / ≤{los_stop_lo*100:.0f}%")
    else:
        print(f"  Games this run:   {n_games:,}")
    print(f"  Prior games:      {prior_games:,}")
    print(f"  Total after run:  {total_after_str}")
    if tc1 == tc2:
        print(f"  TC: {tc1}   Concurrency: {concurrency}   Wait: {wait_ms} ms")
    else:
        print(f"  TC learner: {tc1}   TC opponent: {tc2}   Concurrency: {concurrency}   Wait: {wait_ms} ms")
    if depth1 or depth2:
        d1_str = str(depth1) if depth1 else "none"
        d2_str = str(depth2) if depth2 else "none"
        if depth1 == depth2:
            print(f"  Depth limit:      {d1_str} (both engines)")
        else:
            print(f"  Depth learner:    {d1_str}   Depth opponent: {d2_str}")
    if fischer:
        print( "  Fischer Random:   yes")
    if init_cnt is not None and init_cnt > 0:
        lr_frac = 0.01 + 0.99 / (1.0 + init_cnt / 5000.0)
        print(f"  Initial cnt:      {init_cnt}  (Adam LR0 × {lr_frac:.2f})")
    print(f"  PGN directory:    {pgn_dir}/")
    print("=" * 62)

    if not ask_yes_no("Proceed?", default="y"):
        print("Aborted.")
        sys.exit(0)

    # -----------------------------------------------------------------------
    # Step 5a — Prime .tdleaf.bin with initial cnt (if requested)
    # -----------------------------------------------------------------------
    if init_cnt is not None and init_cnt > 0:
        print()
        lr_frac = 0.01 + 0.99 / (1.0 + init_cnt / 5000.0)
        print(f"Priming .tdleaf.bin with cnt={init_cnt}  "
              f"(Adam LR0 × {lr_frac:.2f} from game 1) ...")
        result = subprocess.run(
            [train_exe, "--set-cnt", str(init_cnt)],
            cwd=learn_dir
        )
        if result.returncode != 0:
            print("--set-cnt failed.", file=sys.stderr)
            sys.exit(result.returncode)

    # -----------------------------------------------------------------------
    # Step 5b — Training loop (single pass or multi-cycle train-validate)
    # -----------------------------------------------------------------------
    match_py      = os.path.join(run_dir, "match.py")
    current_games = prior_games
    cycle_log     = []   # list of (cycle, accepted, vw, vd, vl, los)

    def build_match_cmd(pgn_path):
        cmd = [
            sys.executable, match_py,
            train_exe,
            train_exe2,
            "-n", str(n_games),
            "-tc", tc1,
            "-c", str(concurrency),
            "--wait", str(wait_ms),
            "--pgn-out", pgn_path,
        ]
        if tc2 != tc1:
            cmd += ["--tc2", tc2]
        if fischer:
            cmd.append("--fischer-random")
        if depth1:
            cmd += ["--depth1", str(depth1)]
        if depth2:
            cmd += ["--depth2", str(depth2)]
        return cmd

    def export_nnue(exe, dest_path, label):
        """Export trained weights via --write-nnue; exit on failure."""
        print(f"  Exporting {label} → {os.path.basename(dest_path)} ...")
        r = subprocess.run([exe, "--write-nnue", dest_path], cwd=learn_dir)
        if r.returncode != 0:
            print(f"  --write-nnue ({label}) failed.", file=sys.stderr)
            sys.exit(r.returncode)

    # -----------------------------------------------------------------------
    # Step 5b-pre — Set up initial best.nnue baseline (loop mode only).
    #
    # best.nnue must reflect the weights at the *start* of training so the
    # validation match has a stable baseline.  We set it up ONCE here (before
    # any cycle) rather than at the top of each cycle so that no train_exe
    # subprocess runs immediately before the training match — that was causing
    # intermittent engine-startup failures in cycle 2+.
    #
    # After each *accepted* cycle we re-export best.nnue from the accepted
    # .tdleaf.bin so it advances with the training.
    # -----------------------------------------------------------------------
    if use_loop:
        had_prior_tdleaf = os.path.isfile(tdleaf_bin)
        best_nnue_path = os.path.join(learn_dir, best_nnue_name)
        cand_nnue_path = os.path.join(learn_dir, cand_nnue_name)
        print()
        if had_prior_tdleaf:
            print("Setting up initial best baseline from existing .tdleaf.bin ...")
            wait_until_stable(tdleaf_bin)
            export_nnue(train_exe, best_nnue_path, "initial best")
        else:
            shutil.copy2(net_file, best_nnue_path)
            print(f"  Best baseline → {best_nnue_name}  (base net, no prior .tdleaf.bin)")
            print("  Note: cycle 1 will be auto-accepted (no trained baseline to compare against).")

    cycle_num = 0
    try:
        while True:
            cycle_num += 1

            if use_loop:
                if n_cycles > 0 and cycle_num > n_cycles:
                    break
                print()
                print("─" * 62)
                cycle_label = f"/{n_cycles}" if n_cycles > 0 else "  (Ctrl-C to stop)"
                print(f"  Cycle {cycle_num}{cycle_label}"
                      f"   [{current_games:,} games banked]")
                print("─" * 62)

                # Ensure all file I/O from the previous cycle has settled.
                wait_until_stable(tdleaf_bin)

                # Save a checkpoint so we can revert on rejection.
                checkpoint_bin = tdleaf_bin + ".checkpoint"
                has_checkpoint = os.path.isfile(tdleaf_bin)
                if has_checkpoint:
                    shutil.copy2(tdleaf_bin, checkpoint_bin)

                pgn_path = os.path.join(
                    pgn_dir, f"match_{net_base}_cycle{cycle_num:02d}.pgn")
            else:
                pgn_path = pgn_base_path

            # --- Training match ---
            print()
            result = subprocess.run(build_match_cmd(pgn_path), cwd=run_dir)
            if result.returncode != 0:
                print(f"\nMatch run failed (exit code {result.returncode}).",
                      file=sys.stderr)
                if use_loop:
                    print("Weights from completed cycles preserved. Exiting.")
                else:
                    print("Game count sidecar NOT updated.")
                sys.exit(result.returncode)

            if not use_loop:
                # Single-run mode: update count and break
                write_game_count(sidecar_path, current_games + games_per_cycle)
                current_games += games_per_cycle
                break

            # Wait for all engine processes to finish writing .tdleaf.bin.
            # A fixed delay is unreliable — poll until the file stops changing.
            wait_until_stable(tdleaf_bin)

            # --- Export candidate ---
            print()
            export_nnue(train_exe, cand_nnue_path, "candidate")

            # Brief pause for filesystem to flush the exported .nnue files.
            time.sleep(2)

            # --- Validation match ---
            # Cycle 1 with no prior .tdleaf.bin: the "best" baseline is the
            # raw base network (random weights), which is not a meaningful
            # comparison target — TDLeaf training from a random init can
            # legitimately produce weights that evaluate worse than random for
            # the first several hundred games.  Auto-accept so this cycle
            # establishes a trained baseline for all subsequent comparisons.
            if not had_prior_tdleaf and cycle_num == 1:
                vw = vd = vl = 0
                los = 1.0
                accepted = True
                verdict  = "AUTO-ACCEPTED ✓"
                print(f"\n  Validation: skipped (no prior trained baseline)  → {verdict}")
            else:
                val_pgn = os.path.join(
                    pgn_dir, f"val_{net_base}_cycle{cycle_num:02d}.pgn")
                print(f"  Validation: {val_games} games @ {val_tc}"
                      f"   (candidate vs best)")
                val_cmd = [
                    sys.executable, match_py,
                    eval_cand_exe, eval_best_exe,
                    "-n", str(val_games),
                    "-tc", val_tc,
                    "-c", str(concurrency),
                    "--pgn-out", val_pgn,
                ]
                if fischer:
                    val_cmd.append("--fischer-random")

                vw, vd, vl, vrc = run_match_streaming(
                    val_cmd, los_stop_hi=los_stop_hi, los_stop_lo=los_stop_lo)
                if vrc != 0:
                    print(f"  Validation match failed (exit {vrc}).",
                          file=sys.stderr)
                    sys.exit(vrc)

                vn  = vw + vd + vl
                pct = (vw + 0.5 * vd) / vn * 100.0 if vn else 50.0
                los = compute_los(vw, vd, vl)
                accepted = los >= (los_thresh_pct / 100.0)
                verdict  = "ACCEPTED ✓" if accepted else "REJECTED ✗"

                print(f"\n  Validation: W={vw} D={vd} L={vl}  "
                      f"score={pct:.1f}%  LOS={los * 100:.1f}%  → {verdict}")

            cycle_log.append((cycle_num, accepted, vw, vd, vl, los))

            if accepted:
                current_games += games_per_cycle
                write_game_count(sidecar_path, current_games)
                # Advance the best baseline so the next cycle compares against
                # the newly accepted weights (not the original session start).
                export_nnue(train_exe, best_nnue_path, "new best (accepted)")
                # Save a game-count-stamped snapshot for later tournament use.
                snapshot_name = f"{net_base}-{current_games}g.nnue"
                snapshot_path = os.path.join(learn_dir, snapshot_name)
                export_nnue(train_exe, snapshot_path, f"snapshot @ {current_games}g")
                print(f"  Banked games: {current_games:,}")
            else:
                if has_checkpoint:
                    shutil.copy2(checkpoint_bin, tdleaf_bin)
                    print("  Reverted .tdleaf.bin to pre-cycle checkpoint.")
                else:
                    # No prior checkpoint existed.  Leave the .tdleaf.bin in
                    # place so the next cycle can build on this cycle's
                    # training rather than restarting from the base net.
                    # The next cycle will still compare against the base net
                    # (best.nnue) until some candidate is finally accepted.
                    print("  No prior checkpoint — retaining .tdleaf.bin for"
                          " next cycle.")

    except KeyboardInterrupt:
        print("\n\n[Ctrl-C — exporting current weights before exit ...]")

    # -----------------------------------------------------------------------
    # Step 6 — Export final .nnue
    # -----------------------------------------------------------------------
    print()
    output_net_name = f"{net_base}-{current_games}g.nnue"
    output_net_path = os.path.join(learn_dir, output_net_name)
    print(f"Exporting final weights → {output_net_name}")
    result = subprocess.run(
        [train_exe, "--write-nnue", output_net_path],
        cwd=learn_dir
    )
    if result.returncode != 0:
        print("--write-nnue failed.", file=sys.stderr)
        sys.exit(result.returncode)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print()
    print("=" * 62)
    print("  Training run complete.")
    print(f"  Net in:      {net_filename}")
    print(f"  Net out:     {output_net_name}  (learn/)")
    if partner_choice == "1":
        print( "  Mode:        symmetric self-play")
    elif partner_choice == "2":
        print( "  Mode:        learner vs. read-only opponent")
    else:
        print(f"  Mode:        learner vs. {os.path.basename(train_exe2)}")
    print(f"  Total games: {current_games:,}")
    print(f"  PGN files:   {pgn_dir}/")
    print(f"  .tdleaf.bin: {tdleaf_bin}")

    if use_loop and cycle_log:
        accepted_count = sum(1 for _, acc, *_ in cycle_log if acc)
        print()
        print(f"  Cycle results  ({accepted_count}/{len(cycle_log)} accepted):")
        print(f"  {'Cycle':>5}  {'W':>4} {'D':>4} {'L':>4}  {'Score%':>7}  "
              f"{'LOS%':>6}  {'Result'}")
        print("  " + "-" * 50)
        for cyc, acc, vw, vd, vl, los in cycle_log:
            vn  = vw + vd + vl
            pct = (vw + 0.5 * vd) / vn * 100.0 if vn else 50.0
            tag = "accepted" if acc else "rejected"
            print(f"  {cyc:>5}  {vw:>4} {vd:>4} {vl:>4}  "
                  f"{pct:>6.1f}%  {los*100:>5.1f}%  {tag}")

    print("=" * 62)
    print()


if __name__ == "__main__":
    main()
