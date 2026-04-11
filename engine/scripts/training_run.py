#!/usr/bin/env python3
#
# Leaf TDLeaf(λ) training run manager.
# Run from any directory; paths are resolved relative to this script.
#
#   python3 training_run.py
#
# What this script does:
#   1. Asks for a starting .nnue file or offers to initialise a fresh random one.
#   2. Builds an opponent roster — one or more opponent types that rotate every
#      N games.  Available types: self-play (symmetric), previous checkpoint
#      (read-only .nnue), read-only mirror (frozen .tdleaf.bin during segment),
#      or an external engine.  Builds the necessary binaries and moves them to
#      learn/ so all training files live together.
#   3. Checks for an existing .tdleaf.bin (continue or start fresh).
#   4. Asks for match parameters, then runs match.py for N games.
#      When the roster has multiple opponents (or prev-checkpoint), games are
#      split into segments of `rotation_interval` games each, with a .nnue
#      checkpoint exported at every rotation boundary.
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
import psutil
import re
import shutil
import signal
import stat
import subprocess
import sys
import time

learn_dir = os.path.dirname(os.path.abspath(__file__))
run_dir   = os.path.normpath(os.path.join(learn_dir, "../run"))
src_dir   = os.path.normpath(os.path.join(learn_dir, "../src"))
tools_dir = os.path.normpath(os.path.join(learn_dir, "../../tools"))


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


def discover_external_engines():
    """Scan tools/engines/ for available external engines.

    Returns a list of (display_name, absolute_path) tuples.
    """
    engines = []
    engines_dir = os.path.join(tools_dir, "engines")
    if not os.path.isdir(engines_dir):
        return engines
    for d in sorted(os.listdir(engines_dir)):
        dp = os.path.join(engines_dir, d)
        if not os.path.isdir(dp):
            continue
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
        match = [c for c in candidates if d.lower() in c[0].lower()]
        best = max(match, key=lambda c: c[2]) if match else max(candidates, key=lambda c: c[2])
        engines.append((d, best[1]))
    return engines


def pick_external_engine():
    """Interactive external engine selection. Returns absolute path or None."""
    engines = discover_external_engines()
    if not engines:
        print("    No engines found in tools/engines/.")
        while True:
            opp_path = ask("    Path to opponent executable").strip()
            if os.path.isfile(opp_path):
                return os.path.abspath(opp_path)
            if os.path.isfile(os.path.join(learn_dir, opp_path)):
                return os.path.abspath(os.path.join(learn_dir, opp_path))
            print(f"    File not found: {opp_path}")

    print("    Available external engines:")
    for i, (name, path) in enumerate(engines, 1):
        print(f"      [{i}] {name}")
    print(f"      [c] Custom path")
    print()

    while True:
        choice = input("    Select: ").strip()
        if not choice:
            continue
        if choice.lower() == "c":
            while True:
                opp_path = ask("    Path to opponent executable").strip()
                if os.path.isfile(opp_path):
                    return os.path.abspath(opp_path)
                if os.path.isfile(os.path.join(learn_dir, opp_path)):
                    return os.path.abspath(os.path.join(learn_dir, opp_path))
                print(f"    File not found: {opp_path}")
        try:
            n = int(choice)
            if 1 <= n <= len(engines):
                return engines[n - 1][1]
        except ValueError:
            pass
        print(f"    Invalid selection: {choice}")


def read_game_count(sidecar_path):
    try:
        with open(sidecar_path) as f:
            return int(f.read().strip())
    except Exception:
        return 0


def write_game_count(sidecar_path, count):
    with open(sidecar_path, "w") as f:
        f.write(f"{count}\n")


def format_game_count(n):
    """Convert a game count integer to compact scientific-notation suffix.

    Examples: 100000 → '1e5', 150000 → '1.5e5', 40000 → '4e4', 500 → '500'.
    Counts that are not a clean multiple of a power of 10 (e.g. 123456) are
    left as plain integers.  The threshold for switching to exponent form is
    counts >= 10000 (4 trailing zeros or more expresses as Xe4+).
    """
    if n <= 0:
        return str(n)
    # Strip trailing zeros to find mantissa and exponent
    x, exp = n, 0
    while x % 10 == 0:
        x //= 10
        exp += 1
    if exp < 4:
        # Not enough trailing zeros — keep as plain integer
        return str(n)
    # Normalise to mantissa in [1, 10): e.g. x=15, exp=4 → 1.5e5
    digits = len(str(x))
    total_exp = exp + digits - 1
    mantissa = x / 10 ** (digits - 1)
    if mantissa == int(mantissa):
        return f"{int(mantissa)}e{total_exp}"
    # Format mantissa without trailing zeros (e.g. 1.50 → 1.5)
    m_str = f"{mantissa:.10g}"
    return f"{m_str}e{total_exp}"


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


def parse_pgn_results(pgn_path, engine1_name):
    """Parse a PGN file and return a list of per-game scores from engine1's POV.

    Returns a list of floats: 1.0 (engine1 win), 0.5 (draw), 0.0 (engine1 loss).
    Games with unknown results ('*') are skipped.
    """
    results = []
    if not os.path.isfile(pgn_path):
        return results

    # We need to figure out which color engine1 played in each game and
    # combine that with the game result.
    white_tag = re.compile(r'^\[White\s+"(.+?)"\]')
    result_tag = re.compile(r'^\[Result\s+"(.+?)"\]')

    current_white = None
    with open(pgn_path) as f:
        for line in f:
            line = line.strip()
            m = white_tag.match(line)
            if m:
                current_white = m.group(1)
                continue
            m = result_tag.match(line)
            if m:
                res_str = m.group(1)
                if res_str == "*":
                    current_white = None
                    continue
                # Determine if engine1 was White.
                e1_is_white = (current_white is not None and
                               engine1_name in current_white)
                if res_str == "1-0":
                    results.append(1.0 if e1_is_white else 0.0)
                elif res_str == "0-1":
                    results.append(0.0 if e1_is_white else 1.0)
                elif res_str == "1/2-1/2":
                    results.append(0.5)
                current_white = None
    return results


def analyze_segment_progress(pgn_path, engine1_name, window_frac=0.25,
                             min_window=10):
    """Analyze in-segment learning progress from a PGN file.

    Compares engine1's score in the first `window_frac` of games against
    the last `window_frac` using LOS.  Also fits an OLS linear trend on
    game scores as a diagnostic.

    Returns a dict with:
      n_games, early_wdl, late_wdl, early_score, late_score,
      los (late > early), trend_slope, trend_r2, verdict
    or None if there aren't enough games.
    """
    scores = parse_pgn_results(pgn_path, engine1_name)
    n = len(scores)
    window = max(min_window, int(n * window_frac))

    if n < 2 * window:
        return None   # not enough games to compare

    early = scores[:window]
    late  = scores[-window:]

    def wdl_from_scores(sc):
        w = sum(1 for s in sc if s == 1.0)
        d = sum(1 for s in sc if s == 0.5)
        l = sum(1 for s in sc if s == 0.0)
        return w, d, l

    ew, ed, el = wdl_from_scores(early)
    lw, ld, ll = wdl_from_scores(late)
    early_pct = (ew + 0.5 * ed) / len(early) * 100.0
    late_pct  = (lw + 0.5 * ld) / len(late) * 100.0

    # LOS that late window is better than early window.
    # Treat early as "opponent baseline" (losses) and late as "candidate" (wins).
    # We compute LOS from the late-vs-early differential.
    delta_w = max(0, lw - ew)   # net wins gained
    delta_l = max(0, ll - el) if ll > el else 0
    delta_d = window - delta_w - delta_l
    if delta_d < 0:
        delta_d = 0
    # More robust: use the actual score difference with normal approx.
    diff = (late_pct - early_pct) / 100.0   # fraction
    # Variance of score difference (independent samples)
    var_early = (ew * (1.0 - early_pct/100)**2 +
                 ed * (0.5 - early_pct/100)**2 +
                 el * (0.0 - early_pct/100)**2) / len(early)
    var_late  = (lw * (1.0 - late_pct/100)**2 +
                 ld * (0.5 - late_pct/100)**2 +
                 ll * (0.0 - late_pct/100)**2) / len(late)
    var_diff = var_early / len(early) + var_late / len(late)
    if var_diff > 0:
        z = diff / math.sqrt(var_diff)
        los = 0.5 * math.erfc(-z / math.sqrt(2.0))
    else:
        los = 1.0 if diff > 0 else (0.0 if diff < 0 else 0.5)

    # OLS linear trend: score_i = a + b*i
    x_mean = (n - 1) / 2.0
    y_mean = sum(scores) / n
    ss_xy = sum((i - x_mean) * (s - y_mean) for i, s in enumerate(scores))
    ss_xx = sum((i - x_mean) ** 2 for i in range(n))
    slope = ss_xy / ss_xx if ss_xx > 0 else 0.0
    ss_yy = sum((s - y_mean) ** 2 for s in scores)
    r2 = (ss_xy ** 2) / (ss_xx * ss_yy) if ss_xx > 0 and ss_yy > 0 else 0.0

    return {
        "n_games": n,
        "early_wdl": (ew, ed, el),
        "late_wdl": (lw, ld, ll),
        "early_score": early_pct,
        "late_score": late_pct,
        "los": los,
        "trend_slope": slope,     # score change per game
        "trend_r2": r2,
        "window": window,
    }


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
    cpu_count           = psutil.cpu_count(logical=False) or 1
    default_concurrency = max(1, cpu_count - 1)
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
    print("  [2] Initialise a fresh random network (classical material prior)")
    print("  [3] Initialise a fresh random network (no prior — all pieces 100cp)")
    choice = ask("Choice", "1")

    init_noprior = False
    if choice.strip() in ("2", "3"):
        init_noprior = (choice.strip() == "3")
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

    tdleaf_bin        = os.path.join(learn_dir, net_base + ".tdleaf.bin")
    sidecar_path      = os.path.join(learn_dir, net_base + ".games")
    training_epd_path = os.path.join(learn_dir, "training_openings.epd")
    use_training_epd  = os.path.isfile(training_epd_path)

    # -----------------------------------------------------------------------
    # Step 1b — Opponent roster (rotation across multiple opponent types)
    # -----------------------------------------------------------------------
    #
    # Each roster entry is a dict with:
    #   "type"  : "self-play" | "prev-checkpoint" | "readonly" | "external"
    #   "label" : display name
    #   "exe"   : path to opponent binary (set during build step)
    #   "path"  : for external opponents, the user-supplied path
    #
    # The learner binary (train_exe) is always engine 1.  Opponents rotate
    # every `rotation_interval` games.  A .nnue checkpoint is exported at
    # every rotation boundary.
    #
    print()
    print("Opponent roster (opponents rotate every N games):")
    print("  Available opponent types:")
    print("    [s] Self-play           — both instances learn (symmetric)")
    print("    [p] Previous checkpoint — learner vs. own recent snapshot (read-only)")
    print("    [r] Read-only mirror    — current weights, frozen during segment")
    print("    [e] External engine     — learner vs. user-supplied executable")
    print()

    roster = []
    while True:
        default_type = "s" if not roster else ""
        choice_opp = ask(
            f"  Add opponent #{len(roster)+1} (s/p/r/e, empty to finish)",
            default_type if not roster else None
        ).strip().lower()
        if not choice_opp:
            if roster:
                break
            print("  Roster must have at least one opponent.")
            continue
        if choice_opp == "s":
            roster.append({"type": "self-play", "label": "self-play (symmetric)"})
        elif choice_opp == "p":
            roster.append({"type": "prev-checkpoint", "label": "previous checkpoint (read-only)"})
        elif choice_opp == "r":
            roster.append({"type": "readonly", "label": "read-only mirror (frozen during segment)"})
        elif choice_opp == "e":
            opp_abs = pick_external_engine()
            roster.append({
                "type": "external", "label": os.path.basename(opp_abs),
                "path": opp_abs,
            })
        else:
            print(f"  Unknown type '{choice_opp}' — use s, p, r, or e.")
            continue
        print(f"    Added: {roster[-1]['label']}")

    use_rotation = (len(roster) > 1
                    or any(r["type"] in ("prev-checkpoint", "readonly") for r in roster))
    rotation_interval = 0
    if use_rotation:
        rotation_interval = int(ask("  Games per opponent before rotating", "2000"))

    # Derive which binary types we need to build.
    need_self_play_exe = any(r["type"] == "self-play" for r in roster)
    need_checkpoint_exe = any(r["type"] == "prev-checkpoint" for r in roster)
    need_readonly_exe = any(r["type"] == "readonly" for r in roster)

    # -----------------------------------------------------------------------
    # Step 1c — Train-validate loop (optional)
    # -----------------------------------------------------------------------
    print()
    use_loop       = ask_yes_no("Enable train-validate loop?", default="n")
    n_cycles       = 0
    val_games      = 500
    los_thresh_pct = 50.0
    los_stop_hi    = 0.99   # early-stop if LOS rises above this
    los_stop_lo    = 0.01   # early-stop if LOS falls below this
    val_tc         = None   # set in Step 4 after tc1 is known
    # Eval binary names (set in Step 2 if use_loop)
    best_nnue_name = None
    cand_nnue_name = None
    eval_best_exe  = None
    eval_cand_exe  = None

    if use_loop:
        n_cycles       = int(ask("  Cycles (0 = run forever until Ctrl-C)", "0"))
        val_games      = int(ask("  Validation games per cycle           ", "500"))
        los_thresh_pct = float(ask("  LOS acceptance threshold (%)        ", "50"))
        los_stop_hi    = float(ask("  Early-stop if LOS ≥ (%)            ", "99")) / 100.0
        los_stop_lo    = float(ask("  Early-stop if LOS ≤ (%)            ", "01")) / 100.0
        # val_tc is asked in Step 4 once tc1 is known
        best_nnue_name = f"{net_base}-best.nnue"
        cand_nnue_name = f"{net_base}-cand.nnue"

    # -----------------------------------------------------------------------
    # Step 1d — Segment progress analysis (optional, rotation mode only)
    # -----------------------------------------------------------------------
    # When playing static opponents (prev-checkpoint, external), compare the
    # learner's score in the first N games vs the last N games of each segment
    # to detect in-segment improvement.  If the learner is getting worse,
    # revert the .tdleaf.bin to the pre-segment state.
    has_static_opponents = any(
        r["type"] in ("prev-checkpoint", "readonly", "external") for r in roster
    )
    use_seg_analysis    = False
    seg_accept_los      = 0.60
    seg_reject_los      = 0.40
    seg_window_frac     = 0.25
    seg_min_window      = 20

    if use_rotation and has_static_opponents:
        print()
        use_seg_analysis = ask_yes_no(
            "Enable in-segment progress analysis for static opponents?",
            default="n")
        if use_seg_analysis:
            seg_accept_los  = float(ask(
                "  Accept LOS threshold (%)", "60")) / 100.0
            seg_reject_los  = float(ask(
                "  Reject LOS threshold (%)", "40")) / 100.0
            seg_window_frac = float(ask(
                "  Window fraction (0-0.5) ", "0.25"))
            seg_min_window  = int(ask(
                "  Minimum window size     ", "20"))

    # -----------------------------------------------------------------------
    # Step 2 — Build executables  (comp.pl runs in run/, binaries moved to learn/)
    # -----------------------------------------------------------------------
    print()
    print("Building executables:")
    train_ver   = f"train_{net_base}_a"
    train_exe   = os.path.join(learn_dir, f"Leaf_v{train_ver}")
    nnue_flag   = f"NNUE_NET={net_filename}"
    tdleaf_flags = ["NNUE=1", "TDLEAF=1", nnue_flag]

    # Self-play partner binary (same net, also learns)
    selfplay_ver = f"train_{net_base}_b"
    selfplay_exe = os.path.join(learn_dir, f"Leaf_v{selfplay_ver}")

    # Prev-checkpoint opponent binary — uses a *separate* NNUE_NET so we can
    # swap its .nnue file independently (for prev-checkpoint rotation).
    checkpoint_opp_nnue = f"{net_base}-opponent.nnue"
    checkpoint_ver = f"train_{net_base}_ro"
    checkpoint_exe = os.path.join(learn_dir, f"Leaf_v{checkpoint_ver}")

    # Read-only mirror binary — same NNUE_NET as learner, loads .tdleaf.bin
    # at startup but does not write.  Frozen during a segment; refreshes
    # weights when cutechess-cli restarts it for the next segment.
    readonly_ver = f"train_{net_base}_rom"
    readonly_exe = os.path.join(learn_dir, f"Leaf_v{readonly_ver}")

    # Wire up each roster entry's exe field.
    for entry in roster:
        if entry["type"] == "self-play":
            entry["exe"] = selfplay_exe
        elif entry["type"] == "prev-checkpoint":
            entry["exe"] = checkpoint_exe
        elif entry["type"] == "readonly":
            entry["exe"] = readonly_exe
        elif entry["type"] == "external":
            entry["exe"] = entry["path"]

    # Determine which binaries need building.
    need_binaries = [("learner", train_ver, tdleaf_flags)]
    if need_self_play_exe:
        need_binaries.append(("self-play B", selfplay_ver, tdleaf_flags))
    if need_checkpoint_exe:
        # Pure inference binary — no TDLEAF, just loads the .nnue file.
        # The exported .nnue already contains trained FC weights, so no
        # .tdleaf.bin is needed.
        ro_flags = ["NNUE=1", f"NNUE_NET={checkpoint_opp_nnue}"]
        need_binaries.append(("checkpoint opponent", checkpoint_ver, ro_flags))
    if need_readonly_exe:
        # TDLEAF_READONLY: loads .tdleaf.bin at startup (current learner
        # weights) but skips all ply recording and weight updates.
        rom_flags = ["NNUE=1", "TDLEAF=1", "TDLEAF_READONLY=1", nnue_flag]
        need_binaries.append(("read-only mirror", readonly_ver, rom_flags))

    all_exist = all(
        os.path.isfile(os.path.join(learn_dir, f"Leaf_v{ver}"))
        for _, ver, _ in need_binaries
    )
    rebuild = True
    if all_exist:
        rebuild = ask_yes_no(
            "  Executable(s) already exist.  Rebuild?", default="n")

    if rebuild:
        for label, ver, flags in need_binaries:
            print(f"  Building {label} binary (Leaf_v{ver}) ...")
            if not build_binary(ver, flags):
                print(f"  {label} build failed.", file=sys.stderr)
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
            init_flag = "--init-nnue-noprior" if init_noprior else "--init-nnue"
            label = "no-prior" if init_noprior else "classical"
            print(f"  Initialising fresh network ({label}) → {net_filename}")
            result = subprocess.run(
                [train_exe, init_flag, "--write-nnue", net_file],
                cwd=learn_dir
            )
            if result.returncode != 0:
                print(f"  {init_flag} failed.", file=sys.stderr)
                sys.exit(1)
        else:
            print("  Using existing fresh net.")

    # -----------------------------------------------------------------------
    # Step 3 — .tdleaf.bin continuity check
    # -----------------------------------------------------------------------
    prior_games = read_game_count(sidecar_path)
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
        else:
            # Back up the file before any training writes so it can be
            # recovered if the run produces bad weights or is interrupted
            # at an inopportune moment.
            bak = tdleaf_bin + ".bak"
            shutil.copy2(tdleaf_bin, bak)
            print(f"  Backup saved  → {os.path.basename(bak)}")
    else:
        print(f"No existing .tdleaf.bin for {net_filename} — starting fresh.")

    # -----------------------------------------------------------------------
    # Step 4 — Match parameters
    # -----------------------------------------------------------------------
    print()
    print("Match parameters:")
    if use_loop:
        n_games = int(ask("  Games per cycle     [-n]        ", 10000))
    else:
        n_games = int(ask("  Games                [-n]        ", 10000))
    tc1         = ask(    "  Learner time control   [--tc1]  ", "0:03+0.05")
    tc2_raw     = ask(    "  Opponent time control  [--tc2]  ", tc1)
    tc2         = tc2_raw if tc2_raw.strip() else tc1
    concurrency = int(ask( "  Concurrency         [-c]        ", default_concurrency))
    wait_ms     = int(ask("  Wait between games  [--wait ms] ", 500))
    if use_training_epd:
        print(f"  Opening EPD:      training_openings.epd detected — using FRC+book openings "
              f"(Fischer variant auto-enabled)")
        fischer = True
    else:
        fischer = ask_yes_no("  Fischer Random? [--fischer-random]", default="y")
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

    # Opening book for non-Fischer, non-EPD games
    book_path = os.path.join(learn_dir, "normbk02.bin")
    use_book  = not use_training_epd and not fischer and os.path.isfile(book_path)

    # -----------------------------------------------------------------------
    # PGN directory  —  learn/pgn/<net_base>/
    # -----------------------------------------------------------------------
    pgn_dir       = os.path.join(learn_dir, "pgn", net_base)
    os.makedirs(pgn_dir, exist_ok=True)
    pgn_start_tag = format_game_count(prior_games) + "g"
    pgn_base_path = os.path.join(pgn_dir, f"match_{net_base}_{pgn_start_tag}.pgn")

    # -----------------------------------------------------------------------
    # Confirm
    # -----------------------------------------------------------------------
    print()
    print("=" * 62)
    print(f"  Net in:           {net_filename}")
    print(f"  Learner:          Leaf_v{train_ver}")
    if len(roster) == 1 and not use_rotation:
        print(f"  Opponent:         {roster[0]['label']}")
    else:
        print(f"  Opponent roster:  (rotate every {rotation_interval:,} games)")
        for i, entry in enumerate(roster):
            print(f"    {i+1}. {entry['label']}")
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
    if use_training_epd:
        print(f"  Opening EPD:      training_openings.epd  (FRC+book, Fischer variant)")
    elif fischer:
        print( "  Fischer Random:   yes")
    elif use_book:
        print(f"  Opening book:     {os.path.basename(book_path)}")
    if use_seg_analysis:
        print(f"  Seg. analysis:    accept≥{seg_accept_los*100:.0f}%  "
              f"reject≤{seg_reject_los*100:.0f}%  "
              f"window={seg_window_frac:.0%} (min {seg_min_window})")
    print(f"  PGN directory:    {pgn_dir}/")
    print("=" * 62)

    if not ask_yes_no("Proceed?", default="y"):
        print("Aborted.")
        sys.exit(0)

    # -----------------------------------------------------------------------
    # Step 5 — Training loop (single pass or multi-cycle train-validate)
    # -----------------------------------------------------------------------
    match_py      = os.path.join(run_dir, "match.py")
    current_games = prior_games
    cycle_log     = []   # list of (cycle, accepted, vw, vd, vl, los)

    def build_match_cmd(opp_exe, num_games, pgn_path, no_repeat=False,
                        opp_type="self-play"):
        cmd = [
            sys.executable, match_py,
            train_exe,
            opp_exe,
            "-n", str(num_games),
            "-tc", tc1,
            "-c", str(concurrency),
            "--wait", str(wait_ms),
            "--pgn-out", pgn_path,
        ]
        # Learner (engine1) always uses xboard for TDLeaf hooks.
        # Leaf opponents (self-play, checkpoint, readonly) also use xboard.
        # External engines default to UCI.
        cmd += ["--proto1", "xboard"]
        if opp_type == "external":
            cmd += ["--proto2", "uci"]
        else:
            cmd += ["--proto2", "xboard"]
        if tc2 != tc1:
            cmd += ["--tc2", tc2]
        if use_training_epd:
            cmd += ["--openings", training_epd_path, "--fischer-random"]
        elif fischer:
            cmd.append("--fischer-random")
        elif use_book:
            cmd += ["--openings", book_path]
        if depth1:
            cmd += ["--depth1", str(depth1)]
        if depth2:
            cmd += ["--depth2", str(depth2)]
        if no_repeat:
            cmd.append("--no-repeat")
        return cmd

    def export_nnue(exe, dest_path, label):
        """Export trained weights via --write-nnue; exit on failure."""
        print(f"  Exporting {label} → {os.path.basename(dest_path)} ...")
        r = subprocess.run([exe, "--write-nnue", dest_path], cwd=learn_dir)
        if r.returncode != 0:
            print(f"  --write-nnue ({label}) failed.", file=sys.stderr)
            sys.exit(r.returncode)

    # Path where the read-only opponent binary expects its .nnue.
    checkpoint_opp_nnue_path = os.path.join(learn_dir, checkpoint_opp_nnue)
    # Tracks the last exported checkpoint (for prev-checkpoint opponent).
    # On resume, seed from the highest-numbered existing checkpoint on disk.
    _ckpt_pattern = re.compile(rf"^{re.escape(net_base)}-(\d+)g\.nnue$")
    _existing = []
    for _f in os.listdir(learn_dir):
        _m = _ckpt_pattern.match(_f)
        if _m:
            _existing.append((int(_m.group(1)), os.path.join(learn_dir, _f)))
    if _existing:
        last_checkpoint_nnue = max(_existing)[1]
        if need_checkpoint_exe:
            print(f"  Resuming: prev-checkpoint opponent will start from "
                  f"{os.path.basename(last_checkpoint_nnue)}")
    else:
        last_checkpoint_nnue = None

    def prepare_prev_checkpoint_opponent():
        """Copy the latest checkpoint .nnue to the read-only opponent's expected path.

        If no checkpoint has been exported yet (first segment), use the base net.
        The opponent binary is pure NNUE (no TDLEAF), so only the .nnue is needed.
        """
        nonlocal last_checkpoint_nnue
        src = last_checkpoint_nnue if last_checkpoint_nnue else net_file
        label = os.path.basename(src)
        print(f"  Setting prev-checkpoint opponent → {label}")
        shutil.copy2(src, checkpoint_opp_nnue_path)

    def export_rotation_checkpoint():
        """Export current weights to a game-count-stamped checkpoint."""
        nonlocal last_checkpoint_nnue
        wait_until_stable(tdleaf_bin)
        snap_name = f"{net_base}-{format_game_count(current_games)}g.nnue"
        snap_path = os.path.join(learn_dir, snap_name)
        export_nnue(train_exe, snap_path, f"checkpoint @ {format_game_count(current_games)}g")
        last_checkpoint_nnue = snap_path

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
                    pgn_dir, f"match_{net_base}_{pgn_start_tag}_cycle{cycle_num:02d}.pgn")
            else:
                pgn_path = pgn_base_path

            # --- Training match (with opponent rotation) ---
            print()
            # Track games added in this cycle/run for correct accounting.
            games_before = current_games

            if use_rotation:
                # Split this cycle/run into segments, rotating opponents.
                games_remaining = n_games
                roster_idx = 0
                seg_num = 0
                seg_reverts = 0   # count of reverted segments this cycle/run
                while games_remaining > 0:
                    entry = roster[roster_idx % len(roster)]
                    seg_games = min(rotation_interval, games_remaining)
                    seg_num += 1

                    print(f"  ── Segment {seg_num}: {seg_games} games vs {entry['label']} ──")

                    is_static = entry["type"] in ("prev-checkpoint", "readonly", "external")

                    # Save pre-segment .tdleaf.bin for potential revert.
                    seg_checkpoint = None
                    if use_seg_analysis and is_static and os.path.isfile(tdleaf_bin):
                        seg_checkpoint = tdleaf_bin + ".seg_checkpoint"
                        wait_until_stable(tdleaf_bin)
                        shutil.copy2(tdleaf_bin, seg_checkpoint)

                    # Set up prev-checkpoint opponent if needed.
                    is_prev = entry["type"] == "prev-checkpoint"
                    if is_prev:
                        prepare_prev_checkpoint_opponent()

                    seg_pgn = pgn_path.replace(".pgn", f"_seg{seg_num:02d}.pgn")
                    cmd = build_match_cmd(entry["exe"], seg_games, seg_pgn,
                                         no_repeat=(entry["type"] == "self-play"),
                                         opp_type=entry["type"])
                    # Extract learner binary name for PGN parsing.
                    train_basename = os.path.basename(train_exe)
                    result = subprocess.run(cmd, cwd=run_dir)
                    if result.returncode != 0:
                        print(f"\nMatch segment failed (exit code {result.returncode}).",
                              file=sys.stderr)
                        if use_loop:
                            print("Weights from completed cycles preserved. Exiting.")
                        else:
                            print("Game count sidecar NOT updated.")
                        sys.exit(result.returncode)

                    games_remaining -= seg_games
                    current_games += seg_games

                    # --- In-segment progress analysis ---
                    if use_seg_analysis and is_static and seg_checkpoint:
                        wait_until_stable(tdleaf_bin)
                        analysis = analyze_segment_progress(
                            seg_pgn, train_basename,
                            window_frac=seg_window_frac,
                            min_window=seg_min_window)
                        if analysis is not None:
                            ew, ed, el = analysis["early_wdl"]
                            lw, ld, ll = analysis["late_wdl"]
                            print(f"\n  Segment progress ({analysis['n_games']} games, "
                                  f"window={analysis['window']}):")
                            print(f"    Early {analysis['window']}g: "
                                  f"W={ew} D={ed} L={el}  "
                                  f"score={analysis['early_score']:.1f}%")
                            print(f"    Late  {analysis['window']}g: "
                                  f"W={lw} D={ld} L={ll}  "
                                  f"score={analysis['late_score']:.1f}%")
                            print(f"    LOS(late>early): {analysis['los']*100:.1f}%"
                                  f"   trend: {analysis['trend_slope']*1000:+.2f}/1000g"
                                  f"  R²={analysis['trend_r2']:.3f}")

                            if analysis["los"] >= seg_accept_los:
                                print(f"    → ACCEPT (LOS ≥ {seg_accept_los*100:.0f}%)")
                            elif analysis["los"] < seg_reject_los:
                                print(f"    → REJECT (LOS < {seg_reject_los*100:.0f}%)"
                                      f" — reverting .tdleaf.bin")
                                shutil.copy2(seg_checkpoint, tdleaf_bin)
                                seg_reverts += 1
                            else:
                                print(f"    → INCONCLUSIVE "
                                      f"({seg_reject_los*100:.0f}% ≤ LOS < "
                                      f"{seg_accept_los*100:.0f}%) — keeping")
                        else:
                            print(f"  Segment analysis: not enough games "
                                  f"(need ≥{2*seg_min_window})")

                    # Clean up segment checkpoint.
                    if seg_checkpoint and os.path.isfile(seg_checkpoint):
                        os.remove(seg_checkpoint)

                    # Export checkpoint at rotation boundary (if more segments remain).
                    if games_remaining > 0:
                        export_rotation_checkpoint()

                    roster_idx += 1

                if seg_reverts > 0:
                    print(f"\n  Segments reverted this run: {seg_reverts}")
            else:
                # Single opponent — run all games at once.
                opp_exe = roster[0]["exe"]
                is_prev = roster[0]["type"] == "prev-checkpoint"
                if is_prev:
                    prepare_prev_checkpoint_opponent()
                cmd = build_match_cmd(opp_exe, n_games, pgn_path,
                                     no_repeat=(roster[0]["type"] == "self-play"),
                                     opp_type=roster[0]["type"])
                result = subprocess.run(cmd, cwd=run_dir)
                if result.returncode != 0:
                    print(f"\nMatch run failed (exit code {result.returncode}).",
                          file=sys.stderr)
                    if use_loop:
                        print("Weights from completed cycles preserved. Exiting.")
                    else:
                        print("Game count sidecar NOT updated.")
                    sys.exit(result.returncode)
                current_games += n_games

            if not use_loop:
                # Single-run mode: persist count and break.
                write_game_count(sidecar_path, current_games)
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
                    pgn_dir, f"val_{net_base}_{pgn_start_tag}_cycle{cycle_num:02d}.pgn")
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
                if use_training_epd:
                    val_cmd += ["--openings", training_epd_path, "--fischer-random"]
                elif fischer:
                    val_cmd.append("--fischer-random")
                elif use_book:
                    val_cmd += ["--openings", book_path]

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
                # current_games already advanced during training segments.
                write_game_count(sidecar_path, current_games)
                # Advance the best baseline so the next cycle compares against
                # the newly accepted weights (not the original session start).
                export_nnue(train_exe, best_nnue_path, "new best (accepted)")
                # Save a game-count-stamped snapshot for later tournament use.
                snapshot_name = f"{net_base}-{format_game_count(current_games)}g.nnue"
                snapshot_path = os.path.join(learn_dir, snapshot_name)
                export_nnue(train_exe, snapshot_path, f"snapshot @ {format_game_count(current_games)}g")
                # Keep the prev-checkpoint opponent current: the next cycle's
                # first prev-checkpoint segment should face the just-accepted
                # weights, not a stale intra-cycle snapshot.
                if need_checkpoint_exe:
                    last_checkpoint_nnue = snapshot_path
                print(f"  Banked games: {current_games:,}")
            else:
                # Revert game count — rejected cycle's games don't count.
                current_games = games_before
                write_game_count(sidecar_path, current_games)
                if has_checkpoint:
                    shutil.copy2(checkpoint_bin, tdleaf_bin)
                    print("  Reverted .tdleaf.bin to pre-cycle checkpoint.")
                    # Also revert the prev-checkpoint opponent back to the last
                    # accepted baseline.  The rejected cycle may have advanced
                    # last_checkpoint_nnue to an intra-cycle snapshot that is
                    # now ahead of the reverted .tdleaf.bin, which would make
                    # the opponent artificially stronger than the learner.
                    if need_checkpoint_exe:
                        last_checkpoint_nnue = best_nnue_path
                else:
                    # No prior checkpoint existed.  Leave the .tdleaf.bin in
                    # place so the next cycle can build on this cycle's
                    # training rather than restarting from the base net.
                    # The next cycle will still compare against the base net
                    # (best.nnue) until some candidate is finally accepted.
                    print("  No prior checkpoint — retaining .tdleaf.bin for"
                          " next cycle.")

    except KeyboardInterrupt:
        interrupted = True
        print("\n\n[Ctrl-C — cleaning up ...]")
    else:
        interrupted = False

    # -----------------------------------------------------------------------
    # Step 6 — Export final .nnue
    # -----------------------------------------------------------------------
    # Update the game-count sidecar first so the names we choose below
    # (which embed current_games) always match what the sidecar records.
    # Loop mode keeps .games current after each accepted cycle; this write
    # is the canonical final update for all other exit paths.
    write_game_count(sidecar_path, current_games)

    print()
    if interrupted and use_loop and current_games > prior_games:
        # Validation is active and at least one cycle was accepted.
        # The last validated snapshot already exists at <net>-<current_games>g.nnue.
        # Do NOT overwrite it with unvalidated weights from the interrupted cycle.
        # Instead, export unvalidated weights to a separate file.
        unval_name = f"{net_base}-{format_game_count(current_games)}g-unvalidated.nnue"
        unval_path = os.path.join(learn_dir, unval_name)
        print(f"Exporting unvalidated weights → {unval_name}")
        result = subprocess.run(
            [train_exe, "--write-nnue", unval_path],
            cwd=learn_dir
        )
        if result.returncode != 0:
            print("--write-nnue failed.", file=sys.stderr)
        # Revert .tdleaf.bin to the last validated checkpoint so it matches
        # the validated snapshot .nnue and is ready for the next session.
        checkpoint_bin = tdleaf_bin + ".checkpoint"
        if os.path.isfile(checkpoint_bin):
            shutil.copy2(checkpoint_bin, tdleaf_bin)
            print(f"  Reverted .tdleaf.bin to last validated checkpoint.")
        output_net_name = f"{net_base}-{format_game_count(current_games)}g.nnue"
        print(f"  Last validated: {output_net_name}")
    else:
        # Use -partial suffix on interruption to signal a non-clean exit and
        # avoid overwriting an existing game-count-stamped checkpoint.
        if interrupted:
            output_net_name = f"{net_base}-{format_game_count(current_games)}g-partial.nnue"
        else:
            output_net_name = f"{net_base}-{format_game_count(current_games)}g.nnue"
        output_net_path = os.path.join(learn_dir, output_net_name)
        print(f"Exporting {'partial ' if interrupted else ''}weights → {output_net_name}")
        result = subprocess.run(
            [train_exe, "--write-nnue", output_net_path],
            cwd=learn_dir
        )
        if result.returncode != 0:
            print("--write-nnue failed.", file=sys.stderr)
            if not interrupted:
                sys.exit(result.returncode)

    # Save a .tdleaf.bin snapshot at the final game count for archival / rollback.
    if os.path.isfile(tdleaf_bin):
        tdleaf_snap_name = f"{net_base}.tdleaf.bin-{format_game_count(current_games)}g"
        tdleaf_snap_path = os.path.join(learn_dir, tdleaf_snap_name)
        shutil.copy2(tdleaf_bin, tdleaf_snap_path)
        print(f"  .tdleaf.bin snapshot → {tdleaf_snap_name}")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print()
    print("=" * 62)
    print("  Training run complete.")
    print(f"  Net in:      {net_filename}")
    print(f"  Net out:     {output_net_name}  (learn/)")
    if len(roster) == 1:
        print(f"  Mode:        {roster[0]['label']}")
    else:
        opp_labels = " → ".join(r["label"] for r in roster)
        print(f"  Mode:        rotation ({opp_labels})")
    print(f"  Total games: {current_games:,}")
    print(f"  PGN files:   {pgn_dir}/")
    print(f"  .tdleaf.bin: {tdleaf_bin}")
    print(f"  .tdleaf.bin snapshot: {net_base}.tdleaf.bin-{format_game_count(current_games)}g")

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
