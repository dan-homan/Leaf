#!/usr/bin/env python3
"""
selfplay_run.py — Stage 1 actor/learner training driver.

N frozen actor processes play internal self-play games (--selfplay --traj-out)
and emit per-game .tdg trajectories; ONE learner process (--learn-stream)
consumes them in arrival order and owns the optimizer — a single .tdleaf.bin
writer, no multi-writer merge in the training hot path.

Weight refresh is epoch-style: the learner's --tdleaf-out IS the live state
file the engine loads at startup (atomic tmp+rename saves), so actors simply
exit after --games-per-actor games and are respawned by this driver, reloading
the latest state.  Each respawn gets a fresh deterministic opening shuffle
(seed = base + generation) over its stripe of the book.

Run from learn/ with a training binary installed there, e.g.:

    python3 selfplay_run.py --binary Leaf_vtrain_hl_a \
        --epd training_openings.epd --actors 8 --depth 8 \
        --games-per-actor 1000 --total-games 100000 --traj-dir traj_run1

The learner inherits TDLEAF_* env (TDLEAF_TARGET etc.); actors are forced
frozen (TDLEAF_FREEZE=1).  Stop early by creating <traj-dir>/STOP.
"""

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def log(msg):
    print(f"[selfplay_run {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", required=True,
                    help="Training binary (in cwd) with --selfplay/--learn-stream")
    ap.add_argument("--epd", required=True, help="Opening book EPD")
    ap.add_argument("--actors", type=int, default=4)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--games-per-actor", type=int, default=1000,
                    help="Actor respawn cadence = weight refresh interval")
    ap.add_argument("--total-games", type=int, required=True,
                    help="Learner stops after consuming this many games")
    ap.add_argument("--traj-dir", default="traj",
                    help="Trajectory handoff directory (created if missing)")
    ap.add_argument("--tdleaf-out", default=None,
                    help="Learner state file (default: binary's companion "
                         ".tdleaf.bin next to it — the live state)")
    ap.add_argument("--publish", default=None,
                    help="Optionally bake a .nnue here every --publish-every games")
    ap.add_argument("--publish-every", type=int, default=512)
    ap.add_argument("--seed", type=int, default=1,
                    help="Base opening-shuffle seed")
    ap.add_argument("--delete-consumed", action="store_true",
                    help="Learner deletes consumed .tdg (default: archive to done/)")
    ap.add_argument("--adjudicate", action="store_true",
                    help="Enable actor resign/draw adjudication.  OFF by default "
                         "to match the recipe (train.py generation passes "
                         "--no-adjudication): with learning in the loop, "
                         "adjudicated games form a feedback spiral — evals grow "
                         "more extreme, resignations come earlier, truncated "
                         "outcome-heavy trajectories push evals further "
                         "(d8t-3al collapsed this way: 60%%→97%% resignations, "
                         "27-ply games, entry net lost 400/400).")
    args = ap.parse_args()

    binary = Path(args.binary)
    if not binary.is_file():
        sys.exit(f"binary not found: {binary}")
    traj = Path(args.traj_dir)
    traj.mkdir(exist_ok=True)

    learner_cmd = [f"./{binary}", "--learn-stream", str(traj),
                   "--total-games", str(args.total_games)]
    if args.tdleaf_out:
        learner_cmd += ["--tdleaf-out", args.tdleaf_out]
    if args.publish:
        learner_cmd += ["--publish", args.publish,
                        "--publish-every", str(args.publish_every)]
    if args.delete_consumed:
        learner_cmd += ["--delete"]

    learner_log = open(traj / "learner.log", "a")
    learner = subprocess.Popen(learner_cmd, stdout=learner_log,
                               stderr=learner_log)
    log(f"learner started (pid {learner.pid}) -> {traj}/learner.log")

    actor_env = dict(os.environ)
    actor_env["TDLEAF_FREEZE"] = "1"        # actors never write weights
    actor_env.pop("TDLEAF_DUMP_TSV", None)  # trajectories, not TSV, are the product

    actors = {}      # slot -> (Popen, logfile)
    generation = [0] * args.actors

    def spawn(slot):
        generation[slot] += 1
        seed = args.seed + 1000 * generation[slot]
        cmd = [f"./{binary}", "--selfplay",
               "--epd", args.epd,
               "--epd-shuffle", str(seed),
               "--epd-offset", str(slot), "--epd-stride", str(args.actors),
               "--games", str(args.games_per_actor),
               "--depth", str(args.depth),
               "--traj-out", str(traj)]
        if not args.adjudicate:
            cmd.append("--no-adjudication")
        lf = open(traj / f"actor_{slot}.log", "a")
        p = subprocess.Popen(cmd, env=actor_env, stdout=lf, stderr=lf)
        actors[slot] = (p, lf)
        log(f"actor {slot} gen {generation[slot]} started (pid {p.pid}, seed {seed})")

    def shutdown(*_):
        log("shutting down")
        for p, lf in actors.values():
            p.terminate()
        if learner.poll() is None:
            (traj / "STOP").touch()
            try:
                learner.wait(timeout=120)
            except subprocess.TimeoutExpired:
                learner.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    for slot in range(args.actors):
        spawn(slot)

    while learner.poll() is None:
        time.sleep(2)
        for slot, (p, lf) in list(actors.items()):
            if p.poll() is not None:
                lf.close()
                if p.returncode != 0:
                    log(f"actor {slot} exited rc={p.returncode} — see log")
                spawn(slot)   # respawn = reload latest published state

    log(f"learner finished (rc={learner.returncode}); stopping actors")
    for p, lf in actors.values():
        p.terminate()
    for p, lf in actors.values():
        try:
            p.wait(timeout=30)
        except subprocess.TimeoutExpired:
            p.kill()
        lf.close()
    learner_log.close()
    log("done")


if __name__ == "__main__":
    main()
