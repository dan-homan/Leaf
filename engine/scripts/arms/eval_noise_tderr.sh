#!/bin/bash
# ---------------------------------------------------------------------------
# eval_noise_tderr.sh -- does eval_noise buy LEARNING SIGNAL, or only Elo cost?
#
# The design argument for eval_noise: the actors play perturbed moves, but the
# learner re-scores every leaf with CLEAN weights (--refresh-scores), so a
# perturbed choice the net already understands produces no TD error -- its
# consequence is already priced into the clean label of the leaf the noisy
# search chose.  Only consequences the net did NOT foresee show up as TD error,
# and those are the ones it can learn from.  The phase-2 Elo cost counts every
# consequential mistake, foreseen or not, so it OVERCOUNTS the useful part.
#
# This measures the useful part directly: generate games at each sigma from the
# SAME frozen net through the REAL actor/learner pipeline, with the learner
# frozen too and dumping every trainable record (leaf_ok, gates wide open), and
# reconstruct the learner's TD errors offline (eval_noise_tderr_score.py).
#
#   TD error barely rises with sigma  -> the noise buys mistakes the net
#                                        already understands; cost, no signal.
#   TD error rises clearly            -> the mechanism works as intended.
#   calibration slope goes negative   -> the lambda trace is learning the value
#                                        of the NOISY player (the bias side).
#
# Isolation: runs in its own directory against COPIES of the binary, the seed
# net and the frozen state -- the live chain's m260921.tdleaf.bin is never
# opened.  Actors and learner are both TDLEAF_FREEZE=1.  Fixed depth, so
# sharing the machine with a live leg changes wall time, not results.
#
# Usage:  bash eval_noise_tderr.sh            (from anywhere)
#         SIGMAS="0 20" GAMES=4000 bash eval_noise_tderr.sh
# ---------------------------------------------------------------------------
set -u

WORK="${WORK:-/home/homand/Leaf/engine/learn/tderr_noise}"
ARMS=/home/homand/Leaf/engine/scripts/arms
BIN=Leaf_vtdnoise            # copy of the chain's Leaf_vtrain_hl_a
SIGMAS="${SIGMAS:-0 20 30}"
GAMES="${GAMES:-8000}"
ACTORS="${ACTORS:-8}"
DEPTH="${DEPTH:-8}"
SEED="${SEED:-20260923}"     # shared by every arm: same opening stripes + salts

cd "$WORK" || exit 1
for f in "$BIN" m260921.nnue m260921.tdleaf.bin training_openings.epd selfplay_run.py; do
    [ -e "$f" ] || { echo "missing $WORK/$f"; exit 1; }
done
md5sum m260921.tdleaf.bin > state.md5

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

for sig in $SIGMAS; do
    out=$WORK/td_s$sig
    if [ -f "$out.done" ]; then log "sigma=$sig already generated, skipping"; continue; fi
    rm -rf "$out"; mkdir -p "$out"
    log "sigma=$sig: $GAMES games, $ACTORS actors, d$DEPTH"
    # Learner env: frozen, dumping every leaf_ok record (quiet/max gates wide).
    TDLEAF_FREEZE=1 TDLEAF_DUMP_TSV="$out/dump" \
    TDLEAF_DUMP_QUIET_CP=100000 TDLEAF_DUMP_MAX_CP=100000 \
        python3 selfplay_run.py --binary "$BIN" --epd training_openings.epd \
            --actors "$ACTORS" --depth "$DEPTH" --games-per-actor 1000 \
            --total-games "$GAMES" --traj-dir "$out/traj" \
            --refresh-scores --delete-consumed --seed "$SEED" \
            $( [ "$sig" != 0 ] && echo --eval-noise "$sig" ) \
            > "$out/run.log" 2>&1
    touch "$out.done"
    log "sigma=$sig done"
done

md5sum -c state.md5 || log "WARNING: frozen state changed"
python3 "$ARMS/eval_noise_tderr_score.py" "$WORK" $SIGMAS | tee "$WORK/result.txt"
