#!/bin/bash
# ---------------------------------------------------------------------------
# psqt_noise_tderr.sh -- does DIRECTIONAL parameter-space exploration produce
# coherent TD gradients?
#
# eval_noise (eval_noise_tderr.sh) added no TD error to the trace: its offsets
# are a hash of the pawn structure, so every diverted game goes somewhere
# unrelated and any error the clean net makes there is a one-off.  --psqt-noise
# perturbs along directions the net can represent instead: the positional part
# of each (piece type, PSQT bucket) group of PSQT entries scaled by (1+eps),
# one eps per group, SHARED by every actor.  If the hypothesis is right, games
# steered by a shared hypothesis should push the learner's gradient the same
# way, so the gradient projected on those 48 pattern directions should be more
# COHERENT across games than under clean play (psqt_noise_coherence.py).
#
# Each arm is ONE generation (games-per-actor above the arm size), so one eps
# draw per arm; arms differ by noise seed, and all are paired on openings with
# the clean arm eval_noise_tderr.sh already produced (td_s0: same frozen
# m260921-2.5e6g state, same --seed).  Same isolation as eval_noise_tderr.sh:
# copies of the binary and state, actors and learner both TDLEAF_FREEZE=1.
#
# OPP=clean / OPP=anti run the TWO-SIDED design (selfplay_run --psqt-opponent):
# side A plays +eps against the clean net or against -eps, alternating colour
# over paired openings, so the outcome tests the hypothesis rather than both
# sides enacting it.  Games go to <arm>/pgn with White/Black "hypA"/"hypB", and
# the scorer reports side A's score as Elo (pentanomial over opening pairs).
#
# Usage:  bash psqt_noise_tderr.sh
#         FRAC=0.5 NSEEDS="101 202" GAMES=8000 bash psqt_noise_tderr.sh
#         OPP=clean bash psqt_noise_tderr.sh
# ---------------------------------------------------------------------------
set -u

WORK="${WORK:-/home/homand/Leaf/engine/learn/tderr_noise}"
ARMS=/home/homand/Leaf/engine/scripts/arms
BIN="${BIN:-Leaf_vtdnoise4}"
REF="${REF:-psqt_ref_2.5e6g.txt}"
FRAC="${FRAC:-0.5}"
NSEEDS="${NSEEDS:-101 202}"
OPP="${OPP:-same}"           # same | clean | anti  (selfplay_run --psqt-opponent)
GAMES="${GAMES:-8000}"
ACTORS="${ACTORS:-8}"
DEPTH="${DEPTH:-8}"
SEED="${SEED:-20260923}"     # opening seed: MUST match the clean arm td_s0

cd "$WORK" || exit 1
for f in "$BIN" "$REF" m260921.nnue m260921.tdleaf.bin training_openings.epd selfplay_run.py; do
    [ -e "$f" ] || { echo "missing $WORK/$f"; exit 1; }
done
[ -f td_s0.done ] || { echo "clean arm td_s0 missing: run eval_noise_tderr.sh (SIGMAS=0) first"; exit 1; }
md5sum m260921.tdleaf.bin > state.md5
log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

for ns in $NSEEDS; do
    tag=pq_${FRAC}_n$ns; [ "$OPP" != same ] && tag=pq_${FRAC}_${OPP}_n$ns
    out=$WORK/$tag
    if [ -f "$out.done" ]; then log "frac=$FRAC seed=$ns done, skipping"; continue; fi
    rm -rf "$out"; mkdir -p "$out"
    log "psqt-noise frac=$FRAC opponent=$OPP noise-seed=$ns: $GAMES games, $ACTORS actors, d$DEPTH"
    TDLEAF_FREEZE=1 TDLEAF_DUMP_TSV="$out/dump" \
    TDLEAF_DUMP_QUIET_CP=100000 TDLEAF_DUMP_MAX_CP=100000 \
        python3 selfplay_run.py --binary "$BIN" --epd training_openings.epd \
            --actors "$ACTORS" --depth "$DEPTH" --games-per-actor 1000000 \
            --total-games "$GAMES" --traj-dir "$out/traj" \
            --refresh-scores --delete-consumed --seed "$SEED" \
            --psqt-noise "$FRAC" --psqt-noise-seed "$ns" --psqt-noise-ref "$REF" \
            --psqt-opponent "$OPP" --pgn-dir "$out/pgn" \
            > "$out/run.log" 2>&1
    grep -h "PSQT noise eps" "$out"/traj/actor_0.log > "$out/eps.txt"
    touch "$out.done"
    log "frac=$FRAC seed=$ns done"
done

md5sum -c state.md5 || log "WARNING: frozen state changed"
ARMLIST="td_s0"; for ns in $NSEEDS; do
    t=pq_${FRAC}_n$ns; [ "$OPP" != same ] && t=pq_${FRAC}_${OPP}_n$ns; ARMLIST="$ARMLIST $t"; done
python3 "$ARMS/psqt_noise_coherence.py" "$WORK" "$REF" ../m260921-2.5e6g_final.nnue $ARMLIST \
    | tee "$WORK/pq_result_${FRAC}_${OPP}.txt"
