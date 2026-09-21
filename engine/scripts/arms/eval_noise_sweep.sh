#!/bin/bash
# ---------------------------------------------------------------------------
# eval_noise_sweep.sh -- calibrate the eval_noise sigma.
#
# WHY.  Every sharpness indicator on the m260916 chain drifted monotonically
# across nine legs of d6/800 self-play: draw rate 30.9% -> 22.5%, mean game
# length 159 -> 136 ply, per-ply eval volatility +77%, and the fraction of root
# rows passing the 60cp quiet gate 0.562 -> 0.499 at constant search depth.
# Piece values are flat over the same span, so it is not eval scale inflation:
# the net is steering its own games into sharper positions and deleting a
# growing share of its own quiet training data.  eval_noise perturbs the
# engine's structural preferences during generation to break that loop.
#
# WHAT THIS MEASURES.  Two independent questions, neither involving learning:
#   Phase 1 (sharpness) -- self-play at each sigma, scored on DRAW RATE and
#       GAME LENGTH.  Those are properties of the games played and so are the
#       only metrics the perturbation cannot contaminate; the reported-score
#       metrics are printed but must not be used as targets (see the .awk).
#   Phase 2 (strength cost) -- noisy vs clean head-to-head at the SAME fixed
#       depth generation uses.  This prices the off-policy cost: the TDLeaf
#       label becomes the clean value of the PV a perturbed search chose, so
#       the label bias is bounded by how much worse that search plays.
#
# Weights are FROZEN throughout (TDLEAF_FREEZE=1).  Nothing here trains, and
# nothing writes to the live chain state in learn/.
#
# READING IT.  Look for the sigma that moves draw rate and mean ply back toward
# the 1e6-era values (24.2% / 140 ply) for a strength cost you are willing to
# pay.  Pre-committed caveat: diversifying pawn structures widens the VARIETY
# of positions without necessarily making them quieter, so a flat or wrong-
# signed draw-rate response across the whole sweep is a real possible outcome
# and would say the mechanism does not attack the measured problem.
#
# Usage:  bash eval_noise_sweep.sh [phase1|phase2|all]
# ---------------------------------------------------------------------------
set -u

LEARN=/home/homand/Leaf/engine/learn
ARMS=/home/homand/Leaf/engine/scripts/arms
WORK="${WORK:-$LEARN/evalnoise_sweep}"
BIN=Leaf_vevalnoise
NET=m260916.nnue                          # seed net; the .tdleaf.bin overrides it
STATE=m260916-7e6g_final.tdleaf.bin       # promoted chain endpoint (7e6 games)
EPD=training_openings.epd

SIGMAS="${SIGMAS:-0 10 20 30 40 60}"
SHARDS="${SHARDS:-16}"
GAMES_PER_SHARD="${GAMES_PER_SHARD:-1875}"   # 16 x 1875 = 30,000 games per arm
DEPTH=6                                   # the regime the drift was measured in
NODES=800
MATCH_GAMES="${MATCH_GAMES:-4000}"        # phase 2, per sigma
SEED="${SEED:-20260920}"                  # EPD shuffle seed, shared by all arms

PHASE="${1:-all}"

mkdir -p "$WORK"
cd "$WORK" || exit 1

# Symlink the runtime files in rather than running from learn/ itself: keeps six
# arms of PGN out of the chain directory, and an empty cwd means no main_bk.dat,
# so there is no way for book moves to leak into a measurement run.
for f in "$BIN" "$NET" "$STATE" "$EPD"; do
    [ -e "$f" ] || ln -sf "$LEARN/$f" "$f"
done

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

# ---------------------------------------------------------------------------
# Phase 1 -- sharpness sweep
# ---------------------------------------------------------------------------
phase1() {
log "phase 1: $SIGMAS  |  $SHARDS shards x $GAMES_PER_SHARD games = $((SHARDS*GAMES_PER_SHARD))/arm  |  d$DEPTH/$NODES nodes"
for sig in $SIGMAS; do
    out=$WORK/arm_s$sig
    if [ -f "$out.scan" ]; then log "sigma=$sig already scored, skipping"; continue; fi
    mkdir -p "$out"
    log "sigma=$sig generating..."
    for sh in $(seq 0 $((SHARDS-1))); do
        # Paired across arms: shard i always draws the SAME opening slice at
        # every sigma, so the arms differ only in the perturbation.
        # salt = shard index mirrors the intended per-actor deployment (a salt
        # fixed for a process is score-hash safe; a per-GAME salt is not).
        # --no-adjudication and --epd-shuffle are NOT optional here: they are
        # what selfplay_run.py passes in production.  Without the first, 84% of
        # games end by adjudication and the clean metrics are meaningless
        # (measured: draw 17%, mean 97 ply against production's 22.5% / 136) --
        # and adjudicated self-play is the runaway-decisiveness spiral that
        # online-stability rule 1 exists to forbid.  The shuffle seed is fixed
        # and shared across arms, so the arms stay paired on openings.
        TDLEAF_FREEZE=1 ./$BIN \
            --eval-noise "$sig" --eval-noise-salt "$sh" \
            --selfplay --games "$GAMES_PER_SHARD" \
            --depth "$DEPTH" --nodes "$NODES" --no-adjudication \
            --epd "$EPD" --epd-shuffle "$SEED" \
            --epd-offset "$sh" --epd-stride "$SHARDS" \
            --pgn-out "$out/s$sh.pgn" > "$out/s$sh.log" 2>&1 &
    done
    wait
    log "sigma=$sig scoring..."
    cat "$out"/*.pgn | mawk -f "$ARMS/eval_noise_scan.awk" > "$out.scan" 2>&1
    sed -i "1s/^/sigma=$sig /" "$out.scan"
    cat "$out.scan"
    gzip -f "$out"/*.pgn
done
log "phase 1 done"
}

# ---------------------------------------------------------------------------
# Phase 2 -- strength cost, noisy vs clean, at the generation depth
# ---------------------------------------------------------------------------
phase2() {
log "phase 2: noisy vs clean, $MATCH_GAMES games each, fixed depth $DEPTH"
for sig in $SIGMAS; do
    [ "$sig" = "0" ] && continue
    res=$WORK/match_s$sig.log
    if [ -f "$res" ] && grep -q "Finished match" "$res"; then
        log "sigma=$sig match already done, skipping"; continue
    fi
    log "sigma=$sig match..."
    python3 "$ARMS/../match.py" "$BIN" "$BIN" \
        --name1 "noise$sig" --name2 clean \
        --option1 "EvalNoise=$sig" --option2 "EvalNoise=0" \
        --depth1 "$DEPTH" --depth2 "$DEPTH" \
        -n "$MATCH_GAMES" -c "$SHARDS" \
        --openings "$EPD" --fischer-random \
        --pgn-out "$WORK/match_s$sig.pgn" > "$res" 2>&1
    # match.py logs periodic interim Elo blocks -- at 1000 games into a match
    # one read -47.7 against a final of -91.7 (7.12.4).  The result is the LAST
    # such block, which sits just ABOVE "Finished match", not below it.
    grep -E '^(Elo:|Games:)' "$res" | tail -2
done
log "phase 2 done"
}

case "$PHASE" in
    phase1) phase1 ;;
    phase2) phase2 ;;
    all)    phase1; phase2 ;;
    *) echo "usage: $0 [phase1|phase2|all]"; exit 1 ;;
esac

echo
echo "================ SUMMARY ================"
echo "baseline for comparison (production legs, d6/800, 1M games each):"
echo "  1e6 cum: draw 24.23%  meanply 140.0   <- the un-drifted state"
echo "  6e6 cum: draw 22.48%  meanply 135.6   <- the drifted state"
echo "(context only -- the comparator is the sigma=0 arm below, which is paired"
echo " on openings with every other arm and uses the SAME net.)"
echo
for sig in $SIGMAS; do [ -f "$WORK/arm_s$sig.scan" ] && cat "$WORK/arm_s$sig.scan"; done
echo
for sig in $SIGMAS; do
    [ "$sig" = "0" ] && continue
    [ -f "$WORK/match_s$sig.log" ] || continue
    printf "sigma=%-3s strength vs clean: %s   [%s]\n" "$sig" \
        "$(grep -E '^Elo:' "$WORK/match_s$sig.log" | tail -1)" \
        "$(grep -E '^Games:' "$WORK/match_s$sig.log" | tail -1)"
done
