#!/bin/bash
# ---------------------------------------------------------------------------
# eval_noise_quiet.sh -- does eval_noise recover the QUIET FRACTION?
#
# The sharpness sweep (eval_noise_sweep.sh) measures draw rate and game length.
# This measures the OTHER symptom of the m260916 drift, and the one that bears
# directly on how much positional training data survives: the fraction of root
# positions passing the 60cp quiet gate fell 0.562 -> 0.499 across legs 2e6 to
# 6e6 at constant search depth.  The gate keeps a root row only when
# |root_static - root_search| <= 60cp, so as the net sharpens its own games it
# deletes a growing share of its own quiet corpus.
#
# ⚠️ CONTAMINATED BEFORE 2026-09-23 (change_log 2026_09_23a).  This header
# used to claim the root static and the search score both carry the offset and
# cancel.  They did not: the static is recorded CLEAN (nnue_evaluate), the
# search score was noisy, so the gate compared clean against noisy and every
# q@ figure this produced is biased DOWN under noise.  Since the fix the actor
# subtracts the PV leaf's offset from the root score, so the gate now compares
# clean static against the clean value of the line the noisy search chose --
# which is the quantity that actually becomes the offline label.
#
# WIDE DUMP.  TDLEAF_DUMP_QUIET_CP is left at its default (1000, effectively
# open) instead of production's 60, so every row lands with its `gate` column
# and any narrower gate is an offline filter (the R4 discipline).  That gives
# the gate-width curve for free and makes the arms perfectly paired -- same
# games, same labels, only the admitted row population differs.
#
# Weights FROZEN; nothing trains, nothing touches the live chain state.
# Generation is depth/node limited, not clock limited, so running this
# alongside another job changes wall time but not results.
#
# Usage:  bash eval_noise_quiet.sh
# ---------------------------------------------------------------------------
set -u

LEARN=/home/homand/Leaf/engine/learn
ARMS=/home/homand/Leaf/engine/scripts/arms
WORK="${WORK:-$LEARN/evalnoise_quiet}"
BIN=Leaf_vevalnoise
NET=m260916.nnue
STATE=m260916-7e6g_final.tdleaf.bin
EPD=training_openings.epd

SIGMAS="${SIGMAS:-0 10 20 40}"
SHARDS="${SHARDS:-8}"
GAMES_PER_SHARD="${GAMES_PER_SHARD:-1000}"   # 8 x 1000 = 8000 games per arm
DEPTH=6
NODES=800
SEED="${SEED:-20260920}"                     # same seed as the sharpness sweep

mkdir -p "$WORK"
cd "$WORK" || exit 1
for f in "$BIN" "$NET" "$STATE" "$EPD"; do
    [ -e "$f" ] || ln -sf "$LEARN/$f" "$f"
done

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

log "quiet-fraction arm: $SIGMAS | $SHARDS x $GAMES_PER_SHARD = $((SHARDS*GAMES_PER_SHARD))/arm | d$DEPTH/$NODES"
for sig in $SIGMAS; do
    out=$WORK/q_s$sig
    if [ -f "$out.done" ]; then log "sigma=$sig already generated, skipping"; continue; fi
    rm -rf "$out"; mkdir -p "$out"
    log "sigma=$sig generating (wide dump)..."
    for sh in $(seq 0 $((SHARDS-1))); do
        TDLEAF_FREEZE=1 TDLEAF_DUMP_TSV="$out/d$sh" ./$BIN \
            --eval-noise "$sig" --eval-noise-salt "$sh" \
            --selfplay --games "$GAMES_PER_SHARD" \
            --depth "$DEPTH" --nodes "$NODES" --no-adjudication \
            --epd "$EPD" --epd-shuffle "$SEED" \
            --epd-offset "$sh" --epd-stride "$SHARDS" \
            > "$out/s$sh.log" 2>&1 &
    done
    wait
    touch "$out.done"
    log "sigma=$sig generated: $(ls "$out" | grep -c root) root files"
done

log "scoring"
python3 "$ARMS/eval_noise_quiet_score.py" "$WORK" $SIGMAS
log "done"
