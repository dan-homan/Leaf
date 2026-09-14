#!/bin/bash
# fc0_w LR recalibration arm: TDLEAF_ADAM_LR0 0.005 -> 0.0014 (0.00035 x RMS,
# matching ft_w's 0.00034 and psqt_w's 0.00036 -- the other two STATIONARY
# sections).  FC1 split off to its own constant and held at 0.005, so fc0_w is
# the single changed variable against ladder-ctl.
#
# Protocol is byte-for-byte ladder-ctl's (run_handoff.sh gen()): same head,
# same 30k games, same depth/nodes/concurrency, same --ladder 1000 --skip-train,
# same 7 rating points, same 3+0.05 idle-box gauntlet vs the chain head.
# SEED-PAIRED to ladder-ctl (77881733) -- arm-to-arm seed variance is ~26 Elo
# (7.11.9) and pairing removes it.
cd /home/homand/Leaf/engine/learn || exit 1
tag=lrfc0

echo "=== $(date '+%F %T')  GEN $tag (fc0_w LR 0.005 -> 0.0014, seed-paired to ladder-ctl) ==="
python3 train.py --tag "$tag" --continue m260720-7e6g \
    --games 30000 --depth 8 --nodes 4000 --concurrency 15 \
    --ladder 1000 --skip-train --seed 77881733 > "hand_${tag}.log" 2>&1
rc=$?
echo "=== $(date '+%F %T')  GEN $tag done rc=$rc ==="
cp -f tdleaf_telemetry.log "${tag}_steps.log" 2>/dev/null

# Rate only on an idle box: the 3+0.05 protocol is not comparable across
# machine load (7.8), and the -123/-125/-131/-149 series was all measured idle.
while pgrep -f "run_handoff.sh|train\.py --tag|scripts/match\.py" > /dev/null; do sleep 60; done
echo "=== $(date '+%F %T')  box idle, starting ladder ratings ==="

for g in 1000 2000 3000 5000 10000 20000 30000; do
    src="${tag}_work/${tag}-ladder-${g}g.nnue"
    net="lf${g}.nnue"
    [ -f "$src" ] || { echo "MISSING $src"; continue; }
    cp "$src" "$net"
    ( cd /home/homand/Leaf/engine/run && \
      perl comp.pl "lf${g}" NNUE=1 "NNUE_NET=${net}" OVERWRITE >/dev/null 2>&1 )
    cp "/home/homand/Leaf/engine/run/Leaf_vlf${g}" . || { echo "BUILD FAILED $g"; continue; }
    echo "=== $(date '+%F %T')  rating ${g}g ==="
    python3 /home/homand/Leaf/engine/scripts/match.py \
        "Leaf_vlf${g}" Leaf_vm260720-7e6g-final \
        -n 1000 -c 8 -tc 3+0.05 --openings training_openings.epd --fischer-random \
        --pgn-out "lrfc0_${g}g.pgn" > "lrfc0_${g}g.log" 2>&1
    rm -f "Leaf_vlf${g}" "/home/homand/Leaf/engine/run/Leaf_vlf${g}" "$net"
done
echo "=== $(date '+%F %T')  LRFC0 ARM DONE ==="
