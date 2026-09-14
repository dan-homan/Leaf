#!/bin/bash
# ONLINE-FROM-ONLINE arm.  Tests D. Homan's hypothesis that the ~-125 online
# damage is a HANDOFF artifact of the offline pass, not a cost of online play.
#
# Every leg to date starts its online phase from an OFFLINE-TRAINED net
# (<prev>_final).  This arm starts from an ONLINE endpoint instead
# (m260720.tdleaf.bin-lrfc0-online, i.e. lrfc0's own 30k online net) and is
# otherwise identical to lrfc0: same binary, same seed (77881733), same 30k
# games, same 7 ladder points, same 3+0.05 idle-box gauntlet.  The ONLY thing
# that differs is whether the starting net came out of an offline pass.
#
# Control: lrfc0's own -129.8 +- 9.6 at 30k (from the offline-trained start).
#   ~0     -> handoff confirmed; the -125 is the price of the offline pass
#   ~-125  -> online play damages regardless of starting point
#
# Rating opponent is Leaf_vlrfc030k, built from lrfc0-ladder-30000g.nnue --
# i.e. the arm's own starting net, so the ladder reads damage relative to it.
cd /home/homand/Leaf/engine/learn || exit 1
tag=onon

echo "=== $(date '+%F %T')  GEN $tag (online-from-online, seed-paired to lrfc0) ==="
python3 train.py --tag "$tag" --continue m260720-7e6g \
    --state m260720.tdleaf.bin-lrfc0-online \
    --games 30000 --depth 8 --nodes 4000 --concurrency 15 \
    --ladder 1000 --skip-train --seed 77881733 > "hand_${tag}.log" 2>&1
rc=$?
echo "=== $(date '+%F %T')  GEN $tag done rc=$rc ==="
cp -f tdleaf_telemetry.log "${tag}_steps.log" 2>/dev/null

# train.py logs the piece-value canary AFTER generation, so this is where the
# net ENDED, not a check on where it started.  The start state is evidenced by
# train.py's own "promoting m260720.tdleaf.bin-lrfc0-online" line.  lrfc0
# ended at P=119 N=382 B=407 R=611 Q=1203; large drift from that is the
# outcome-imbalance canary, not a setup error.
echo "--- post-generation piece-value canary (lrfc0 ended at P=119 Q=1203) ---"
grep "piece-value canary" "hand_${tag}.log"

while pgrep -f "run_handoff.sh|train\.py --tag|scripts/match\.py" > /dev/null; do sleep 60; done
echo "=== $(date '+%F %T')  box idle, starting ladder ratings ==="

for g in 1000 2000 3000 5000 10000 20000 30000; do
    src="${tag}_work/${tag}-ladder-${g}g.nnue"
    net="on${g}.nnue"
    [ -f "$src" ] || { echo "MISSING $src"; continue; }
    cp "$src" "$net"
    ( cd /home/homand/Leaf/engine/run && \
      perl comp.pl "on${g}" NNUE=1 "NNUE_NET=${net}" OVERWRITE >/dev/null 2>&1 )
    cp "/home/homand/Leaf/engine/run/Leaf_von${g}" . || { echo "BUILD FAILED $g"; continue; }
    echo "=== $(date '+%F %T')  rating ${g}g ==="
    python3 /home/homand/Leaf/engine/scripts/match.py \
        "Leaf_von${g}" Leaf_vlrfc030k \
        -n 1000 -c 8 -tc 3+0.05 --openings training_openings.epd --fischer-random \
        --pgn-out "onon_${g}g.pgn" > "onon_${g}g.log" 2>&1
    rm -f "Leaf_von${g}" "/home/homand/Leaf/engine/run/Leaf_von${g}" "$net"
done
echo "=== $(date '+%F %T')  ONON ARM DONE ==="
