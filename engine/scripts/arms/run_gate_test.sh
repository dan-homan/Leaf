#!/bin/bash
# THE UNGATED TEST (7.13.6 discriminator).
#
# Hypothesis A ("different targets"): the offline pass fits the FC head on root
# positions gated to |cp-gate| <= 60, while the online phase fits ungated PV
# leaves -- a genuine target difference that the handoff must traverse.
# Hypothesis B ("same target, different variance"): Sigma, i.e. 140k FC params
# estimated from 8 correlated games.
#
# Test: consolidate the SAME state from two corpora that are identical in row
# count (20,122,155), sources and construction, differing ONLY in the position
# distribution -- G is 100% in-gate, U is the full wide dump (58.3% in-gate,
# mean |cp| 263 vs 163).  Then measure handoff damage out of each with a 5k
# online arm (7.11.8's cheap protocol).
#
#   damage_U << damage_G  -> the gate creates a target difference (A)
#   damage_U ~= damage_G  -> estimator variance (B); lever is decorrelation
#
# NOTE: the standard 190M chain corpus CANNOT be used -- train.py assembles it
# already cut at 60 (max|cp-gate| = 60, zero rows over), so the wide rows were
# never written.  These corpora are pooled from the six surviving WIDE raw root
# dumps (gate column present, dumped at TDLEAF_DUMP_QUIET_CP=1000), with gids
# renumbered per source.  train.py never passes --bt-quiet-cp to the trainer --
# it is a corpus-assembly knob only -- so the gate is controlled at file level.
cd /home/homand/Leaf/engine/learn || exit 1
GT=/home/homand/Leaf/engine/learn/gatetest
ROWS=20122155
STATE=m260720.tdleaf.bin-lrfc0-online

cons () {   # $1 = tag, $2 = corpus
    echo "=== $(date '+%F %T')  CONSOLIDATE $1 from $2 ==="
    python3 train.py --tag "$1" --continue m260720-7e6g --skip-online \
        --state "$STATE" --corpus "$2" \
        --corpus-window 0 --corpus-rows $ROWS --epochs 1 \
        --gauntlet-anchors Leaf_vclassic_eval > "hand_$1.log" 2>&1
    echo "=== $(date '+%F %T')  CONSOLIDATE $1 done rc=$? ==="
}

damage () { # $1 = arm tag (gateG/gateU)
    local t="$1on"
    echo "=== $(date '+%F %T')  5k ONLINE from $1 ==="
    python3 train.py --tag "$t" --continue m260720-7e6g \
        --state "$1_final.tdleaf.bin" \
        --games 5000 --depth 8 --nodes 4000 --concurrency 15 \
        --ladder 5000 --skip-train --seed 77881733 > "hand_$t.log" 2>&1
    echo "=== $(date '+%F %T')  5k ONLINE $t done rc=$? ==="
    cp "${t}_work/${t}-ladder-5000g.nnue" "${t}5k.nnue" || return 1
    ( cd /home/homand/Leaf/engine/run && \
      perl comp.pl "${t}5k" NNUE=1 "NNUE_NET=${t}5k.nnue" OVERWRITE >/dev/null 2>&1 )
    cp "/home/homand/Leaf/engine/run/Leaf_v${t}5k" . || return 1
    echo "=== $(date '+%F %T')  rating $t 5k vs its own start (Leaf_v$1-final) ==="
    python3 /home/homand/Leaf/engine/scripts/match.py \
        "Leaf_v${t}5k" "Leaf_v$1-final" \
        -n 1000 -c 8 -tc 3+0.05 --openings training_openings.epd --fischer-random \
        --pgn-out "gate_${1}_dmg.pgn" > "gate_${1}_dmg.log" 2>&1
    rm -f "Leaf_v${t}5k" "/home/homand/Leaf/engine/run/Leaf_v${t}5k" "${t}5k.nnue"
}

cons gateG "$GT/corpus_G.tsv"
cons gateU "$GT/corpus_U.tsv"

echo "=== $(date '+%F %T')  DIRECT MATCH gateU vs gateG (relative strength) ==="
python3 /home/homand/Leaf/engine/scripts/match.py \
    Leaf_vgateU-final Leaf_vgateG-final \
    -n 1000 -c 8 -tc 3+0.05 --openings training_openings.epd --fischer-random \
    --pgn-out gate_U_vs_G.pgn > gate_U_vs_G.log 2>&1

damage gateG
damage gateU
echo "=== $(date '+%F %T')  GATE TEST DONE ==="
