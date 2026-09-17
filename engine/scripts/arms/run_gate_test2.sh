#!/bin/bash
# CORRECTION to run_gate_test.sh.  That script's "gateU" arm was NOT ungated:
# train.py applies --bt-quiet-cp during corpus ASSEMBLY (train.py:380), so the
# out-of-gate rows were dropped and gateU assembled 11,731,426 rows that are
# 100% in-gate (verified: max|cp-gate| = 60, 0% over).  It is a ROW-COUNT
# control, not a gate control.  --bt-quiet-cp 0 disables the assembly filter.
#
# gateW is the real ungated arm: same corpus_U.tsv, same 20,122,155-row budget
# as gateG, but --bt-quiet-cp 0 so the 41.7% out-of-gate rows survive assembly.
#
# Resulting 3-point design:
#   gateG  20.1M rows, 100% in-gate   -> damage -119.9 +- 9.8   (power check: PASSED)
#   gateU  11.7M rows, 100% in-gate   -> isolates ROW COUNT
#   gateW  20.1M rows,  58.3% in-gate -> isolates THE GATE against gateG
cd /home/homand/Leaf/engine/learn || exit 1
GT=/home/homand/Leaf/engine/learn/gatetest

until grep -q "GATE TEST DONE" gate_driver.log 2>/dev/null; do sleep 60; done
echo "=== $(date '+%F %T')  prior test done, starting gateW ==="

python3 train.py --tag gateW --continue m260720-7e6g --skip-online \
    --state m260720.tdleaf.bin-lrfc0-online --corpus "$GT/corpus_U.tsv" \
    --corpus-window 0 --corpus-rows 20122155 --epochs 1 --bt-quiet-cp 0 \
    --gauntlet-anchors Leaf_vclassic_eval > hand_gateW.log 2>&1
echo "=== $(date '+%F %T')  CONSOLIDATE gateW done rc=$? ==="
grep -m1 "positions assembled" hand_gateW.log

echo "=== $(date '+%F %T')  DIRECT MATCH gateW vs gateG ==="
python3 /home/homand/Leaf/engine/scripts/match.py \
    Leaf_vgateW-final Leaf_vgateG-final \
    -n 1000 -c 8 -tc 3+0.05 --openings training_openings.epd --fischer-random \
    --pgn-out gate_W_vs_G.pgn > gate_W_vs_G.log 2>&1

echo "=== $(date '+%F %T')  5k ONLINE from gateW ==="
python3 train.py --tag gateWon --continue m260720-7e6g \
    --state gateW_final.tdleaf.bin \
    --games 5000 --depth 8 --nodes 4000 --concurrency 15 \
    --ladder 5000 --skip-train --seed 77881733 > hand_gateWon.log 2>&1
cp gateWon_work/gateWon-ladder-5000g.nnue gateWon5k.nnue || exit 1
( cd /home/homand/Leaf/engine/run && \
  perl comp.pl gateWon5k NNUE=1 NNUE_NET=gateWon5k.nnue OVERWRITE >/dev/null 2>&1 )
cp /home/homand/Leaf/engine/run/Leaf_vgateWon5k . || exit 1
echo "=== $(date '+%F %T')  rating gateW damage ==="
python3 /home/homand/Leaf/engine/scripts/match.py \
    Leaf_vgateWon5k Leaf_vgateW-final \
    -n 1000 -c 8 -tc 3+0.05 --openings training_openings.epd --fischer-random \
    --pgn-out gate_gateW_dmg.pgn > gate_gateW_dmg.log 2>&1
rm -f Leaf_vgateWon5k /home/homand/Leaf/engine/run/Leaf_vgateWon5k gateWon5k.nnue
echo "=== $(date '+%F %T')  GATE TEST 2 DONE ==="
