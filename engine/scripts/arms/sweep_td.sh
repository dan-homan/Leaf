#!/bin/zsh
# 6-arm sweep of the outcome-weight ceilings under the new distance-decayed
# result weight  w = lambda_eff * td_lambda^(N-ply),  td_lambda = 0.98
# (the trainer default = TDLEAF_LAMBDA; no flag needed).
#
# Calibration: measured corpus-mean decay = 0.502 (identical for roots and
# leaves, mean gap ~42 recorded plies).  The flat sweep's plateau was at
# corpus-mean outcome weight ~0.19-0.33, so the matching diagonal ceiling is
# lambda ~0.4-0.65 -> the diagonal arms {0.3, 0.5, 0.7, 1.0} bracket the
# predicted optimum (~0.5) and include the pure lambda-return end (1.0).
# The two off-diagonal arms sit at the same corpus-mean weight as the 0.5
# diagonal arm; if they land on the diagonal's curve, "mean weight is the
# knob" holds under decay too.
#
# Same protocol as the flat sweep: from the LIVE (iter2-online) state, full
# 57M iter2 dump corpus, K=220, unsharded, 1 epoch, ladder-only 1000 games
# at 1+0.01 vs Leaf_vbtsp-final (comparable with the sw* flat arms and the
# iter2s2 epoch ladder).  ~50 min/arm => ~5 h total.
#
# Rerun after an interrupted arm: delete that arm's <tag>_work dir first.

set -e
cd /Users/homand/Leaf/engine/learn

CORPUS=()
for f in iter2_work/iter2.*.tsv; do CORPUS+=(--corpus $f); done

run_arm() {
  local L=$1 F=$2
  local TAG="tdL${L/./}F${F/./}"      # 0.5/0.5 -> tdL05F05
  echo "=== $TAG : bt-lambda=$L  bt-leaf-lambda=$F  (td_lambda=0.98 default) ==="
  python3 hybrid_loop.py --tag $TAG --skip-online --shards 1 --epochs 1 \
      --bt-K 220 --bt-lambda $L --bt-leaf-lambda $F \
      $CORPUS \
      --gauntlet-epochs --no-final-gauntlet --gauntlet Leaf_vbtsp-final \
      --epoch-games 1000
  rm -f ${TAG}_work/shard_0.tsv       # reclaim ~4 GB per arm
}

# Diagonal (mean-weight axis)
run_arm 0.3 0.3
run_arm 0.5 0.5
run_arm 0.7 0.7
run_arm 1.0 1.0
# Symmetry checks at the diagonal-0.5 mean
run_arm 0.3 0.7
run_arm 0.7 0.3

echo "=== sweep complete ==="
for T in tdL03F03 tdL05F05 tdL07F07 tdL10F10 tdL03F07 tdL07F03; do
  P=match_${T}-ep1_vs_btsp-final.pgn
  [ -f $P ] && echo "$T: $P"
done
