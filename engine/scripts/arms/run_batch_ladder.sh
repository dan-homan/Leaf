#!/bin/bash
# THE SIGMA LADDER (7.15).  Batch size 8/16/32/64 at MATCHED ADAM STEPS.
#
# Sigma is the last surviving mechanism (7.14).  Batch size is the intervention:
# gradients are SUMMED and Adam normalises the step, so B does not change step
# SIZE -- it changes samples-per-step, i.e. the gradient-noise covariance.
#
# 6.15's batch-16 arm was confounded: at fixed games it also HALVED the Adam
# step count (18,750 vs 37,500), so "cleaner steps" and "fewer steps" moved
# together and displacement came out invariant.  Here every arm runs exactly
# 1000 Adam steps -- Adam's own equilibration time 1/(1-beta2), and >3x the
# ~300 steps damage needs to equilibrate (7.13.1) -- so samples-per-step is
# the ONLY variable.
#
# Guards against the other confounds enumerated before launch:
#   - --opt-reset is NEVER passed.  It would activate TDLEAF_ADAM_WARMUP (1000
#     steps), which at these step counts would span an entire arm.
#   - TDLEAF_FT_SESSION_WARMUP (100 steps, fires every session) is a MATCHED
#     fraction of the run because steps are matched.  At matched games it would
#     have been 2.7% of steps at B=8 and 43% at B=128.
#   - TDLEAF_BATCH_SIZE is compile-time and train.py always builds the binary
#     named train_hl_a, so arms CANNOT run concurrently and a stale binary would
#     silently run the wrong B.  Each arm rebuilds, and the learner's own
#     startup banner is checked; mismatch aborts the arm.
#   - the accumulated gradient norm grows as sqrt(B) against a fixed
#     TDLEAF_GRAD_CLIP_NORM = 1.0 (0.147 at B=8, 0.213 at B=16, ~0.42 expected
#     at B=64).  Clip telemetry is saved per arm; if clips fire at B=64 the top
#     of the ladder is in a different regime and must be reported as such.
#
# All arms start from the chain head (7e6g_final, via --continue) and are rated
# against it, so the number is handoff damage, seed-paired at 77881733.
cd /home/homand/Leaf/engine/learn || exit 1
STEPS=1000
HEAD=Leaf_vm260720-7e6g-final

for B in 8 16 32 64; do
    N=$((STEPS * B))
    tag="bs$B"
    echo "=== $(date '+%F %T')  ARM $tag : batch=$B, $N games, $STEPS Adam steps ==="

    ( cd /home/homand/Leaf/engine/run && \
      perl comp.pl train_hl_a NNUE=1 NNUE_NET=m260720.nnue TDLEAF=1 \
           TDLEAF_LOG_STEP_CLIPS=1 "TDLEAF_BATCH_SIZE_DEFAULT=$B" OVERWRITE ) \
      > "build_$tag.log" 2>&1 || { echo "BUILD FAILED $tag"; continue; }
    cp /home/homand/Leaf/engine/run/Leaf_vtrain_hl_a . || { echo "COPY FAILED $tag"; continue; }

    banner=$(echo quit | ./Leaf_vtrain_hl_a 2>&1 | grep -o "batch=[0-9]*" | head -1)
    if [ "$banner" != "batch=$B" ]; then
        echo "ABORT $tag: binary reports $banner, expected batch=$B"; continue
    fi
    echo "  verified: $banner"

    rm -f tdleaf_telemetry.log
    python3 train.py --tag "$tag" --continue m260720-7e6g \
        --games "$N" --depth 8 --nodes 4000 --concurrency 15 \
        --ladder "$N" --skip-train --seed 77881733 > "hand_$tag.log" 2>&1
    echo "=== $(date '+%F %T')  GEN $tag done rc=$? ==="
    cp -f tdleaf_telemetry.log "${tag}_steps.log" 2>/dev/null
    grep -m1 "TDLeaf config" "${tag}_work/traj/learner.log" 2>/dev/null

    src="${tag}_work/${tag}-ladder-${N}g.nnue"
    [ -f "$src" ] || { echo "MISSING $src"; continue; }
    cp "$src" "${tag}e.nnue"
    ( cd /home/homand/Leaf/engine/run && \
      perl comp.pl "${tag}e" NNUE=1 "NNUE_NET=${tag}e.nnue" OVERWRITE >/dev/null 2>&1 )
    cp "/home/homand/Leaf/engine/run/Leaf_v${tag}e" . || { echo "RATE BUILD FAILED $tag"; continue; }
    echo "=== $(date '+%F %T')  rating $tag vs chain head ==="
    python3 /home/homand/Leaf/engine/scripts/match.py \
        "Leaf_v${tag}e" "$HEAD" \
        -n 1000 -c 8 -tc 3+0.05 --openings training_openings.epd --fischer-random \
        --pgn-out "bsl_${tag}.pgn" > "bsl_${tag}.log" 2>&1
    rm -f "Leaf_v${tag}e" "/home/homand/Leaf/engine/run/Leaf_v${tag}e" "${tag}e.nnue"
done
echo "=== $(date '+%F %T')  BATCH LADDER DONE ==="
