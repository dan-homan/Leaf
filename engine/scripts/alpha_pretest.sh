#!/bin/bash
# alpha_pretest.sh — weight-level pre-test of TDLEAF_STACK_NORM_ALPHA.
#
# Runs one short actor/learner generation per alpha from an IDENTICAL seed and
# diffs seed -> post-online per material bucket.  No gauntlet: the question is
# whether the per-game per-stack normalisation flattens the endgame
# over-coherence measured in docs/Online_Learning_Investigation.md 6.4
# (online PSQT med|dw|/sqrt(updates) 14.85 in bucket 0 vs 7.94 in bucket 7).
#
# Sizing (measured, see 6.7): the b0/b7 violence ratio does NOT scale with game
# count — it has a RESOLUTION FLOOR somewhere between 20k and 100k games and is
# flat above it (four d6 iterations from 100k to 1M games all read 1.32-1.45).
# So >= 100k games/arm, and arms above the floor are comparable to each other
# and to the chain's historical iterations.  Default depth 6: the signature is
# weaker there than at d8 (~1.4 vs ~1.9) but its baseline is far tighter
# (sigma ~0.06 over four runs) at ~5x the throughput.  Validate the winning
# alpha at d8 with a foreign anchor afterwards.
#
# Usage (from engine/learn/):
#   bash ../scripts/alpha_pretest.sh [GAMES] [DEPTH] [ACTORS]
#   ALPHAS="0 1" bash ../scripts/alpha_pretest.sh 20000 8 11

set -euo pipefail

GAMES=${1:-200000}
DEPTH=${2:-6}
ACTORS=${3:-11}
ALPHAS=${ALPHAS:-"0 0.5 1"}
# Actor weight-refresh cadence.  train.py production default is 1000; a
# shorter interval means actors track the learner more closely, which reduces
# score staleness and therefore TD-error size — a confound for any violence
# measurement compared against production iterations.  Match production unless
# deliberately probing this.
GAMES_PER_ACTOR=${GAMES_PER_ACTOR:-1000}

BINARY=Leaf_valphapre
# The .tdleaf.bin carries a content hash of the .nnue it was trained against and
# REFUSES TO LOAD against any other one — and the engine then plays on happily
# from freshly-derived shadow weights, so the run looks fine and measures
# nothing.  The chain's states are all anchored to the BASE net m260720.nnue,
# not to the baked m260720-*_final.nnue exports.  Hence: net = base net, state
# = base-net name, seeded by copying the chain head onto it.  The guard below
# catches any recurrence.
NET=m260720.nnue
STATE=m260720.tdleaf.bin
# Seed state.  Overridable: the endgame-violence profile is SEED-DEPENDENT
# (6.8), so which state an arm starts from is a first-class variable, not a
# constant.  Must be anchored to $NET or it will refuse to load (see above).
SEED_STATE=${SEED_STATE:-m260720-3e6g_final.tdleaf.bin}
EPD=training_openings.epd
OUT=${OUT:-alphapre_work}

LEARN_DIR=$(pwd)
SCRIPTS=$(cd "$(dirname "$0")" && pwd)
ENGINE_RUN=$LEARN_DIR/../run

for f in "$NET" "$SEED_STATE" "$EPD"; do
    [ -f "$f" ] || { echo "missing $f in $LEARN_DIR" >&2; exit 1; }
done
[ -x "$ENGINE_RUN/$BINARY" ] || {
    echo "missing $ENGINE_RUN/$BINARY — build it with:" >&2
    echo "  cd ../run && perl comp.pl alphapre NNUE=1 TDLEAF=1 NNUE_NET=$NET OVERWRITE" >&2
    exit 1; }

mkdir -p "$OUT"
echo "alpha pre-test: $GAMES games, depth $DEPTH, $ACTORS actors, alphas: $ALPHAS"
echo "refresh: $GAMES_PER_ACTOR games/actor   out: $OUT"
echo "seed state: $SEED_STATE   base net: $NET"
echo

for A in $ALPHAS; do
    ARM="$OUT/a$A"
    echo "=== alpha=$A -> $ARM ==="
    rm -rf "$ARM"; mkdir -p "$ARM"
    # Bookless arm dir on purpose: run/ carries main_bk.dat, and an opening book
    # would collapse the opening diversity this measurement depends on.
    cp "$ENGINE_RUN/$BINARY" "$ARM/"
    cp "$NET" "$ARM/"
    cp "$SEED_STATE" "$ARM/$STATE"     # every arm starts from a fresh seed copy
    [ -f "$ENGINE_RUN/search.par" ] && cp "$ENGINE_RUN/search.par" "$ARM/"
    ln -sf "$LEARN_DIR/$EPD" "$ARM/$EPD"

    ( cd "$ARM" && TDLEAF_STACK_NORM_ALPHA=$A python3 "$SCRIPTS/selfplay_run.py" \
        --binary "$BINARY" --epd "$EPD" \
        --actors "$ACTORS" --depth "$DEPTH" \
        --games-per-actor "$GAMES_PER_ACTOR" --total-games "$GAMES" \
        --traj-dir traj --refresh-scores --delete-consumed \
        --seed 20260813 > run.log 2>&1 )

    # Hard guards: a silently-unloaded state or a wrong alpha invalidates the arm.
    if grep -rq "Refusing to load" "$ARM"/run.log "$ARM"/traj/*.log 2>/dev/null; then
        echo "  FATAL: state failed to load (.nnue content-hash mismatch)." >&2
        grep -rh -A1 "Refusing to load" "$ARM"/run.log "$ARM"/traj/*.log 2>/dev/null | head -3 >&2
        exit 1
    fi
    # Anchored at EOL: the banner ends with the alpha, and an unanchored "0"
    # would also match a "0.5" banner.
    if ! grep -rhq "stack_norm_alpha=$A\$" "$ARM"/traj/*.log "$ARM"/run.log 2>/dev/null; then
        echo "  FATAL: no process reported stack_norm_alpha=$A." >&2
        exit 1
    fi
    echo "  ok (state loaded, alpha confirmed in banner)"
done

echo
echo "############ RESULTS ############"
for A in $ALPHAS; do
    ARM="$OUT/a$A"
    echo
    echo "================= alpha=$A ================="
    # Draw-rate canary — DEPTH-DEPENDENT: ~35-40% at d8, ~26-28% at d6 (the
    # chain's d6 iterations ran 26-27%).  A collapse invalidates the arm no
    # matter what the bucket profile says.
    # No `| head -N` here: under `set -euo pipefail` an early-closing head sends
    # SIGPIPE to the loop and kills the whole script mid-report (it did).
    echo "--- generation health: final +W =D -L per actor (first 4) ---"
    n=0
    for L in "$ARM"/traj/actor_*.log; do
        [ -f "$L" ] || continue
        n=$((n + 1)); [ "$n" -le 4 ] || continue
        printf '  %-9s %s\n' "$(basename "$L" .log)" \
            "$(grep -oE '\+[0-9]+ =[0-9]+ -[0-9]+' "$L" | tail -1)"
    done
    echo "--- seed -> post-online, per material bucket ---"
    # 3rd arg = post-online again: the off/proj/cos columns are meaningless here
    # (there is no offline phase); the on/upd column is the measurement.
    python3 "$SCRIPTS/bucket_phase_analysis.py" \
        "$SEED_STATE" "$ARM/$STATE" "$ARM/$STATE" 2>&1 | sed 's/^/  /'
    echo "--- headline: b0/b7 violence ratio (control ~1.4 at d6; 1.0 = flat) ---"
    python3 - "$SEED_STATE" "$ARM/$STATE" <<'PYEOF' | sed 's/^/  /'
import sys, importlib.util, numpy as np
spec = importlib.util.spec_from_file_location(
    "cnl", "/Users/homand/Leaf/engine/scripts/compare_nnue_learning.py")
cnl = importlib.util.module_from_spec(spec); spec.loader.exec_module(cnl)
s = cnl.read_tdleaf_fc(sys.argv[1]); o = cnl.read_tdleaf_fc(sys.argv[2])
ia = {fi: k for k, fi in enumerate(s['ft_fi'])}
ib = {fi: k for k, fi in enumerate(o['ft_fi'])}
fis = [fi for fi in s['ft_fi'] if fi in ib]
ka = np.array([ia[f] for f in fis]); kb = np.array([ib[f] for f in fis])
d = o['psqt_w'][kb] - s['psqt_w'][ka]
m = o['psqt_cnt'][kb].astype(np.int64) - s['psqt_cnt'][ka].astype(np.int64)
v = []
for b in range(8):
    sel = m[:, b] > 0
    v.append(float(np.median(np.abs(d[sel, b]) / np.sqrt(m[sel, b]))) if sel.any() else float('nan'))
print("per-bucket: " + " ".join(f"{x:.2f}" for x in v))
print(f"b0/b7 = {v[0]/v[7]:.2f}    spread max/min = {max(v)/min(v):.2f}")
PYEOF
    echo "--- aggregate displacement (guards against 'alpha is just an LR cut') ---"
    python3 "$SCRIPTS/diff_tdleaf_checkpoints.py" \
        "$SEED_STATE" "$ARM/$STATE" 2>&1 | tail -18 | sed 's/^/  /'
done

cat <<'NOTE'

############ HOW TO READ THIS ############
PASS  : the online `on/upd` column flattens — bucket0:bucket7 ratio drops from
        whatever the alpha=0 arm shows toward ~1 — AND the aggregate
        displacement stays the same order of magnitude as alpha=0.
FAIL-1: profile flattens but aggregate displacement collapses -> alpha is acting
        as a blanket LR cut, not a decorrelator.  Re-run the winning alpha with
        the section LRs scaled up to match alpha=0's displacement before
        believing anything.
FAIL-2: profile unchanged -> within-game per-stack coherence is not the
        mechanism; go back to 6.5 and test the id_weight amplifier instead.
Also confirm the draw rate held its depth's band in every arm (~35-40% at d8,
~26-28% at d6): a quiet run flatters every other number on the page.
Only a PASS earns a full iteration with a foreign-anchor gauntlet.
NOTE
