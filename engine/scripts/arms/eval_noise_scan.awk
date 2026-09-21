# eval_noise_scan.awk -- score a Leaf self-play PGN for "how tactical are these
# games".  Reads one or more concatenated generation PGNs on stdin.
#
# CLEAN metrics depend only on the GAMES PLAYED and are unaffected by the
# eval_noise perturbation:
#     draw%, W%, L%, mean/median ply, short%  (games under 80 ply)
# These are the calibration targets.
#
# CONTAMINATED metrics read the engine's REPORTED score, which under
# --eval-noise includes the offset itself.  mean|cp| is shifted outright, and
# mean|dcp| picks up a spurious jump on every pawn move (the offset re-draws
# there -- roughly 15-20% of plies).  Printed for continuity with the
# nine-leg baseline table, NEVER as a calibration target at sigma > 0.
#
# Usage:  cat *.pgn | mawk -f eval_noise_scan.awk

/^\[Event/      { flush(); ns=0; ply=0; res=""; next }
/^\[Result /    { split($0,a,"\""); res=a[2]; next }
/^\[PlyCount /  { split($0,a,"\""); ply=a[2]+0; next }
/^\[/           { next }
/^$/            { next }
{
    s=$0
    while (match(s, /\{[+-][0-9]+\.[0-9]+\/[0-9]+/)) {
        tok = substr(s, RSTART+1, RLENGTH-1)
        split(tok, b, "/")
        ns++
        sc[ns] = b[1]+0          # own-POV score, pawns
        dp[ns] = b[2]+0          # achieved depth
        s = substr(s, RSTART+RLENGTH)
    }
}
function flush(   i,w,prev,d) {
    if (ns == 0) return
    games++
    if (res=="1/2-1/2") draws++; else if (res=="1-0") wins++; else if (res=="0-1") losses++
    plysum += ply
    plies[games] = ply
    if (ply < 80) shortg++
    prev = ""
    for (i=1; i<=ns; i++) {
        w = (i % 2 == 1) ? sc[i] : -sc[i]     # to white POV
        absum += (w<0 ? -w : w)
        depsum += dp[i]; nsc++
        if (prev != "") {
            d = w - prev; if (d<0) d=-d
            dsum += d; nd++
            if (d >= 0.5) swing50++
        }
        prev = w
    }
}
END {
    flush()
    if (games == 0) { print "no games"; exit 1 }
    n = asort_plies()
    med = plies_sorted[int((games+1)/2)]
    printf "games=%d | CLEAN draw%%=%.2f W%%=%.2f L%%=%.2f meanply=%.1f medianply=%d short%%=%.2f | depth=%.2f | CONTAM mean|cp|=%.1f mean|dcp|=%.2f swing50%%=%.2f\n", \
      games, 100*draws/games, 100*wins/games, 100*losses/games, plysum/games, med, 100*shortg/games, \
      depsum/nsc, 100*absum/nsc, 100*dsum/nd, 100*swing50/nd
}
# mawk has no asort(); simple insertion into an ordered copy is too slow for
# 30k games, so bucket-count instead (ply is a small non-negative integer).
function asort_plies(   i,v,c,idx) {
    for (i=1; i<=games; i++) cnt[plies[i]]++
    idx=0
    for (v=0; v<=1024; v++) {
        c = cnt[v]
        while (c-- > 0) { idx++; plies_sorted[idx]=v }
    }
    return idx
}
