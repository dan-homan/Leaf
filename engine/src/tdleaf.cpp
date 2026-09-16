// Leaf TDLeaf(λ) online learning — implementation
// Compiled only when TDLEAF=1 (included by Leaf.cc after nnue.cpp).

#include "define.h"

#if TDLEAF

#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <climits>
#include <unistd.h>     // getpid — leaf-dump per-process file naming
#include "chess.h"
#include "nnue.h"
#include "tdleaf.h"

// value[] lives in score.h (included earlier in the unity build).  Declared
// extern here so TDLEAF_SCORE_CLIP_PAWNS × max(value[PAWN], 100) can be
// evaluated at runtime.  Under the default NNUE_FIXED_PIECE_VALUES value[PAWN]
// stays at the classical 100, so the threshold is constant and does NOT stretch
// with the net's material scale — which is the point: the clip must not widen as
// PSQT drifts.  The runtime read and the max() floor matter only if that flag is
// turned off.
extern int value[7];

// ---------------------------------------------------------------------------
// tdleaf_check_env — startup guardrail + config banner for TDLEAF builds.
//
// Hard-errors if any TDLEAF_* environment variable outside the known allowlist
// is set.  The retired experimental knobs (blend/hybrid targets, online root
// learning, TDLEAF_LR_* sweeps, TDLEAF_FREEZE_PASSTHROUGH) were deleted, so a
// leftover or mistyped TDLEAF_* var must never silently alter a training run —
// it fails loudly instead.  Then prints the effective online-training config so
// every run's log records exactly what it trained with.
// ---------------------------------------------------------------------------
extern char **environ;

void tdleaf_check_env()
{
    static const char *const allowed[] = {
        "TDLEAF_FREEZE",          // frozen actor / generate-only play
        "TDLEAF_DUMP_TSV",        // corpus dump prefix
        "TDLEAF_DUMP_QUIET_CP",   // corpus dump quiet gate (cp)
        "TDLEAF_DUMP_MAX_CP",     // corpus dump |cp| cap
        "TDLEAF_CHECK_ACC",       // diagnostic: walked-vs-rebuilt accumulator check
        "TDLEAF_TRACE_UPDATE",    // diagnostic: per-record gradient trace file
    };
    int bad = 0;
    for (char **e = environ; e && *e; e++) {
        if (strncmp(*e, "TDLEAF_", 7) != 0) continue;
        size_t nlen = 0;
        while ((*e)[nlen] && (*e)[nlen] != '=') nlen++;
        bool ok = false;
        for (const char *name : allowed)
            if (strlen(name) == nlen && strncmp(*e, name, nlen) == 0) { ok = true; break; }
        if (!ok) {
            fprintf(stderr, "TDLeaf: unrecognized environment variable %.*s — "
                            "retired experimental knobs (blend/hybrid targets, "
                            "root learning, TDLEAF_LR_*, TDLEAF_FREEZE_PASSTHROUGH) "
                            "were removed.  Unset it (see docs/SIMPLIFICATION_PLAN.md).\n",
                    (int)nlen, *e);
            bad++;
        }
    }
    if (bad) {
        fprintf(stderr, "TDLeaf: refusing to start with %d unrecognized TDLEAF_* "
                        "variable(s).\n", bad);
        exit(1);
    }

    fprintf(stderr,
            "TDLeaf config: K=%.0f lambda=%.4f batch=%d grad_clip=%.2f wd=%.1e "
            "score_clip=%.1fxP id_var_sigma2=%.0f\n"
            "TDLeaf LR0: FC0=%.4g FC1=%.4g FC2=%.4g FC_bias=%.4g FT=%.4g FT_bias=%.4g PSQT=%.4g\n",
            (double)TDLEAF_K, (double)TDLEAF_LAMBDA, TDLEAF_BATCH_SIZE,
            (double)TDLEAF_GRAD_CLIP_NORM, (double)TDLEAF_WEIGHT_DECAY,
            (double)TDLEAF_SCORE_CLIP_PAWNS, (double)TDLEAF_ID_VAR_SIGMA2,
            (double)TDLEAF_ADAM_LR0, (double)TDLEAF_ADAM_FC1_LR0, (double)TDLEAF_ADAM_FC2_LR0,
            (double)TDLEAF_ADAM_FC_BIAS_LR0, (double)TDLEAF_ADAM_FT_LR0,
            (double)TDLEAF_ADAM_FT_BIAS_LR0, (double)TDLEAF_ADAM_PSQT_LR0);
}

// True when the leaf/root TSV dump is enabled (TDLEAF_DUMP_TSV env var).
// Cached once; consulted by tdleaf_record_ply to decide whether to snapshot
// the root position and compute its static eval (one extra nnue_evaluate
// per recorded ply — skipped entirely when dumping is off).
static bool tdleaf_dump_wanted()
{
    static int wanted = -1;
    if (wanted < 0) {
        const char *p = getenv("TDLEAF_DUMP_TSV");
        wanted = (p && *p) ? 1 : 0;
    }
    return wanted == 1;
}

static void tdleaf_dump_fen(const position &pos, bool wtm, char *out);

// Set by selfplay.cpp when trajectory emission (--traj-out) is active: the
// .tdg format ships root_pos/root_static, so record_ply must capture them
// even when neither the TSV dump nor root learning would.
bool tdleaf_capture_root = false;

// Set by search(): 1 when pc[0] is not from a resolved root update.
extern int tdleaf_pv_is_stub;

// ---------------------------------------------------------------------------
// PV-walk / score-consistency telemetry.  Counters only, always on: the cost is
// a handful of adds per recorded ply against a full search, and the questions it
// answers ("how often does the PV walk fail to reach a real leaf?", "how far is
// the leaf's static eval from the propagated search score?") are otherwise
// unmeasurable after the fact.  Reported by tdleaf_report_pv_stats().
//
// A short walk is not a bug — it means the record's label d_t is the static eval
// of a shallower position than the search actually evaluated, in the limit the
// ROOT itself (pv_len == 0), which makes that record self-distillation rather
// than a search label.  Its own TD error is then wrong at full strength; earlier
// records are contaminated only at (1-lambda) = 1.5% because a wrong d_t enters
// two adjacent deltas with opposite signs and telescopes out.
// ---------------------------------------------------------------------------
enum { TDPV_STOP_NOMOVE = 0, TDPV_STOP_ILLEGAL = 1, TDPV_STOP_MAXD = 2 };

struct TDPvStats {
    uint64_t n;                 // records seen
    uint64_t len_exact[33];     // exact pv_len 0..31, [32] = 32+
    uint64_t stop_illegal;      // walk ended on an illegal PV move
    uint64_t stop_maxd;         // walk hit the MAXD bound
    uint64_t shorter_than_depth;// pv_len < achieved search depth
    uint64_t len_sum;
    uint64_t depth_sum;         // achieved search depth, for the shortfall ratio
    // |leaf static eval - propagated root search score|, in cp.  Not an error
    // bound: the search score includes quiescence, so a nonzero miss is expected.
    // Mate-scored records are counted separately: a MATE sentinel against a
    // finite static eval is a scale artifact, not a measurable disagreement.
    uint64_t miss_n;
    double   miss_sum;
    int      miss_max;
    uint64_t miss_over[5];      // > 25, 50, 100, 300, 1000 cp
    uint64_t miss_over_by_short;// of the >100cp records, those with pv_len <= 2
    uint64_t mate_n;            // records whose root score is a mate sentinel
    // Miss split by whether the walk reached the full achieved depth.
    uint64_t full_n;   double full_sum;
    uint64_t short_n;  double short_sum;
    uint64_t n_by_stm[2], len2_by_stm[2];   // does the len==2 spike track STM?
    uint64_t n_by_par[2];                   // ...or record parity (ply order)?
    // SIGNED (leaf_static - propagated) in STM POV, split by whether the walk
    // reached full depth.  This is the measurement that decides whether the
    // truncation error is zero-mean noise (averages away offline) or a
    // systematic STM-correlated bias (becomes an alternating sawtooth in
    // white-POV d[], and TD errors are differences of consecutive d).
    uint64_t len2_depth_sum, len2_n, oth_depth_sum, oth_n;  // depth cross-tab
    uint64_t len2_neardraw, oth_neardraw;   // |propagated| < 25 cp
    // Is the ROOT a better stand-in for the search value than a truncated leaf?
    double   sh_leafmiss, sh_rootmiss;  uint64_t sh_cmp_n;
    double   sh_leafsgn, sh_rootsgn;
    uint64_t fallback_n;   // records re-anchored at the root
    uint64_t skipped_stub; // plies skipped: PV was an unresolved stub
    // Classify the position the walk STOPPED on (truncated records only):
    // what would the search have done there?
    uint64_t tr_n, tr_incheck, tr_hascaps, tr_quiet, tr_fifty_hi, tr_lowpieces;
    // Are the residual short PVs the LEGITIMATE terminal ones -- draws and mates?
    uint64_t tr_neardraw, tr_exactzero, tr_mate, tr_matescore_n;
    uint64_t tr_rep_pos;
    // How often does leaf_static EQUAL the propagated root score?  For a fully
    // resolved PV that runs through qsearch it should, exactly -- the backed-up
    // value IS the leaf's static eval.  Mismatches mark records where the search
    // value came from somewhere other than this leaf.
    uint64_t mm_n, mm_exact, mm_le2, mm_le10, mm_le25, mm_le50, mm_gt50;
    uint64_t mm_mate, mm_zero;          // classified BEFORE the tolerance test
    uint64_t mm_gt50_short, mm_gt50_full;
    // Signed delta in ROOT-STM POV: (leaf eval propagated back to the root's
    // frame) - (root search score).  Parity-invariant, so a systematic search
    // miss shows as a one-sided mean; a pure PV-approximation shows symmetric.
    uint64_t gate_seen, gate_skipped;   // leaf-match gate
    uint64_t rp_n, rp_hi, rp_lo, rp_eq;
    double   rp_sum, rp_sq;
    uint64_t rp_n_w, rp_n_b;  double rp_sum_w, rp_sum_b;   // split by root STM
    uint64_t rp_hi_big, rp_lo_big;                          // |delta| > 25 cp
    uint64_t ob_n, ob_leaf_exact, ob_prev_exact, ob_prev_better;
    double   ob_leaf_sum, ob_prev_sum;
    // Exact-match rate split by whether the LEAF is quiescent.  If the PV runs
    // through qsearch as intended, the leaf should be quiet and match exactly.
    uint64_t q_n[4], q_exact[4];   // [0]=quiet [1]=has caps [2]=in check [3]=caps+check
    double   q_absmiss[4];
    double   len2_absprop, oth_absprop;
    double   sgn_full_sum, sgn_short_sum;
    uint64_t sgn_full_n,   sgn_short_n;
    double   sgn_full_sq,  sgn_short_sq;
};
static TDPvStats td_pv;

static void tdleaf_pv_stats_record(int pv_len, int walk_stop, int search_depth,
                                   int leaf_score_stm, int score_root_stm,
                                   int root_wtm, int rec_index, int root_static_diag,
                                   int leaf_incheck_diag, int leaf_hascaps_diag)
{
    td_pv.n++;
    td_pv.n_by_stm[root_wtm & 1]++;
    if (pv_len == 2) td_pv.len2_by_stm[root_wtm & 1]++;
    if (pv_len == 2) td_pv.n_by_par[rec_index & 1]++;
    td_pv.len_exact[pv_len < 32 ? pv_len : 32]++;
    td_pv.len_sum += (uint64_t)pv_len;
    if (search_depth > 0) {
        td_pv.depth_sum += (uint64_t)search_depth;
        if (pv_len == 2) { td_pv.len2_depth_sum += (uint64_t)search_depth; td_pv.len2_n++; }
        else             { td_pv.oth_depth_sum  += (uint64_t)search_depth; td_pv.oth_n++;  }
    }
    if (walk_stop == TDPV_STOP_ILLEGAL) td_pv.stop_illegal++;
    if (walk_stop == TDPV_STOP_MAXD)    td_pv.stop_maxd++;
    bool reached_depth = !(search_depth > 0 && pv_len < search_depth);
    if (!reached_depth) td_pv.shorter_than_depth++;

    // Leaf-vs-root correspondence over ALL records, mates and draws classified
    // separately rather than silently dropped.
    {
        int prop = (pv_len & 1) ? -score_root_stm : score_root_stm;
        int dd = leaf_score_stm - prop; if (dd < 0) dd = -dd;
        int asr = score_root_stm < 0 ? -score_root_stm : score_root_stm;
        td_pv.mm_n++;
        if (asr > MATE - 1000)        td_pv.mm_mate++;
        else if (score_root_stm == 0) td_pv.mm_zero++;
        else {
            // Root-STM-POV signed delta.
            int leaf_in_root = (pv_len & 1) ? -leaf_score_stm : leaf_score_stm;
            int dr = leaf_in_root - score_root_stm;
            td_pv.rp_n++;  td_pv.rp_sum += dr;  td_pv.rp_sq += (double)dr*dr;
            if (dr > 0) { td_pv.rp_hi++; if (dr >  25) td_pv.rp_hi_big++; }
            else if (dr < 0) { td_pv.rp_lo++; if (dr < -25) td_pv.rp_lo_big++; }
            else td_pv.rp_eq++;
            if (root_wtm) { td_pv.rp_n_w++; td_pv.rp_sum_w += dr; }
            else          { td_pv.rp_n_b++; td_pv.rp_sum_b += dr; }
            // Dump the first 25 mismatches with everything that feeds them.
            static int dumped = 0;
            if (dd != 0 && dumped < 25) {
                dumped++;
                fprintf(stderr, "MISMATCH#%02d pv_len=%2d depth=%2d  leaf_static=%6d  "
                        "root_score=%6d  propagated=%6d  diff=%+6d  incheck=%d caps=%d\n",
                        dumped, pv_len, search_depth, leaf_score_stm,
                        score_root_stm, prop, leaf_score_stm - prop,
                        leaf_incheck_diag, leaf_hascaps_diag);
            }
            int cls = (leaf_incheck_diag ? 2 : 0) | (leaf_hascaps_diag ? 1 : 0);
            if (cls > 3) cls = 3;
            td_pv.q_n[cls]++;
            td_pv.q_absmiss[cls] += dd;
            if (dd == 0) td_pv.q_exact[cls]++;
            if (dd == 0)  td_pv.mm_exact++;
            if (dd <= 2)  td_pv.mm_le2++;
            if (dd <= 10) td_pv.mm_le10++;
            if (dd <= 25) td_pv.mm_le25++;
            if (dd <= 50) td_pv.mm_le50++;
            if (dd > 50) {
                td_pv.mm_gt50++;
                if (search_depth > 0 && pv_len < search_depth) td_pv.mm_gt50_short++;
                else                                          td_pv.mm_gt50_full++;
            }
        }
    }

    // A mate sentinel (|score| near MATE) against a finite static eval is a
    // scale artifact, not a disagreement — count and exclude.
    if (score_root_stm > MATE - 1000 || score_root_stm < -(MATE - 1000)) {
        td_pv.mate_n++;
        return;
    }

    // Propagate the root score to the leaf's POV: one sign flip per ply walked.
    int propagated = (pv_len & 1) ? -score_root_stm : score_root_stm;
    { int ap = propagated < 0 ? -propagated : propagated;
      if (pv_len == 2) { td_pv.len2_absprop += ap; if (ap < 25) td_pv.len2_neardraw++; }
      else             { td_pv.oth_absprop  += ap; if (ap < 25) td_pv.oth_neardraw++;  } }
    int diff = leaf_score_stm - propagated;
    int adiff = diff < 0 ? -diff : diff;
    // For SHORT walks only: compare the leaf's miss against the ROOT's own
    // static-vs-search miss.  If the root is closer, falling back to it is a
    // strict improvement for those records.
    if (search_depth > 0 && pv_len < search_depth && root_static_diag != INT_MIN) {
        int rds = root_static_diag - score_root_stm;
        int rd = rds < 0 ? -rds : rds;
        td_pv.sh_leafmiss += adiff; td_pv.sh_rootmiss += rd; td_pv.sh_cmp_n++;
        td_pv.sh_leafsgn += diff;   td_pv.sh_rootsgn += rds;
    }
    td_pv.miss_n++;
    td_pv.miss_sum += (double)adiff;
    if (adiff > td_pv.miss_max) td_pv.miss_max = adiff;
    if (reached_depth) {
        td_pv.full_n++;  td_pv.full_sum  += (double)adiff;
        td_pv.sgn_full_n++;  td_pv.sgn_full_sum  += (double)diff;
        td_pv.sgn_full_sq   += (double)diff * (double)diff;
    } else {
        td_pv.short_n++; td_pv.short_sum += (double)adiff;
        td_pv.sgn_short_n++; td_pv.sgn_short_sum += (double)diff;
        td_pv.sgn_short_sq  += (double)diff * (double)diff;
    }
    static const int thr[5] = { 25, 50, 100, 300, 1000 };
    for (int i = 0; i < 5; i++)
        if (adiff > thr[i]) {
            td_pv.miss_over[i]++;
            if (i == 2 && pv_len <= 2) td_pv.miss_over_by_short++;
        }
}

void tdleaf_report_pv_stats(const char *tag)
{
    if (!td_pv.n) return;
    const double N = (double)td_pv.n;
    const char *T = tag ? tag : "";
    fprintf(stderr, "TDLeaf PV stats %s: records=%llu mean_pv_len=%.2f "
            "mean_depth=%.2f stop_illegal=%llu (%.4f%%) stop_maxd=%llu "
            "shorter_than_depth=%llu (%.2f%%) mate_scored=%llu (%.2f%%)\n",
            T, (unsigned long long)td_pv.n, (double)td_pv.len_sum / N,
            (double)td_pv.depth_sum / N,
            (unsigned long long)td_pv.stop_illegal, 100.0 * td_pv.stop_illegal / N,
            (unsigned long long)td_pv.stop_maxd,
            (unsigned long long)td_pv.shorter_than_depth,
            100.0 * td_pv.shorter_than_depth / N,
            (unsigned long long)td_pv.mate_n, 100.0 * td_pv.mate_n / N);
    fprintf(stderr, "TDLeaf PV len2-split %s: by_stm black=%llu/%llu white=%llu/%llu | "
            "len2 by record parity even=%llu odd=%llu\n", T,
            (unsigned long long)td_pv.len2_by_stm[0], (unsigned long long)td_pv.n_by_stm[0],
            (unsigned long long)td_pv.len2_by_stm[1], (unsigned long long)td_pv.n_by_stm[1],
            (unsigned long long)td_pv.n_by_par[0], (unsigned long long)td_pv.n_by_par[1]);
    fprintf(stderr, "TDLeaf PV len %s:", T);
    for (int i = 0; i <= 32; i++)
        if (td_pv.len_exact[i])
            fprintf(stderr, " %d:%llu(%.2f%%)", i,
                    (unsigned long long)td_pv.len_exact[i],
                    100.0 * td_pv.len_exact[i] / N);
    fprintf(stderr, "\n");
    if (td_pv.miss_n) {
        const double M = (double)td_pv.miss_n;
        fprintf(stderr, "TDLeaf score-miss %s: n=%llu mean|d|=%.1f cp max=%d cp | "
                ">25=%.2f%% >50=%.2f%% >100=%.2f%% >300=%.3f%% >1000=%.4f%% | "
                "of >100cp with pv_len<=2: %llu | mean|d| full-depth=%.1f (n=%llu) "
                "short=%.1f (n=%llu)\n",
                T, (unsigned long long)td_pv.miss_n, td_pv.miss_sum / M, td_pv.miss_max,
                100.0 * td_pv.miss_over[0] / M, 100.0 * td_pv.miss_over[1] / M,
                100.0 * td_pv.miss_over[2] / M, 100.0 * td_pv.miss_over[3] / M,
                100.0 * td_pv.miss_over[4] / M,
                (unsigned long long)td_pv.miss_over_by_short,
                td_pv.full_n  ? td_pv.full_sum  / (double)td_pv.full_n  : 0.0,
                (unsigned long long)td_pv.full_n,
                td_pv.short_n ? td_pv.short_sum / (double)td_pv.short_n : 0.0,
                (unsigned long long)td_pv.short_n);
    }
    if (td_pv.len2_n && td_pv.oth_n)
        fprintf(stderr, "TDLeaf len2-vs-depth %s: mean achieved depth  len==2: %.2f (n=%llu)"
                "   other: %.2f (n=%llu)\n", T,
                (double)td_pv.len2_depth_sum/(double)td_pv.len2_n, (unsigned long long)td_pv.len2_n,
                (double)td_pv.oth_depth_sum/(double)td_pv.oth_n, (unsigned long long)td_pv.oth_n);
    if (td_pv.len2_n && td_pv.oth_n)
        fprintf(stderr, "TDLeaf len2-score %s: mean|root score| len==2: %.1f cp "
                "(near-draw <25cp: %.1f%%)   other: %.1f cp (near-draw: %.1f%%)\n", T,
                td_pv.len2_absprop/(double)td_pv.len2_n,
                100.0*td_pv.len2_neardraw/(double)td_pv.len2_n,
                td_pv.oth_absprop/(double)td_pv.oth_n,
                100.0*td_pv.oth_neardraw/(double)td_pv.oth_n);
    {
        static const char *cn[4] = {"QUIET (no caps, no check)","has captures",
                                    "in check","captures+check"};
        for (int i = 0; i < 4; i++)
            if (td_pv.q_n[i])
                fprintf(stderr, "TDLeaf leaf-class %s %-26s: n=%8llu  EXACT %6.2f%%  "
                        "mean|miss| %7.1f cp\n", T, cn[i],
                        (unsigned long long)td_pv.q_n[i],
                        100.0*td_pv.q_exact[i]/(double)td_pv.q_n[i],
                        td_pv.q_absmiss[i]/(double)td_pv.q_n[i]);
    }
    if (td_pv.gate_seen)
        fprintf(stderr, "TDLeaf leaf-match gate %s (<=%d cp): %llu of %llu records "
                "excluded from the online trace (%.2f%%), %.2f%% retained\n", T,
                TDLEAF_LEAF_MATCH_CP,
                (unsigned long long)td_pv.gate_skipped,
                (unsigned long long)td_pv.gate_seen,
                100.0*td_pv.gate_skipped/(double)td_pv.gate_seen,
                100.0*(td_pv.gate_seen-td_pv.gate_skipped)/(double)td_pv.gate_seen);
    if (td_pv.rp_n) {
        double N2 = (double)td_pv.rp_n;
        double m = td_pv.rp_sum / N2;
        double sd = sqrt(td_pv.rp_sq / N2 - m * m);
        fprintf(stderr, "TDLeaf ROOT-POV delta %s (leaf_eval_in_root_frame - root_score): "
                "n=%llu  mean=%+.3f cp  sd=%.1f  SE=%.3f  -> %.1f sigma from zero\n", T,
                (unsigned long long)td_pv.rp_n, m, sd, sd/sqrt(N2),
                sd > 0 ? fabs(m)/(sd/sqrt(N2)) : 0.0);
        fprintf(stderr, "TDLeaf ROOT-POV sides %s: leaf HIGHER than search %.2f%% "
                "(>25cp %.2f%%) | LOWER %.2f%% (>25cp %.2f%%) | equal %.2f%% || "
                "mean by root STM: white %+.2f (n=%llu)  black %+.2f (n=%llu)\n", T,
                100.0*td_pv.rp_hi/N2, 100.0*td_pv.rp_hi_big/N2,
                100.0*td_pv.rp_lo/N2, 100.0*td_pv.rp_lo_big/N2,
                100.0*td_pv.rp_eq/N2,
                td_pv.rp_n_w ? td_pv.rp_sum_w/(double)td_pv.rp_n_w : 0.0,
                (unsigned long long)td_pv.rp_n_w,
                td_pv.rp_n_b ? td_pv.rp_sum_b/(double)td_pv.rp_n_b : 0.0,
                (unsigned long long)td_pv.rp_n_b);
    }
    if (td_pv.ob_n) {
        double O = (double)td_pv.ob_n;
        fprintf(stderr, "TDLeaf off-by-one probe %s: n=%llu | LEAF exact %.2f%% "
                "mean|miss| %.1f | ONE-PLY-BACK exact %.2f%% mean|miss| %.1f | "
                "prev closer in %.2f%% of records\n", T, (unsigned long long)td_pv.ob_n,
                100.0*td_pv.ob_leaf_exact/O, td_pv.ob_leaf_sum/O,
                100.0*td_pv.ob_prev_exact/O, td_pv.ob_prev_sum/O,
                100.0*td_pv.ob_prev_better/O);
    }
    if (td_pv.mm_n) {
        double M = (double)td_pv.mm_n;
        double R = (double)(td_pv.mm_n - td_pv.mm_mate - td_pv.mm_zero);
        fprintf(stderr, "TDLeaf leaf-vs-root correspondence %s: n=%llu | "
                "MATE-scored %.2f%%  score==0 %.2f%% | of the remaining %.0f: "
                "EXACT %.2f%%  <=2cp %.2f%%  <=10 %.2f%%  <=25 %.2f%%  <=50 %.2f%%  "
                ">50 %.2f%% (short %llu / full %llu)\n", T,
                (unsigned long long)td_pv.mm_n,
                100.0*td_pv.mm_mate/M, 100.0*td_pv.mm_zero/M, R,
                100.0*td_pv.mm_exact/R, 100.0*td_pv.mm_le2/R, 100.0*td_pv.mm_le10/R,
                100.0*td_pv.mm_le25/R, 100.0*td_pv.mm_le50/R, 100.0*td_pv.mm_gt50/R,
                (unsigned long long)td_pv.mm_gt50_short,
                (unsigned long long)td_pv.mm_gt50_full);
    }
    if (td_pv.tr_n) {
        double T2 = (double)td_pv.tr_n;
        fprintf(stderr, "TDLeaf truncation-point %s: n=%llu | in_check=%.1f%% "
                "has_captures=%.1f%% QUIET(no check,no caps)=%.1f%% | "
                "fifty>=80=%.1f%% pieces<=6=%.1f%%\n", T,
                (unsigned long long)td_pv.tr_n,
                100.0*td_pv.tr_incheck/T2, 100.0*td_pv.tr_hascaps/T2,
                100.0*td_pv.tr_quiet/T2, 100.0*td_pv.tr_fifty_hi/T2,
                100.0*td_pv.tr_lowpieces/T2);
        fprintf(stderr, "TDLeaf truncation-score %s: MATE-scored=%.1f%% | of the "
                "non-mate rest (n=%llu): EXACTLY 0 = %.1f%%, |score|<25cp = %.1f%%\n",
                T, 100.0*td_pv.tr_mate/T2,
                (unsigned long long)td_pv.tr_matescore_n,
                td_pv.tr_matescore_n ? 100.0*td_pv.tr_exactzero/(double)td_pv.tr_matescore_n : 0.0,
                td_pv.tr_matescore_n ? 100.0*td_pv.tr_neardraw/(double)td_pv.tr_matescore_n : 0.0);
    }
    if (td_pv.skipped_stub)
        fprintf(stderr, "TDLeaf stub-skip %s: %llu plies not recorded (%.1f%% of "
                "%llu offered) -- PV was not from a resolved root search\n", T,
                (unsigned long long)td_pv.skipped_stub,
                100.0*td_pv.skipped_stub/(double)(td_pv.skipped_stub + td_pv.n),
                (unsigned long long)(td_pv.skipped_stub + td_pv.n));
    if (td_pv.fallback_n)
        fprintf(stderr, "TDLeaf root-fallback %s: %llu records (%.2f%%) re-anchored at "
                "the root with the root search score as label\n", T,
                (unsigned long long)td_pv.fallback_n, 100.0*td_pv.fallback_n/N);
    if (td_pv.sh_cmp_n)
        fprintf(stderr, "TDLeaf short-walk leaf-vs-root %s: mean |leaf_static-propagated| "
                "= %.1f cp (signed %+.2f)   vs   mean |root_static-root_search| = %.1f cp "
                "(signed %+.2f)   (n=%llu)\n", T,
                td_pv.sh_leafmiss/(double)td_pv.sh_cmp_n,
                td_pv.sh_leafsgn/(double)td_pv.sh_cmp_n,
                td_pv.sh_rootmiss/(double)td_pv.sh_cmp_n,
                td_pv.sh_rootsgn/(double)td_pv.sh_cmp_n,
                (unsigned long long)td_pv.sh_cmp_n);
    if (td_pv.sgn_short_n) {
        double mf = td_pv.sgn_full_sum  / (double)td_pv.sgn_full_n;
        double ms = td_pv.sgn_short_sum / (double)td_pv.sgn_short_n;
        double sf = sqrt(td_pv.sgn_full_sq  / (double)td_pv.sgn_full_n  - mf*mf);
        double ss = sqrt(td_pv.sgn_short_sq / (double)td_pv.sgn_short_n - ms*ms);
        fprintf(stderr, "TDLeaf signed-bias %s (STM POV, leaf_static - propagated): "
                "full-depth mean=%+.2f sd=%.1f n=%llu | short mean=%+.2f sd=%.1f "
                "n=%llu | bias/sd short=%.4f\n", T,
                mf, sf, (unsigned long long)td_pv.sgn_full_n,
                ms, ss, (unsigned long long)td_pv.sgn_short_n,
                ss > 0 ? ms/ss : 0.0);
        // Overall bias across every record that actually gets a gradient --
        // the bottom line, since this is the component that does not average out.
        double tot_n   = (double)(td_pv.sgn_full_n + td_pv.sgn_short_n);
        double tot_sum = td_pv.sgn_full_sum + td_pv.sgn_short_sum;
        double tot_sq  = td_pv.sgn_full_sq  + td_pv.sgn_short_sq;
        double mt = tot_sum / tot_n;
        double st = sqrt(tot_sq / tot_n - mt * mt);
        fprintf(stderr, "TDLeaf OVERALL bias %s: mean=%+.2f cp  sd=%.1f  n=%.0f  "
                "bias/sd=%.4f\n", T, mt, st, tot_n, st > 0 ? mt/st : 0.0);
    }
    fflush(stderr);
}

// ---------------------------------------------------------------------------
// tdleaf_record_ply — walk the PV to the leaf, then snapshot its accumulator
// ---------------------------------------------------------------------------
void tdleaf_record_ply(TDGameRecord &rec,
                       const position &root_pos,
                       const NNUEAccumulator &root_acc,
                       const move *pv,
                       int score_root_stm,
                       const int *id_scores,
                       int id_score_count,
                       int search_depth,
                       int game_ply)
{
    if (rec.n_plies >= MAX_GAME_PLY) return;  // safety guard

#if TDLEAF_SKIP_STUB_PV
    // The PV did not come from a resolved root search (fail-high stub, or an
    // early return that left pc[0] stale/guessed).  Its second move was never
    // searched and its score is a bound, so there is no leaf to differentiate.
    // Skip the ply entirely: the trace's pow(lambda, dply) absorbs the gap.
    if (tdleaf_pv_is_stub) { td_pv.skipped_stub++; return; }
#endif

    // Capture engine color on the first ply of a fresh game.  Every recorded
    // ply has root_pos.wtm == engine's color (we only record on engine moves).
    if (rec.n_plies == 0) rec.engine_color = (int8_t)root_pos.wtm;

    // Walk the PV, updating the position and accumulator incrementally.
    // We use two alternating accumulator slots to avoid unnecessary copies.
    NNUEAccumulator acc_a = root_acc;   // current leaf accumulator
    NNUEAccumulator acc_b;              // scratch for next step
    position cur = root_pos;
    NNUEAccumulator acc_prev = root_acc;   // accumulator one ply back (diagnostic)
    position prev_pos = root_pos;
    int pv_len = 0;
    int walk_stop = TDPV_STOP_NOMOVE;   // why the walk ended (telemetry only)

    for (int k = 0; k < MAXD && pv[k].t != NOMOVE; k++) {
        position next = cur;
        if (!next.exec_move(pv[k], 0)) {   // illegal — stop here
            walk_stop = TDPV_STOP_ILLEGAL;
            break;
        }
        nnue_record_delta(acc_b, cur, next, pv[k]);
        nnue_apply_delta(acc_b, acc_a, next);
        acc_prev = acc_a; prev_pos = cur;
        cur   = next;
        acc_a = acc_b;
        pv_len++;
        if (pv_len >= MAXD) walk_stop = TDPV_STOP_MAXD;
    }
    // acc_a now holds the fully computed leaf accumulator; cur is the leaf position.

    bool leaf_wtm = (bool)root_pos.wtm ^ (bool)(pv_len & 1);

    // Leaf piece count for stack selection.
    int pc = 2;  // kings
    for (int sd = 0; sd < 2; sd++)
        for (int pt = PAWN; pt <= QUEEN; pt++)
            pc += cur.plist[sd][pt][0];
    pc = (pc < 1) ? 1 : (pc > 32) ? 32 : pc;

    // Use the NNUE static evaluation of the leaf position directly.
    // This ensures d[t] is computed from what nnue_forward_fp32 actually produces
    // at that position, making the gradient self-consistent.
    // (The propagated search score includes quiescence and may differ.)
    int leaf_score_stm = nnue_evaluate(acc_a, (int)leaf_wtm, pc);

    int root_static_diag = INT_MIN;
#if PVTRUNC_DIAG
    {   int pcr = 0;
        for (int sd = 0; sd < 2; sd++)
            for (int pt = PAWN; pt <= KING; pt++) pcr += root_pos.plist[sd][pt][0];
        pcr = (pcr < 1) ? 1 : (pcr > 32) ? 32 : pcr;
        root_static_diag = nnue_evaluate(root_acc, (int)root_pos.wtm, pcr);   }
#endif
    int leaf_incheck_diag = 0, leaf_hascaps_diag = 0;
#if PVTRUNC_DIAG
    // Off-by-one probe: does the root score match the LEAF, or the position one
    // ply before it?  A systematic win for prev would mean the walk overshoots.
    if (pv_len >= 1) {
        int pcp = 2;
        for (int sd = 0; sd < 2; sd++)
            for (int pt = PAWN; pt <= QUEEN; pt++) pcp += prev_pos.plist[sd][pt][0];
        pcp = (pcp < 1) ? 1 : (pcp > 32) ? 32 : pcp;
        bool prev_wtm = (bool)root_pos.wtm ^ (bool)((pv_len - 1) & 1);
        int prev_eval = nnue_evaluate(acc_prev, (int)prev_wtm, pcp);
        int prop_leaf = (pv_len & 1) ? -score_root_stm : score_root_stm;
        int prop_prev = ((pv_len - 1) & 1) ? -score_root_stm : score_root_stm;
        int dl = leaf_score_stm - prop_leaf; if (dl < 0) dl = -dl;
        int dp = prev_eval - prop_prev;      if (dp < 0) dp = -dp;
        td_pv.ob_n++;
        td_pv.ob_leaf_sum += dl; td_pv.ob_prev_sum += dp;
        if (dl == 0) td_pv.ob_leaf_exact++;
        if (dp == 0) td_pv.ob_prev_exact++;
        if (dp < dl) td_pv.ob_prev_better++;
    }
    leaf_incheck_diag = cur.in_check() ? 1 : 0;
    { move_list ml; cur.captures(&ml, -10000); leaf_hascaps_diag = ml.count > 0; }
#endif
    tdleaf_pv_stats_record(pv_len, walk_stop, search_depth,
                           leaf_score_stm, score_root_stm,
                           (int)root_pos.wtm, rec.n_plies, root_static_diag,
                           leaf_incheck_diag, leaf_hascaps_diag);

#if TDLEAF_CHECK_SCORE
    {
        // Sanity check: propagated root score (with per-ply sign flip) vs direct eval.
        int propagated = (pv_len & 1) ? -score_root_stm : score_root_stm;
        int diff = leaf_score_stm - propagated;
        fprintf(stderr, "TDLeaf check: pv_len=%d  leaf_wtm=%d  direct=%d  propagated=%d  diff=%d%s\n",
                pv_len, (int)leaf_wtm, leaf_score_stm, propagated, diff,
                (diff < -300 || diff > 300) ? "  *** LARGE ***" : "");
    }
#endif

    // Compute variance of the last N iterative-deepening scores.
    float id_var = 0.0f;
    if (id_score_count >= 2) {
        float id_mean = 0.0f;
        for (int i = 0; i < id_score_count; i++) id_mean += id_scores[i];
        id_mean /= id_score_count;
        for (int i = 0; i < id_score_count; i++) {
            float delta = id_scores[i] - id_mean;
            id_var += delta * delta;
        }
        id_var /= id_score_count;
    }

    // Diagnostic (env TDLEAF_CHECK_ACC=1): verify the incrementally walked
    // leaf accumulator equals a from-scratch rebuild of the leaf position —
    // the invariant the trajectory learner's tdleaf_rebuild_record relies on
    // for bit-exact gradient reconstruction.
    {
        static int check_acc = -1;
        if (check_acc < 0) {
            const char *p = getenv("TDLEAF_CHECK_ACC");
            check_acc = (p && *p && *p != '0') ? 1 : 0;
        }
        if (check_acc) {
            NNUEAccumulator fresh;
            nnue_init_accumulator(fresh, cur);
            bool p_ok[2], q_ok[2];
            for (int p = 0; p < 2; p++) {
                p_ok[p] = memcmp(fresh.acc[p],  acc_a.acc[p],
                                 NNUE_HALF_DIMS * sizeof(int16_t)) == 0;
                q_ok[p] = memcmp(fresh.psqt[p], acc_a.psqt[p],
                                 NNUE_PSQT_BKTS * sizeof(int32_t)) == 0;
            }
            if (!p_ok[0] || !p_ok[1] || !q_ok[0] || !q_ok[1]) {
                char fen[110];
                tdleaf_dump_fen(cur, leaf_wtm, fen);
                fprintf(stderr, "TDLeaf CHECK_ACC MISMATCH ply=%d pv_len=%d "
                                "accB=%d accW=%d psqtB=%d psqtW=%d  pv=",
                        game_ply, pv_len,
                        (int)p_ok[0], (int)p_ok[1], (int)q_ok[0], (int)q_ok[1]);
                for (int k = 0; k < pv_len; k++)
                    fprintf(stderr, "%d>%d/t%d ", (int)pv[k].b.from,
                            (int)pv[k].b.to, (int)pv[k].b.type);
                fprintf(stderr, " leaf=%s\n", fen);
            }
        }
    }

    // ---- Root fallback (TDLEAF_ROOT_FALLBACK) -----------------------------
    // The walk could not reach what the search actually evaluated.  Anchor the
    // record at the ROOT and label it with the root's SEARCH score, rather than
    // training on the static eval of a position the search never scored.
    bool used_root_fallback = false;
#if TDLEAF_ROOT_FALLBACK
    if (search_depth > 0 && (search_depth - pv_len) >= TDLEAF_ROOT_FALLBACK) {
        used_root_fallback = true;
        acc_a    = root_acc;
        cur      = root_pos;
        leaf_wtm = (bool)root_pos.wtm;
        pc = 0;
        for (int sd = 0; sd < 2; sd++)
            for (int pt = PAWN; pt <= KING; pt++) pc += root_pos.plist[sd][pt][0];
        pc = (pc < 1) ? 1 : (pc > 32) ? 32 : pc;
        leaf_score_stm = score_root_stm;   // the SEARCH score, not a static eval
        td_pv.fallback_n++;
    }
#endif

#if PVTRUNC_DIAG
    // Characterise the truncation point.  If the PV ended because the
    // continuation was a quiescent stand-pat, the position should be QUIET
    // (not in check, no captures).  If it ended on a draw exit, it should show
    // a high fifty counter or very few pieces.
    if (search_depth > 0 && pv_len < search_depth) {
        td_pv.tr_n++;
        bool incheck = cur.in_check();
        int caps = 0;
        { move_list ml; cur.captures(&ml, -10000); caps = ml.count; }
        if (incheck) td_pv.tr_incheck++;
        if (caps > 0) td_pv.tr_hascaps++;
        if (!incheck && caps == 0) td_pv.tr_quiet++;
        if (cur.fifty >= 80) td_pv.tr_fifty_hi++;
        { int np = 0;
          for (int sd = 0; sd < 2; sd++)
            for (int pt = PAWN; pt <= KING; pt++) np += cur.plist[sd][pt][0];
          if (np <= 6) td_pv.tr_lowpieces++; }
        // Score classification of the ROOT search value for this record.
        { int sr = score_root_stm;
          int asr = sr < 0 ? -sr : sr;
          if (asr > MATE - 1000) td_pv.tr_mate++;
          else {
            td_pv.tr_matescore_n++;
            if (sr == 0)   td_pv.tr_exactzero++;
            if (asr < 25)  td_pv.tr_neardraw++;
          } }
    }
#endif

    TDRecord &r = rec.plies[rec.n_plies++];
    memcpy(r.acc[0],  acc_a.acc[0],  NNUE_HALF_DIMS  * sizeof(int16_t));
    memcpy(r.acc[1],  acc_a.acc[1],  NNUE_HALF_DIMS  * sizeof(int16_t));
    memcpy(r.psqt[0], acc_a.psqt[0], NNUE_PSQT_BKTS * sizeof(int32_t));
    memcpy(r.psqt[1], acc_a.psqt[1], NNUE_PSQT_BKTS * sizeof(int32_t));
    r.score_stm         = leaf_score_stm;
    r.score_root_stm    = score_root_stm;   // engine-POV root score (cp) for adjudication
    r.wtm               = leaf_wtm;
    r.root_wtm          = (bool)root_pos.wtm;  // per-record root STM (POV for the dump)
    r.game_ply          = game_ply;            // 1-based game-ply of the root position
    r.stack             = (pc - 1) / 4;
    r.id_score_variance = id_var;
    r.pos               = cur;  // leaf position (trajectory learner rebuilds from it)
    r.id_depth          = (int8_t)((search_depth < 1) ? 1 :
                                   (search_depth > 127) ? 127 : search_depth);
    // Does the leaf's static eval match the root score propagated to the leaf's
    // POV?  If not, the PV did not locate the position the score came from and
    // this record's gradient is uninformative online.  See TDLEAF_LEAF_MATCH_CP.
    {
        int prop = (pv_len & 1) ? -score_root_stm : score_root_stm;
        int d = leaf_score_stm - prop; if (d < 0) d = -d;
        r.leaf_ok = (TDLEAF_LEAF_MATCH_CP <= 0) || (d <= TDLEAF_LEAF_MATCH_CP);
        if (!r.leaf_ok) td_pv.gate_skipped++;
        td_pv.gate_seen++;
    }
    if (tdleaf_dump_wanted() || tdleaf_capture_root) {
        // Root snapshot + static eval for the root-row TSV dump and the .tdg
        // trajectory format.
        r.root_pos = root_pos;
        int pc_root = 0;
        for (int sd = 0; sd < 2; sd++)
            for (int pt = PAWN; pt <= KING; pt++)
                pc_root += root_pos.plist[sd][pt][0];
        pc_root = (pc_root < 1) ? 1 : (pc_root > 32) ? 32 : pc_root;
        r.root_static = nnue_evaluate(root_acc, (int)root_pos.wtm, pc_root);
    }

    // Enumerate active features at the leaf position for FT/PSQT backprop.
    // Indices are by actual perspective (0=BLACK, 1=WHITE) matching halfkav2_feature().
    for (int p = 0; p < 2; p++) {
        int ksq = cur.plist[p][KING][1];
        r.n_ft[p] = 0;
        for (int sd = 0; sd < 2; sd++)
            for (int pt = PAWN; pt <= KING; pt++)
                for (int i = 1; i <= cur.plist[sd][pt][0]; i++) {
                    if (r.n_ft[p] >= NNUE_MAX_FT_PER_PERSP) goto ft_done;
                    int fi = halfkav2_feature(p, ksq, cur.plist[sd][pt][i], pt, sd);
                    if (fi >= 0) r.ft_idx[p][r.n_ft[p]++] = fi;
                }
        ft_done:;
    }
}

// ---------------------------------------------------------------------------
// Runtime freeze (TDLEAF_FREEZE=1 env): play and dump exactly as a learning
// binary would, but skip gradient accumulation, weight application, and all
// .tdleaf.bin writes.  Unlike the compile-time TDLEAF_READONLY flag (which
// compiles out the record/update hooks entirely, so no corpus is dumped),
// this keeps the corpus dump alive — actors in the actor/learner split run
// frozen (only the learner owns the optimizer and writes weights).
// ---------------------------------------------------------------------------
static bool tdleaf_frozen()
{
    static int frozen = -1;
    if (frozen < 0) {
        const char *v = getenv("TDLEAF_FREEZE");
        frozen = (v && *v && atoi(v) != 0) ? 1 : 0;
        if (frozen)
            fprintf(stderr, "TDLeaf: TDLEAF_FREEZE=1 — weights frozen "
                            "(recording + TSV dump only; no gradient updates, "
                            "no .tdleaf.bin writes)\n");
    }
    return frozen == 1;
}

// ---------------------------------------------------------------------------
// tdleaf_accumulate_game — steps 1-3: compute d[], e[], accumulate gradients.
// Does NOT apply or save.  Called by tdleaf_update_after_game.
// ---------------------------------------------------------------------------
static uint32_t td_grad_samples = 0;  // POSITIONS accumulated since last apply

static void tdleaf_accumulate_game(TDGameRecord &rec, float result)
{
    // 0. Compact to records that pass the leaf-match gate.  A gated-out record
    //    has a leaf the search never valued, so it contributes neither a
    //    gradient nor a link in the eligibility trace.  Dropping it simply
    //    widens the game-ply gap to its neighbours, which pow(lambda, dply)
    //    already handles by construction -- exactly as under UCI play, where
    //    only every other ply is recorded.
    static int ix[MAX_GAME_PLY];
    int T = 0;
    for (int t = 0; t < rec.n_plies; t++)
        if (rec.plies[t].leaf_ok) ix[T++] = t;
    if (T < 1) return;          // nothing trainable in this game

    // 1. Convert scores to White-POV sigmoid values d[t] ∈ (0,1)
    static float d[MAX_GAME_PLY];
    static float score_w_cp[MAX_GAME_PLY];
    for (int t = 0; t < T; t++) {
        score_w_cp[t] = rec.plies[ix[t]].wtm
                        ?  (float)rec.plies[ix[t]].score_stm
                        : -(float)rec.plies[ix[t]].score_stm;
        d[t] = 1.0f / (1.0f + expf(-score_w_cp[t] / TDLEAF_K));
    }

    // 2. Compute TD errors backward
    const float lambda = TDLEAF_LAMBDA;
    // Under NNUE_FIXED_PIECE_VALUES value[PAWN] stays at the classical 100 cp,
    // so this threshold is constant at SCORE_CLIP_PAWNS × 100 cp.  The 100 cp
    // floor is belt-and-braces only.
    const float score_clip_cp =
        TDLEAF_SCORE_CLIP_PAWNS * std::max((float)value[PAWN], 100.0f);

    static float e[MAX_GAME_PLY];
    e[T - 1] = result - d[T - 1];
    // Classic λ-decayed eligibility trace (white-POV sigmoid values), with the
    // score-change clip applied to each bootstrap delta.
    for (int t = T - 2; t >= 0; t--) {
        float delta_d  = d[t + 1] - d[t];

        float delta_cp = fabsf(score_w_cp[t + 1] - score_w_cp[t]);
        if (delta_cp > score_clip_cp && delta_cp > 0.0f)
            delta_d *= score_clip_cp / delta_cp;
        // Decay per GAME-PLY: pow(lambda, dply).  dply = 2 in the two-process
        // harness (own moves only), 1 under internal self-play — so one lambda
        // expresses the same real-game horizon in both modes.  Guard dply >= 1
        // against any out-of-order/duplicate ply.
        int dply = rec.plies[ix[t + 1]].game_ply - rec.plies[ix[t]].game_ply;
        if (dply < 1) dply = 1;
        float trace_decay = (dply == 1) ? lambda : powf(lambda, (float)dply);
        e[t] = delta_d + trace_decay * e[t + 1];
    }

    // 3. For each ply, run FP32 forward pass + accumulate gradients
    const float cp_factor = 100.0f / 5776.0f;

    // Diagnostic (env TDLEAF_TRACE_UPDATE=<file>): append one line per record
    // with every quantity that feeds the gradient, floats in exact hex — for
    // diffing the online arm against the trajectory learner.
    static FILE *trace_f = nullptr;
    {
        static int trace_init = 0;
        if (!trace_init) {
            trace_init = 1;
            const char *p = getenv("TDLEAF_TRACE_UPDATE");
            if (p && *p) trace_f = fopen(p, "a");
        }
    }

    for (int t = 0; t < T; t++) {
        float sig_grad = d[t] * (1.0f - d[t]) / TDLEAF_K;
        // wtm_sign converts ∂d_t/∂w (white-POV utility we want to ascend)
        // into the descent-form gradient expected by nnue_apply_gradients
        // (which does w -= LR × step on the supplied "loss" gradient).
        // score_white = wtm ? +score_stm : -score_stm; nnue_forward_fp32
        // backprops ∂(stm-POV score)/∂w, so the white-POV sign is
        // (wtm ? +1 : -1) and the loss-form sign we pass downstream is its
        // negative — hence (wtm ? -1 : +1).
        float wtm_sign = rec.plies[ix[t]].wtm ? -1.0f : 1.0f;
        float id_weight = 1.0f / (1.0f + rec.plies[ix[t]].id_score_variance / TDLEAF_ID_VAR_SIGMA2);
        float grad_scale = e[t] * sig_grad * cp_factor * wtm_sign * id_weight;

        if (trace_f) {
            const TDRecord &r = rec.plies[ix[t]];
            long acc_sum = 0;
            for (int p = 0; p < 2; p++)
                for (int i = 0; i < NNUE_HALF_DIMS; i++) acc_sum += r.acc[p][i];
            long ft_sum = 0;
            for (int p = 0; p < 2; p++)
                for (int i = 0; i < r.n_ft[p]; i++) ft_sum += r.ft_idx[p][i];
            fprintf(trace_f,
                    "t=%d ply=%d stack=%d wtm=%d score=%d var=%a e=%a gs=%a "
                    "accsum=%ld nft=%d/%d ftsum=%ld\n",
                    t, r.game_ply, r.stack, (int)r.wtm, r.score_stm,
                    (double)r.id_score_variance, (double)e[t], (double)grad_scale,
                    acc_sum, (int)r.n_ft[0], (int)r.n_ft[1], ft_sum);
            fflush(trace_f);
        }

        if (grad_scale != 0.0f) {
            NNUEActivations act;
            act.stack = rec.plies[ix[t]].stack;
            nnue_forward_fp32(rec.plies[ix[t]].acc, rec.plies[ix[t]].psqt,
                              rec.plies[ix[t]].wtm, act);
            memcpy(act.acc_raw[0], rec.plies[ix[t]].acc[0], NNUE_HALF_DIMS * sizeof(int16_t));
            memcpy(act.acc_raw[1], rec.plies[ix[t]].acc[1], NNUE_HALF_DIMS * sizeof(int16_t));
            act.n_ft[0] = rec.plies[ix[t]].n_ft[0];
            act.n_ft[1] = rec.plies[ix[t]].n_ft[1];
            memcpy(act.ft_idx[0], rec.plies[ix[t]].ft_idx[0], rec.plies[ix[t]].n_ft[0] * sizeof(int));
            memcpy(act.ft_idx[1], rec.plies[ix[t]].ft_idx[1], rec.plies[ix[t]].n_ft[1] * sizeof(int));

            // Dense piece value gradient: stm_count − opp_count per piece type.
            int stm_p = rec.plies[ix[t]].wtm ? 1 : 0;
            for (int pt = PAWN; pt <= KING; pt++)
                act.piece_count_diff[pt - 1] = (int8_t)(rec.plies[ix[t]].pos.plist[stm_p][pt][0]
                                                       - rec.plies[ix[t]].pos.plist[stm_p ^ 1][pt][0]);

            nnue_accumulate_gradients(act, grad_scale);
            td_grad_samples++;
        }
    }
}

// ---------------------------------------------------------------------------
// Leaf + root TSV dump — build offline-training corpora during play.
//
// Env-gated: TDLEAF_DUMP_TSV=<prefix> writes two per-process files in the
// scripts/extract_quiet_positions.py format
//     fen \t cp \t result \t ply \t depth \t gid
//
//   <prefix>.<pid>.leaf.tsv — the PV-leaf position of every recorded ply.
//     cp = leaf STATIC eval (white POV) — the current net's own output
//     (self-distillation), so leaf rows carry training signal in the OUTCOME
//     label only.  depth column = 0, which the batch trainer treats as
//     "no search label: train this record outcome-only (lambda = 1)".
//     Quietness: |leaf static − propagated root search score| <= QUIET_CP.
//
//   <prefix>.<pid>.root.tsv — the root (played) position of every recorded
//     ply.  cp = root SEARCH score (white POV) — a search-amplified label,
//     the same kind the PGN extraction pipeline produces; depth column =
//     achieved ID depth.  Quietness: |root static − root search| <= QUIET_CP
//     (an operational test — unresolved tactics show up as static-vs-search
//     disagreement).
//
// Both apply |cp| <= TDLEAF_DUMP_MAX_CP (default 1500).  QUIET_CP
// (TDLEAF_DUMP_QUIET_CP) defaults to 1000 — effectively open, see below.
//
// Column 8, "gate": the value the quietness test compared cp against, in the
// SAME POV as cp — root static for root rows, the propagated root search score
// for leaf rows.  The gate condition is therefore exactly
//     |cp - gate| <= TDLEAF_DUMP_QUIET_CP
// for both files, which makes the gate RE-CUTTABLE OFFLINE: dump once with a
// wide QUIET_CP and every narrower gate is a filter over the same rows.  That
// turns the gate-width question into a paired offline experiment (same games,
// same labels, only the admitted row population differs) instead of two
// divergent generation runs.  Consumers that only know 7 columns ignore it.
// ---------------------------------------------------------------------------

// FEN board+stm from a stored position (castling/ep are not NNUE features and
// the trainer's parser ignores them — emit "- -").
static void tdleaf_dump_fen(const position &pos, bool wtm, char *out)
{
    int fi = 0;
    for (int ry = 7; ry >= 0; ry--) {
        int run = 0;
        for (int rx = 0; rx < 8; rx++) {
            int code = pos.sq[SQR(rx, ry)];
            int pt = PTYPE(code);
            if (pt == 0) { run++; continue; }
            if (run) out[fi++] = (char)('0' + run);
            run = 0;
            static const char pc[] = " pnbrqk";
            char ch = pc[pt];
            out[fi++] = PSIDE(code) ? (char)(ch - 32) : ch;
        }
        if (run) out[fi++] = (char)('0' + run);
        if (ry) out[fi++] = '/';
    }
    snprintf(out + fi, 16, " %c - - 0 1", wtm ? 'w' : 'b');
}

static void tdleaf_dump_game(const TDGameRecord &rec, float result)
{
    static FILE    *leaf_f = nullptr;
    static FILE    *root_f = nullptr;
#if TDLEAF_REFRESH_DIAG
    static FILE    *diag_f = nullptr;
    static FILE    *stale_f = nullptr;   // paired root corpus, actor-vintage label
#endif
    // Default WIDE (2026-09-03).  The gate used to be applied here and was
    // irreversible; now that every row carries the `gate` column it can be
    // re-cut at training time with --bt-quiet-cp, so dumping narrow only
    // destroys information.  It costs the online phase nothing — the gate is
    // consulted in the dump path only, never in the TD update.  Set
    // TDLEAF_DUMP_QUIET_CP=60 to reproduce the historical corpora.
    static int      dump_quiet_cp = 1000;
    static int      dump_max_cp   = 1500;
    static uint32_t dump_gid      = 0;
    static bool     dump_init     = false;
    if (!dump_init) {
        dump_init = true;
        const char *prefix = getenv("TDLEAF_DUMP_TSV");
        if (prefix && *prefix) {
            const char *v;
            if ((v = getenv("TDLEAF_DUMP_QUIET_CP")) && *v) dump_quiet_cp = atoi(v);
            if ((v = getenv("TDLEAF_DUMP_MAX_CP"))   && *v) dump_max_cp   = atoi(v);
            char path[FILENAME_MAX];
            auto open_dump = [&](const char *kind) -> FILE* {
                snprintf(path, sizeof(path), "%s.%d.%s.tsv", prefix, (int)getpid(), kind);
                FILE *f = fopen(path, "a");
                if (f) {
                    if (ftell(f) == 0) {
                        // Axis marker: the ply/endply columns are true GAME-ply
                        // (game-ply λ^Δ era).  --batch-train keys its result-decay
                        // axis off this line; legacy corpora without it use the
                        // old record-index axis.
                        fprintf(f, "# tdleaf-corpus axis=game-ply\n");
                        fprintf(f, "fen\tcp\tresult\tply\tdepth\tgid\tendply\tgate\n");
                    }
                } else {
                    fprintf(stderr, "TDLeaf: cannot open dump file %s\n", path);
                }
                return f;
            };
            leaf_f = open_dump("leaf");
            root_f = open_dump("root");
#if TDLEAF_REFRESH_DIAG
            // Unfiltered per-record refresh telemetry: actor-vintage vs
            // refreshed leaf static (L), root search score (R) and root
            // static (rs).  Gate decisions left to offline analysis.
            stale_f = open_dump("rootstale");
            snprintf(path, sizeof(path), "%s.%d.diag.tsv", prefix, (int)getpid());
            diag_f = fopen(path, "a");
            if (diag_f && ftell(diag_f) == 0)
                fprintf(diag_f, "gid\tply\tendply\tdepth\twtm\trootwtm\t"
                                "L0\tL1\tR0\tR1\trs0\trs1\tresult\n");
#endif
            if (leaf_f && root_f)
                fprintf(stderr, "TDLeaf: dumping leaf+root positions to %s.%d.{leaf,root}.tsv "
                                "(quiet<=%d cp, max=%d cp)\n",
                        prefix, (int)getpid(), dump_quiet_cp, dump_max_cp);
            // gid layout: 12-bit pid tag in the high bits, 20-bit per-process
            // game counter in the low bits.  The corpus's train/val split and
            // dedup key off gid, so two distinct games must not share one.
            //   - Within a process: the 20-bit counter wraps after 1,048,575
            //     games and would then silently collide with this run's own
            //     early games.  Guarded below.
            //   - Across processes: only 12 bits of pid, so N concurrent
            //     dumping processes collide with probability ~N(N-1)/2 / 4096
            //     (~0.7% at N=8).  This only arises when several --selfplay
            //     actors dump TSV directly; the actor/learner recipe has a
            //     single learner writing the corpus, so it is collision-free.
            dump_gid = ((uint32_t)getpid() & 0xFFF) << 20;
        }
    }
    if (!leaf_f && !root_f) return;

    dump_gid++;
    // Counter overflow into the pid tag: warn once rather than silently
    // emitting a gid that duplicates one from early in this same run.
    if ((dump_gid & 0xFFFFF) == 0) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            fprintf(stderr, "TDLeaf: WARNING — corpus gid counter wrapped past 2^20 games "
                            "in this process; gids are no longer unique and the "
                            "train/val split will merge games.\n");
        }
    }
    const char *res_str = (result > 0.75f) ? "1" : (result < 0.25f) ? "0" : "0.5";
    char fen[110];
    // Result-decay reference N_game: the last recorded root game-ply (the engine's
    // final recorded move).  Slightly short of the game's true terminal ply, same
    // approximation as the historical per-gid-max fallback.
    int final_game_ply = (rec.n_plies > 0) ? rec.plies[rec.n_plies - 1].game_ply : 0;

    for (int t = 0; t < rec.n_plies; t++) {
        const TDRecord &r = rec.plies[t];
        // Per-record root STM.  In harness mode this equals rec.engine_color for
        // every record; under internal self-play it alternates.
        int root_wtm = (int)r.root_wtm;

        // ---- Leaf row: static-eval label, depth 0 (outcome-only) ---------
        if (leaf_f) {
            int root_leaf_pov = ((int)r.wtm == root_wtm) ? r.score_root_stm
                                                         : -r.score_root_stm;
            // Leaf rows use the SAME test as the online trace (r.leaf_ok, i.e.
            // TDLEAF_LEAF_MATCH_CP), so the offline leaf population is exactly
            // the online one.  A leaf the search never valued is no more useful
            // offline than online.  dump_quiet_cp still applies as an additional
            // cap, so the gate is the tighter of the two and disabling the
            // leaf-match gate restores the historical behaviour.
            if (r.leaf_ok && abs(r.score_stm - root_leaf_pov) <= dump_quiet_cp) {
                int cp_white = r.wtm ? r.score_stm : -r.score_stm;
                // Column 8 "gate": what the quietness test compared cp against,
                // same POV as cp, so the gate is |cp - gate| <= QUIET_CP and can
                // be RE-CUT OFFLINE.  Dump wide once and every narrower gate is
                // a filter away — no second generation run, and the arms are
                // then perfectly paired (same games, same labels).
                int gate_white = r.wtm ? root_leaf_pov : -root_leaf_pov;
                if (cp_white <= dump_max_cp && cp_white >= -dump_max_cp) {
                    tdleaf_dump_fen(r.pos, r.wtm, fen);
                    fprintf(leaf_f, "%s\t%d\t%s\t%d\t0\t%u\t%d\t%d\n",
                            fen, cp_white, res_str, r.game_ply, dump_gid,
                            final_game_ply, gate_white);
                }
            }
        }

#if TDLEAF_REFRESH_DIAG
        if (diag_f)
            fprintf(diag_f, "%u\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%s\n",
                    dump_gid, r.game_ply, final_game_ply, (int)r.id_depth,
                    (int)r.wtm, root_wtm,
                    r.score_stm_actor, r.score_stm,
                    r.score_root_stm_actor, r.score_root_stm,
                    r.root_static_actor, r.root_static, res_str);
#endif

        // ---- Root row: search-score label, depth = achieved ID depth -----
        if (root_f) {
            if (abs(r.root_static - r.score_root_stm) <= dump_quiet_cp) {
                int cp_white = root_wtm ? r.score_root_stm : -r.score_root_stm;
                // Column 8 "gate": the root STATIC eval, same POV as cp.  The
                // gate was |cp - gate| <= QUIET_CP, so a wide dump re-cuts to
                // any narrower gate offline.  This is the quantity Part 1 found
                // the label's value is proportional to.
                int gate_white = root_wtm ? r.root_static : -r.root_static;
                if (cp_white <= dump_max_cp && cp_white >= -dump_max_cp) {
                    tdleaf_dump_fen(r.root_pos, (bool)root_wtm, fen);
                    fprintf(root_f, "%s\t%d\t%s\t%d\t%d\t%u\t%d\t%d\n",
                            fen, cp_white, res_str, r.game_ply, (int)r.id_depth,
                            dump_gid, final_game_ply, gate_white);
#if TDLEAF_REFRESH_DIAG
                    // Same row, same gate decision, ACTOR-VINTAGE label — the
                    // paired control that isolates the score refresh from the
                    // quietness-gate change.
                    if (stale_f) {
                        int cp_stale = root_wtm ? r.score_root_stm_actor
                                                : -r.score_root_stm_actor;
                        fprintf(stale_f, "%s\t%d\t%s\t%d\t%d\t%u\t%d\n",
                                fen, cp_stale, res_str, r.game_ply,
                                (int)r.id_depth, dump_gid, final_game_ply);
                    }
#endif
                }
            }
        }
    }
    if (leaf_f) fflush(leaf_f);   // survive process kills at match end
    if (root_f) fflush(root_f);
#if TDLEAF_REFRESH_DIAG
    if (diag_f) fflush(diag_f);
    if (stale_f) fflush(stale_f);
#endif
}

// ---------------------------------------------------------------------------
// Mini-batch: accumulate gradients across TDLEAF_BATCH_SIZE games before
// applying the Adam step.  This gives Adam a more reliable gradient signal
// per step, reducing single-game noise.
// ---------------------------------------------------------------------------
static int td_batch_pending = 0;  // games accumulated since last apply

// Uniform online step multiplier (see tdleaf.h).  Default 1.0 = unchanged.
float tdleaf_lr_scale = 1.0f;

// ---------------------------------------------------------------------------
// tdleaf_update_after_game — live pass: accumulate; apply every BATCH_SIZE games
// ---------------------------------------------------------------------------
void tdleaf_update_after_game(TDGameRecord &rec, float result, const char *save_path)
{
    int T = rec.n_plies;
    if (T < TDLEAF_MIN_PLIES) {
        fprintf(stderr, "TDLeaf: skipping short game (%d plies)\n", T);
        return;
    }

    // Optional leaf-position TSV dump (env TDLEAF_DUMP_TSV) — same games
    // that feed the TD update, so corpus and learning stay consistent.
    tdleaf_dump_game(rec, result);

    // Frozen (TDLEAF_FREEZE=1): the dump above still runs, but no gradients
    // accumulate — so no batch ever applies, and tdleaf_flush_batch stays a
    // no-op.  The .tdleaf.bin is never touched.
    if (tdleaf_frozen()) return;

    tdleaf_accumulate_game(rec, result);
    td_batch_pending++;

    if (td_batch_pending >= TDLEAF_BATCH_SIZE) {
        nnue_clip_gradients(TDLEAF_GRAD_CLIP_NORM);
        if (nnue_grad_normalize && td_grad_samples)
            nnue_scale_gradients(1.0f / (float)td_grad_samples);
        td_grad_samples = 0;
        nnue_apply_gradients(tdleaf_lr_scale);
        nnue_requantize_fc();

        if (save_path && save_path[0]) {
            if (!nnue_save_fc_weights(save_path))
                fprintf(stderr, "TDLeaf: failed to save weights to %s\n", save_path);
        }

        fprintf(stderr, "TDLeaf: applied batch of %d game(s), latest %d plies (result=%.1f)\n",
                td_batch_pending, T, (double)result);
        td_batch_pending = 0;
    } else {
        fprintf(stderr, "TDLeaf: accumulated %d-ply game (result=%.1f), batch %d/%d\n",
                T, (double)result, td_batch_pending, TDLEAF_BATCH_SIZE);
    }
}

// ---------------------------------------------------------------------------
// tdleaf_rebuild_record — reconstruct the derived snapshot fields of a
// TDRecord from its stored leaf position using the CURRENT weights: leaf
// accumulator/PSQT sums, active features, stack index.  Used by the trajectory
// learner, which ships only positions + scores (refresh off preserves exact
// online semantics; refresh_score=true — the learner's --refresh-scores — also
// re-evaluates score_stm so d[t] reflects the current network).
//
// Integer accumulator rebuilds equal the incremental PV-walked snapshots
// exactly (same FT weight rows, integer adds), so with unchanged weights this
// reproduces the online-recorded snapshot bit-for-bit.
// ---------------------------------------------------------------------------
void tdleaf_rebuild_record(TDRecord &r, bool refresh_score)
{
    // Rebuild leaf accumulator from the stored position.
    NNUEAccumulator fresh_acc;
    nnue_init_accumulator(fresh_acc, r.pos);
    memcpy(r.acc[0],  fresh_acc.acc[0],  NNUE_HALF_DIMS  * sizeof(int16_t));
    memcpy(r.acc[1],  fresh_acc.acc[1],  NNUE_HALF_DIMS  * sizeof(int16_t));
    memcpy(r.psqt[0], fresh_acc.psqt[0], NNUE_PSQT_BKTS * sizeof(int32_t));
    memcpy(r.psqt[1], fresh_acc.psqt[1], NNUE_PSQT_BKTS * sizeof(int32_t));

    // Re-enumerate active features (must match rebuilt accumulator).
    for (int p = 0; p < 2; p++) {
        int ksq = r.pos.plist[p][KING][1];
        r.n_ft[p] = 0;
        for (int sd = 0; sd < 2; sd++)
            for (int pt = PAWN; pt <= KING; pt++)
                for (int i = 1; i <= r.pos.plist[sd][pt][0]; i++) {
                    if (r.n_ft[p] >= NNUE_MAX_FT_PER_PERSP) goto ft_done_rebuild;
                    int fi = halfkav2_feature(p, ksq, r.pos.plist[sd][pt][i], pt, sd);
                    if (fi >= 0) r.ft_idx[p][r.n_ft[p]++] = fi;
                }
        ft_done_rebuild:;
    }

    // Stack index — same piece-count formula as tdleaf_record_ply's leaf path
    // (kings as the constant 2, PAWN..QUEEN from the piece lists).  The bucket
    // (pc-1)/4 is what the eval consumes, matching a re-eval at pc = stack*4 + 2.
    int pc = 2;
    for (int sd = 0; sd < 2; sd++)
        for (int pt = PAWN; pt <= QUEEN; pt++)
            pc += r.pos.plist[sd][pt][0];
    pc = (pc < 1) ? 1 : (pc > 32) ? 32 : pc;
    r.stack = (pc - 1) / 4;

#if TDLEAF_REFRESH_DIAG
    r.score_stm_actor       = r.score_stm;
    r.score_root_stm_actor  = r.score_root_stm;
    r.root_static_actor     = r.root_static;
#endif

    if (refresh_score) {
        const int leaf_actor = r.score_stm;
        r.score_stm = nnue_evaluate_acc_raw(r.acc, r.psqt, (int)r.wtm, pc);

        // Re-express the ROOT search score on current weights.  The root score
        // is the value backed up from the PV leaf, so it equals s*leaf_static
        // plus a residual delta (quiescence tail beyond the stored PV, TT
        // cutoff, aspiration bound):
        //     s     = +1 if leaf STM == root STM else -1   (leaf POV -> root POV)
        //     delta = score_root_stm - s*leaf_actor        (actor-vintage)
        //     new   = s*leaf_current + delta = old + s*(leaf_current - leaf_actor)
        // Holding delta fixed keeps the part the SEARCH contributed (which leaf
        // it chose) and re-values that leaf with the current net — so the label
        // stops lagging the learner by an actor-refresh epoch, while staying an
        // exact identity when the weights have not moved.
        //
        // Mate announcements (|score| near MATE) are not evaluations; shifting
        // them by an eval delta is meaningless, so they pass through untouched.
        if (r.score_root_stm > -MATE / 2 && r.score_root_stm < MATE / 2) {
            const int sgn = ((int)r.wtm == (int)r.root_wtm) ? 1 : -1;
            r.score_root_stm += sgn * (r.score_stm - leaf_actor);
        }

        // Root static eval on current weights, from an accumulator rebuilt on
        // the stored root position.  Without this the root quietness test
        // (|root_static - score_root_stm|) would compare a refreshed score
        // against an actor-vintage static — the same mixed-vintage error the
        // refresh exists to remove.  (rebuild_record is learner-only, and the
        // .tdg format always ships root_pos.)  Same piece-count formula as
        // tdleaf_record_ply's root path: PAWN..KING over both sides.
        NNUEAccumulator root_acc;
        nnue_init_accumulator(root_acc, r.root_pos);
        int pc_root = 0;
        for (int sd = 0; sd < 2; sd++)
            for (int pt = PAWN; pt <= KING; pt++)
                pc_root += r.root_pos.plist[sd][pt][0];
        pc_root = (pc_root < 1) ? 1 : (pc_root > 32) ? 32 : pc_root;
        r.root_static = nnue_evaluate_acc_raw(root_acc.acc, root_acc.psqt,
                                              (int)r.root_pos.wtm, pc_root);
    }
}

// ---------------------------------------------------------------------------
// tdleaf_flush_batch — apply any pending accumulated gradients (e.g., at
// session end or weight export).  No-op if no gradients are pending.
// ---------------------------------------------------------------------------
void tdleaf_flush_batch(const char *save_path)
{
    if (td_batch_pending <= 0) return;

    nnue_clip_gradients(TDLEAF_GRAD_CLIP_NORM);
    if (nnue_grad_normalize && td_grad_samples)
        nnue_scale_gradients(1.0f / (float)td_grad_samples);
    td_grad_samples = 0;
    nnue_apply_gradients(tdleaf_lr_scale);
    nnue_requantize_fc();

    if (save_path && save_path[0]) {
        if (!nnue_save_fc_weights(save_path))
            fprintf(stderr, "TDLeaf flush: failed to save weights to %s\n", save_path);
    }

    fprintf(stderr, "TDLeaf flush: applied partial batch of %d game(s)\n", td_batch_pending);
    td_batch_pending = 0;

    // End-of-session dump of the L2-clip telemetry so we still get a summary
    // even if the periodic cadence didn't tick on this run's call count.
    nnue_clip_gradient_stats_report();
}

// ---------------------------------------------------------------------------
// tdleaf_self_adjudicate — derive a game result without a protocol "result"
// command, so UCI mode (no game-over signal) can still feed the learner.
//
// Priority:
//   1. Terminal position on final_pos:
//        - no legal moves + in_check → mate (loser = side to move)
//        - no legal moves, not in check → stalemate (draw)
//        - fifty-move counter >= 100 → draw
//        - 3-fold repetition over `plist` (stride-2 same-STM hashes) → draw
//   2. Score-history adjudication (mirrors cutechess/fastchess defaults):
//        - last 6 plies' engine-POV score >= +600 cp → engine won
//        - last 6 plies' engine-POV score <= -600 cp → engine lost
//        - past move 40, last 8 plies' |engine-POV score| <= 10 cp → draw
//   3. Otherwise return false (caller should skip learning).
//
// Engine-POV score per ply: TDRecord stores leaf STM score; if the leaf STM
// matches rec.engine_color the leaf score is already engine-POV, else negate.
// ---------------------------------------------------------------------------
// Insufficient mating material: each side has no pawns/rooks/queens and at
// most one minor piece (KvK, KvKN, KvKB, KNvK, KBvK, KNvKN, KBvKB, KNvKB).
// Shared by UCI self-adjudication and the internal selfplay game loop.
bool tdleaf_insufficient_material(const position &p)
{
    for (int side = 0; side <= 1; side++) {
        int heavy = p.plist[side][PAWN][0] + p.plist[side][ROOK][0] +
                    p.plist[side][QUEEN][0];
        int minor = p.plist[side][KNIGHT][0] + p.plist[side][BISHOP][0];
        if (heavy != 0 || minor > 1) return false;
    }
    return true;
}

bool tdleaf_self_adjudicate(const TDGameRecord &rec,
                            const position &final_pos,
                            const uint64_t *plist,
                            int game_T,
                            float &out_result_white_pov)
{
    if (rec.n_plies == 0 || rec.engine_color < 0) return false;

    // ---- (1) Terminal position checks ------------------------------------
    {
        position scratch = final_pos;
        int mate = scratch.in_check_mate();   // 1 = mate, 2 = stalemate, 0 = neither
        if (mate == 1) {
            // Side to move on final_pos is the loser.
            out_result_white_pov = final_pos.wtm ? 0.0f : 1.0f;
            return true;
        }
        if (mate == 2) {
            out_result_white_pov = 0.5f;
            return true;
        }
    }
    if (final_pos.fifty >= 100) { out_result_white_pov = 0.5f; return true; }

    // 3-fold: count matches of final_pos.hcode within the last `fifty` plies
    // at stride 2 (same-STM repetitions).  `plist[game_T-1]` is the current
    // hash; we already count that as one occurrence.
    {
        int reps = 1;
        int floor = game_T - 1 - final_pos.fifty;
        if (floor < 0) floor = 0;
        for (int ri = game_T - 3; ri >= floor; ri -= 2) {
            if (plist[ri] == final_pos.hcode) {
                reps++;
                if (reps >= 3) { out_result_white_pov = 0.5f; return true; }
            }
        }
    }

    if (tdleaf_insufficient_material(final_pos)) {
        out_result_white_pov = 0.5f;
        return true;
    }

    // ---- (2) Score-history self-adjudication -----------------------------
    // Constants shared with the internal selfplay adjudicator (tdleaf.h).
    const int RESIGN_PLIES      = TDLEAF_RESIGN_PLIES;
    const int RESIGN_CP         = TDLEAF_RESIGN_CP;
    const int DRAW_PLIES        = TDLEAF_DRAW_PLIES;
    const int DRAW_CP           = TDLEAF_DRAW_CP;
    const int DRAW_MOVE_NUMBER  = TDLEAF_DRAW_MOVE_NUMBER;

    int n = rec.n_plies;

    // Root-position score is already from engine's POV (we only record on
    // engine moves, so root STM == engine_color at every entry).  We use
    // score_root_stm (not the leaf score) because it matches what the engine
    // reported via UCI `info ... score cp X` — i.e., exactly what cutechess /
    // fastchess sees when applying its own adjudication thresholds.

    if (n >= RESIGN_PLIES) {
        bool all_won = true, all_lost = true;
        for (int i = n - RESIGN_PLIES; i < n; i++) {
            int s = rec.plies[i].score_root_stm;
            if (s <  RESIGN_CP) all_won  = false;
            if (s > -RESIGN_CP) all_lost = false;
        }
        if (all_won) {
            out_result_white_pov = rec.engine_color ? 1.0f : 0.0f;
            return true;
        }
        if (all_lost) {
            out_result_white_pov = rec.engine_color ? 0.0f : 1.0f;
            return true;
        }
    }

    // game_T counts plies from game start (1-based).  Move number = (T-1)/2 + 1.
    int move_number = (game_T - 1) / 2 + 1;
    if (n >= DRAW_PLIES && move_number >= DRAW_MOVE_NUMBER) {
        bool all_drawish = true;
        for (int i = n - DRAW_PLIES; i < n; i++) {
            int s = rec.plies[i].score_root_stm;
            if (s > DRAW_CP || s < -DRAW_CP) { all_drawish = false; break; }
        }
        if (all_drawish) { out_result_white_pov = 0.5f; return true; }
    }

    // Ambiguous (e.g. time forfeit, unusual termination): skip learning.
    return false;
}

#endif // TDLEAF
