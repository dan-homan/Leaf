// Leaf TDLeaf(λ) online learning — implementation
// Compiled only when TDLEAF=1 (included by Leaf.cc after nnue.cpp).

#include "define.h"

#if TDLEAF

#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <unistd.h>     // getpid — leaf-dump per-process file naming
#include "chess.h"
#include "nnue.h"
#include "tdleaf.h"

// value[] lives in score.h (included earlier in the unity build).  Declared
// extern here so TDLEAF_SCORE_CLIP_PAWNS × value[PAWN] can be evaluated at
// runtime — the threshold tracks piece-value drift under TDLeaf.
extern int value[7];

// ---------------------------------------------------------------------------
// tdleaf_stack_norm_alpha — per-game per-stack record normalisation exponent.
// Defaults to TDLEAF_STACK_NORM_ALPHA; overridable by the env var of the same
// name for the A/B sweep.  Cached on first call.  Negative values are rejected
// (they would AMPLIFY the coherence this knob exists to suppress).
// ---------------------------------------------------------------------------
float tdleaf_stack_norm_alpha()
{
    static float alpha = -1.0f;
    if (alpha < 0.0f) {
        alpha = TDLEAF_STACK_NORM_ALPHA;
        const char *p = getenv("TDLEAF_STACK_NORM_ALPHA");
        if (p && *p) {
            char *end = nullptr;
            float v = strtof(p, &end);
            if (end == p || *end || !(v >= 0.0f)) {
                fprintf(stderr, "TDLeaf: TDLEAF_STACK_NORM_ALPHA=%s is not a "
                                "non-negative number.\n", p);
                exit(1);
            }
            alpha = v;
        }
    }
    return alpha;
}

// tdleaf_feature_dedup — per-feature within-game vote normalisation exponent.
// Defaults to TDLEAF_FEATURE_DEDUP; overridable by the env var of the same
// name.  Cached on first call.  Negative values are rejected (they would
// AMPLIFY repeated features rather than pool them).
// ---------------------------------------------------------------------------
float tdleaf_feature_dedup()
{
    static float beta = -1.0f;
    if (beta < 0.0f) {
        beta = TDLEAF_FEATURE_DEDUP;
        const char *p = getenv("TDLEAF_FEATURE_DEDUP");
        if (p && *p) {
            char *end = nullptr;
            float v = strtof(p, &end);
            if (end == p || *end || !(v >= 0.0f)) {
                fprintf(stderr, "TDLeaf: TDLEAF_FEATURE_DEDUP=%s is not a "
                                "non-negative number.\n", p);
                exit(1);
            }
            beta = v;
        }
    }
    return beta;
}

// tdleaf_feature_rbar — min prior games before the scale-neutral rbar/r weight
// is trusted for a cell (0 = mode disabled).  Mutually exclusive with
// TDLEAF_FEATURE_DEDUP; setting both is a hard error rather than a silent
// precedence rule.
// ---------------------------------------------------------------------------
int tdleaf_feature_rbar()
{
    static int min_games = -1;
    if (min_games < 0) {
        min_games = TDLEAF_FEATURE_RBAR;
        const char *p = getenv("TDLEAF_FEATURE_RBAR");
        if (p && *p) {
            char *end = nullptr;
            long v = strtol(p, &end, 10);
            if (end == p || *end || v < 0 || v > 1000000) {
                fprintf(stderr, "TDLeaf: TDLEAF_FEATURE_RBAR=%s is not a "
                                "non-negative integer.\n", p);
                exit(1);
            }
            min_games = (int)v;
        }
        if (min_games > 0 && tdleaf_feature_dedup() > 0.0f) {
            fprintf(stderr, "TDLeaf: TDLEAF_FEATURE_RBAR and TDLEAF_FEATURE_DEDUP "
                            "are mutually exclusive — set exactly one.\n");
            exit(1);
        }
    }
    return min_games;
}

// ---------------------------------------------------------------------------
// Feature-repetition histogram (env TDLEAF_REP_HIST=1) — calibration for the
// above.  Buckets the per-game occurrence count r of every feature CONTRIBUTION
// (so a feature used 13 times adds 13 counts to r's bucket), separately for the
// FT-row and PSQT (row,bucket) granularities, and weights each by |grad_scale|
// so the report reads as a share of gradient MASS, not just of contributions.
// Independent of tdleaf_feature_dedup() so the baseline distribution can be
// measured with the normalisation off.
// ---------------------------------------------------------------------------
bool tdleaf_rep_hist_enabled()
{
    static int on = -1;
    if (on < 0) {
        const char *p = getenv("TDLEAF_REP_HIST");
        on = (p && *p && strcmp(p, "0") != 0) ? 1 : 0;
    }
    return on == 1;
}

// Bucket edges: r == 1, 2, 3, 4-7, 8-15, 16-31, 32+
static const int TD_REP_NB = 7;
static const char *td_rep_label[TD_REP_NB] =
    { "1", "2", "3", "4-7", "8-15", "16-31", "32+" };
static double td_rep_n_ft  [TD_REP_NB];   // contributions, FT-row granularity
static double td_rep_m_ft  [TD_REP_NB];   // |grad| mass,   FT-row granularity
static double td_rep_n_pq  [TD_REP_NB];   // contributions, PSQT (row,bucket)
static double td_rep_m_pq  [TD_REP_NB];   // |grad| mass,   PSQT (row,bucket)
// Coverage: how many PRIOR games the cell had when it contributed.  This is
// what governs the quality of the rbar estimate (its residual drift is
// ~CV/sqrt(n)), so it says how much of the learning signal is actually being
// normalised well at a given pseudo-count k.  Bands: 0, 1-3, 4-15, 16-63, 64+.
static const int TD_COV_NB = 5;
static const char *td_cov_label[TD_COV_NB] = { "0", "1-3", "4-15", "16-63", "64+" };
static double td_cov_n[TD_COV_NB];

// ---------------------------------------------------------------------------
// Per-feature ACROSS-GAME statistics of r — the quantity the variance argument
// actually turns on.  The marginal histogram above sizes the intervention (how
// much gradient mass sits at high r); it cannot say whether Adam absorbs it.
//
// Writing a cell's batch gradient as r*g0, the Adam step is r/sqrt(E[r^2]).
// With mean m and standard deviation s of r for THAT cell:
//     mean step = 1 / sqrt(1 + CV^2)      CV = s/m
//     step CV   = CV
// So CV(r) is *exactly* the step-size coefficient of variation.  A feature
// whose r is a stable 20 has its multiplier fully absorbed by v and costs
// nothing; one swinging 1..40 is the one that injects noise.  Small CV
// everywhere ⇒ the mechanism does not bite and TDLEAF_FEATURE_DEDUP is
// treating a non-problem.
//
// Accumulated once per (cell, game) via the first-visit trick in the reset
// walk.  Lazily allocated so a build with the diagnostic off carries no dead
// BSS (the ~40 MB replay-buffer incident in Part 2 is the cautionary tale).
// Weighted by total contributions (sum of r), not |grad| mass, which would
// need a further per-game side array; the mass-weighted view is already in the
// marginal histogram.
// ---------------------------------------------------------------------------
static uint32_t *rv_n   = nullptr;   // games in which this PSQT cell appeared
static double   *rv_sr  = nullptr;   // sum of r
static double   *rv_sr2 = nullptr;   // sum of r^2
// FT rows are indexed by fi alone (no bucket), so the rbar mode needs its own
// per-row running mean.  sr2 is not tracked here — the CV report is PSQT-only.
static uint32_t *rvf_n  = nullptr;
static double   *rvf_sr = nullptr;
// Shrinkage priors for the rbar mode: the mean r of a cell's MATERIAL BUCKET
// (endgame features persist far longer than opening ones, so the bucket is a
// strong predictor), and a single global mean for FT rows, which have no
// bucket.  Both are (cell, game) sample means, accumulated in the teardown
// walk alongside rv_*.  One game supplies thousands of samples, so the warmup
// guard below clears after the first game or two.
static double rb_bkt_sr[NNUE_PSQT_BKTS];
static double rb_bkt_n [NNUE_PSQT_BKTS];
static double rb_ft_sr = 0.0, rb_ft_n = 0.0;
static const double RB_PRIOR_MIN_SAMPLES = 1000.0;

static bool tdleaf_rv_alloc()
{
    if (rv_n) return true;
    size_t n = (size_t)NNUE_FT_INPUTS * NNUE_PSQT_BKTS;
    rv_n   = (uint32_t *)calloc(n, sizeof(uint32_t));
    rv_sr  = (double   *)calloc(n, sizeof(double));
    rv_sr2 = (double   *)calloc(n, sizeof(double));
    rvf_n  = (uint32_t *)calloc(NNUE_FT_INPUTS, sizeof(uint32_t));
    rvf_sr = (double   *)calloc(NNUE_FT_INPUTS, sizeof(double));
    if (!rv_n || !rv_sr || !rv_sr2 || !rvf_n || !rvf_sr) {
        fprintf(stderr, "TDLeaf: rep-hist variance tables allocation failed\n");
        free(rv_n); free(rv_sr); free(rv_sr2); free(rvf_n); free(rvf_sr);
        rv_n = nullptr; rv_sr = nullptr; rv_sr2 = nullptr;
        rvf_n = nullptr; rvf_sr = nullptr;
        return false;
    }
    return true;
}

static inline int td_rep_bucket(int r)
{
    if (r <= 1)  return 0;
    if (r == 2)  return 1;
    if (r == 3)  return 2;
    if (r < 8)   return 3;
    if (r < 16)  return 4;
    if (r < 32)  return 5;
    return 6;
}

void tdleaf_rep_hist_report(FILE *out)
{
    double tn_ft = 0, tm_ft = 0, tn_pq = 0, tm_pq = 0;
    for (int i = 0; i < TD_REP_NB; i++) {
        tn_ft += td_rep_n_ft[i]; tm_ft += td_rep_m_ft[i];
        tn_pq += td_rep_n_pq[i]; tm_pq += td_rep_m_pq[i];
    }
    if (tn_pq <= 0.0) return;
    fprintf(out, "TDLeaf rep-hist (cumulative; %% of contributions / %% of |grad| mass)\n");
    fprintf(out, "  r        FT-row              PSQT(row,bkt)\n");
    for (int i = 0; i < TD_REP_NB; i++)
        fprintf(out, "  %-6s %6.2f%% / %6.2f%%     %6.2f%% / %6.2f%%\n",
                td_rep_label[i],
                100.0 * td_rep_n_ft[i] / tn_ft, 100.0 * td_rep_m_ft[i] / std::max(tm_ft, 1e-30),
                100.0 * td_rep_n_pq[i] / tn_pq, 100.0 * td_rep_m_pq[i] / std::max(tm_pq, 1e-30));
    double share_ft = 0, share_pq = 0;
    for (int i = 1; i < TD_REP_NB; i++) { share_ft += td_rep_m_ft[i]; share_pq += td_rep_m_pq[i]; }
    fprintf(out, "  mass at r>=2:  FT %.2f%%   PSQT %.2f%%   (n=%.0f contributions)\n",
            100.0 * share_ft / std::max(tm_ft, 1e-30),
            100.0 * share_pq / std::max(tm_pq, 1e-30), tn_pq);

    // ---- Across-game CV(r) per PSQT cell -----------------------------------
    // CV is the step-size coefficient of variation (see the note above), so
    // these bands read directly as "how much of the learning signal sits on
    // cells whose Adam step swings by this much".
    if (rv_n) {
        static const int NCV = 6;
        static const char *cvlab[NCV] =
            { "<0.1", "0.1-0.25", "0.25-0.5", "0.5-1.0", "1.0-2.0", ">2.0" };
        double w[NCV] = {0}, wtot = 0;      // weighted by total contributions
        double cv_mean_num = 0;
        size_t cells = 0, cells_ge2 = 0;
        size_t ncell = (size_t)NNUE_FT_INPUTS * NNUE_PSQT_BKTS;
        for (size_t i = 0; i < ncell; i++) {
            if (rv_n[i] == 0) continue;
            cells++;
            if (rv_n[i] < 2) continue;      // need >=2 games to have a variance
            cells_ge2++;
            double n  = (double)rv_n[i];
            double m  = rv_sr[i] / n;
            double va = rv_sr2[i] / n - m * m;
            if (va < 0) va = 0;             // rounding
            double cv = (m > 0) ? sqrt(va) / m : 0.0;
            double wt = rv_sr[i];           // total contributions from this cell
            int b = (cv < 0.1) ? 0 : (cv < 0.25) ? 1 : (cv < 0.5) ? 2
                  : (cv < 1.0) ? 3 : (cv < 2.0) ? 4 : 5;
            w[b] += wt; wtot += wt; cv_mean_num += cv * wt;
        }
        if (wtot > 0) {
            fprintf(out, "  CV(r) across games, PSQT cells (%% of contributions):\n   ");
            for (int i = 0; i < NCV; i++)
                fprintf(out, " %s=%.1f%%", cvlab[i], 100.0 * w[i] / wtot);
            double cvbar = cv_mean_num / wtot;
            fprintf(out, "\n  weighted mean CV = %.2f  ->  mean step attenuation "
                         "1/sqrt(1+CV^2) = %.2f  (cells seen %zu, with >=2 games %zu)\n",
                    cvbar, 1.0 / sqrt(1.0 + cvbar * cvbar), cells, cells_ge2);
        }
        double covtot = 0;
        for (int i = 0; i < TD_COV_NB; i++) covtot += td_cov_n[i];
        if (covtot > 0) {
            fprintf(out, "  prior games per contributing cell (rbar estimate "
                         "quality, %% of contributions):\n   ");
            for (int i = 0; i < TD_COV_NB; i++)
                fprintf(out, " n=%s:%.1f%%", td_cov_label[i],
                        100.0 * td_cov_n[i] / covtot);
            fputc('\n', out);
        }
    }
    fflush(out);
}

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
        "TDLEAF_STACK_NORM_ALPHA",// per-game per-stack record normalisation exponent (REJECTED, see 6.10)
        "TDLEAF_FEATURE_DEDUP",   // per-feature within-game vote normalisation exponent
        "TDLEAF_FEATURE_RBAR",    // scale-neutral rbar/r mode (min prior games)
        "TDLEAF_REP_HIST",        // diagnostic: feature-repetition histogram
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
            "score_clip=%.1fxP id_var_sigma2=%.0f stack_norm_alpha=%.3g "
            "feature_dedup=%.3g feature_rbar=%d rep_hist=%d\n"
            "TDLeaf LR0: FC=%.4g FC2=%.4g FC_bias=%.4g FT=%.4g FT_bias=%.4g PSQT=%.4g\n",
            (double)TDLEAF_K, (double)TDLEAF_LAMBDA, TDLEAF_BATCH_SIZE,
            (double)TDLEAF_GRAD_CLIP_NORM, (double)TDLEAF_WEIGHT_DECAY,
            (double)TDLEAF_SCORE_CLIP_PAWNS, (double)TDLEAF_ID_VAR_SIGMA2,
            (double)tdleaf_stack_norm_alpha(),
            (double)tdleaf_feature_dedup(), tdleaf_feature_rbar(),
            (int)tdleaf_rep_hist_enabled(),
            (double)TDLEAF_ADAM_LR0, (double)TDLEAF_ADAM_FC2_LR0,
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

    // Capture engine color on the first ply of a fresh game.  Every recorded
    // ply has root_pos.wtm == engine's color (we only record on engine moves).
    if (rec.n_plies == 0) rec.engine_color = (int8_t)root_pos.wtm;

    // Walk the PV, updating the position and accumulator incrementally.
    // We use two alternating accumulator slots to avoid unnecessary copies.
    NNUEAccumulator acc_a = root_acc;   // current leaf accumulator
    NNUEAccumulator acc_b;              // scratch for next step
    position cur = root_pos;
    int pv_len = 0;

    for (int k = 0; k < MAXD && pv[k].t != NOMOVE; k++) {
        position next = cur;
        if (!next.exec_move(pv[k], 0)) break;  // illegal — stop here
        nnue_record_delta(acc_b, cur, next, pv[k]);
        nnue_apply_delta(acc_b, acc_a, next);
        cur   = next;
        acc_a = acc_b;
        pv_len++;
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
static void tdleaf_accumulate_game(TDGameRecord &rec, float result)
{
    int T = rec.n_plies;

    // 1. Convert scores to White-POV sigmoid values d[t] ∈ (0,1)
    static float d[MAX_GAME_PLY];
    static float score_w_cp[MAX_GAME_PLY];
    for (int t = 0; t < T; t++) {
        score_w_cp[t] = rec.plies[t].wtm
                        ?  (float)rec.plies[t].score_stm
                        : -(float)rec.plies[t].score_stm;
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
        int dply = rec.plies[t + 1].game_ply - rec.plies[t].game_ply;
        if (dply < 1) dply = 1;
        float trace_decay = (dply == 1) ? lambda : powf(lambda, (float)dply);
        e[t] = delta_d + trace_decay * e[t + 1];
    }

    // 3. For each ply, run FP32 forward pass + accumulate gradients
    const float cp_factor = 100.0f / 5776.0f;

    // Per-game, per-stack record normalisation (see TDLEAF_STACK_NORM_ALPHA in
    // tdleaf.h).  Count this game's records per FC/PSQT material bucket, then
    // divide each record's gradient by pow(count, alpha) so a game's endgame
    // tail — dozens of near-identical positions carrying the same-sign
    // saturated outcome term into the same bucket — cannot outvote its sparse,
    // decorrelated opening records within one Adam step.
    // At alpha == 0 the divisor is skipped entirely rather than computed as
    // pow(n, 0): that keeps the default arithmetic byte-for-byte identical to
    // the pre-knob code, which the actor/learner bit-exactness gate relies on.
    const float stack_alpha = tdleaf_stack_norm_alpha();
    float stack_norm[NNUE_LAYER_STACKS];
    for (int s = 0; s < NNUE_LAYER_STACKS; s++) stack_norm[s] = 1.0f;
    if (stack_alpha > 0.0f) {
        int n_stack[NNUE_LAYER_STACKS] = {0};
        for (int t = 0; t < T; t++) {
            int s = rec.plies[t].stack;
            if (s >= 0 && s < NNUE_LAYER_STACKS) n_stack[s]++;
        }
        for (int s = 0; s < NNUE_LAYER_STACKS; s++)
            if (n_stack[s] > 1)
                stack_norm[s] = 1.0f / powf((float)n_stack[s], stack_alpha);
    }

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

    // Per-record gradient scales, hoisted out of the accumulation loop so the
    // feature-occupancy pass below can count exactly the records that will
    // contribute (grad_scale != 0) and weight the histogram by |grad_scale|.
    // The arithmetic is unchanged from the in-loop version it replaces.
    //
    // wtm_sign converts ∂d_t/∂w (white-POV utility we want to ascend) into the
    // descent-form gradient expected by nnue_apply_gradients (which does
    // w -= LR × step on the supplied "loss" gradient).  score_white = wtm ?
    // +score_stm : -score_stm; nnue_forward_fp32 backprops ∂(stm-POV score)/∂w,
    // so the white-POV sign is (wtm ? +1 : -1) and the loss-form sign we pass
    // downstream is its negative — hence (wtm ? -1 : +1).
    static float gs[MAX_GAME_PLY];
    for (int t = 0; t < T; t++) {
        float sig_grad  = d[t] * (1.0f - d[t]) / TDLEAF_K;
        float wtm_sign  = rec.plies[t].wtm ? -1.0f : 1.0f;
        float id_weight = 1.0f / (1.0f + rec.plies[t].id_score_variance / TDLEAF_ID_VAR_SIGMA2);
        float g = e[t] * sig_grad * cp_factor * wtm_sign * id_weight;
        int s = rec.plies[t].stack;
        if (s >= 0 && s < NNUE_LAYER_STACKS) g *= stack_norm[s];
        gs[t] = g;
    }

    // ---- Per-feature within-game occupancy (TDLEAF_FEATURE_DEDUP / _REP_HIST)
    // r = the number of contributions THIS GAME makes to a given FT row, and to
    // a given PSQT (row, bucket).  Counted over exactly the records the
    // accumulation loop will use, both perspectives, matching its indexing:
    // nnue_accumulate_gradients writes grad_psqt_w[fi*PSQT_BKTS + s] with the
    // record's own stack s for BOTH perspectives.
    // Each feature can occur at most once per perspective per record, so
    // r <= 2*T <= 2*MAX_GAME_PLY, which bounds the reciprocal table below.
    const float dedup_beta = tdleaf_feature_dedup();
    const int   rbar_k     = tdleaf_feature_rbar();
    const bool  want_hist  = tdleaf_rep_hist_enabled();
    const bool  want_rep   = (dedup_beta > 0.0f) || (rbar_k > 0) || want_hist;
    // The rbar mode needs the across-game running means maintained every game,
    // not only when the histogram is on.
    const bool  want_stats = want_hist || (rbar_k > 0);
    static uint16_t feat_rep_ft[NNUE_FT_INPUTS];                          //  44 KB
    static uint16_t feat_rep_pq[(size_t)NNUE_FT_INPUTS * NNUE_PSQT_BKTS]; // 352 KB
    static const int REP_MAX = 2 * MAX_GAME_PLY + 1;
    static float feat_rep_recip[REP_MAX + 1];
    // Precomputed per-cell weights handed to nnue_accumulate_gradients.  Only
    // cells this game touches are written, and only those are read back during
    // its accumulation, so no reset pass is needed for them.
    static float feat_w_ft[NNUE_FT_INPUTS];                               //  88 KB
    static float feat_w_pq[(size_t)NNUE_FT_INPUTS * NNUE_PSQT_BKTS];      // 704 KB

    // Walk helper: visit every (record, perspective, feature) the accumulation
    // loop will touch.  Used three times — count, histogram, reset — so the
    // three passes cannot drift out of sync with each other.
    auto walk_features = [&](auto &&fn) {
        for (int t = 0; t < T; t++) {
            if (gs[t] == 0.0f) continue;
            const TDRecord &r = rec.plies[t];
            int s = r.stack;
            if (s < 0 || s >= NNUE_PSQT_BKTS) continue;
            for (int p = 0; p < 2; p++)
                for (int k = 0; k < r.n_ft[p]; k++) {
                    int fi = r.ft_idx[p][k];
                    if (fi < 0 || fi >= NNUE_FT_INPUTS) continue;
                    fn(t, fi, (size_t)fi * NNUE_PSQT_BKTS + (size_t)s);
                }
        }
    };

    if (want_rep) {
        walk_features([&](int, int fi, size_t pi) {
            if (feat_rep_ft[fi] < REP_MAX) feat_rep_ft[fi]++;
            if (feat_rep_pq[pi] < REP_MAX) feat_rep_pq[pi]++;
        });
    }
    if (want_hist) {
        walk_features([&](int t, int fi, size_t pi) {
            double mass = fabs((double)gs[t]);
            int bf = td_rep_bucket(feat_rep_ft[fi]);
            td_rep_n_ft[bf] += 1.0; td_rep_m_ft[bf] += mass;
            int bp = td_rep_bucket(feat_rep_pq[pi]);
            td_rep_n_pq[bp] += 1.0; td_rep_m_pq[bp] += mass;
            if (rv_n) {
                uint32_t np = rv_n[pi];   // prior games (teardown runs after this)
                int cb = (np == 0) ? 0 : (np < 4) ? 1 : (np < 16) ? 2
                       : (np < 64) ? 3 : 4;
                td_cov_n[cb] += 1.0;
            }
        });
    }
    if (dedup_beta > 0.0f) {
        // Exponent mode: w = r^-beta, from a table built once per process.
        static float recip_built_for = -1.0f;
        if (recip_built_for != dedup_beta) {
            recip_built_for = dedup_beta;
            feat_rep_recip[0] = 1.0f;
            for (int r = 1; r <= REP_MAX; r++)
                feat_rep_recip[r] = (r == 1) ? 1.0f
                                             : 1.0f / powf((float)r, dedup_beta);
        }
        walk_features([&](int, int fi, size_t pi) {
            feat_w_ft[fi] = feat_rep_recip[feat_rep_ft[fi]];
            feat_w_pq[pi] = feat_rep_recip[feat_rep_pq[pi]];
        });
        g_feat_w_ft   = feat_w_ft;
        g_feat_w_psqt = feat_w_pq;
    } else if (rbar_k > 0 && tdleaf_rv_alloc()) {
        // Scale-neutral mode: w = rbar_shrunk / r, so the current game's r
        // cancels exactly and only the (slowly drifting) rbar survives.
        // rbar_shrunk = (n*rbar_cell + k*rbar_prior) / (n + k), with the stats
        // taken over PRIOR games only — they are updated in the teardown walk
        // below, after this game's accumulation, so there is no self-reference.
        const double k = (double)rbar_k;
        double prior_bkt[NNUE_PSQT_BKTS];
        for (int b = 0; b < NNUE_PSQT_BKTS; b++)
            prior_bkt[b] = (rb_bkt_n[b] >= RB_PRIOR_MIN_SAMPLES)
                         ? rb_bkt_sr[b] / rb_bkt_n[b] : 0.0;   // 0 = not ready
        const double prior_ft = (rb_ft_n >= RB_PRIOR_MIN_SAMPLES)
                              ? rb_ft_sr / rb_ft_n : 0.0;
        walk_features([&](int, int fi, size_t pi) {
            double r_ft = (double)feat_rep_ft[fi];
            double r_pq = (double)feat_rep_pq[pi];
            if (prior_ft > 0.0 && r_ft > 0.0) {
                double rbar = (rvf_sr[fi] + k * prior_ft) / ((double)rvf_n[fi] + k);
                feat_w_ft[fi] = (float)(rbar / r_ft);
            } else feat_w_ft[fi] = 1.0f;
            double pb = prior_bkt[pi % NNUE_PSQT_BKTS];
            if (pb > 0.0 && r_pq > 0.0) {
                double rbar = (rv_sr[pi] + k * pb) / ((double)rv_n[pi] + k);
                feat_w_pq[pi] = (float)(rbar / r_pq);
            } else feat_w_pq[pi] = 1.0f;
        });
        g_feat_w_ft   = feat_w_ft;
        g_feat_w_psqt = feat_w_pq;
    }

    for (int t = 0; t < T; t++) {
        float grad_scale = gs[t];

        if (trace_f) {
            const TDRecord &r = rec.plies[t];
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
            act.stack = rec.plies[t].stack;
            nnue_forward_fp32(rec.plies[t].acc, rec.plies[t].psqt,
                              rec.plies[t].wtm, act);
            memcpy(act.acc_raw[0], rec.plies[t].acc[0], NNUE_HALF_DIMS * sizeof(int16_t));
            memcpy(act.acc_raw[1], rec.plies[t].acc[1], NNUE_HALF_DIMS * sizeof(int16_t));
            act.n_ft[0] = rec.plies[t].n_ft[0];
            act.n_ft[1] = rec.plies[t].n_ft[1];
            memcpy(act.ft_idx[0], rec.plies[t].ft_idx[0], rec.plies[t].n_ft[0] * sizeof(int));
            memcpy(act.ft_idx[1], rec.plies[t].ft_idx[1], rec.plies[t].n_ft[1] * sizeof(int));

            // Dense piece value gradient: stm_count − opp_count per piece type.
            int stm_p = rec.plies[t].wtm ? 1 : 0;
            for (int pt = PAWN; pt <= KING; pt++)
                act.piece_count_diff[pt - 1] = (int8_t)(rec.plies[t].pos.plist[stm_p][pt][0]
                                                       - rec.plies[t].pos.plist[stm_p ^ 1][pt][0]);

            nnue_accumulate_gradients(act, grad_scale);
        }
    }

    // ---- Tear down the per-game occupancy tables --------------------------
    // Zero exactly the entries the counting pass incremented (same walk, so no
    // drift) rather than memset-ing 396 KB per game, and drop the pointers so
    // every other caller of nnue_accumulate_gradients — the offline batch
    // trainer above all — sees the mechanism disabled.
    // First-visit trick: the counting pass left every touched cell non-zero, and
    // this walk zeroes it, so the first visit to a cell is the only one that
    // sees a non-zero r.  That gives exactly one (cell, game) sample for the
    // across-game statistics without a second dirty list.
    if (want_rep) {
        const bool rv = want_stats && tdleaf_rv_alloc();
        walk_features([&](int, int fi, size_t pi) {
            if (rv && feat_rep_pq[pi] != 0) {
                double r = (double)feat_rep_pq[pi];
                rv_n[pi]++; rv_sr[pi] += r; rv_sr2[pi] += r * r;
                int b = (int)(pi % NNUE_PSQT_BKTS);
                rb_bkt_n[b] += 1.0; rb_bkt_sr[b] += r;      // per-bucket prior
            }
            if (rv && feat_rep_ft[fi] != 0) {
                double r = (double)feat_rep_ft[fi];
                rvf_n[fi]++; rvf_sr[fi] += r;
                rb_ft_n += 1.0; rb_ft_sr += r;              // global FT prior
            }
            feat_rep_ft[fi] = 0;
            feat_rep_pq[pi] = 0;
        });
    }
    g_feat_w_ft   = nullptr;
    g_feat_w_psqt = nullptr;
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
// Both apply |cp| <= TDLEAF_DUMP_MAX_CP (default 1500).  QUIET_CP default 60
// (TDLEAF_DUMP_QUIET_CP).
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
    static int      dump_quiet_cp = 60;
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
                        fprintf(f, "fen\tcp\tresult\tply\tdepth\tgid\tendply\n");
                    }
                } else {
                    fprintf(stderr, "TDLeaf: cannot open dump file %s\n", path);
                }
                return f;
            };
            leaf_f = open_dump("leaf");
            root_f = open_dump("root");
            if (leaf_f && root_f)
                fprintf(stderr, "TDLeaf: dumping leaf+root positions to %s.%d.{leaf,root}.tsv "
                                "(quiet<=%d cp, max=%d cp)\n",
                        prefix, (int)getpid(), dump_quiet_cp, dump_max_cp);
            // gid: unique across concurrent processes (pid in high bits).
            dump_gid = ((uint32_t)getpid() & 0xFFF) << 20;
        }
    }
    if (!leaf_f && !root_f) return;

    dump_gid++;
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
            if (abs(r.score_stm - root_leaf_pov) <= dump_quiet_cp) {
                int cp_white = r.wtm ? r.score_stm : -r.score_stm;
                if (cp_white <= dump_max_cp && cp_white >= -dump_max_cp) {
                    tdleaf_dump_fen(r.pos, r.wtm, fen);
                    fprintf(leaf_f, "%s\t%d\t%s\t%d\t0\t%u\t%d\n",
                            fen, cp_white, res_str, r.game_ply, dump_gid,
                            final_game_ply);
                }
            }
        }

        // ---- Root row: search-score label, depth = achieved ID depth -----
        if (root_f) {
            if (abs(r.root_static - r.score_root_stm) <= dump_quiet_cp) {
                int cp_white = root_wtm ? r.score_root_stm : -r.score_root_stm;
                if (cp_white <= dump_max_cp && cp_white >= -dump_max_cp) {
                    tdleaf_dump_fen(r.root_pos, (bool)root_wtm, fen);
                    fprintf(root_f, "%s\t%d\t%s\t%d\t%d\t%u\t%d\n",
                            fen, cp_white, res_str, r.game_ply, (int)r.id_depth,
                            dump_gid, final_game_ply);
                }
            }
        }
    }
    if (leaf_f) fflush(leaf_f);   // survive process kills at match end
    if (root_f) fflush(root_f);
}

// ---------------------------------------------------------------------------
// Mini-batch: accumulate gradients across TDLEAF_BATCH_SIZE games before
// applying the Adam step.  This gives Adam a more reliable gradient signal
// per step, reducing single-game noise.
// ---------------------------------------------------------------------------
static int td_batch_pending = 0;  // games accumulated since last apply

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
        nnue_apply_gradients();
        nnue_requantize_fc();

        if (save_path && save_path[0]) {
            if (!nnue_save_fc_weights(save_path))
                fprintf(stderr, "TDLeaf: failed to save weights to %s\n", save_path);
        }

        fprintf(stderr, "TDLeaf: applied batch of %d game(s), latest %d plies (result=%.1f)\n",
                td_batch_pending, T, (double)result);
        // Feature-repetition calibration (TDLEAF_REP_HIST=1).  Cumulative, so
        // one report per batch is enough to watch it converge; grep the last
        // one out of the learner log.
        if (tdleaf_rep_hist_enabled()) tdleaf_rep_hist_report(stderr);
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

    if (refresh_score)
        r.score_stm = nnue_evaluate_acc_raw(r.acc, r.psqt, (int)r.wtm, pc);
}

// ---------------------------------------------------------------------------
// tdleaf_flush_batch — apply any pending accumulated gradients (e.g., at
// session end or weight export).  No-op if no gradients are pending.
// ---------------------------------------------------------------------------
void tdleaf_flush_batch(const char *save_path)
{
    if (td_batch_pending <= 0) return;

    nnue_clip_gradients(TDLEAF_GRAD_CLIP_NORM);
    nnue_apply_gradients();
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
