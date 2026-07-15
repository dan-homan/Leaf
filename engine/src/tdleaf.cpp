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

// Defined with the learning-target selection block below; needed here so
// tdleaf_record_ply can snapshot root data when root learning is on.
static bool tdleaf_root_enabled();

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
    h_code key_own = 0, key_reply = 0;  // prediction-gate keys (hybrid target)

    for (int k = 0; k < MAXD && pv[k].t != NOMOVE; k++) {
        position next = cur;
        if (!next.exec_move(pv[k], 0)) break;  // illegal — stop here
        nnue_record_delta(acc_b, cur, next, pv[k]);
        nnue_apply_delta(acc_b, acc_a, next);
        cur   = next;
        acc_a = acc_b;
        pv_len++;
        if      (pv_len == 1) key_own   = cur.hcode;
        else if (pv_len == 2) key_reply = cur.hcode;
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
    r.root_key          = root_pos.hcode;      // prediction-gate keys (hybrid target)
    r.key_own           = key_own;
    r.key_reply         = key_reply;
    r.stack             = (pc - 1) / 4;
    r.id_score_variance = id_var;
    r.pos               = cur;  // store leaf position for Flavor A replay
    r.id_depth          = (int8_t)((search_depth < 1) ? 1 :
                                   (search_depth > 127) ? 127 : search_depth);
    const bool want_root = tdleaf_root_enabled();
    if (tdleaf_dump_wanted() || want_root) {
        // Root snapshot + static eval for the root-row TSV dump and for
        // online root learning (TDLEAF_ROOT=1).
        r.root_pos = root_pos;
        int pc_root = 0;
        for (int sd = 0; sd < 2; sd++)
            for (int pt = PAWN; pt <= KING; pt++)
                pc_root += root_pos.plist[sd][pt][0];
        pc_root = (pc_root < 1) ? 1 : (pc_root > 32) ? 32 : pc_root;
        r.root_static = nnue_evaluate(root_acc, (int)root_pos.wtm, pc_root);
        if (want_root) {
            // Root gradient snapshot: accumulator, PSQT sums, stack, and
            // active features — mirrors the leaf snapshot so the update pass
            // can run nnue_forward_fp32 + nnue_accumulate_gradients on the
            // root exactly as it does on the leaf.
            memcpy(r.root_acc[0],  root_acc.acc[0],  NNUE_HALF_DIMS * sizeof(int16_t));
            memcpy(r.root_acc[1],  root_acc.acc[1],  NNUE_HALF_DIMS * sizeof(int16_t));
            memcpy(r.root_psqt[0], root_acc.psqt[0], NNUE_PSQT_BKTS * sizeof(int32_t));
            memcpy(r.root_psqt[1], root_acc.psqt[1], NNUE_PSQT_BKTS * sizeof(int32_t));
            r.root_stack = (pc_root - 1) / 4;
            for (int p = 0; p < 2; p++) {
                int ksq = root_pos.plist[p][KING][1];
                r.root_n_ft[p] = 0;
                for (int sd = 0; sd < 2; sd++)
                    for (int pt = PAWN; pt <= KING; pt++)
                        for (int i = 1; i <= root_pos.plist[sd][pt][0]; i++) {
                            if (r.root_n_ft[p] >= NNUE_MAX_FT_PER_PERSP) goto root_ft_done;
                            int fi = halfkav2_feature(p, ksq, root_pos.plist[sd][pt][i], pt, sd);
                            if (fi >= 0) r.root_ft_idx[p][r.root_n_ft[p]++] = fi;
                        }
                root_ft_done:;
            }
        }
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
// Learning-target selection.  Default (0) is the classic lambda-decayed
// eligibility trace; TDLEAF_TARGET=blend (1) is the two-point blend target;
// TDLEAF_TARGET=hybrid (2) is blend with a short prediction-gated trace in
// the bootstrap slot (see tdleaf.h).  Read from the environment once.
// ---------------------------------------------------------------------------
static int   td_target_mode  = -1;               // -1 until env is read
static float td_quiet_cp     = TDLEAF_QUIET_CP;
static float td_trace_lambda = TDLEAF_TRACE_LAMBDA;
static long  td_blend_used   = 0;                // blend: transitions passing the quiet gate
static long  td_blend_skip   = 0;                // blend: transitions rejected by it
static long  td_pred_hit     = 0;                // hybrid: predicted opponent replies
static long  td_pred_miss    = 0;                // hybrid: unpredicted replies
static long  td_gate_pass    = 0;                // hybrid: trace flowed (predicted OR quiet)
static long  td_gate_break   = 0;                // hybrid: trace broken (loud AND unpredicted)
static int   td_root         = 0;                // TDLEAF_ROOT=1: online root learning
static float td_root_weight  = TDLEAF_ROOT_WEIGHT;
static long  td_root_used    = 0;                // root records passing the quiet gate
static long  td_root_skip    = 0;                // root records rejected by it

static int tdleaf_target_mode()
{
    if (td_target_mode < 0) {
        const char *v = getenv("TDLEAF_TARGET");
        td_target_mode = (v && strcmp(v, "blend")  == 0) ? 1
                       : (v && strcmp(v, "hybrid") == 0) ? 2 : 0;
        if ((v = getenv("TDLEAF_QUIET_CP")) && *v)     td_quiet_cp     = (float)atof(v);
        if ((v = getenv("TDLEAF_TRACE_LAMBDA")) && *v) td_trace_lambda = (float)atof(v);
        if ((v = getenv("TDLEAF_ROOT")) && *v && atoi(v) != 0) td_root = 1;
        if ((v = getenv("TDLEAF_ROOT_WEIGHT")) && *v)  td_root_weight  = (float)atof(v);
        if (td_root && td_target_mode == 0) {
            fprintf(stderr, "TDLeaf: TDLEAF_ROOT=1 requires TDLEAF_TARGET=blend|hybrid "
                            "— root learning disabled\n");
            td_root = 0;
        }
        if (td_target_mode == 1)
            fprintf(stderr, "TDLeaf: target=blend (two-point lambda blend), "
                            "quiet gate %.0f cp\n", (double)td_quiet_cp);
        else if (td_target_mode == 2)
            fprintf(stderr, "TDLeaf: target=hybrid (blend + gated trace), "
                            "trace lambda %.2f, gate: predicted or |dcp| <= %.0f cp\n",
                    (double)td_trace_lambda, (double)td_quiet_cp);
        if (td_root)
            fprintf(stderr, "TDLeaf: root learning enabled (weight %.2f, "
                            "gate |static-search| <= %.0f cp)\n",
                    (double)td_root_weight, (double)td_quiet_cp);
    }
    return td_target_mode;
}

// Root learning enablement for tdleaf_record_ply (triggers env init on the
// first recorded ply, before the first game's update runs).
static bool tdleaf_root_enabled()
{
    tdleaf_target_mode();
    return td_root == 1;
}

// ---------------------------------------------------------------------------
// tdleaf_accumulate_game — steps 1-3: compute d[], e[], accumulate gradients.
// Does NOT apply or save.  Called by both tdleaf_update_after_game and replay.
// ---------------------------------------------------------------------------
static void tdleaf_accumulate_game(TDGameRecord &rec, float result,
                                   bool replay_mode = false)
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
    const int target_mode = tdleaf_target_mode();
    if (target_mode == 2) {
        // Hybrid target: e_t = w·(result − d_t) + (1−w)·trace_t with a short,
        // strongly damped trace in the bootstrap slot (see tdleaf.h).  The
        // trace flows through record t when the opponent played the reply the
        // engine PREDICTED (search t's pv[1], verified by hash) OR the
        // transition was quiet (|Δcp| ≤ TDLEAF_QUIET_CP) — i.e. it is broken
        // only by loud, uncalculated transitions.  A break (trace_t = 0)
        // propagates upstream through the recursion, but the record still
        // trains on its outcome term.  No score clip here: predicted swings
        // are calculated, not accidental.
        const int N_ply = rec.plies[T - 1].game_ply;
        static float tr[MAX_GAME_PLY];
        tr[T - 1] = 0.0f;
        for (int t = T - 2; t >= 0; t--) {
            int dply = rec.plies[t + 1].game_ply - rec.plies[t].game_ply;
            h_code expected = (dply == 1) ? rec.plies[t].key_own
                            : (dply == 2) ? rec.plies[t].key_reply : 0;
            bool predicted = (expected != 0 && expected == rec.plies[t + 1].root_key);
            if (predicted) td_pred_hit++; else td_pred_miss++;
            float delta_cp = fabsf(score_w_cp[t + 1] - score_w_cp[t]);
            if (predicted || delta_cp <= td_quiet_cp) {
                tr[t] = (d[t + 1] - d[t]) + td_trace_lambda * tr[t + 1];
                td_gate_pass++;
            } else {
                tr[t] = 0.0f;
                td_gate_break++;
            }
        }
        for (int t = 0; t < T - 1; t++) {
            int dend = N_ply - rec.plies[t].game_ply;
            if (dend < 0) dend = 0;
            float w = powf(lambda, (float)dend);
            e[t] = w * (result - d[t]) + (1.0f - w) * tr[t];
        }
    } else if (target_mode == 1) {
        // Two-point blend target: e_t = w·(result − d_t) + (1−w)·(d_{t+1} − d_t)
        // with w = λ^(N − game_ply_t) — the outcome's weight decays with
        // game-ply distance from the end, the remainder bootstraps from the
        // NEXT search's score only.  Local per record (no trace recursion), so
        // distant eval swings never leak into e_t.  Gated on transition
        // quietness (direct consecutive-score test; the opponent's intervening
        // reply means position-quietness at t cannot certify the transition):
        // loud records get e_t = 0 and contribute no gradient.  The score-
        // change clip is subsumed by the gate.  N = last recorded root
        // game-ply, so the final record's w = λ^0 = 1 reproduces the
        // outcome-only e[T−1] above.
        const int N_ply = rec.plies[T - 1].game_ply;
        for (int t = 0; t < T - 1; t++) {
            float delta_cp = fabsf(score_w_cp[t + 1] - score_w_cp[t]);
            if (delta_cp > td_quiet_cp) {
                e[t] = 0.0f;
                td_blend_skip++;
                continue;
            }
            td_blend_used++;
            int dend = N_ply - rec.plies[t].game_ply;
            if (dend < 0) dend = 0;
            float w = powf(lambda, (float)dend);
            e[t] = w * (result - d[t]) + (1.0f - w) * (d[t + 1] - d[t]);
        }
    } else {
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
    }

    // 3. For each ply, run FP32 forward pass + accumulate gradients
    const float cp_factor = 100.0f / 5776.0f;
    const bool  root_learn = (td_root == 1 && target_mode >= 1);
    const int   N_ply_end  = rec.plies[T - 1].game_ply;

    for (int t = 0; t < T; t++) {
        float sig_grad = d[t] * (1.0f - d[t]) / TDLEAF_K;
        // wtm_sign converts ∂d_t/∂w (white-POV utility we want to ascend)
        // into the descent-form gradient expected by nnue_apply_gradients
        // (which does w -= LR × step on the supplied "loss" gradient).
        // score_white = wtm ? +score_stm : -score_stm; nnue_forward_fp32
        // backprops ∂(stm-POV score)/∂w, so the white-POV sign is
        // (wtm ? +1 : -1) and the loss-form sign we pass downstream is its
        // negative — hence (wtm ? -1 : +1).
        float wtm_sign = rec.plies[t].wtm ? -1.0f : 1.0f;
        float id_weight = 1.0f / (1.0f + rec.plies[t].id_score_variance / TDLEAF_ID_VAR_SIGMA2);
        float grad_scale = e[t] * sig_grad * cp_factor * wtm_sign * id_weight;

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

            nnue_accumulate_gradients(act, grad_scale, replay_mode);
        }

        // Online root learning (TDLEAF_ROOT=1, blend/hybrid): second gradient
        // at the ROOT position — the online mirror of the offline corpus root
        // rows.  e_root = w·(result − d_root) + (1−w)·(d_leaf − d_root): pull
        // the root's static eval toward the outcome-blended search score
        // (search-amplified self-distillation).  Gated on root quietness
        // |root_static − score_root_stm| ≤ TDLEAF_QUIET_CP — a within-search
        // test (no opponent move intervenes), matching the TSV dump's root
        // gate.  Runs independently of the leaf error (including on records
        // whose leaf gradient was gated out).
        if (root_learn) {
            const TDRecord &r = rec.plies[t];
            if (fabsf((float)(r.root_static - r.score_root_stm)) > td_quiet_cp) {
                td_root_skip++;
            } else {
                td_root_used++;
                float root_w_cp = r.root_wtm ?  (float)r.root_static
                                             : -(float)r.root_static;
                float d_root    = 1.0f / (1.0f + expf(-root_w_cp / TDLEAF_K));
                int dend = N_ply_end - r.game_ply;
                if (dend < 0) dend = 0;
                float w      = powf(lambda, (float)dend);
                float e_root = w * (result - d_root) + (1.0f - w) * (d[t] - d_root);
                float root_sig_grad = d_root * (1.0f - d_root) / TDLEAF_K;
                float root_sign     = r.root_wtm ? -1.0f : 1.0f;
                float root_scale    = e_root * root_sig_grad * cp_factor
                                      * root_sign * id_weight * td_root_weight;
                if (root_scale != 0.0f) {
                    NNUEActivations act;
                    act.stack = r.root_stack;
                    nnue_forward_fp32(r.root_acc, r.root_psqt, r.root_wtm, act);
                    memcpy(act.acc_raw[0], r.root_acc[0], NNUE_HALF_DIMS * sizeof(int16_t));
                    memcpy(act.acc_raw[1], r.root_acc[1], NNUE_HALF_DIMS * sizeof(int16_t));
                    act.n_ft[0] = r.root_n_ft[0];
                    act.n_ft[1] = r.root_n_ft[1];
                    memcpy(act.ft_idx[0], r.root_ft_idx[0], r.root_n_ft[0] * sizeof(int));
                    memcpy(act.ft_idx[1], r.root_ft_idx[1], r.root_n_ft[1] * sizeof(int));

                    int stm_p = r.root_wtm ? 1 : 0;
                    for (int pt = PAWN; pt <= KING; pt++)
                        act.piece_count_diff[pt - 1] = (int8_t)(r.root_pos.plist[stm_p][pt][0]
                                                               - r.root_pos.plist[stm_p ^ 1][pt][0]);

                    nnue_accumulate_gradients(act, root_scale, replay_mode);
                }
            }
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
                    if (ftell(f) == 0)
                        // Axis marker: the ply/endply columns are true GAME-ply
                        // (game-ply λ^Δ era).  --batch-train keys its result-decay
                        // axis off this line; legacy corpora without it use the
                        // old record-index axis.
                        fprintf(f, "# tdleaf-corpus axis=game-ply\n");
                        fprintf(f, "fen\tcp\tresult\tply\tdepth\tgid\tendply\n");
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

        if (td_target_mode == 1 && (td_blend_used + td_blend_skip) > 0)
            fprintf(stderr, "TDLeaf: applied batch of %d game(s), latest %d plies (result=%.1f), "
                            "quiet-accept %.1f%% (%ld/%ld transitions)\n",
                    td_batch_pending, T, (double)result,
                    100.0 * (double)td_blend_used / (double)(td_blend_used + td_blend_skip),
                    td_blend_used, td_blend_used + td_blend_skip);
        else if (td_target_mode == 2 && (td_gate_pass + td_gate_break) > 0)
            fprintf(stderr, "TDLeaf: applied batch of %d game(s), latest %d plies (result=%.1f), "
                            "trace-gate %.1f%% pass (predicted %.1f%%, %ld/%ld transitions)\n",
                    td_batch_pending, T, (double)result,
                    100.0 * (double)td_gate_pass / (double)(td_gate_pass + td_gate_break),
                    100.0 * (double)td_pred_hit / (double)(td_pred_hit + td_pred_miss),
                    td_gate_pass, td_gate_pass + td_gate_break);
        else
            fprintf(stderr, "TDLeaf: applied batch of %d game(s), latest %d plies (result=%.1f)\n",
                    td_batch_pending, T, (double)result);
        if (td_root == 1 && (td_root_used + td_root_skip) > 0)
            fprintf(stderr, "TDLeaf: root-accept %.1f%% (%ld/%ld records)\n",
                    100.0 * (double)td_root_used / (double)(td_root_used + td_root_skip),
                    td_root_used, td_root_used + td_root_skip);
        td_batch_pending = 0;
    } else {
        fprintf(stderr, "TDLeaf: accumulated %d-ply game (result=%.1f), batch %d/%d\n",
                T, (double)result, td_batch_pending, TDLEAF_BATCH_SIZE);
    }
}

// ---------------------------------------------------------------------------
// Replay ring buffer
// ---------------------------------------------------------------------------
int tdleaf_replay_k = TDLEAF_REPLAY_K;

struct TDReplayEntry {
    TDGameRecord rec;
    float        result;
    bool         valid;
};
static TDReplayEntry td_replay_buf[TDLEAF_REPLAY_BUF_N];
static int           td_replay_head  = 0;  // next slot to write
static int           td_replay_count = 0;  // slots filled (saturates at BUF_N)

// ---------------------------------------------------------------------------
// tdleaf_refresh_scores — rewrite score_stm in every ply of rec using
// the current quantized weights.  Must be called before tdleaf_accumulate_game
// in each replay pass so d[t] reflects the current network, not stale weights.
//
// Flavor A: rebuild accumulators from stored positions using current FT weights,
// re-enumerate active features, and re-evaluate scores.  This ensures FT
// gradients during replay reflect the current weights, not stale game-time values.
// ---------------------------------------------------------------------------
static void tdleaf_refresh_scores(TDGameRecord &rec)
{
    for (int t = 0; t < rec.n_plies; t++) {
        TDRecord &r = rec.plies[t];

        // Rebuild accumulator from stored position using current FT weights.
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
                        if (r.n_ft[p] >= NNUE_MAX_FT_PER_PERSP) goto ft_done_refresh;
                        int fi = halfkav2_feature(p, ksq, r.pos.plist[sd][pt][i], pt, sd);
                        if (fi >= 0) r.ft_idx[p][r.n_ft[p]++] = fi;
                    }
            ft_done_refresh:;
        }

        // Re-evaluate score from rebuilt accumulator.
        int pc = r.stack * 4 + 2;
        pc = (pc < 1) ? 1 : (pc > 32) ? 32 : pc;
        r.score_stm = nnue_evaluate_acc_raw(r.acc, r.psqt, (int)r.wtm, pc);
    }
}

// ---------------------------------------------------------------------------
// tdleaf_replay — push completed game into ring buffer, then run
// tdleaf_replay_k additional passes over all buffered games.
// Must be called after tdleaf_update_after_game().
// ---------------------------------------------------------------------------
void tdleaf_replay(TDGameRecord &rec, float result, const char *save_path)
{
    if (rec.n_plies < TDLEAF_MIN_PLIES) return;
    // Skip the ring-buffer copy entirely when replay is disabled (the settled
    // recipe: TDLEAF_REPLAY_K=0).  TDGameRecord is several MB — copying into
    // the 8-slot buffer would page in ~40 MB of BSS per process for nothing.
    if (tdleaf_replay_k <= 0) return;

    // Push the completed game into the ring buffer.
    int slot = td_replay_head;
    td_replay_buf[slot].rec    = rec;
    td_replay_buf[slot].result = result;
    td_replay_buf[slot].valid  = true;
    td_replay_head = (td_replay_head + 1) % TDLEAF_REPLAY_BUF_N;
    if (td_replay_count < TDLEAF_REPLAY_BUF_N) td_replay_count++;

    // Only run replay passes on batch boundaries (when td_batch_pending was
    // just reset to 0 by tdleaf_update_after_game).  This ensures the live
    // batch and replay batch are applied at the same cadence.
    if (tdleaf_replay_k <= 0 || td_batch_pending != 0) return;

    int n_valid = td_replay_count;

    for (int pass = 0; pass < tdleaf_replay_k; pass++) {
        // Iterate entries in chronological order (oldest first).
        for (int i = 0; i < n_valid; i++) {
            int idx = (td_replay_head - n_valid + i + TDLEAF_REPLAY_BUF_N)
                      % TDLEAF_REPLAY_BUF_N;
            TDReplayEntry &entry = td_replay_buf[idx];
            if (!entry.valid) continue;

            // Refresh scores from current weights before forming d[t].
            tdleaf_refresh_scores(entry.rec);
            // replay_mode: suppress FT/PSQT/FT-bias gradients (those feed into
            // nnue_init_accumulator so updating them would race the next
            // refresh).  FC weights are still updated because they do not feed
            // into the accumulator rebuild.
            tdleaf_accumulate_game(entry.rec, entry.result, /*replay_mode=*/true);
        }
        // Apply the summed replay gradients, then requantize so the next
        // pass's tdleaf_refresh_scores() sees the updated weights.
        // LR scaled down via TDLEAF_REPLAY_LR_SCALE to soften overfitting to
        // the small replay buffer.
        nnue_clip_gradients(TDLEAF_GRAD_CLIP_NORM);
        nnue_apply_gradients(TDLEAF_REPLAY_LR_SCALE);
        nnue_requantize_fc();
    }

    if (save_path && save_path[0]) {
        if (!nnue_save_fc_weights(save_path))
            fprintf(stderr, "TDLeaf replay: failed to save weights to %s\n", save_path);
    }

    fprintf(stderr, "TDLeaf replay: %d pass(es) x %d game(s) in buffer\n",
            tdleaf_replay_k, n_valid);
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

    // Insufficient mating material: each side has no pawns/rooks/queens and
    // at most one minor piece (KvK, KvKN, KvKB, KNvK, KBvK, KNvKN, KBvKB,
    // KNvKB).  Mirrors fastchess's standalone "draw by insufficient mating
    // material" rule.  Slightly over-broad on KNvKB but those cannot force
    // mate in normal play.
    {
        auto insuff = [](const position &p, int side) {
            int heavy = p.plist[side][PAWN][0] + p.plist[side][ROOK][0] +
                        p.plist[side][QUEEN][0];
            int minor = p.plist[side][KNIGHT][0] + p.plist[side][BISHOP][0];
            return heavy == 0 && minor <= 1;
        };
        if (insuff(final_pos, 0) && insuff(final_pos, 1)) {
            out_result_white_pov = 0.5f;
            return true;
        }
    }

    // ---- (2) Score-history self-adjudication -----------------------------
    // Cutechess defaults: -resign movecount=6 score=600, -draw movenumber=40
    // movecount=8 score=10.  Fastchess uses the same semantics.
    const int RESIGN_PLIES      = 6;
    const int RESIGN_CP         = 600;
    const int DRAW_PLIES        = 8;
    const int DRAW_CP           = 10;
    const int DRAW_MOVE_NUMBER  = 40;

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
