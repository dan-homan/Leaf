// Leaf TDLeaf(λ) online learning — implementation
// Compiled only when TDLEAF=1 (included by Leaf.cc after nnue.cpp).

#include "define.h"

#if TDLEAF

#include <cmath>
#include <cstring>
#include <cstdio>
#include <algorithm>
#include "chess.h"
#include "nnue.h"
#include "tdleaf.h"

// value[] lives in score.h (included earlier in the unity build).  Declared
// extern here so TDLEAF_SCORE_CLIP_PAWNS × value[PAWN] can be evaluated at
// runtime — the threshold tracks piece-value drift under TDLeaf.
extern int value[7];

// ---------------------------------------------------------------------------
// tdleaf_record_ply — walk the PV to the leaf, then snapshot its accumulator
// ---------------------------------------------------------------------------
void tdleaf_record_ply(TDGameRecord &rec,
                       const position &root_pos,
                       const NNUEAccumulator &root_acc,
                       const move *pv,
                       int score_root_stm,
                       const int *id_scores,
                       int id_score_count)
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
    int leaf_score_stm = nnue_evaluate(acc_a, (int)leaf_wtm, pc)
                       + nnue_dense_piece_val(cur, (int)leaf_wtm, pc);

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
    r.stack             = (pc - 1) / 4;
    r.id_score_variance = id_var;
    r.pos               = cur;  // store leaf position for Flavor A replay

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
    // Floor value[PAWN] at 100 cp (the classical default) before applying the
    // SCORE_CLIP_PAWNS multiplier, so the --init-nnue-noprior bootstrap (where
    // value[PAWN] can be clamped to 1 by nnue_extract_piece_values when PSQT and
    // piece_val are both zeroed) doesn't collapse the threshold.  This keeps
    // SCORE_CLIP_PAWNS as the dominant control across the full piece-value range:
    // at value[PAWN]=100 the threshold is exactly SCORE_CLIP_PAWNS × 100 cp; for
    // value[PAWN] > 100 the threshold scales linearly with the current pawn value.
    const float score_clip_cp =
        TDLEAF_SCORE_CLIP_PAWNS * std::max((float)value[PAWN], 100.0f);

    static float e[MAX_GAME_PLY];
    e[T - 1] = result - d[T - 1];
    for (int t = T - 2; t >= 0; t--) {
        float delta_d  = d[t + 1] - d[t];

        float delta_cp = fabsf(score_w_cp[t + 1] - score_w_cp[t]);
        if (delta_cp > score_clip_cp && delta_cp > 0.0f)
            delta_d *= score_clip_cp / delta_cp;
        e[t] = delta_d + lambda * e[t + 1];
    }

    // 3. For each ply, run FP32 forward pass + accumulate gradients
    const float cp_factor = 100.0f / 5776.0f;

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

        if (grad_scale == 0.0f) continue;

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

    tdleaf_accumulate_game(rec, result);
    td_batch_pending++;

    if (td_batch_pending >= TDLEAF_BATCH_SIZE) {
        nnue_mean_center_psqt_gradients();
        nnue_clip_gradients(TDLEAF_GRAD_CLIP_NORM);
        nnue_apply_gradients();
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
        r.score_stm = nnue_evaluate_acc_raw(r.acc, r.psqt, (int)r.wtm, pc)
                     + nnue_dense_piece_val(r.pos, (int)r.wtm, pc);
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

    // Always push the completed game into the ring buffer.
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
            // refresh).  FC weights and piece_val are still updated because
            // they do not feed into the accumulator rebuild.
            tdleaf_accumulate_game(entry.rec, entry.result, /*replay_mode=*/true);
        }
        // Apply the summed replay gradients, then requantize so the next
        // pass's tdleaf_refresh_scores() sees the updated weights.
        // LR scaled down via TDLEAF_REPLAY_LR_SCALE to soften overfitting to
        // the small replay buffer.
        nnue_mean_center_psqt_gradients();
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

    nnue_mean_center_psqt_gradients();
    nnue_clip_gradients(TDLEAF_GRAD_CLIP_NORM);
    nnue_apply_gradients();
    nnue_requantize_fc();

    if (save_path && save_path[0]) {
        if (!nnue_save_fc_weights(save_path))
            fprintf(stderr, "TDLeaf flush: failed to save weights to %s\n", save_path);
    }

    fprintf(stderr, "TDLeaf flush: applied partial batch of %d game(s)\n", td_batch_pending);
    td_batch_pending = 0;
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
