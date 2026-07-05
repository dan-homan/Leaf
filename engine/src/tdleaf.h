// Leaf TDLeaf(λ) online learning for the NNUE FC layers.
//
// Algorithm (Baxter, Tridgell & Weaver, 2000):
//   After a game of T half-moves, let d_t = sigmoid(score_white_t / K).
//   TD errors (backward view):
//     e_{T-1} = result - d_{T-1}
//     e_t     = (d_{t+1} - d_t) + lambda * e_{t+1}
//   Weight update:
//     Δw = Σ_t  e_t * ∇_w d_t   (step size governed by Adam LR schedule)
//
// FC layers (FC0/FC1/FC2) and FT biases (1024 int16) are trained.  FT weights
// and PSQT are also trained (FT weights 46 MB, PSQT 720 KB).
// FP32 shadow copies of the FC weights are maintained in nnue.cpp; after each
// game the int8 inference arrays are updated via nnue_requantize_fc().
//
// Build: perl comp.pl <version> NNUE=1 TDLEAF=1

#ifndef TDLEAF_H
#define TDLEAF_H

#include "define.h"
#include "nnue.h"

// ---------------------------------------------------------------------------
// Hyperparameters (can be overridden by setvalue/environment at runtime)
// ---------------------------------------------------------------------------
static const float TDLEAF_LAMBDA           = 0.98f;  // eligibility trace decay (single value for
                                                     // decisive and draw games — empirically fitted
                                                     // from 1.6M self-play games; autocorrelation
                                                     // and d_t-vs-result methods give ~0.97–0.99
                                                     // for both game types)
static const float TDLEAF_K               = 220.0f; // sigmoid temperature (centipawns)
                                                     // MLE over 58M positions from the classical
                                                     // eval side of match_nn-fresh-260514-
                                                     // 1.39e6g_9.5e5g.pgn (1.015M games,
                                                     // 2026-05-25): optimum 217.71 cp, rounded
                                                     // to 220.
static const int   TDLEAF_MIN_PLIES       = 8;      // skip games shorter than this
static const int   TDLEAF_MIN_PLIES_REP   = 40;     // skip 3-rep draws shorter than this
// Approach 1 — TD error clipping.
// When the white-POV score change between consecutive moves exceeds
// TDLEAF_SCORE_CLIP_PAWNS × max(value[PAWN], 100 cp), the (d[t+1]−d[t])
// contribution to the eligibility trace is scaled down proportionally.
// With TDLEAF_PIN_PAWN_VALUE the threshold is effectively constant at
// TDLEAF_SCORE_CLIP_PAWNS × 100 cp; the floor on value[PAWN] is retained
// as belt-and-braces against any future configuration that disables the pin.
// Set to a large value to disable.
static const float TDLEAF_SCORE_CLIP_PAWNS = 1.0f;
// Approach 2 — iterative-deepening score stability weight.
// w_t = 1 / (1 + id_score_variance / TDLEAF_ID_VAR_SIGMA2)
// Expressed in cp²: 10000 corresponds to a 100 cp std-dev reference.
// Larger values are more tolerant of ID score instability.
static const float TDLEAF_ID_VAR_SIGMA2  = 10000.0f;
// Gradient clipping: if global L2 norm of all gradients exceeds this threshold,
// scale all gradients by max_norm/norm.  Set to 0 to disable.
static const float TDLEAF_GRAD_CLIP_NORM = 1.0f;
// Adam step clipping: bound the unit-less Adam step |m_hat / sqrt(v_hat)| (or
// |g / sqrt(v_hat)| for the RMSProp FT path) to this value before multiplying
// by the category LR.  Targets the rare-feature pathology where a low running
// v makes a normal gradient produce an oversized parameter change.  Uniform
// across FC / FT / FT-bias / PSQT / piece_val because the Adam step is scale-
// normalised by design.  Set to a large value to disable.
static const float TDLEAF_ADAM_STEP_CLIP = 30.0f; 

// ---------------------------------------------------------------------------
// Adam hyperparameters
//
// FT weights use RMSProp (per-weight v, no m); all other layers use full Adam.
// v arrays (second moment / gradient scale) and t_adam are persisted to .tdleaf.bin
// (v6+) so gradient scale knowledge survives across sessions.  Multi-writer merge
// uses max(v_file, v_local) per element.  m (momentum) is session-local.
// FT weight v (~92 MB) is sparsely persisted in v8+ (only non-zero rows saved).
// LR warmup: ramps from 0 to full LR over first WARMUP Adam steps.
// Mini-batch: gradients accumulated across BATCH_SIZE games before each Adam step.
// ---------------------------------------------------------------------------
// Per-section Adam LRs.  Targets ~0.1% fractional change per Adam step at
// each section's typical weight magnitude in the int-equivalent FP32 shadow
// space (rule: LR ≈ 0.001 × median(|w|) measured on nn-ad9b42354671).
// Per-section magnitudes in that net (signed std vs median of absolute val):
//                      median(|w|)   std(w)   std/med
//   FC0 weights              3          8       2.8×    (sparse, heavy-tail)
//   FC1 weights              7         18       2.6×    (sparse, heavy-tail)
//   FC2 weights             68         76       1.1×    (dense, final 32→1)
//   FC0 bias              2067       2937       1.4×
//   FC1 bias              1582       2510       1.6×
//   FC2 bias               861       1218       1.4×
//   FT weights              15         44       3.0×    (~92% near zero;
//                                                        std dominated by tail)
//   FT bias                 51         97       1.9×
//   PSQT                 13319      20519       1.5×    (int32)
//   PSQT deviation         665       1207       1.8x    (slot mean PV removed)
//
// Median (not std) drives LR sizing: in the sparse sections (FC0/FC1/FT
// weights), std is dominated by the heavy upper tail rather than the bulk
// where most updates land, so a std-based rule would over-LR by 2–3×.  std
// is recorded here for regime-drift monitoring — if std/med ever collapses
// toward 1 for the sparse sections, the bulk has spread and the "0.1% of
// typical weight" interpretation no longer holds.
// FC2 weights are dramatically larger than FC0/FC1 because the 32→1 fan-in
// gives each FC2 weight unusually high leverage on the score; they need
// their own LR.  FC biases are int32-scale and need a different LR than
// the int8-scale FC weights.
//
// PSQT / piece_val subspace nuance: nnue_mean_center_psqt_gradients()
// constrains PSQT updates to the per-piece-slot deviation subspace; the
// slot-mean (material) component flows to piece_val instead.  In that
// active subspace median(|deviation|) ≈ 665, not 13319 — so PSQT_LR0 = 13
// is sized to the raw weight magnitude, not the subspace PSQT actually
// moves in.  Effective Adam step is ~2% of typical deviation magnitude
// per step (the 0.001 × median(|w|) rule applied to the subspace would
// give PSQT_LR0 ≈ 0.67).  Empirically stable; flag for any future LR
// sweep.  piece_val now absorbs the slot-mean component (median ~12 901)
// and PV_LR0 = 13 ≈ 0.001 × 12.9k is correctly aligned with its active
// subspace.
//
// Gradient mean-centering alone does NOT preserve the slot-total invariant
// across an Adam step: per-weight 1/sqrt(v_hat) makes the applied dw
// non-zero-sum within a slot (sparse features take proportionally larger
// steps).  nnue_apply_gradients adds a post-Adam pass that re-centers the
// applied dw per slot (aggregated across all NNUE_PSQT_BKTS buckets).  Both
// gradient and dw centering use the per-slot AGGREGATE (sum over buckets),
// not per-(slot, bucket) — only the slot's absolute material level is
// anchored; the bucket structure is left free so PSQT can learn
// phase-dependent piece values (e.g. "pawn is worth more in deep endgame"),
// which is the entire reason HalfKAv2_hm uses 8 PSQT buckets.  Without
// post-Adam centering, PSQT pawn-slot mean drifted +13% over 50k games even
// with piece_val[PAWN] hard-pinned at 0.
static const float TDLEAF_ADAM_LR0         = 0.005f;  // FC0/FC1 weights (int8, median ~5)
static const float TDLEAF_ADAM_FC2_LR0     = 0.07f;   // FC2 weights (int8, median ~68 — final 32→1 layer)
static const float TDLEAF_ADAM_FC_BIAS_LR0 = 1.5f;    // FC biases (int32, median ~1500 across stacks)
static const float TDLEAF_ADAM_FT_LR0      = 0.015f;  // FT weights (int16, median ~16)
static const float TDLEAF_ADAM_FT_BIAS_LR0 = 0.02f;   // FT biases  (int16, median ~51; hedged below
                                                       // 0.001×median to limit dying-ReLU risk)
static const float TDLEAF_ADAM_PSQT_LR0    = 13.0f;   // PSQT (int32; sized to raw ~13 319 — active
                                                       // post-centering subspace is ~665, see note above)
static const float TDLEAF_ADAM_PV_LR0      = 13.0f;   // dense piece values (int32; absorbs PSQT slot-mean
                                                       // ~12 901 after mean-centering, well calibrated)
// Pin piece_val[PAWN] at its init value (skip its Adam update).  Together with
// nnue_mean_center_psqt_gradients() (which zeroes the slot-mean PSQT gradient),
// this completes the gauge fix: PSQT carries spatial deviation, piece_val
// carries per-piece material, and PAWN is the unit reference — fixing the
// overall material scale that TDLEAF_K=220 would otherwise leave free.  N/B/R/Q
// piece_val still adapt freely (their values become ratios to the pinned pawn).
// Without this pin, piece_val[PAWN] climbs without ceiling because the loss has
// no anchor for absolute material magnitude; the resulting value[PAWN] drift
// affects downstream code that consumes it as cp (SEE, endgame draw detection,
// UCI score display, this file's SCORE_CLIP_PAWNS multiplier).
static const bool  TDLEAF_PIN_PAWN_VALUE = true;
// Freeze the entire material gauge: PSQT weights AND all piece_val entries
// receive no training updates (gradients are not accumulated, Adam steps are
// skipped, mean-centering / slot-mean recentering become no-ops).  The PSQT
// channel becomes a fixed material + piece-square prior (set at --init-nnue
// time; use --init-nnue-classical so it matches the classical eval and the
// search constants in value[]), and ALL learning lives in the FC / FT layers.
// This supersedes TDLEAF_PIN_PAWN_VALUE and the whole gauge-anchoring
// apparatus: with no trainable material channel there is no PSQT/piece_val
// redundancy, no slot-mean drift for the multi-writer merge to amplify, and
// no piece_val death-spiral failure mode.  nnue_extract_piece_values() still
// derives value[] from the (frozen) PSQT, reproducing the classical values.
static const bool  TDLEAF_FREEZE_MATERIAL = true;
static const float TDLEAF_ADAM_BETA1    = 0.9f;    // first-moment decay  (FC + FT bias + PSQT)
static const float TDLEAF_ADAM_BETA2    = 0.999f;  // second-moment decay (all layers)
static const float TDLEAF_ADAM_EPS      = 1e-8f;   // numerical floor
// AdamW decoupled weight decay: w -= λ × lr × w after each Adam step.
// Applied to FC weights and FT weights only (not biases, not PSQT).
// Set to 0.0 to disable.
static const float TDLEAF_WEIGHT_DECAY  = 1e-4f; //1e-4f;   // decoupled weight decay coefficient
static const int   TDLEAF_ADAM_WARMUP        = 50;  // linear LR warmup over first N Adam steps (0 = disabled)
                                                     // Keyed on t_adam (persisted) so only fires in first session.
static const int   TDLEAF_FT_SESSION_WARMUP  = 100; // per-session FT LR ramp over first N Adam steps.
                                                     // Applied every restart via t_ft_session (not persisted).
                                                     // Damps FT updates during the v_ft_w accumulation phase.
static const int   TDLEAF_BATCH_SIZE    = 8;        // accumulate gradients across N games before Adam step

// Replay LR scale: multiplicative factor applied to all category LRs during
// replay-pass Adam steps (1.0 = no softening, 0.0 = no-op replay).  Lower
// values reduce overfitting to the small replay buffer (BUF_N games).
// Adam is scale-invariant in the gradient, so LR is the only effective knob.
static const float TDLEAF_REPLAY_LR_SCALE = 0.3f;

// ---------------------------------------------------------------------------
// Per-ply record: accumulator snapshot + search score
// ---------------------------------------------------------------------------
struct TDRecord {
    int16_t acc [2][NNUE_HALF_DIMS];   // raw accumulator [perspective][dim]
    int32_t psqt[2][NNUE_PSQT_BKTS];  // PSQT sums [perspective][bucket]
    int     score_stm;                 // search score (centipawns, side-to-move POV)
    int     score_root_stm;            // root-position search score (engine POV, cp).
                                        // For self-adjudication only — does not feed
                                        // TDLeaf gradients (those use score_stm at leaf).
    int     stack;                     // layer stack index used (piece_count-1)/4
    bool    wtm;                       // White to move at the leaf position
    float   id_score_variance;         // variance of last N ID depth scores (cp²); 0 if < 2 depths
    // Active feature indices at the leaf position (indexed by actual perspective 0=BLACK,1=WHITE).
    // Used for FT and PSQT gradient backprop.
    int     ft_idx[2][NNUE_MAX_FT_PER_PERSP];
    int8_t  n_ft[2];
    // Leaf position for Flavor A replay: allows full accumulator rebuild from
    // current FT weights during replay, rather than using stale accumulators.
    position pos;
    // Root-position snapshot for the TSV dump (TDLEAF_DUMP_TSV): the root's
    // search score (score_root_stm) is a search-amplified label for root_pos,
    // unlike the leaf's static eval which is self-distillation.  root_static
    // is the root's STATIC eval (STM POV) — |root_static − score_root_stm|
    // is the root quietness test.  Filled only when dumping is enabled.
    position root_pos;
    int      root_static;
    int8_t   id_depth;    // ID iteration count ≈ achieved search depth
};

// ---------------------------------------------------------------------------
// Per-game record: array of TDRecord entries + outcome
// ---------------------------------------------------------------------------
struct TDGameRecord {
    TDRecord plies[MAX_GAME_PLY];
    int      n_plies;
    // n_plies is reset to 0 at game start; entries filled by tdleaf_record_ply()

    // Engine's color this game (root STM at every recorded ply, since we only
    // record when the engine is about to move).  -1 = unset; 0 = black, 1 = white.
    // Used by UCI self-adjudication to map "engine won/lost" → white-POV result.
    int8_t engine_color;
};

// ---------------------------------------------------------------------------
// Public interface
// ---------------------------------------------------------------------------

// Record one ply after each ts.search() call.
// Walks the principal variation to the leaf position; records the leaf
// accumulator, leaf wtm, and leaf-perspective score (not the root's).
//
// root_pos:        game.pos
// root_acc:        game.ts.tdata[0].n[0].acc
// pv:              game.ts.tdata[0].pc[0]   (NOMOVE-terminated)
// score_root_stm:  game.ts.g_last           (score, root STM perspective)
void tdleaf_record_ply(TDGameRecord &rec,
                       const struct position &root_pos,
                       const NNUEAccumulator &root_acc,
                       const move *pv,
                       int score_root_stm,
                       const int *id_scores,
                       int id_score_count,
                       int search_depth);

// Run the full TDLeaf(λ) update after a game ends.
// result: game outcome from White's perspective (1.0=White wins, 0.5=draw, 0.0=Black wins).
// Calls nnue_apply_gradients(), nnue_requantize_fc(), and nnue_save_fc_weights().
void tdleaf_update_after_game(TDGameRecord &rec, float result, const char *save_path);

// Runtime-mutable replay pass count (initialised from TDLEAF_REPLAY_K).
// Can be changed at runtime via setvalue; set to 0 to disable replay.
extern int tdleaf_replay_k;

// Push the completed game into the replay ring buffer, then run tdleaf_replay_k
// additional passes over all buffered games with score_stm refreshed from the
// current weights before each pass.  Must be called after tdleaf_update_after_game().
// No-op if tdleaf_replay_k == 0.
void tdleaf_replay(TDGameRecord &rec, float result, const char *save_path);

// Flush any pending mini-batch gradients (e.g., at session end or weight export).
void tdleaf_flush_batch(const char *save_path);

// Self-adjudicate a UCI game outcome from in-engine state.  UCI has no protocol
// command for game results, so we reconstruct one from:
//   (1) the terminal position on `final_pos` (mate / stalemate / 50-move / 3-rep), or
//   (2) the engine's own recent score history (mirrors cutechess/fastchess
//       `-resign movecount=6 score=600` and `-draw movenumber=40 movecount=8 score=10`).
// Returns true with `out_result_white_pov` set (1.0/0.5/0.0) when confident.
// Returns false when the outcome is ambiguous — caller should skip learning.
//
// `plist` is one thread's repetition list (game.ts.tdata[0].plist); `game_T` is
// game.T (1-based ply index, so most recent recorded hash is at plist[game_T-1]).
bool tdleaf_self_adjudicate(const TDGameRecord &rec,
                            const struct position &final_pos,
                            const uint64_t *plist,
                            int game_T,
                            float &out_result_white_pov);

#endif // TDLEAF_H
