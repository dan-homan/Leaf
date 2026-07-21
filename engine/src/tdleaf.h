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
// Eligibility trace decay, expressed PER GAME-PLY.  The online trace applies
// pow(TDLEAF_LAMBDA, dply) where dply is the game-ply gap between consecutive
// records (2 in the two-process harness, since the engine records only its own
// moves; 1 under internal self-play), so one lambda expresses the same
// real-game horizon in both modes.  0.985 (down from the earlier 0.98994949 =
// sqrt(0.98)) comes from offline batch-training convergence testing, which
// found better convergence at 0.985; since that's close to the prior default,
// the same constant is now used everywhere (online trace and --bt-td-lambda's
// default alike) rather than keeping separate per-mode values.
static const float TDLEAF_LAMBDA           = 0.985f;  // per game ply, from offline convergence testing
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
// Under NNUE_FIXED_PIECE_VALUES the threshold is constant at
// TDLEAF_SCORE_CLIP_PAWNS × 100 cp (value[PAWN] stays at the classical 100);
// the max() floor is belt-and-braces only.
// Set to a large value to disable.
static const float TDLEAF_SCORE_CLIP_PAWNS = 1.0f;
// Approach 2 — iterative-deepening score stability weight.
// w_t = 1 / (1 + id_score_variance / TDLEAF_ID_VAR_SIGMA2)
// Expressed in cp²: 10000 corresponds to a 100 cp std-dev reference.
// Larger values are more tolerant of ID score instability.
static const float TDLEAF_ID_VAR_SIGMA2  = 10000.0f;
// The learning target is the classic λ-decayed eligibility trace (per game-ply
// decay pow(λ, dply), with the score-change clip and ID-variance stability
// weight above).  Earlier opt-in "blend"/"hybrid" targets and online root
// learning were retired; see docs/history/ for that experiment.
// Gradient clipping: if global L2 norm of all gradients exceeds this threshold,
// scale all gradients by max_norm/norm.  Set to 0 to disable.
static const float TDLEAF_GRAD_CLIP_NORM = 1.0f;
// Adam step clipping: bound the unit-less Adam step |m_hat / sqrt(v_hat)| (or
// |g / sqrt(v_hat)| for the RMSProp FT path) to this value before multiplying
// by the category LR.  Targets the rare-feature pathology where a low running
// v makes a normal gradient produce an oversized parameter change.  Uniform
// across FC / FT / FT-bias / PSQT because the Adam step is scale-
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
// PSQT_LR0 = 13 is sized to the raw weight magnitude (median ~13 319).  Under
// pure-PSQT the bucketed PSQT is the sole material channel and moves freely —
// its full range (material level + spatial deviation + phase-dependent per-bucket
// structure) is learnable; the 8 PSQT buckets are what let it encode e.g. "pawn
// worth more in deep endgame".  No gradient mean-centering or post-Adam dw
// centering is applied.
static const float TDLEAF_ADAM_LR0         = 0.005f;  // FC0/FC1 weights (int8, median ~5)
static const float TDLEAF_ADAM_FC2_LR0     = 0.07f;   // FC2 weights (int8, median ~68 — final 32→1 layer)
static const float TDLEAF_ADAM_FC_BIAS_LR0 = 1.5f;    // FC biases (int32, median ~1500 across stacks)
static const float TDLEAF_ADAM_FT_LR0      = 0.015f;  // FT weights (int16, median ~16)
static const float TDLEAF_ADAM_FT_BIAS_LR0 = 0.02f;   // FT biases  (int16, median ~51; hedged below
                                                       // 0.001×median to limit dying-ReLU risk)
static const float TDLEAF_ADAM_PSQT_LR0    = 13.0f;   // PSQT (int32; sized to raw ~13 319 — active
                                                       // post-centering subspace is ~665, see note above)
// Material representation: pure-PSQT — the bucketed PSQT is the SOLE trainable
// material channel.  There is no dense piece_val channel and no gauge machinery
// (pin / gradient mean-centering / post-Adam dw centering / persisted slot-mean
// recentering) — with a single material channel there is no gauge null direction
// for the multi-writer merge to amplify.  Absolute eval scale is anchored by the
// outcome term through TDLEAF_K (outcome-dominated lambda-return targets force
// sigmoid(v/K) to match empirical win rates).  Search is decoupled from the eval
// scale by NNUE_FIXED_PIECE_VALUES (define.h): value[] stays at the classical
// constants, so SEE, pruning margins, and TDLEAF_SCORE_CLIP are unaffected by any
// slow PSQT scale drift.  DO NOT reintroduce a second material channel or freeze
// PSQT — see docs/history/TRAINING_HISTORY.md "Material Representation —
// Dense Piece Values & the Gauge Machinery" and "PSQT Freezing" for why both fail.
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
    bool    root_wtm;                  // White to move at the ROOT (recorded) position.
                                        // In harness mode == engine_color for every
                                        // record; per-record (alternates) under internal
                                        // self-play.  Used by the TSV dump for POV.
    int     game_ply;                  // 1-based game-ply of the ROOT position.  Gap
                                        // between consecutive records = 2 in the harness
                                        // (own moves only), 1 under internal self-play.
                                        // Drives the pow(lambda, dply) trace decay.
    float   id_score_variance;         // variance of last N ID depth scores (cp²); 0 if < 2 depths
    // Active feature indices at the leaf position (indexed by actual perspective 0=BLACK,1=WHITE).
    // Used for FT and PSQT gradient backprop.
    int     ft_idx[2][NNUE_MAX_FT_PER_PERSP];
    int8_t  n_ft[2];
    // Leaf position: lets the trajectory learner rebuild the accumulator from
    // current FT weights (tdleaf_rebuild_record), rather than shipping/using a
    // stale accumulator snapshot.
    position pos;
    // Root-position snapshot for the TSV dump (TDLEAF_DUMP_TSV) and the .tdg
    // trajectory format: the root's search score (score_root_stm) is a
    // search-amplified label for root_pos, unlike the leaf's static eval which
    // is self-distillation.  root_static is the root's STATIC eval (STM POV) —
    // |root_static − score_root_stm| is the root quietness test.  Filled when
    // dumping or trajectory capture is enabled.
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
// Score-history adjudication constants — cutechess/fastchess defaults
// (-resign movecount=6 score=600, -draw movenumber=40 movecount=8 score=10).
// Shared by tdleaf_self_adjudicate (UCI harness games) and the internal
// selfplay adjudicator (selfplay.cpp) so both modes stay in sync.
// ---------------------------------------------------------------------------
static const int TDLEAF_RESIGN_PLIES     = 6;
static const int TDLEAF_RESIGN_CP        = 600;
static const int TDLEAF_DRAW_PLIES       = 8;
static const int TDLEAF_DRAW_CP          = 10;
static const int TDLEAF_DRAW_MOVE_NUMBER = 40;

// ---------------------------------------------------------------------------
// Public interface
// ---------------------------------------------------------------------------

// Draw by insufficient mating material: no pawns/rooks/queens and at most one
// minor per side.  Mirrors fastchess's standalone rule (slightly over-broad on
// KNvKB, which cannot force mate in normal play anyway).
bool tdleaf_insufficient_material(const struct position &p);

// Internal self-play driver (selfplay.cpp), dispatched from main() on
// --selfplay.  Plays whole games in-process with TDLeaf recording every ply.
int selfplay_main(int argc, char *argv[]);

// Trajectory learner (selfplay.cpp), dispatched from main() on --learn-stream:
// consumes actor-emitted .tdg game files in arrival order and runs the exact
// online update with ONE optimizer (single .tdleaf.bin writer).
int learner_main(int argc, char *argv[]);

// Reconstruct a TDRecord's derived snapshot fields (leaf accumulator/PSQT,
// active features, stack) from its stored leaf position using the current
// weights.  refresh_score additionally re-evaluates score_stm (the learner's
// --refresh-scores).  Bit-exact vs the online-recorded snapshot when weights
// are unchanged.
void tdleaf_rebuild_record(struct TDRecord &r, bool refresh_score);

// Set by selfplay.cpp when --traj-out is active: forces tdleaf_record_ply to
// capture root_pos/root_static (shipped in the .tdg format).
extern bool tdleaf_capture_root;

// Startup guardrail + config banner (call once at main() entry in TDLEAF
// builds).  Hard-errors on any TDLEAF_* env var outside the known allowlist;
// then logs the effective online-training constants.
void tdleaf_check_env();

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
                       int search_depth,
                       int game_ply);

// Run the full TDLeaf(λ) update after a game ends.
// result: game outcome from White's perspective (1.0=White wins, 0.5=draw, 0.0=Black wins).
// Calls nnue_apply_gradients(), nnue_requantize_fc(), and nnue_save_fc_weights().
void tdleaf_update_after_game(TDGameRecord &rec, float result, const char *save_path);

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
