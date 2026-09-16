// Leaf TDLeaf(λ) online learning for the NNUE weights.
//
// Algorithm (Baxter, Tridgell & Weaver, 2000):
//   After a game of T half-moves, let d_t = sigmoid(score_white_t / K).
//   TD errors (backward view):
//     e_{T-1} = result - d_{T-1}
//     e_t     = (d_{t+1} - d_t) + lambda * e_{t+1}
//   Weight update:
//     Δw = Σ_t  e_t * ∇_w d_t   (step size governed by Adam LR schedule)
//
// Every weight section is trained: FC0/FC1/FC2, all FC biases, FT weights
// (46 MB), FT biases (1024 int16) and PSQT (720 KB).
// FP32 shadow copies of the FC weights are maintained in nnue.cpp; after each
// game the int8 inference arrays are updated via nnue_requantize_fc().
//
// Build: perl comp.pl <version> NNUE=1 TDLEAF=1

#ifndef TDLEAF_H
#define TDLEAF_H

#include "define.h"
#include "nnue.h"

// ---------------------------------------------------------------------------
// Hyperparameters — all compile-time.  None of these is settable at runtime:
// tdleaf_check_env() hard-errors on any TDLEAF_* environment variable outside a
// short allowlist (freeze, corpus dump, two diagnostics), so a stray or mistyped
// variable can never silently alter a training run.  The only runtime knobs are
// the learner's --lr-scale (uniform multiplier, see tdleaf_lr_scale below) and
// the compile-time TDLEAF_BATCH_SIZE_DEFAULT override.
// ---------------------------------------------------------------------------
// Eligibility trace decay, expressed PER GAME-PLY: the trace applies
// pow(TDLEAF_LAMBDA, dply), where dply is the game-ply gap between consecutive
// records — 1 under internal self-play (every ply recorded), 2 under UCI play
// where only the engine's own moves are recorded.  Expressing it per game-ply
// means one constant gives the same real-game horizon in both.
// 0.985 comes from offline batch-training convergence testing, and is used
// everywhere (online trace and --bt-td-lambda's default alike).
static const float TDLEAF_LAMBDA           = 0.985f;  // per game ply
// Sigmoid temperature (centipawns).  MLE over 58M positions from the classical
// eval side of a 1.015M-game match (2026-05-25): optimum 217.71 cp, rounded.
// Also the loss anchor for absolute eval scale under pure-PSQT — see the
// material-representation note below.
static const float TDLEAF_K               = 220.0f;
static const int   TDLEAF_MIN_PLIES       = 8;      // skip games shorter than this
static const int   TDLEAF_MIN_PLIES_REP   = 40;     // skip 3-rep draws shorter than this
// Horizon-noise mitigation 1 — TD error clipping.
// When the white-POV score change between consecutive moves exceeds
// TDLEAF_SCORE_CLIP_PAWNS × max(value[PAWN], 100 cp), the (d[t+1]−d[t])
// contribution to the eligibility trace is scaled down proportionally.
// GOTCHA: under the default NNUE_FIXED_PIECE_VALUES this threshold is a
// CONSTANT 100 cp — it deliberately does not stretch as the net's material
// scale drifts.  The max() floor and the runtime read of value[] matter only if
// that flag is turned off.  Set to a large value to disable.
static const float TDLEAF_SCORE_CLIP_PAWNS = 1.0f;
// Leaf-match gate (ONLINE only).  A record contributes a gradient only when its
// leaf's static eval agrees with the root search score propagated to the leaf's
// POV, within this many centipawns.  0 disables.
//
// WHY.  The PV stored in the triangular array is an APPROXIMATION of the minimax
// line.  The leaf position and its accumulator are provably exact (verified by
// TDLEAF_CHECK_ACC and an off-by-one probe), but the root score is not that
// leaf's static eval in roughly half of records -- a real alpha-beta search with
// a TT, extensions, reductions and pruning can take its value from a node the
// stored PV does not name.  A gradient computed at a position unrelated to the
// score it is being trained against carries no usable information.
//
// The error is SYMMETRIC (leaf higher 27.4% / lower 27.4%, mean +0.7 cp against
// sd 74), so it is variance rather than bias -- and variance on the labels is a
// direct contributor to Sigma, the gradient-noise covariance that is the one
// lever left standing after 7.14, and the one 7.15 showed pays (batch 8 -> 32 =
// +64.0 Elo, 5.0 sigma).
//
// WHY A CONSTANT WORKS.  Measured across the m260720 chain from 1e5 to 7e6 games,
// the exact-match SPIKE grows with maturity (38% -> 55%) but the error SCALE does
// not: sd flat at 63-72 cp and the <=10 cp band flat at 78-82% over a 70x range
// of training.  So the gate needs one constant, not a maturity schedule.
//
// AT 10 cp this keeps ~80% of records.  That is a good trade because plies within
// one game are highly correlated -- dropping 20% of them costs far less
// INDEPENDENT information than the count suggests, while removing the records
// whose gradients point somewhere the score did not come from.
// Retention by threshold: 10 cp ~80%, 25 cp ~89%, 50 cp ~95%.
//
// Mates and drawn lines are excluded for free: their value comes from a terminal
// RULE, not from evaluating the leaf, so the mismatch is enormous (mate sentinel)
// or the eval simply disagrees with the 0.
//
// APPLIES TO THE DUMPED LEAF ROW TOO (via TDRecord::leaf_ok), so the offline
// leaf population is exactly the online one -- a leaf the search never valued is
// no more useful offline than online.  dump_quiet_cp remains as an additional
// cap, so the effective gate is the tighter of the two.
//
// TRADE-OFF, stated because it gives something up: the wide-dump design (Offline
// Part 4.1) exists so the leaf gate can be RE-CUT OFFLINE from the `gate` column.
// Gating the dump at 10 cp keeps re-cutting TIGHTER but makes WIDER impossible
// without regenerating.  That is deliberate -- rows failing this test have a
// label the search never endorsed at any threshold -- but note Offline Part 4
// measured gate 60/120/200 as FLAT and "none" as -27.9 Elo, and never tested
// tighter than 60.  Moving the offline leaf population from 60 cp to 10 cp is
// therefore an untested corpus change and should be rated, not assumed.
//
// OFFLINE: the dumped LEAF row already has a gate on this SAME quantity --
// |leaf_static - propagated root SEARCH score| (tdleaf.cpp, leaf-row block),
// re-cuttable via --bt-quiet-cp -- but at 60 cp by default, 6x looser than this.
// Do NOT confuse it with the ROOT row gate, which is |root_static -
// root_search|, i.e. root QUIETNESS, a different quantity.  Root rows are not
// affected by leaf approximation quality at all.
// Offline Part 4 measured gate 60/120/200 as flat and "none" as -27.9 Elo, and
// explicitly never tested TIGHTER than 60 -- which is this same question from
// the other end, and is a one-flag experiment on corpora already on disk.
// Compile-time overridable (`perl comp.pl <v> ... TDLEAF_LEAF_MATCH_CP_DEFAULT=25`)
// so the threshold can be swept without editing this file.  0 disables the gate
// and reproduces the pre-gate behaviour exactly.
#ifndef TDLEAF_LEAF_MATCH_CP_DEFAULT
 #define TDLEAF_LEAF_MATCH_CP_DEFAULT 10
#endif
static const int TDLEAF_LEAF_MATCH_CP = TDLEAF_LEAF_MATCH_CP_DEFAULT;
// Horizon-noise mitigation 2 — iterative-deepening score stability weight.
// w_t = 1 / (1 + id_score_variance / TDLEAF_ID_VAR_SIGMA2)
// Expressed in cp²: 10000 corresponds to a 100 cp std-dev reference.
// Larger values are more tolerant of ID score instability.
static const float TDLEAF_ID_VAR_SIGMA2  = 10000.0f;
// The learning target is the λ-decayed eligibility trace above, and is the only
// target: the opt-in "blend"/"hybrid" targets and online root learning were
// tested and deleted.  Do not re-derive them without reading why they failed —
// three unrelated error formulas all lost 50–95 Elo, which is what exonerated
// the target math and pointed at the update machinery instead.
// See docs/history/Online_Learning_Investigation.md 3.1.
//
// Gradient clipping: if the global L2 norm of all gradients exceeds this
// threshold, scale all gradients by max_norm/norm.  Set to 0 to disable.
//
// RULE: gradients are SUMMED across the batch, so the accumulated norm grows as
// sqrt(B).  If TDLEAF_BATCH_SIZE changes, rescale this with sqrt(B) and check
// the `TDLeaf clip stats` fire rate — this clip is meant to catch pathological
// gradients, and if it fires on ordinary steps it silently caps exactly what a
// larger batch is meant to buy.  Measured at 1000 Adam steps per batch size:
//
//     B      norm mean   norm max   fires
//     8        0.160       0.583     0.0%
//    16        0.231       0.572     0.0%
//    32        0.331       0.719     0.0%
//    64        0.475       1.029     0.1%
//
// At B = 50 that gives mean ~0.40, max ~0.91; 2.0 keeps roughly the 1.7x margin
// a threshold of 1.0 gave at B = 8.
static const float TDLEAF_GRAD_CLIP_NORM = 2.0f;
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
// v arrays (second moment / gradient scale) and t_adam are persisted to
// .tdleaf.bin (v6+) so gradient-scale knowledge survives across sessions; m
// (momentum) is session-local.  The learner is the SOLE writer and saves are a
// plain atomic write of in-memory state — the old cross-writer max/average merge
// of v and m is gone.  FT weight v (~92 MB) is sparsely persisted in v8+ (only
// non-zero rows saved).
// Mini-batch: gradients accumulated across BATCH_SIZE games before each Adam step.
// ---------------------------------------------------------------------------
// Per-section Adam LRs.
//
// Two kinds of section, and only one obeys a magnitude rule:
//   STATIONARY — fc0_w, ft_w, psqt_w (99.99% of parameters).  Scale is set by
//     the init constants and does not move across millions of games, so "LR as a
//     fraction of typical weight" is well defined; all three are held to one
//     ratio, ~0.00035 of RMS.
//   SCALE-FINDING — fc2_w and the five bias sections (~1,672 parameters).  All
//     initialise at or near ZERO and spend the run finding their own scale, so
//     the LR is a GROWTH RATE and a fixed absolute LR self-anneals.
//
// TRAP: never recalibrate a scale-finding section against its CONVERGED
// magnitude — sizing the FC biases off fc0_b ~ 1400 would give a fresh net a
// bias LR that cannot move a section sitting at zero.  And measure on the FP32
// shadows (.tdleaf.bin is scaled by TDLEAF_SCALE = 128) over TOUCHED weights,
// never on the rounded, structurally padded .nnue.
// Evidence and per-section trajectories: docs/Learning_Investigation.md §1 J.
// ---------------------------------------------------------------------------
// LR SCALE CONVENTION.  ONE learning-rate set, used by BOTH phases at scale 1.0
// — the learner's `--lr-scale` and the batch trainer's `--bt-lr` multiply these
// same constants.
//
// GOTCHA: before 2026-09-15 the phases ran at different scales (online 1.0,
// offline 0.25, against constants 4x larger than these), so **any absolute LR
// quoted from a run before that date must be divided by 4** to compare with the
// numbers below.  The 4x online cut that implies is measured and safe.
// ---------------------------------------------------------------------------
// 0.00035 x RMS 3.88.  Deliberately NOT the value inherited from Stockfish-net
// statistics (which lands at 0.00125 here): that assumed a median of ~5 and left
// fc0_w at 3.7x the ratio ft_w and psqt_w agree on, making it the net's largest
// single source of online weight displacement.
static const float TDLEAF_ADAM_LR0         = 0.00035f;// FC0 weights (int8, RMS ~3.88)
// FC1 weights sit at the ratio fc0_w used to have, deliberately: fc1_w is NOT
// stationary (RMS 3.00 at init -> 6.65 at 7M games), so there is no fixed
// magnitude to calibrate it against and the fc0_w reasoning does not transfer.
static const float TDLEAF_ADAM_FC1_LR0     = 0.00125f;// FC1 weights (int8, RMS 3.0 -> 6.7)
static const float TDLEAF_ADAM_FC2_LR0     = 0.0175f; // FC2 weights (scale-finding, 1.9 -> 36;
                                                       // 32→1 fan-in gives high score leverage)
static const float TDLEAF_ADAM_FC_BIAS_LR0 = 0.375f;  // FC biases (int32; scale-finding, init 0)
static const float TDLEAF_ADAM_FT_LR0      = 0.00375f;// FT weights (int16, RMS ~44 -> 40)
static const float TDLEAF_ADAM_FT_BIAS_LR0 = 0.005f;  // FT biases  (int16; scale-finding, init 0;
                                                       // kept low to limit dying-ReLU risk)
static const float TDLEAF_ADAM_PSQT_LR0    = 3.25f;   // PSQT (int32, RMS ~3.6e4)
// Material representation: pure-PSQT — the bucketed PSQT is the SOLE trainable
// material channel.  There is no dense piece_val channel and no gauge machinery
// (pin / gradient mean-centering / post-Adam dw centering / persisted slot-mean
// recentering); all of it was deleted, not disabled.  Absolute eval scale is
// anchored by the outcome term through TDLEAF_K, and search is decoupled from
// that scale by NNUE_FIXED_PIECE_VALUES (define.h), so SEE, pruning margins and
// TDLEAF_SCORE_CLIP are unaffected by PSQT drift.
// DO NOT reintroduce a second material channel, and DO NOT freeze PSQT — both
// were tried and both fail (freezing costs ~200 Elo).  See
// docs/history/TRAINING_HISTORY.md "Material Representation — Dense Piece Values
// & the Gauge Machinery" and "PSQT Freezing".
static const float TDLEAF_ADAM_BETA1    = 0.9f;    // first-moment decay  (FC + FT bias + PSQT)
static const float TDLEAF_ADAM_BETA2    = 0.999f;  // second-moment decay (all layers)
// Numerical floor in m_hat / (sqrt(v_hat) + EPS).  This value is bracketed from
// BOTH sides and neither bound is obvious:
//   too HIGH (it was 1e-8) — EPS stops being negligible against real sqrt(v_hat),
//     silently gating the low-gradient tail out of Adam's normalisation and
//     making the optimizer sensitive to the ACCUMULATED GRADIENT SCALE, hence to
//     batch size.  That sensitivity was mistaken for a real effect more than once.
//   too LOW — the build is -ffast-math, so flush-to-zero makes v == 0 reachable,
//     and then m_hat/(0 + EPS) saturates TDLEAF_ADAM_STEP_CLIP: a full-size step
//     from a meaningless gradient.
static const float TDLEAF_ADAM_EPS      = 1e-12f;
// AdamW decoupled weight decay: w -= λ × lr × w after each Adam step.
// Applied to FC weights and FT weights only (not biases, not PSQT).
// Set to 0.0 to disable.
static const float TDLEAF_WEIGHT_DECAY  = 1e-4f;   // decoupled weight decay coefficient
// Linear LR warmup over the first N Adam steps, applied to ALL categories so no
// section is released at full rate while others are still ramping.
//
// GOTCHA: this is counted in Adam STEPS, so its length in GAMES scales with
// TDLEAF_BATCH_SIZE — 100 steps is 5,000 games at batch 50.  Keep it short for
// that reason; at batch 50 the historical 1000 would ramp through 50,000 games.
//
// WHEN IT FIRES: keyed on the SESSION clock after --opt-reset, otherwise on the
// PERSISTED t_adam — so on an established net it is inert, and it fires for real
// only on a fresh --init-nnue net (t_adam = 0) or after an explicit reset.
static const int   TDLEAF_ADAM_WARMUP        = 100;
// Per-session FT-weight LR ramp, applied every restart via t_ft_session (not
// persisted), damping FT updates while v_ft_w re-accumulates.  Suppressed while
// the global warmup above is ramping: applying both would give FT (t/N)^2 where
// every other section gets t/N, breaking the "same warmup everywhere" property.
static const int   TDLEAF_FT_SESSION_WARMUP  = 100;
// Accumulate gradients across N games before each Adam step.  Overridable at
// compile time (`perl comp.pl <v> ... TDLEAF_BATCH_SIZE_DEFAULT=16`) so a
// batch-size ladder can be built without editing this file.  NOTE when
// sweeping it: gradients are SUMMED across the batch and Adam normalises the
// step, so B does not change the step SIZE -- it changes samples-per-step and,
// at fixed games, the step COUNT (games/B).  Match Adam STEPS across arms, not
// games, or the sweep re-runs 6.15.1's confound.  The accumulated gradient norm
// grows as sqrt(B) against the fixed TDLEAF_GRAD_CLIP_NORM, so check the clip
// telemetry at large B.  See docs/history/Online_Learning_Investigation.md 7.15.
#ifndef TDLEAF_BATCH_SIZE_DEFAULT
#define TDLEAF_BATCH_SIZE_DEFAULT 50
#endif
static const int   TDLEAF_BATCH_SIZE    = TDLEAF_BATCH_SIZE_DEFAULT;

// ---------------------------------------------------------------------------
// Per-ply record: accumulator snapshot + search score
// ---------------------------------------------------------------------------
// Optional per-record refresh telemetry (compile with TDLEAF_REFRESH_DIAG=1):
// keeps the ACTOR-vintage leaf static / root search score / root static
// alongside the refreshed ones, and makes tdleaf_dump_game emit a
// <prefix>.<pid>.diag.tsv carrying both.  Off (0) in production.
#ifndef TDLEAF_REFRESH_DIAG
#define TDLEAF_REFRESH_DIAG 0
#endif

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
                                        // Alternates per record under internal
                                        // self-play (all training); == engine_color for
                                        // every record under UCI play, where only the
                                        // engine's own moves are recorded.  Used by the
                                        // TSV dump for POV.
    int     game_ply;                  // 1-based game-ply of the ROOT position.  Gap
                                        // between consecutive records (dply) = 1 under
                                        // internal self-play, 2 under UCI play (own moves
                                        // only).  Drives the pow(lambda, dply) decay.
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
    bool     leaf_ok;     // leaf static eval agrees with the propagated root
                          // score within TDLEAF_LEAF_MATCH_CP.  False = the PV
                          // did not locate the position the score came from, so
                          // the record is excluded from the ONLINE trace (it is
                          // still dumped; the dump has its own gate).
#if TDLEAF_REFRESH_DIAG
    // Actor-vintage values, preserved by tdleaf_rebuild_record before the
    // --refresh-scores rewrite.  Telemetry only.
    int      score_stm_actor;
    int      score_root_stm_actor;
    int      root_static_actor;
#endif
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
// Shared by tdleaf_self_adjudicate (UCI games) and the internal
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
// weights.  refresh_score (the learner's --refresh-scores) additionally
// re-evaluates every stored EVALUATION on current weights:
//   score_stm      leaf static, from the rebuilt leaf accumulator
//   score_root_stm root search score, re-expressed as s*leaf_static + delta,
//                  holding the search-vs-static residual delta fixed (the
//                  search picked the leaf; the current net values it)
//   root_static    root static, from an accumulator rebuilt on root_pos
// Bit-exact vs the online-recorded snapshot when weights are unchanged (the
// refresh is then an identity on all three).
void tdleaf_rebuild_record(struct TDRecord &r, bool refresh_score);

// Set by selfplay.cpp when --traj-out is active: forces tdleaf_record_ply to
// capture root_pos/root_static (shipped in the .tdg format).
extern bool tdleaf_capture_root;

// Uniform multiplier on every online Adam/RMSProp step (all weight categories),
// mirroring the offline trainer's --bt-lr.  Set by the learner's --lr-scale;
// 1.0 reproduces the historical online behaviour exactly.  Scaling this toward
// 0 does NOT converge on "no generation" — actors stay frozen between refreshes
// either way — it converges on generating from a net that stops drifting.
extern float tdleaf_lr_scale;

// Report the PV-walk and score-consistency counters accumulated by
// tdleaf_record_ply (to stderr).  Diagnostic only — nothing reads these back.
// `tag` is an optional label for distinguishing concurrent processes.
void tdleaf_report_pv_stats(const char *tag);

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
