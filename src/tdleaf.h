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
#include "chess.h"
#include "nnue.h"

// ---------------------------------------------------------------------------
// Hyperparameters (can be overridden by setvalue/environment at runtime)
// ---------------------------------------------------------------------------
static const float TDLEAF_LAMBDA          = 0.7f;   // eligibility trace decay
static const float TDLEAF_K               = 400.0f; // sigmoid temperature (centipawns)
static const int   TDLEAF_MIN_PLIES       = 8;      // skip games shorter than this
// Approach 1 — TD error clipping.
// When the white-POV score change between consecutive moves exceeds this
// threshold (centipawns), the (d[t+1]−d[t]) contribution to the eligibility
// trace is scaled down proportionally.  Set to a large value to disable.
static const float TDLEAF_SCORE_CLIP_CP  = 200.0f;
// Approach 2 — iterative-deepening score stability weight.
// w_t = 1 / (1 + id_score_variance / TDLEAF_ID_VAR_SIGMA2)
// Expressed in cp²: 10000 corresponds to a 100 cp std-dev reference.
// Larger values are more tolerant of ID score instability.
static const float TDLEAF_ID_VAR_SIGMA2  = 10000.0f;

// ---------------------------------------------------------------------------
// Adam + per-weight LR decay hyperparameters
//
// LR schedule with long-term floor:
//   lr(cnt) = LR0 × (floor + (1 − floor) / (1 + cnt / C))
//   cnt = 0  → LR0 × 1.0         (full initial rate)
//   cnt = C  → LR0 × (0.5 + 0.5×floor)  (half-life point)
//   cnt → ∞  → LR0 × floor       (long-term floor; never reaches zero)
//
// FT weights use RMSProp (per-weight v, no m); all other layers use full Adam.
// v arrays are session-local (process memory only, not persisted to .tdleaf.bin).
// t_adam is also session-local; resets alongside v so bias correction is always valid.
// LR warmup: ramps from 0 to full LR over first WARMUP Adam steps.
// Mini-batch: gradients accumulated across BATCH_SIZE games before each Adam step.
// ---------------------------------------------------------------------------
static const float TDLEAF_ADAM_LR0      = 0.2f;    // initial step size for FC/FT layers (float weight units)
static const float TDLEAF_ADAM_PSQT_LR0 = 2.0f;   // initial step size for PSQT (int32 scale ~36k std; needs larger LR)
static const float TDLEAF_ADAM_C        = 5000.0f;  // LR half-life in per-weight updates (shared)
// Long-term LR floor: the learning rate settles to LR0 × LR_FLOOR as cnt → ∞
// rather than approaching zero.  Full decay schedule:
//   lr(cnt) = LR0 × (LR_FLOOR + (1 − LR_FLOOR) / (1 + cnt/C))
//   cnt=0  → LR0 × 1.0    (full initial rate)
//   cnt=C  → LR0 × (0.5 + 0.5×LR_FLOOR)  (half-life point)
//   cnt→∞  → LR0 × LR_FLOOR  (long-term floor)
// Set to 0.0 to restore the original decay-to-zero behaviour.
static const float TDLEAF_ADAM_LR_FLOOR = 0.05f;   // long-term fraction of LR0 (5%)
// AdamW decoupled weight decay: w -= λ × lr × w after each Adam step.
// Applied to FC weights and FT weights only (not biases, not PSQT).
// Set to 0.0 to disable.
static const float TDLEAF_WEIGHT_DECAY  = 1e-4f;   // decoupled weight decay coefficient
static const float TDLEAF_ADAM_BETA1    = 0.9f;    // first-moment decay  (FC + FT bias + PSQT)
static const float TDLEAF_ADAM_BETA2    = 0.999f;  // second-moment decay (all layers)
static const float TDLEAF_ADAM_EPS      = 1e-8f;   // numerical floor
static const int   TDLEAF_ADAM_WARMUP   = 50;      // linear LR warmup over first N games (0 = disabled)
static const int   TDLEAF_BATCH_SIZE    = 4;       // accumulate gradients across N games before Adam step

// ---------------------------------------------------------------------------
// Per-ply record: accumulator snapshot + search score
// ---------------------------------------------------------------------------
struct TDRecord {
    int16_t acc [2][NNUE_HALF_DIMS];   // raw accumulator [perspective][dim]
    int32_t psqt[2][NNUE_PSQT_BKTS];  // PSQT sums [perspective][bucket]
    int     score_stm;                 // search score (centipawns, side-to-move POV)
    int     stack;                     // layer stack index used (piece_count-1)/4
    bool    wtm;                       // White to move at the leaf position
    float   id_score_variance;         // variance of last N ID depth scores (cp²); 0 if < 2 depths
    // Active feature indices at the leaf position (indexed by actual perspective 0=BLACK,1=WHITE).
    // Used for FT and PSQT gradient backprop.
    int     ft_idx[2][NNUE_MAX_FT_PER_PERSP];
    int8_t  n_ft[2];
};

// ---------------------------------------------------------------------------
// Per-game record: array of TDRecord entries + outcome
// ---------------------------------------------------------------------------
struct TDGameRecord {
    TDRecord plies[MAX_GAME_PLY];
    int      n_plies;
    // n_plies is reset to 0 at game start; entries filled by tdleaf_record_ply()
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
                       int id_score_count);

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

#endif // TDLEAF_H
