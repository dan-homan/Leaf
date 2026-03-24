// Leaf NNUE evaluation — HalfKAv2_hm format (Stockfish 15.1 exact)
//
// Architecture (confirmed from file structure — each stack is 17,640 bytes):
//   Features = HalfKAv2_hm — king-square bucket × all-piece-square
//   Feature space : 22,528  (32 king-buckets × 704 piece-sq indices)
//   Accumulator   : 1024 int16 per perspective  +  8 int32 PSQT per perspective
//   Network       : 8 layer-stacks (selected by material count), each:
//     - FC0: 1,024 → 16  (SqrCReLU only: 512 per perspective × 2 sides = 1024)
//     - FC1: 30   → 32  (30 = 15 SqrCReLU + 15 CReLU from FC0 outputs 0-14)
//     - FC2: 32   → 1   (output; FC0 output-15 adds via passthrough)
//   PSQT : accumulated separately; final score = (psqt_diff/2 + positional) / 16
//   Note: accumulator is NOT shifted before SqrCReLU; clamp directly to [0,127]
//
// Compatible net: nn-ad9b42354671.nnue (Stockfish 15.1 exact release net)

#ifndef NNUE_H
#define NNUE_H

#include <cstdint>

// ---------------------------------------------------------------------------
// Architecture constants
// ---------------------------------------------------------------------------
static const int NNUE_HALF_DIMS    = 1024;  // accumulator units per perspective
static const int NNUE_FT_INPUTS    = 22528; // 32 king-buckets × 704 piece-sq
static const int NNUE_LAYER_STACKS = 8;     // separate nets per material bucket
static const int NNUE_PSQT_BKTS   = 8;     // PSQT buckets (== LAYER_STACKS)

// FC0: SqrCReLU-only input (512 per perspective × 2 sides)
static const int NNUE_L0_SIZE     = 16;   // FC0 output neurons (incl. direct-out)
static const int NNUE_L0_DIRECT   = 15;   // FC0 outputs going through activations
static const int NNUE_L0_INPUT    = 1024; // = 2 × 512 SqrCReLU per side (no CReLU)

// FC1: dense (takes dual-activation of FC0 outputs 0..14 → 15 sqr + 15 clip = 30)
static const int NNUE_L1_SIZE     = 32;   // FC1 output neurons
static const int NNUE_L1_PADDED   = 32;   // padded input dim (ceil(30, 16) = 32)

// FC2 (output): input = NNUE_L1_SIZE = 32 neurons
static const int NNUE_L2_PADDED   = 32;   // padded input dim for FC2

// Quantization scales
static const int NNUE_WEIGHT_SHIFT = 6;   // FC weights: raw >> 6 → [0,127] int8
static const int NNUE_SQR_SHIFT    = 7;   // SqrCReLU: (v*v) >> 19 → [0,127]  (2*WEIGHT_SHIFT+SQR_SHIFT=19)
// Stockfish 15.1 output formula:
//   value_internal = (psqt_diff/2 + positional) / OutputScale(16)   [Stockfish Value units]
//   centipawns = value_internal * 100 / NormalizeToPawnValue(361) = * 100 / 5776
// Passthrough (FC0 output 15): fwdOut = fc0_raw[15] * 9600 / 8128
//   (600 * OutputScale) / (127 * WeightScaleBits=64) = 9600/8128

// ---------------------------------------------------------------------------
// Accumulator (one per search node, lazily updated)
// ---------------------------------------------------------------------------
struct NNUEAccumulator {
    int16_t acc [2][NNUE_HALF_DIMS];  // [perspective][unit]  WHITE=1, BLACK=0
    int32_t psqt[2][NNUE_PSQT_BKTS]; // [perspective][bucket]
    bool    dirty[2];                 // true → full refresh needed (legacy fallback)

    // Lazy evaluation: instead of copying and updating at every node, each node
    // records only the feature-index deltas from its parent.  The full accumulator
    // is materialised (via nnue_apply_delta) only when score_pos is actually called.
    bool    computed;           // true → acc[][] is fully up-to-date
    int     add[2][4];          // per-perspective feature indices to add (max 4)
    int     sub[2][4];          // per-perspective feature indices to subtract (max 4)
    int8_t  n_add[2];           // count of adds per perspective
    int8_t  n_sub[2];           // count of subs per perspective
    bool    need_refresh[2];    // true → perspective needs full rebuild (king moved)

    NNUEAccumulator() {
        dirty[0] = dirty[1] = true;
        computed = true;
        n_add[0] = n_add[1] = n_sub[0] = n_sub[1] = 0;
        need_refresh[0] = need_refresh[1] = false;
    }
};

// ---------------------------------------------------------------------------
// Public interface
// ---------------------------------------------------------------------------
extern bool nnue_available;

// Allocate FT heap arrays and set nnue_available=true without loading a file.
// Used by --init-nnue mode; also called internally by nnue_load().
void nnue_alloc_arrays();

// Load a HalfKAv2_hm .nnue file. Returns true on success.
bool nnue_load(const char *path);

// Write current FC weights into a complete .nnue file (FT copied verbatim from
// the loaded source).  Useful for exporting TDLeaf-trained weights.
// Backs up dst_path → dst_path.bak before writing.
bool nnue_write_nnue(const char *dst_path);

// Rename path → path.bak if path exists.  Called before any destructive write.
void nnue_backup_file(const char *path);

// Full accumulator refresh from the current position.
void nnue_init_accumulator(NNUEAccumulator &acc, const struct position &pos);

// Incremental update after exec_move (eager copy-make; kept for reference).
void nnue_update_accumulator(NNUEAccumulator &next_acc,
                             const struct position &before,
                             const struct position &after,
                             union move mv);

// Record feature-index deltas for a move without touching ft_weights.
// Fills acc.add/sub/n_add/n_sub/need_refresh and sets acc.computed = false.
// Call after exec_move with before=pre-move pos and after=post-move pos.
void nnue_record_delta(NNUEAccumulator &acc,
                       const struct position &before,
                       const struct position &after,
                       union move mv);

// Materialise a lazy accumulator from the parent's computed accumulator.
// Copies parent_acc for each non-refresh perspective and applies the stored
// deltas; rebuilds from scratch for need_refresh perspectives.
// Sets acc.computed = true on return.
void nnue_apply_delta(NNUEAccumulator &acc,
                      const NNUEAccumulator &parent_acc,
                      const struct position &pos);

// Forward pass. Returns centipawns from side-to-move's perspective.
// piece_count: total pieces on board (for layer-stack selection).
int nnue_evaluate(const NNUEAccumulator &acc, int stm, int piece_count);

// ---------------------------------------------------------------------------
// TDLeaf(λ) support — only compiled when TDLEAF=1
// ---------------------------------------------------------------------------
#if TDLEAF

// Max active HalfKAv2 features per perspective (32 pieces max → 32 features; 64 is safe).
static const int NNUE_MAX_FT_PER_PERSP = 64;

// Intermediate activations saved during the FP32 forward pass (for backprop).
struct NNUEActivations {
    float l0_in  [NNUE_L0_INPUT];    // SqrCReLU output from acc pairs
    float fc0_raw[NNUE_L0_SIZE];     // FC0 pre-activation (int32 cast to float)
    float fc1_in [NNUE_L1_PADDED];   // dual-activation output (indices 0..29 active)
    float fc1_raw[NNUE_L1_SIZE];     // FC1 pre-activation
    float fc2_in [NNUE_L2_PADDED];   // CReLU output
    float fc2_raw;                    // FC2 dot-product output (before passthrough add)
    float fwdOut;                     // passthrough: fc0_raw[15] * 9600/8128
    float positional;                 // fc2_raw + fwdOut
    int   stack;                      // layer stack index
    // FT/PSQT backprop fields — filled by tdleaf_update_after_game before nnue_accumulate_gradients:
    int16_t acc_raw[2][NNUE_HALF_DIMS];       // raw int16 accumulator (for SqrCReLU gradient)
    int     ft_idx[2][NNUE_MAX_FT_PER_PERSP]; // active feature indices, indexed by actual persp
    int8_t  n_ft[2];                           // active feature count per perspective
    int8_t  stm_persp;                         // STM perspective index (WHITE=1, BLACK=0)
};

// Initialise all FC and FT weights to zero, PSQT to 100 cp/piece equivalent.
// Used when starting training from scratch (no .tdleaf.bin found).
// Writes zero to FC int8 inference arrays and PSQT_100CP to int32 PSQT arrays.
void nnue_init_zero_weights();

// Initialise FP32 shadow copies from the just-loaded int8 arrays.
// Called once at end of nnue_load().
void nnue_init_fp32_weights();

// FP32 forward pass — mirrors nnue_evaluate() but saves activations for backprop.
// acc[0] = BLACK perspective, acc[1] = WHITE perspective (raw int16).
// wtm: true if White to move (determines stm perspective layout for FC0 input).
void nnue_forward_fp32(const int16_t acc[2][NNUE_HALF_DIMS],
                       const int32_t psqt[2][NNUE_PSQT_BKTS],
                       bool wtm, NNUEActivations &act);

// Accumulate per-weight gradients for one position into the static grad arrays.
// grad_scale = alpha * e_t * sigmoid_gradient — applied inside.
void nnue_accumulate_gradients(const NNUEActivations &act, float grad_scale);

// Clip gradients by global L2 norm.  Returns pre-clip norm (0 if disabled).
float nnue_clip_gradients(float max_norm);

// Apply accumulated gradients (zero them afterwards).
void nnue_apply_gradients();

// Requantize FP32 weights → int8 arrays used by the live forward pass.
// Must be called after nnue_apply_gradients().  Also clears the score hash.
void nnue_requantize_fc();

// Save FC-only weights (all 8 stacks) to a companion file.  Returns true on success.
bool nnue_save_fc_weights(const char *path);

// Load FC-only weights from companion file, overriding what nnue_load() loaded.
bool nnue_load_fc_weights(const char *path);

// Evaluate from raw accumulator arrays (used by TDLeaf replay to refresh
// score_stm from stored acc[][] against current weights, without constructing
// a full NNUEAccumulator object).
int nnue_evaluate_acc_raw(const int16_t acc[2][NNUE_HALF_DIMS],
                           const int32_t psqt[2][NNUE_PSQT_BKTS],
                           int stm, int piece_count);

#endif // TDLEAF

#endif // NNUE_H
