// nnue_training.cpp — NNUE TDLeaf(λ) training: FP32 shadow weights, gradient
// accumulation, Adam optimizer, requantization, and .tdleaf.bin I/O.
//
// Unity-build position: after nnue.cpp (needs static weight arrays), before
// tdleaf.cpp.  Only compiled when TDLEAF=1.
// ===========================================================================
#if TDLEAF

#include <cmath>
#include <sys/file.h>   // flock, LOCK_EX, LOCK_SH, LOCK_UN
#include <fcntl.h>      // open, O_RDONLY, O_RDWR, O_CREAT
#include <unistd.h>     // close
#include "tdleaf.h"
#include "hash.h"   // score_table, SCORE_SIZE, score_rec — for cache invalidation

// ---------------------------------------------------------------------------
// FP32 shadow copies of FC weights (natural output-major layout)
// ---------------------------------------------------------------------------
static float l0_weights_f32[NNUE_LAYER_STACKS][NNUE_L0_SIZE * NNUE_L0_INPUT];
static float l0_biases_f32 [NNUE_LAYER_STACKS][NNUE_L0_SIZE];
static float l1_weights_f32[NNUE_LAYER_STACKS][NNUE_L1_SIZE * NNUE_L1_PADDED];
static float l1_biases_f32 [NNUE_LAYER_STACKS][NNUE_L1_SIZE];
static float l2_weights_f32[NNUE_LAYER_STACKS][NNUE_L2_PADDED];
static float l2_bias_f32   [NNUE_LAYER_STACKS];

// Update counts — per weight/bias, incremented each game a non-zero gradient was applied.
// Saved to / loaded from .tdleaf.bin so training history accumulates across sessions.
static uint32_t l0_weights_cnt[NNUE_LAYER_STACKS][NNUE_L0_SIZE * NNUE_L0_INPUT];
static uint32_t l0_biases_cnt [NNUE_LAYER_STACKS][NNUE_L0_SIZE];
static uint32_t l1_weights_cnt[NNUE_LAYER_STACKS][NNUE_L1_SIZE * NNUE_L1_PADDED];
static uint32_t l1_biases_cnt [NNUE_LAYER_STACKS][NNUE_L1_SIZE];
static uint32_t l2_weights_cnt[NNUE_LAYER_STACKS][NNUE_L2_PADDED];
static uint32_t l2_bias_cnt   [NNUE_LAYER_STACKS];

// Gradient accumulators (zeroed before each game, filled by nnue_accumulate_gradients)
static float grad_l0_w[NNUE_LAYER_STACKS][NNUE_L0_SIZE * NNUE_L0_INPUT];
static float grad_l0_b[NNUE_LAYER_STACKS][NNUE_L0_SIZE];
static float grad_l1_w[NNUE_LAYER_STACKS][NNUE_L1_SIZE * NNUE_L1_PADDED];
static float grad_l1_b[NNUE_LAYER_STACKS][NNUE_L1_SIZE];
static float grad_l2_w[NNUE_LAYER_STACKS][NNUE_L2_PADDED];
static float grad_l2_b[NNUE_LAYER_STACKS];

// Per-session delta accumulators: Σ of gradient changes applied to float shadows since the
// last file sync.  On write, merged = re_read_file_value + our_delta, which correctly
// incorporates concurrent updates from other Leaf instances.  Cleared after each sync.
static float delta_l0_w[NNUE_LAYER_STACKS][NNUE_L0_SIZE * NNUE_L0_INPUT];
static float delta_l0_b[NNUE_LAYER_STACKS][NNUE_L0_SIZE];
static float delta_l1_w[NNUE_LAYER_STACKS][NNUE_L1_SIZE * NNUE_L1_PADDED];
static float delta_l1_b[NNUE_LAYER_STACKS][NNUE_L1_SIZE];
static float delta_l2_w[NNUE_LAYER_STACKS][NNUE_L2_PADDED];
static float delta_l2_b[NNUE_LAYER_STACKS];
// Per-session delta counts: incremented alongside absolute counts in nnue_apply_gradients.
// On save, merged count = file_count + delta_count (additive, not max-based).
// Cleared after each file sync.  FT weights keep max-based merge (counts not used
// for per-weight BC, and 92 MB delta array would be too expensive).
static uint32_t delta_l0_w_cnt[NNUE_LAYER_STACKS][NNUE_L0_SIZE * NNUE_L0_INPUT];
static uint32_t delta_l0_b_cnt[NNUE_LAYER_STACKS][NNUE_L0_SIZE];
static uint32_t delta_l1_w_cnt[NNUE_LAYER_STACKS][NNUE_L1_SIZE * NNUE_L1_PADDED];
static uint32_t delta_l1_b_cnt[NNUE_LAYER_STACKS][NNUE_L1_SIZE];
static uint32_t delta_l2_w_cnt[NNUE_LAYER_STACKS][NNUE_L2_PADDED];
static uint32_t delta_l2_b_cnt[NNUE_LAYER_STACKS];
static uint32_t delta_ft_bias_cnt[NNUE_HALF_DIMS];
static uint32_t delta_piece_val_cnt[6];
// PSQT delta counts: heap-allocated (~720 KB), parallels psqt_weights_cnt.
static uint32_t *delta_psqt_cnt = nullptr;

// FT/PSQT float shadow arrays (heap — OS lazy-paged, physical use ∝ active features)
// ft_weights_f32 / grad_ft_w: [FT_INPUTS × HALF_DIMS]  ~92 MB each
// psqt_weights_f32 / grad_psqt_w: [FT_INPUTS × PSQT_BKTS] ~720 KB each
// ft_dirty: [FT_INPUTS] — which feature rows received gradient this game
// ft_delta_f32 / psqt_delta_f32: accumulated FT/PSQT deltas since last file sync
static float    *ft_weights_f32   = nullptr;
// psqt_weights_f32 forward-declared near top of file (used in nnue_write_nnue)
static uint32_t *ft_weights_cnt   = nullptr;  // update count per FT weight
static uint32_t *psqt_weights_cnt = nullptr;  // update count per PSQT weight
static float    *grad_ft_w        = nullptr;  // FT weight gradients
static float    *grad_psqt_w      = nullptr;  // PSQT weight gradients
static bool     *ft_dirty         = nullptr;  // which feature rows are non-zero
static float    *ft_delta_f32     = nullptr;  // FT delta since last file sync [FT_INPUTS×HALF_DIMS]
static float    *psqt_delta_f32   = nullptr;  // PSQT delta since last file sync [FT_INPUTS×PSQT_BKTS]

// FT bias float shadow, gradient accumulator, update count, and delta (all NNUE_HALF_DIMS).
// Static (16 KB total) — no heap allocation needed.
static float    ft_biases_f32 [NNUE_HALF_DIMS] = {};
static float    grad_ft_bias  [NNUE_HALF_DIMS] = {};
static uint32_t ft_bias_cnt   [NNUE_HALF_DIMS] = {};
static float    ft_bias_delta [NNUE_HALF_DIMS] = {};

// ---------------------------------------------------------------------------
// Dense piece values: shared material parameters per piece type per PSQT bucket.
// 6 piece types (PAWN..KING, index = pt-1) × 8 buckets = 48 floats.
// Receives dense gradient updates from every position in every game (unlike
// per-feature PSQT which is sparse).  In NNUE internal units (cp × 5776/100).
// ---------------------------------------------------------------------------
// piece_val_f32 forward-declared near top of file (used in nnue_write_nnue)
static float    grad_piece_val  [6] = {};
static float    m_piece_val     [6] = {};
static float    v_piece_val     [6] = {};
static uint32_t piece_val_cnt   [6] = {};
static float    delta_piece_val [6] = {};
// piece_val_active forward-declared near top of file (used in nnue_write_nnue)

// ---------------------------------------------------------------------------
// Adam moment arrays — session-local (process memory only; not saved to .tdleaf.bin).
// All zeroed at session start in nnue_init_fp32_weights / nnue_init_zero_weights.
//
// FC layers + FT biases: true per-weight m and v (static, ~1.1 MB total).
// FT weights:            per-weight v (RMSProp; m omitted — per-dim m would
//                        require 92 MB heap and the per-row mean is too coarse
//                        to be directionally useful).  Heap, ~92 MB (OS lazy-paged).
// PSQT:                  per-weight m and v (heap, ~1.4 MB; only 8 buckets/row
//                        so per-weight is affordable and per-row is too coarse).
// ---------------------------------------------------------------------------
static float v_l0_w[NNUE_LAYER_STACKS][NNUE_L0_SIZE * NNUE_L0_INPUT] = {};
static float v_l0_b[NNUE_LAYER_STACKS][NNUE_L0_SIZE]                  = {};
static float v_l1_w[NNUE_LAYER_STACKS][NNUE_L1_SIZE * NNUE_L1_PADDED] = {};
static float v_l1_b[NNUE_LAYER_STACKS][NNUE_L1_SIZE]                  = {};
static float v_l2_w[NNUE_LAYER_STACKS][NNUE_L2_PADDED]                = {};
static float v_l2_b[NNUE_LAYER_STACKS]                                 = {};
static float v_ft_bias[NNUE_HALF_DIMS]                                  = {};

static float m_l0_w[NNUE_LAYER_STACKS][NNUE_L0_SIZE * NNUE_L0_INPUT] = {};
static float m_l0_b[NNUE_LAYER_STACKS][NNUE_L0_SIZE]                  = {};
static float m_l1_w[NNUE_LAYER_STACKS][NNUE_L1_SIZE * NNUE_L1_PADDED] = {};
static float m_l1_b[NNUE_LAYER_STACKS][NNUE_L1_SIZE]                  = {};
static float m_l2_w[NNUE_LAYER_STACKS][NNUE_L2_PADDED]                = {};
static float m_l2_b[NNUE_LAYER_STACKS]                                 = {};
static float m_ft_bias[NNUE_HALF_DIMS]                                  = {};

static float    *v_ft_w    = nullptr;  // [NNUE_FT_INPUTS × NNUE_HALF_DIMS] — FT per-weight second moment (~92 MB, OS lazy-paged)
static bool     *ft_v_warmed = nullptr; // [NNUE_FT_INPUTS] — true if v_ft_w row was loaded from disk (v8+).
                                        // Warmed rows use t_adam for bc2; fresh rows use min(t_adam,t_ft_session).
static float    *v_psqt_w  = nullptr;  // [NNUE_FT_INPUTS × PSQT_BKTS] — PSQT per-weight v (~720 KB)
static float    *m_psqt_w  = nullptr;  // [NNUE_FT_INPUTS × PSQT_BKTS] — PSQT per-weight m (~720 KB)

// Global Adam step counter — persisted in .tdleaf.bin so warmup and FC/PSQT
// bias correction survive session restarts.
static uint32_t  t_adam    = 0;

// ---------------------------------------------------------------------------
// Adam step-clip telemetry — one line to stderr per batch, reset after log.
// Tracks the max unit-less Adam step magnitude (|m_hat/sqrt(v_hat)| or, for
// RMSProp FT, |g/sqrt(v_hat)|) and the number of parameters whose step was
// clipped to TDLEAF_ADAM_STEP_CLIP.  Written as a single fprintf so
// concurrent engines sharing a 2>> log produce intact lines.
// ---------------------------------------------------------------------------
static float step_max_fc      = 0.0f;
static float step_max_ft      = 0.0f;
static float step_max_ft_bias = 0.0f;
static float step_max_psqt    = 0.0f;
static float step_max_pv      = 0.0f;
static uint64_t step_clips_fc      = 0;
static uint64_t step_clips_ft      = 0;
static uint64_t step_clips_ft_bias = 0;
static uint64_t step_clips_psqt    = 0;
static uint64_t step_clips_pv      = 0;

// Clip the unit-less Adam step and update the per-category max / clip-count.
// Returns the (possibly scaled-down) step.  Symmetric about zero.
static inline float clip_adam_step(float step, float &max_track, uint64_t &clip_count)
{
    float a = fabsf(step);
    if (a > max_track) max_track = a;
    if (a > TDLEAF_ADAM_STEP_CLIP) {
        clip_count++;
        return (step > 0.0f ? TDLEAF_ADAM_STEP_CLIP : -TDLEAF_ADAM_STEP_CLIP);
    }
    return step;
}

// Session-local FT step counter — intentionally NOT persisted.  v_ft_w is zeroed
// at every startup (too large to persist), so bc2 must be computed relative to the
// current session's step count, not the global t_adam.  Using min(t_adam, t_ft_session)
// for ft_bc2 gives the standard Adam bias-correction formula (v / (1-β²^T)) for
// the freshly-zeroed v, preventing 31×-oversized FT steps on the first batch.
static uint32_t  t_ft_session = 0;

// ---------------------------------------------------------------------------
// nnue_init_zero_weights — fresh-start FC/FT initialisation + classical PSQT
//
// === Design philosophy: start quiet, let TDLeaf build structure from signal ===
// Initial NNUE positional output should be near zero so that classical material
// dominates early play (reasonable game quality).  The network gradually grows
// its influence as TDLeaf learns real patterns from self-play.
//
// === Weight means: all zero ===
// Non-zero means inherited from a trained network (Stockfish 15.1) are the
// endpoint of a fully-converged training run, not a useful prior for TDLeaf.
// Zero mean (the core He/Kaiming principle) is the correct starting point.
//
// === FT weights: std calibrated for healthy CReLU/SqrCReLU activation ===
// ~30 features active per position.  Accumulator = sum of ~30 rows, so
// acc std ≈ √30 × σ.  CReLU divides by 64 (>>6 shift), so the accumulator
// needs values in [0, ~8128] for non-zero CReLU output, and |acc| > 724
// for non-zero SqrCReLU output.  With σ=44, acc std ≈ 241, giving ~40%
// non-zero CReLU activations with mean ~3 — rich input for FC0 learning.
// Too-small σ (e.g. 5) kills >99% of activations, causing mode collapse.
//
// === FC weights: calibrated for signal to survive the /64 CReLU cascade ===
// Each CReLU layer divides by 64 (>>6 shift).  FC0 raw must be large enough
// that FC0_CReLU = FC0_raw/64 gives useful FC1 inputs, otherwise the multi-
// layer FC chain (FC0→FC1→FC2) is dormant and only the passthrough carries
// signal.  With σ=4 and ~400 active CReLU inputs of mean ~3:
//   FC0 raw std ≈ √400 × 3 × 4 ≈ 240, CReLU ≈ 3.8 — healthy FC1 input.
//   fwdOut std ≈ 240 × 1.18 ≈ 283 internal ≈ 5 cp — still quiet.
// FC1 (fan-in 30): σ=3, moderate — FC1 CReLU ≈ 0.5, FC2 sees useful input.
// FC2 (fan-in 32, output): σ=2, positional ≈ 5 cp — quiet enough.
//
// === PSQT ===
// Initialised with pure classical material values (no piece-square bonuses).
// The material term already provides a strong prior; TDLeaf learns positional
// adjustments on top.  Each of the 8 buckets receives the same material value.
//
// === Rejection sampling ===
// Int8 weights use truncated-Gaussian rejection sampling (discard and redraw
// if |w| > 127) rather than clipping, to avoid density spikes at the int8
// boundaries.  With the current stds the rejection rate is negligible.
//
// Both ft_weights (int16, inference) and ft_weights_f32 (float, backprop)
// are initialised together; nnue_apply_gradients keeps them in sync for
// dirty feature rows.
// ---------------------------------------------------------------------------
#define INIT_FT_W_STD     44.0f     // acc std ≈ √30 × 44 ≈ 241; ~40% CReLU active
#define INIT_FC0_W_STD    4.0f      // FC0 CReLU ≈ 3.8; keeps FC1→FC2 chain active
#define INIT_FC1_W_STD    3.0f      // moderate — fan-in 30, low saturation risk
#define INIT_FC2_W_STD    2.0f      // small — keep initial positional output ≈ 0 cp

// Map NNUE PSQT bucket (0..7, selected by piece_count) to a classical gstage
// (0..15, used to index piece_sq[stage][piece][sq]).  Smooth gstage ladder so
// the gstage interpolation gives a continuous PSQT shape across buckets.
//   bucket 0 (1-4 pieces, deep endgame)     → gstage 15
//   bucket 1 (5-8 pieces, late endgame)     → gstage 14
//   bucket 2 (9-12 pieces, early endgame)   → gstage 12
//   bucket 3 (13-16 pieces, late midgame)   → gstage 10
//   bucket 4 (17-20 pieces, midgame)        → gstage  8
//   bucket 5 (21-24 pieces, early midgame)  → gstage  6
//   bucket 6 (25-28 pieces, opening trans.) → gstage  3
//   bucket 7 (29-32 pieces, opening)        → gstage  0
static const int NNUE_BUCKET_TO_GSTAGE[NNUE_PSQT_BKTS] = { 15, 14, 12, 10, 8, 6, 3, 0 };

// Classical PSQT lookup with gstage interpolation, matching score.cpp:380.
//   sq: square index in "black-POV" layout (matches piece_sq[] table layout).
//       For NNUE persp-relative square Y: use Y for enemy-side (black-equivalent),
//       use Y^56 for own-side (white-equivalent, via whitef[] convention).
//   ptype: 1..5 (PAWN..QUEEN); KING (6) returns 0 here — king positional
//          contribution is captured implicitly by the HalfKAv2_hm king bucket.
static inline float classical_pst_cp(int gstage, int ptype, int sq)
{
    if (ptype < 1 || ptype > 5) return 0.f;
    if (gstage >= 12)
        return (float)piece_sq[3][ptype][sq];
    int stage = gstage / 4;
    int rem   = gstage % 4;
    return ((4 - rem) * (float)piece_sq[stage][ptype][sq]
            + rem    * (float)piece_sq[stage + 1][ptype][sq]) / 4.0f;
}

void nnue_init_zero_weights(int prior_mode)
{
    const bool noprior   = (prior_mode == NNUE_PRIOR_NOPRIOR);
    const bool classical = (prior_mode == NNUE_PRIOR_CLASSICAL);
    // ---- FC layers: zero-mean weights (He principle); FC0 std He-adjusted; biases zero ----
    std::mt19937 rng(42);  // fixed seed for reproducibility
    // Truncated normal for int8 weights: reject samples outside [-127, 127]
    // rather than clipping, to avoid artificial density spikes at the boundaries.
    auto rnd_w = [&](float std) -> float {
        std::normal_distribution<float> d(0.0f, std);
        float v;
        do { v = d(rng); } while (v < -127.f || v > 127.f);
        return v;
    };

    for (int s = 0; s < NNUE_LAYER_STACKS; s++) {
        for (int i = 0; i < NNUE_L0_SIZE * NNUE_L0_INPUT; i++)
            l0_weights_f32[s][i] = rnd_w(INIT_FC0_W_STD);
        // Zero the passthrough row (output NNUE_L0_DIRECT): fc0_raw[L0_DIRECT] feeds
        // directly to the final score as fwdOut = fc0_raw[15] * 9600/8128.  With random
        // weights this creates ~81 cp score noise at init (std 4 × √1024 inputs ×
        // SqrCReLU mean ≈ 20 → fc0_raw std ≈ 4 000; scaled: ≈ 81 cp), overwhelming the
        // 100 cp/pawn piece_val signal and preventing material-aware play from game 1.
        // Starting at zero eliminates this noise; gradient still flows through the
        // passthrough (g_fc0_raw[15] += g_pos × 9600/8128) so it learns normally.
        for (int i = 0; i < NNUE_L0_INPUT; i++)
            l0_weights_f32[s][NNUE_L0_DIRECT * NNUE_L0_INPUT + i] = 0.0f;
        for (int i = 0; i < NNUE_L0_SIZE; i++)
            l0_biases_f32[s][i]  = 0.0f;
        for (int i = 0; i < NNUE_L1_SIZE * NNUE_L1_PADDED; i++)
            l1_weights_f32[s][i] = rnd_w(INIT_FC1_W_STD);
        for (int i = 0; i < NNUE_L1_SIZE; i++)
            l1_biases_f32[s][i]  = 0.0f;
        for (int i = 0; i < NNUE_L2_PADDED; i++)
            l2_weights_f32[s][i] = rnd_w(INIT_FC2_W_STD);
        l2_bias_f32[s] = 0.0f;
    }
    memset(l0_weights_cnt, 0, sizeof(l0_weights_cnt));
    memset(l0_biases_cnt,  0, sizeof(l0_biases_cnt));
    memset(l1_weights_cnt, 0, sizeof(l1_weights_cnt));
    memset(l1_biases_cnt,  0, sizeof(l1_biases_cnt));
    memset(l2_weights_cnt, 0, sizeof(l2_weights_cnt));
    memset(l2_bias_cnt,    0, sizeof(l2_bias_cnt));
    memset(grad_l0_w, 0, sizeof(grad_l0_w));
    memset(grad_l0_b, 0, sizeof(grad_l0_b));
    memset(grad_l1_w, 0, sizeof(grad_l1_w));
    memset(grad_l1_b, 0, sizeof(grad_l1_b));
    memset(grad_l2_w, 0, sizeof(grad_l2_w));
    memset(grad_l2_b, 0, sizeof(grad_l2_b));
    // Delta counts — zero for fresh network.
    memset(delta_l0_w_cnt, 0, sizeof(delta_l0_w_cnt));
    memset(delta_l0_b_cnt, 0, sizeof(delta_l0_b_cnt));
    memset(delta_l1_w_cnt, 0, sizeof(delta_l1_w_cnt));
    memset(delta_l1_b_cnt, 0, sizeof(delta_l1_b_cnt));
    memset(delta_l2_w_cnt, 0, sizeof(delta_l2_w_cnt));
    memset(delta_l2_b_cnt, 0, sizeof(delta_l2_b_cnt));
    memset(delta_ft_bias_cnt,   0, sizeof(delta_ft_bias_cnt));
    memset(delta_piece_val_cnt, 0, sizeof(delta_piece_val_cnt));

    // Adam moment arrays — session-local, reset to zero for fresh network.
    memset(v_l0_w,    0, sizeof(v_l0_w));
    memset(v_l0_b,    0, sizeof(v_l0_b));
    memset(v_l1_w,    0, sizeof(v_l1_w));
    memset(v_l1_b,    0, sizeof(v_l1_b));
    memset(v_l2_w,    0, sizeof(v_l2_w));
    memset(v_l2_b,    0, sizeof(v_l2_b));
    memset(v_ft_bias, 0, sizeof(v_ft_bias));
    memset(m_l0_w,    0, sizeof(m_l0_w));
    memset(m_l0_b,    0, sizeof(m_l0_b));
    memset(m_l1_w,    0, sizeof(m_l1_w));
    memset(m_l1_b,    0, sizeof(m_l1_b));
    memset(m_l2_w,    0, sizeof(m_l2_w));
    memset(m_l2_b,    0, sizeof(m_l2_b));
    memset(m_ft_bias, 0, sizeof(m_ft_bias));
    t_adam = 0;
    if (ft_v_warmed) memset(ft_v_warmed, 0, NNUE_FT_INPUTS * sizeof(bool));

    // ---- FT biases: zero init ----
    // FT weights already break symmetry across dimensions, so zero FT biases
    // are sufficient to get varied SqrCReLU activations from game 1.
    // ft_biases_f32 must also be explicitly zeroed here: nnue_init_fp32_weights()
    // (called just before this) copies ft_biases → ft_biases_f32, but ft_biases
    // was only just allocated (uninitialized memory) when called from --init-nnue
    // mode.  Without this zero, the initial nnue_save_fc_weights() call saves
    // garbage values as FT biases, corrupting every subsequent training session.
    if (ft_biases)
        memset(ft_biases, 0, NNUE_HALF_DIMS * sizeof(int16_t));
    memset(ft_biases_f32, 0, NNUE_HALF_DIMS * sizeof(float));

    // ---- FT weights: random; PSQT: classical material (if !noprior), else zero ----
    // PSQT provides the base material signal.  Despite sigmoid saturation for large
    // material advantages (sigmoid(1197/290) ≈ 0.985), PSQT recovers from corrosive
    // gradients because its 180,224 parameters are updated HETEROGENEOUSLY and SPARSELY:
    // each (king_sq, piece_sq) feature row is updated only when that exact configuration
    // appears in a leaf position.  Different features erode at different rates; positional
    // features (piece activity, king safety) can receive constructive gradient even while
    // material features erode.  A partial material signal persists until FC/FT learns
    // enough chess to provide recovery (~7000 games).
    //
    // Contrast with dense piece_val as the material source: 48 global scalars, updated
    // ~200×/game.  Once corrosive gradient drives them negative the entire material
    // evaluation inverts, the engine plays to lose pieces, and FC/FT training is corrupted
    // by abnormal game positions — no recovery is possible.
    //
    // PSQT scale: own-piece features set to 2 × cp × 5776/100 so that
    //   (psqt_diff / 2) × 100/5776 = cp contribution to the score.
    // Enemy-piece features stay at zero: psqt_diff = own_val − 0 = own_val, giving the
    // correct one-sided material contribution without double-counting.
    //
    // noprior: PSQT = uniform 100 cp per piece type (P=N=B=R=Q=100), piece_val = 0.
    // Materially blind from move 1 but value[PAWN] stays at 100 cp, preserving
    // SEE/material-accounting semantics; N/B/R/Q differentiate from 100 cp via
    // piece_val updates during training.
    if (ft_weights_f32) {
        size_t ft_sz   = (size_t)NNUE_FT_INPUTS * NNUE_HALF_DIMS;
        size_t psqt_sz = (size_t)NNUE_FT_INPUTS * NNUE_PSQT_BKTS;
        {
            std::normal_distribution<float> ft_w_dist(0.0f, INIT_FT_W_STD);
            for (size_t i = 0; i < ft_sz; i++) {
                float v = ft_w_dist(rng);
                if (v < -32767.f) v = -32767.f;
                if (v >  32767.f) v =  32767.f;
                ft_weights_f32[i] = v;
                ft_weights[i]     = (int16_t)v;
            }
        }
        // PSQT init: symmetric — own-piece features = +V, enemy-piece features = -V.
        // V = cp × 5776/100 so that psqt_diff contribution of one own piece is:
        //   psqt[stm] += +V, psqt[opp] += -V  →  psqt_diff = 2V
        //   score = psqt_diff/2 × 100/5776 = V × 100/5776 = cp  ✓
        // Both signs required: own positive (piece good for me), enemy negative (piece bad for me).
        // Feature layout within each king-bucket (PS_NB=704 entries per bucket):
        //   ps_slot = (fi % PS_NB) / 128 → 0=pawn, 1=knight, 2=bishop, 3=rook, 4=queen, 5=king
        //   is_own  = (fi % PS_NB) % 128 < 64  (own-piece squares are [0,63], enemy are [64,127])
        // Per-piece-type baseline PSQT (cp).  MATERIAL/CLASSICAL use classical
        // values; CLASSICAL additionally adds the 4-stage piece-square table.
        // NOPRIOR uses uniform 100 cp across all pieces: a minimal material
        // prior that anchors value[PAWN] = value[N] = value[B] = value[R]
        // = value[Q] = 100 cp from the start.  This preserves the SEE /
        // material-accounting semantics that consume value[PAWN] as cp and
        // works with the PAWN-pinned gauge fix (TDLEAF_PIN_PAWN_VALUE in
        // tdleaf.h).  N/B/R/Q differentiate via piece_val_f32 updates during
        // training; the engine is materially blind from move 1 but never
        // sees value[PAWN] clamp to 1.
        static const float MATERIAL_CP_NORMAL[6] = {
            100.f, 377.f, 399.f, 596.f, 1197.f, 0.f,
        };
        static const float MATERIAL_CP_NOPRIOR[6] = {
            100.f, 100.f, 100.f, 100.f, 100.f, 0.f,
        };
        const float *MATERIAL_CP = noprior ? MATERIAL_CP_NOPRIOR : MATERIAL_CP_NORMAL;
        const float SCALE = 5776.f / 100.f;  // cp → PSQT-int32 units

        // Classical PST mean-centering: classical piece-square tables have
        // non-zero mean across the 64 squares (e.g. pawns favour the centre,
        // knights penalise edges).  Without centering, that bias contaminates
        // the slot-mean target persisted in .tdleaf.bin (v11+) and the engine
        // would treat the positional bias as part of the material level.  We
        // pre-compute the per-(slot, bucket) PST mean and subtract it, leaving
        // the slot-mean target as MATERIAL_CP[slot] (pure material) and the
        // PST contribution as zero-mean positional deviation only.
        float pst_mean[5][NNUE_PSQT_BKTS] = {};
        if (classical) {
            for (int slot = 0; slot < 5; slot++) {
                for (int b = 0; b < NNUE_PSQT_BKTS; b++) {
                    double s = 0.0;
                    for (int sq = 0; sq < 64; sq++)
                        s += classical_pst_cp(NNUE_BUCKET_TO_GSTAGE[b],
                                              slot + 1, sq);
                    pst_mean[slot][b] = (float)(s / 64.0);
                }
            }
        }

        for (int fi = 0; fi < NNUE_FT_INPUTS; fi++) {
            int fi_in_bkt = fi % PS_NB;
            int ps_slot   = fi_in_bkt / 128;
            bool is_own   = (fi_in_bkt % 128) < 64;
            int  persp_sq = (fi_in_bkt % 128) & 63;
            // sq_lookup is in piece_sq[] table coordinates.  Table is indexed
            // "from black's POV"; classical does pst[whitef[sq]] for white pieces
            // and pst[sq] for black.  In NNUE persp-relative coords:
            //   own piece at persp_sq Y → equivalent to white piece at real Y
            //     (white-persp: no flip; black-persp: real sq = Y^56, then
            //      whitef[Y^56] = Y — both reduce to table-index Y^56).
            //   enemy piece at persp_sq Y → equivalent to black piece at real Y
            //     (table-index Y, no flip).
            int sq_lookup = is_own ? (persp_sq ^ 56) : persp_sq;
            float   *fp = psqt_weights_f32 + (size_t)fi * NNUE_PSQT_BKTS;
            int32_t *ip = psqt_weights     + (size_t)fi * NNUE_PSQT_BKTS;
            for (int b = 0; b < NNUE_PSQT_BKTS; b++) {
                float pst_cp = 0.f;
                if (classical && ps_slot < 5)
                    pst_cp = classical_pst_cp(NNUE_BUCKET_TO_GSTAGE[b],
                                              ps_slot + 1, sq_lookup)
                           - pst_mean[ps_slot][b];
                float cp  = MATERIAL_CP[ps_slot] + pst_cp;
                float val = (is_own ? +1.f : -1.f) * cp * SCALE;
                fp[b] = val;
                ip[b] = (int32_t)roundf(val);
            }
        }
        memset(ft_weights_cnt,   0, ft_sz   * sizeof(uint32_t));
        memset(psqt_weights_cnt, 0, psqt_sz * sizeof(uint32_t));
        memset(grad_ft_w,        0, ft_sz   * sizeof(float));
        memset(grad_psqt_w,      0, psqt_sz * sizeof(float));
        memset(ft_dirty,         0, NNUE_FT_INPUTS * sizeof(bool));

        // Adam heap arrays — allocate on first use; zero for fresh network.
        if (!v_ft_w)    v_ft_w    = new float[ft_sz]();
        if (!v_psqt_w)  v_psqt_w  = new float[psqt_sz]();
        if (!m_psqt_w)  m_psqt_w  = new float[psqt_sz]();
        memset(v_ft_w,    0, ft_sz   * sizeof(float));
        memset(v_psqt_w,  0, psqt_sz * sizeof(float));
        memset(m_psqt_w,  0, psqt_sz * sizeof(float));
        // PSQT delta counts — heap allocated, parallels psqt_weights_cnt.
        if (!delta_psqt_cnt) delta_psqt_cnt = new uint32_t[psqt_sz]();
        memset(delta_psqt_cnt, 0, psqt_sz * sizeof(uint32_t));
    }

    // Dense piece values: start at zero — learns corrections on top of PSQT material.
    // piece_val handles fast-converging material corrections via dense updates (~200/game);
    // PSQT provides the base material prior and all positional knowledge.
    // Clamped >= 0 in nnue_apply_gradients to prevent anti-material death spiral.
    memset(piece_val_f32,   0, sizeof(piece_val_f32));
    memset(grad_piece_val,  0, sizeof(grad_piece_val));
    memset(m_piece_val,     0, sizeof(m_piece_val));
    memset(v_piece_val,     0, sizeof(v_piece_val));
    memset(piece_val_cnt,   0, sizeof(piece_val_cnt));
    memset(delta_piece_val, 0, sizeof(delta_piece_val));
    piece_val_active = true;

    // Sync all int8/int16/int32 inference arrays from the zeroed float shadows.
    nnue_requantize_fc();

    // Fingerprint the fresh FT weights so the companion .tdleaf.bin (written
    // immediately after --init-nnue) carries the matching content hash.
    nnue_update_content_hash();

    nnue_zero_initialized = true;
    nnue_init_prior_mode  = prior_mode;

    // Snapshot per-(slot, bucket) init PSQT slot-means.  These are persisted in
    // .tdleaf.bin (v11+) and used as the re-centering target at every subsequent
    // load — keeps PSQT slot-mean (= material level) invariant against
    // multi-writer merge drift and accumulated numerical error.
    nnue_capture_psqt_init_slot_means();

    const char *psqt_desc;
    if (noprior)         psqt_desc = "symmetric uniform 100 cp "
                                     "(own=+V,enemy=-V; P=N=B=R=Q=100 cp) (noprior)";
    else if (classical)  psqt_desc = "classical material + 4-stage piece-square tables "
                                     "(gstage-interpolated across 8 buckets; "
                                     "P=100 N=377 B=399 R=596 Q=1197 cp + pst)";
    else                 psqt_desc = "symmetric classical material "
                                     "(own=+V,enemy=-V; P=100 N=377 B=399 R=596 Q=1197 cp)";
    printf("NNUE TDLeaf: FC weights=N(0,{%.0f,%.0f,%.0f}) passthrough[15]=0; "
           "FT weights=N(0,%.0f); biases=zero; "
           "PSQT=%s; piece_val=zero (learns corrections)\n",
           INIT_FC0_W_STD, INIT_FC1_W_STD, INIT_FC2_W_STD, INIT_FT_W_STD,
           psqt_desc);
}

// Forward decl — defined near nnue_apply_gradients; used here to log envvar
// overrides at startup rather than after the first training batch.
struct TDLeafLRMultipliers;
static const TDLeafLRMultipliers &tdleaf_lr_multipliers();

// ---------------------------------------------------------------------------
// nnue_init_fp32_weights — dequantize int8 → float after nnue_load()
// ---------------------------------------------------------------------------
void nnue_init_fp32_weights()
{
    // Store weights at raw int8/int32 scale (no q_scale division).
    // The float forward pass uses the same arithmetic as the int path:
    //   fc_raw = bias_int32 + sum(w_int8 * input_[0,127])
    //   activation = clamp(fc_raw / 64, 0, 127)   (CReLU / SqrCReLU)
    // Requantisation is then just round(clamp(w_f32, -127, 127)).
    for (int s = 0; s < NNUE_LAYER_STACKS; s++) {
        // FC0: vdotq layout → natural [o * L0_INPUT + i]
        for (int o = 0; o < NNUE_L0_SIZE; o++) {
            int ob = o / 4, k = o % 4;
            l0_biases_f32[s][o] = (float)l0_biases[s][o];
            for (int i = 0; i < NNUE_L0_INPUT; i++) {
                int ib = i / 4, j = i % 4;
                l0_weights_f32[s][o * NNUE_L0_INPUT + i] =
                    (float)l0_weights[s][ib * 64 + ob * 16 + k * 4 + j];
            }
        }
        // FC1: vdotq layout → natural [o * PADDED + i]
        for (int o = 0; o < NNUE_L1_SIZE; o++) {
            int ob = o / 4, k = o % 4;
            l1_biases_f32[s][o] = (float)l1_biases[s][o];
            for (int i = 0; i < NNUE_L1_PADDED; i++) {
                int ib = i / 4, j = i % 4;
                l1_weights_f32[s][o * NNUE_L1_PADDED + i] =
                    (float)l1_weights[s][ib * 128 + ob * 16 + k * 4 + j];
            }
        }
        // FC2: natural layout (32 weights, 1 output)
        l2_bias_f32[s] = (float)out_biases[s];
        for (int i = 0; i < NNUE_L2_PADDED; i++)
            l2_weights_f32[s][i] = (float)out_weights[s][i];
    }
    memset(l0_weights_cnt, 0, sizeof(l0_weights_cnt));
    memset(l0_biases_cnt,  0, sizeof(l0_biases_cnt));
    memset(l1_weights_cnt, 0, sizeof(l1_weights_cnt));
    memset(l1_biases_cnt,  0, sizeof(l1_biases_cnt));
    memset(l2_weights_cnt, 0, sizeof(l2_weights_cnt));
    memset(l2_bias_cnt,    0, sizeof(l2_bias_cnt));
    memset(grad_l0_w, 0, sizeof(grad_l0_w));
    memset(grad_l0_b, 0, sizeof(grad_l0_b));
    memset(grad_l1_w, 0, sizeof(grad_l1_w));
    memset(grad_l1_b, 0, sizeof(grad_l1_b));
    memset(grad_l2_w, 0, sizeof(grad_l2_w));
    memset(grad_l2_b, 0, sizeof(grad_l2_b));

    // FT/PSQT float shadows — heap allocated; OS lazy-pages them.
    size_t ft_sz   = (size_t)NNUE_FT_INPUTS * NNUE_HALF_DIMS;
    size_t psqt_sz = (size_t)NNUE_FT_INPUTS * NNUE_PSQT_BKTS;
    if (!ft_weights_f32) {
        ft_weights_f32   = new float   [ft_sz];
        psqt_weights_f32 = new float   [psqt_sz];
        ft_weights_cnt   = new uint32_t[ft_sz];
        psqt_weights_cnt = new uint32_t[psqt_sz];
        grad_ft_w        = new float   [ft_sz];
        grad_psqt_w      = new float   [psqt_sz];
        ft_dirty         = new bool    [NNUE_FT_INPUTS];
        ft_delta_f32     = new float   [ft_sz]();    // zero-initialised
        psqt_delta_f32   = new float   [psqt_sz]();  // zero-initialised
        // Adam heap arrays — session-local moment arrays for FT and PSQT.
        ft_v_warmed = new bool[NNUE_FT_INPUTS](); // zero-init: no rows warmed yet
        v_ft_w    = new float[ft_sz]();    // per-weight FT second moment (~92 MB, OS lazy-paged)
        v_psqt_w  = new float[psqt_sz]();
        m_psqt_w  = new float[psqt_sz]();
    }
    // Initialize float shadows from the loaded int16/int32 arrays.
    for (size_t i = 0; i < ft_sz;   i++) ft_weights_f32[i]   = (float)ft_weights[i];
    for (size_t i = 0; i < psqt_sz; i++) psqt_weights_f32[i] = (float)psqt_weights[i];
    memset(ft_weights_cnt,   0, ft_sz   * sizeof(uint32_t));
    memset(psqt_weights_cnt, 0, psqt_sz * sizeof(uint32_t));
    memset(grad_ft_w,        0, ft_sz   * sizeof(float));
    memset(grad_psqt_w,      0, psqt_sz * sizeof(float));
    memset(ft_dirty,         0, NNUE_FT_INPUTS * sizeof(bool));
    memset(ft_v_warmed,      0, NNUE_FT_INPUTS * sizeof(bool));
    // Zero delta accumulators — fresh session, no pending changes yet.
    memset(delta_l0_w,     0, sizeof(delta_l0_w));
    memset(delta_l0_b,     0, sizeof(delta_l0_b));
    memset(delta_l1_w,     0, sizeof(delta_l1_w));
    memset(delta_l1_b,     0, sizeof(delta_l1_b));
    memset(delta_l2_w,     0, sizeof(delta_l2_w));
    memset(delta_l2_b,     0, sizeof(delta_l2_b));
    memset(delta_l0_w_cnt, 0, sizeof(delta_l0_w_cnt));
    memset(delta_l0_b_cnt, 0, sizeof(delta_l0_b_cnt));
    memset(delta_l1_w_cnt, 0, sizeof(delta_l1_w_cnt));
    memset(delta_l1_b_cnt, 0, sizeof(delta_l1_b_cnt));
    memset(delta_l2_w_cnt, 0, sizeof(delta_l2_w_cnt));
    memset(delta_l2_b_cnt, 0, sizeof(delta_l2_b_cnt));
    memset(delta_ft_bias_cnt, 0, sizeof(delta_ft_bias_cnt));
    memset(delta_piece_val_cnt, 0, sizeof(delta_piece_val_cnt));
    if (!delta_psqt_cnt) delta_psqt_cnt = new uint32_t[psqt_sz]();
    memset(delta_psqt_cnt, 0, psqt_sz * sizeof(uint32_t));
    if (ft_delta_f32)   memset(ft_delta_f32,   0, ft_sz   * sizeof(float));
    if (psqt_delta_f32) memset(psqt_delta_f32, 0, psqt_sz * sizeof(float));

    // FT bias float shadow: initialise from the loaded int16 array.
    for (int d = 0; d < NNUE_HALF_DIMS; d++) ft_biases_f32[d] = (float)ft_biases[d];
    memset(grad_ft_bias,  0, sizeof(grad_ft_bias));
    memset(ft_bias_cnt,   0, sizeof(ft_bias_cnt));
    memset(ft_bias_delta, 0, sizeof(ft_bias_delta));

    // Adam moment arrays — session-local, reset at each session start.
    memset(v_l0_w,    0, sizeof(v_l0_w));
    memset(v_l0_b,    0, sizeof(v_l0_b));
    memset(v_l1_w,    0, sizeof(v_l1_w));
    memset(v_l1_b,    0, sizeof(v_l1_b));
    memset(v_l2_w,    0, sizeof(v_l2_w));
    memset(v_l2_b,    0, sizeof(v_l2_b));
    memset(v_ft_bias, 0, sizeof(v_ft_bias));
    memset(m_l0_w,    0, sizeof(m_l0_w));
    memset(m_l0_b,    0, sizeof(m_l0_b));
    memset(m_l1_w,    0, sizeof(m_l1_w));
    memset(m_l1_b,    0, sizeof(m_l1_b));
    memset(m_l2_w,    0, sizeof(m_l2_w));
    memset(m_l2_b,    0, sizeof(m_l2_b));
    memset(m_ft_bias, 0, sizeof(m_ft_bias));
    if (v_ft_w)    memset(v_ft_w,    0, ft_sz * sizeof(float));
    if (v_psqt_w)  memset(v_psqt_w,  0, psqt_sz * sizeof(float));
    if (m_psqt_w)  memset(m_psqt_w,  0, psqt_sz * sizeof(float));
    t_adam = 0;

    // Dense piece values: zero for pre-existing .nnue (material is in per-feature PSQT).
    // Will be overridden from .tdleaf.bin v5+ if present.
    memset(piece_val_f32,   0, sizeof(piece_val_f32));
    memset(grad_piece_val,  0, sizeof(grad_piece_val));
    memset(m_piece_val,     0, sizeof(m_piece_val));
    memset(v_piece_val,     0, sizeof(v_piece_val));
    memset(piece_val_cnt,   0, sizeof(piece_val_cnt));
    memset(delta_piece_val, 0, sizeof(delta_piece_val));
    piece_val_active = false;  // activated by init_zero_weights or v5+ load

    printf("NNUE TDLeaf: FP32 weights initialised (%d stacks + FT/PSQT + FT biases)\n", NNUE_LAYER_STACKS);

    // Touch the LR-multiplier singleton so any envvar overrides log at startup
    // (visibility before kicking off a long training run), not after the first
    // batch.  No effect when all multipliers are at default 1.0.
    (void)tdleaf_lr_multipliers();
}

// ---------------------------------------------------------------------------
// nnue_forward_fp32 — FP32 forward pass, saves activations for backprop
// ---------------------------------------------------------------------------
void nnue_forward_fp32(const int16_t acc[2][NNUE_HALF_DIMS],
                       const int32_t psqt[2][NNUE_PSQT_BKTS],
                       bool wtm, NNUEActivations &act)
{
    int stm = wtm ? 1 : 0;  // WHITE=1, BLACK=0

    // 1. SqrCReLU from raw accumulator: pair (acc[i], acc[i+512]) for each perspective
    //    Layout: [stm_512 | opp_512]
    for (int p = 0; p < 2; p++) {
        int persp = (p == 0) ? stm : (stm ^ 1);
        float *out = act.l0_in + p * 512;
        const int16_t *a = acc[persp];
        for (int i = 0; i < 512; i++) {
            float va = (float)a[i];
            float vb = (float)a[i + 512];
            if (va < 0.0f) va = 0.0f; else if (va > 127.0f) va = 127.0f;
            if (vb < 0.0f) vb = 0.0f; else if (vb > 127.0f) vb = 127.0f;
            float sq = (va * vb) * (1.0f / 128.0f);  // >> 7
            out[i] = (sq > 127.0f) ? 127.0f : sq;
        }
    }

    // Layer stack from piece count is not available here; use stored stack index.
    // The caller fills act.stack before calling this function.
    int s = act.stack;

    // 2. FC0: [L0_INPUT → L0_SIZE] with FP32 weights
    for (int o = 0; o < NNUE_L0_SIZE; o++) {
        float sum = l0_biases_f32[s][o];
        const float *w = &l0_weights_f32[s][o * NNUE_L0_INPUT];
        for (int i = 0; i < NNUE_L0_INPUT; i++)
            sum += w[i] * act.l0_in[i];
        act.fc0_raw[o] = sum;
    }

    // 3. Dual activation of FC0 outputs 0..14 → FC1 input [0..29], padding [30..31]=0
    //    SqrCReLU: clamp(0,127, fc0_raw^2 >> 19)   (same formula as int path)
    //    CReLU:    clamp(0,127, fc0_raw >> 6)
    for (int o = 0; o < NNUE_L0_DIRECT; o++) {
        float raw = act.fc0_raw[o];
        // SqrCReLU (using float arithmetic, equivalent to int version)
        float sq = (raw * raw) * (1.0f / (1 << 19));
        act.fc1_in[o] = (sq < 0.0f) ? 0.0f : (sq > 127.0f) ? 127.0f : sq;
        // CReLU
        float v = raw * (1.0f / 64.0f);
        act.fc1_in[NNUE_L0_DIRECT + o] = (v < 0.0f) ? 0.0f : (v > 127.0f) ? 127.0f : v;
    }
    act.fc1_in[NNUE_L0_DIRECT * 2]     = 0.0f;
    act.fc1_in[NNUE_L0_DIRECT * 2 + 1] = 0.0f;

    // 4. FC1: [L1_PADDED → L1_SIZE]
    for (int o = 0; o < NNUE_L1_SIZE; o++) {
        float sum = l1_biases_f32[s][o];
        const float *w = &l1_weights_f32[s][o * NNUE_L1_PADDED];
        for (int i = 0; i < NNUE_L1_PADDED; i++)
            sum += w[i] * act.fc1_in[i];
        act.fc1_raw[o] = sum;
    }

    // 5. CReLU on FC1 output → FC2 input
    for (int o = 0; o < NNUE_L1_SIZE; o++) {
        float v = act.fc1_raw[o] * (1.0f / 64.0f);
        act.fc2_in[o] = (v < 0.0f) ? 0.0f : (v > 127.0f) ? 127.0f : v;
    }

    // 6. FC2: [L2_PADDED → 1]
    float fc2 = l2_bias_f32[s];
    for (int i = 0; i < NNUE_L2_PADDED; i++)
        fc2 += l2_weights_f32[s][i] * act.fc2_in[i];
    act.fc2_raw = fc2;

    // 7. Passthrough + final positional
    act.fwdOut    = act.fc0_raw[NNUE_L0_DIRECT] * (9600.0f / 8128.0f);
    act.positional = act.fc2_raw + act.fwdOut;

    // Record STM perspective for FT/PSQT backprop (WHITE=1, BLACK=0).
    act.stm_persp = (int8_t)stm;

    // PSQT diff is not included in activations (it is constant w.r.t. FC weights)
    (void)psqt;
}

// ---------------------------------------------------------------------------
// nnue_accumulate_gradients — backprop one position, add to grad arrays
// grad_scale = alpha * e_t * d_t * (1-d_t) / K * (100 / 5776)
// ---------------------------------------------------------------------------
void nnue_accumulate_gradients(const NNUEActivations &act, float grad_scale,
                               bool replay_mode)
{
    int s = act.stack;

    // 7. ∂loss/∂positional = grad_scale  (d_score_cp/d_positional = 100/5776 absorbed)
    float g_pos = grad_scale;

    // 6. FC2 backward: g_pos → grad of fc2_raw = g_pos
    float g_fc2_raw = g_pos;
    // grad w.r.t. FC2 weights and bias
    for (int i = 0; i < NNUE_L2_PADDED; i++)
        grad_l2_w[s][i] += g_fc2_raw * act.fc2_in[i];
    grad_l2_b[s] += g_fc2_raw;
    // grad w.r.t. fc2_in[i]
    float g_fc2_in[NNUE_L2_PADDED];
    for (int i = 0; i < NNUE_L2_PADDED; i++)
        g_fc2_in[i] = g_fc2_raw * l2_weights_f32[s][i];

    // 5. CReLU inverse: ∂fc2_in/∂fc1_raw = 1/64 if fc1_raw in (0, 127*64)
    float g_fc1_raw[NNUE_L1_SIZE];
    for (int o = 0; o < NNUE_L1_SIZE; o++) {
        float v = act.fc1_raw[o] * (1.0f / 64.0f);
        g_fc1_raw[o] = (v > 0.0f && v < 127.0f) ? g_fc2_in[o] * (1.0f / 64.0f) : 0.0f;
    }

    // 4. FC1 backward
    for (int o = 0; o < NNUE_L1_SIZE; o++) {
        if (g_fc1_raw[o] == 0.0f) continue;
        for (int i = 0; i < NNUE_L1_PADDED; i++)
            grad_l1_w[s][o * NNUE_L1_PADDED + i] += g_fc1_raw[o] * act.fc1_in[i];
        grad_l1_b[s][o] += g_fc1_raw[o];
    }
    // grad w.r.t. fc1_in[i]
    float g_fc1_in[NNUE_L1_PADDED] = {};
    for (int o = 0; o < NNUE_L1_SIZE; o++) {
        if (g_fc1_raw[o] == 0.0f) continue;
        const float *w = &l1_weights_f32[s][o * NNUE_L1_PADDED];
        for (int i = 0; i < NNUE_L1_PADDED; i++)
            g_fc1_in[i] += g_fc1_raw[o] * w[i];
    }

    // 3. Dual-activation inverse → g_fc0_raw[0..14]
    float g_fc0_raw[NNUE_L0_SIZE] = {};
    for (int o = 0; o < NNUE_L0_DIRECT; o++) {
        float raw = act.fc0_raw[o];
        // SqrCReLU gradient: d(clamp(0,127, raw^2/2^19))/d(raw) = 2*raw/2^19 when in range
        float sq = (raw * raw) * (1.0f / (1 << 19));
        if (sq > 0.0f && sq < 127.0f)
            g_fc0_raw[o] += g_fc1_in[o] * 2.0f * raw * (1.0f / (float)(1 << 19));
        // CReLU gradient: d(clamp(0,127, raw/64))/d(raw) = 1/64 when in range
        float v = raw * (1.0f / 64.0f);
        if (v > 0.0f && v < 127.0f)
            g_fc0_raw[o] += g_fc1_in[NNUE_L0_DIRECT + o] * (1.0f / 64.0f);
    }
    // Passthrough output[15]: ∂fwdOut/∂fc0_raw[15] = 9600/8128
    g_fc0_raw[NNUE_L0_DIRECT] += g_pos * (9600.0f / 8128.0f);

    // 2. FC0 backward — weights and biases
    for (int o = 0; o < NNUE_L0_SIZE; o++) {
        if (g_fc0_raw[o] == 0.0f) continue;
        for (int i = 0; i < NNUE_L0_INPUT; i++)
            grad_l0_w[s][o * NNUE_L0_INPUT + i] += g_fc0_raw[o] * act.l0_in[i];
        grad_l0_b[s][o] += g_fc0_raw[o];
    }

    // Dense piece value gradient:
    //   piece_val_diff = Σ_pt piece_val[pt] × (stm_count[pt] − opp_count[pt])
    //   ∂score/∂piece_val[pt] = count_diff[pt] × (100/5776) × 0.5
    //   grad_scale already includes cp_factor = 100/5776.
    // Runs in both live and replay paths: piece_val is an output-side additive
    // term (see nnue_dense_piece_val) and does NOT feed into nnue_init_accumulator,
    // so replay updates do not create the FT/PSQT/ft_bias feedback loop below.
    if (piece_val_active && !TDLEAF_FREEZE_MATERIAL) {
        float g_pv = grad_scale * 0.5f;
        for (int pt = 0; pt < 6; pt++) {
            if (act.piece_count_diff[pt] != 0)
                grad_piece_val[pt] += g_pv * (float)act.piece_count_diff[pt];
        }
    }

    // Replay mode: skip FT weights, PSQT, and FT biases.  These three feed
    // into nnue_init_accumulator (ft_biases directly, ft_weights/psqt via
    // add_feat) — updating them during replay would change what the next
    // tdleaf_refresh_scores() produces and drive a positive feedback loop.
    if (replay_mode) return;

    // 1. Continue backward: FC0 inputs → accumulator → FT/PSQT weights
    // g_l0_in[i] = Σ_o g_fc0_raw[o] × l0_weights_f32[s][o×L0_INPUT+i]
    float g_l0_in[NNUE_L0_INPUT] = {};
    for (int o = 0; o < NNUE_L0_SIZE; o++) {
        if (g_fc0_raw[o] == 0.0f) continue;
        const float *w = &l0_weights_f32[s][o * NNUE_L0_INPUT];
        for (int i = 0; i < NNUE_L0_INPUT; i++)
            g_l0_in[i] += g_fc0_raw[o] * w[i];
    }

    // SqrCReLU backward:
    //   l0_in[p*512+j] = clamp(a[j],0,127) * clamp(a[j+512],0,127) / 128
    //   ∂l0_in/∂a[j]     = clamp(a[j+512],0,127) / 128  IFF 0 < a[j]     < 127, else 0
    //   ∂l0_in/∂a[j+512] = clamp(a[j],    0,127) / 128  IFF 0 < a[j+512] < 127, else 0
    // IMPORTANT: gradient is 0 when acc is saturated (≥127) OR dead (≤0).
    // Passing gradient through saturated neurons (the old `clo > 0` bug) caused a
    // positive-feedback loop: large biases → more saturation → incorrect gradient
    // pushes biases higher → crash.
    // l0_in layout: [stm_512 | opp_512]  where p=0 is stm, p=1 is opp.
    float g_acc[2][NNUE_HALF_DIMS] = {};
    int stm_p = (int)act.stm_persp;
    for (int p = 0; p < 2; p++) {
        int persp       = (p == 0) ? stm_p : (stm_p ^ 1);
        const float *gi = g_l0_in + p * 512;
        const int16_t *a = act.acc_raw[persp];
        float *g_a       = g_acc[persp];
        for (int j = 0; j < 512; j++) {
            float vlo = (float)a[j];
            float vhi = (float)a[j + 512];
            float clo = (vlo < 0.0f) ? 0.0f : (vlo > 127.0f) ? 127.0f : vlo;
            float chi = (vhi < 0.0f) ? 0.0f : (vhi > 127.0f) ? 127.0f : vhi;
            float g   = gi[j] * (1.0f / 128.0f);
            if (vlo > 0.0f && vlo < 127.0f) g_a[j]       += g * chi;
            if (vhi > 0.0f && vhi < 127.0f) g_a[j + 512] += g * clo;
        }
    }

    // FT weight gradient: for each active feature fi in perspective p:
    //   grad_ft_w[fi × HALF_DIMS + d] += g_acc[p][d]
    // PSQT gradient (only bucket `stack` is used by this position):
    //   psqt_diff = psqt[stm][stack] - psqt[opp][stack]
    //   ∂score_cp/∂psqt_diff = cp_factor/2 (half of the cp_factor used for positional)
    //   g_psqt_diff = grad_scale × 0.5  (grad_scale already includes cp_factor for positional)
    //   grad_psqt_w[fi × PSQT_BKTS + stack] += g_psqt_diff × (+1 for stm, -1 for opp)
    //   Adam normalises gradient magnitude; TDLEAF_ADAM_PSQT_LR0 governs the per-step size.
    float g_psqt_diff = grad_scale * 0.5f;
    for (int p = 0; p < 2; p++) {
        int persp       = (p == 0) ? stm_p : (stm_p ^ 1);
        float psqt_sign = (persp == stm_p) ? 1.0f : -1.0f;
        float *g_a      = g_acc[persp];
        for (int k = 0; k < (int)act.n_ft[persp]; k++) {
            int fi = act.ft_idx[persp][k];
            if (fi < 0 || fi >= NNUE_FT_INPUTS) continue;
            ft_dirty[fi] = true;
            float *gfw = grad_ft_w + (size_t)fi * NNUE_HALF_DIMS;
            for (int d = 0; d < NNUE_HALF_DIMS; d++)
                gfw[d] += g_a[d];
            if (!TDLEAF_FREEZE_MATERIAL)
                grad_psqt_w[fi * NNUE_PSQT_BKTS + s] +=
                    g_psqt_diff * psqt_sign;
        }
    }

    // FT bias gradient: ∂loss/∂ft_biases[d] = Σ_persp g_acc[persp][d]
    // Both perspectives share the same bias vector, so gradients sum across them.
    for (int d = 0; d < NNUE_HALF_DIMS; d++)
        grad_ft_bias[d] += (g_acc[0][d] + g_acc[1][d]);
}

// ---------------------------------------------------------------------------
// Runtime LR overrides (read once from environment).
//
// TDLEAF_LR_{FC,FT,FT_BIAS,PSQT,PV} multiply the corresponding TDLEAF_ADAM_*_LR0
// constant.  TDLEAF_FREEZE_PSQT=1 forces the PSQT multiplier to 0 (PSQT weights
// don't move; Adam state still advances, so a later session without the freeze
// resumes normally).  TDLEAF_FREEZE_PASSTHROUGH=1 holds the FC0 passthrough row
// (output NNUE_L0_DIRECT = 15, weights + bias) at its init value — gradient and
// Adam state for those weights are zeroed before clipping/apply, so they stay
// exactly stationary.  Defaults are all 1.0 (no override).
//
// Intended for LR sweeps without recompiling.  Example:
//   TDLEAF_LR_PSQT=0.1 TDLEAF_LR_FT=3.0 ./Leaf_vtrain ...
//   TDLEAF_FREEZE_PASSTHROUGH=1 ./Leaf_vtrain ...
// ---------------------------------------------------------------------------
struct TDLeafLRMultipliers {
    float fc, fc2, fc_bias, ft, ft_bias, psqt, pv;
    bool  freeze_passthrough;   // FC0 row NNUE_L0_DIRECT (15) weights+bias held at init
};
static const TDLeafLRMultipliers &tdleaf_lr_multipliers()
{
    static const TDLeafLRMultipliers m = []() {
        auto read_env = [](const char *name, float def) {
            const char *v = getenv(name);
            if (!v || !*v) return def;
            char *end = nullptr;
            float f = strtof(v, &end);
            return (end == v) ? def : f;
        };
        auto read_bool_env = [](const char *name) {
            const char *v = getenv(name);
            return v && (*v == '1' || *v == 't' || *v == 'T' || *v == 'y' || *v == 'Y');
        };
        TDLeafLRMultipliers x;
        x.fc      = read_env("TDLEAF_LR_FC",      1.0f);  // FC0/FC1 weights
        x.fc2     = read_env("TDLEAF_LR_FC2",     1.0f);  // FC2 (final 32→1) weights
        x.fc_bias = read_env("TDLEAF_LR_FC_BIAS", 1.0f);  // FC biases (all three)
        x.ft      = read_env("TDLEAF_LR_FT",      1.0f);
        x.ft_bias = read_env("TDLEAF_LR_FT_BIAS", 1.0f);
        x.psqt    = read_env("TDLEAF_LR_PSQT",    1.0f);
        x.pv      = read_env("TDLEAF_LR_PV",      1.0f);
        bool freeze_psqt = read_bool_env("TDLEAF_FREEZE_PSQT");
        if (freeze_psqt) x.psqt = 0.0f;
        x.freeze_passthrough = read_bool_env("TDLEAF_FREEZE_PASSTHROUGH");
        bool any = (x.fc != 1.0f || x.fc2 != 1.0f || x.fc_bias != 1.0f
                    || x.ft != 1.0f || x.ft_bias != 1.0f
                    || x.psqt != 1.0f || x.pv != 1.0f || x.freeze_passthrough);
        if (any) {
            fprintf(stderr, "TDLeaf LR overrides:");
            if (x.fc      != 1.0f) fprintf(stderr, " FC=%.4g",      (double)x.fc);
            if (x.fc2     != 1.0f) fprintf(stderr, " FC2=%.4g",     (double)x.fc2);
            if (x.fc_bias != 1.0f) fprintf(stderr, " FC_BIAS=%.4g", (double)x.fc_bias);
            if (x.ft      != 1.0f) fprintf(stderr, " FT=%.4g",      (double)x.ft);
            if (x.ft_bias != 1.0f) fprintf(stderr, " FT_BIAS=%.4g", (double)x.ft_bias);
            if (x.psqt    != 1.0f) fprintf(stderr, " PSQT=%.4g%s",  (double)x.psqt,
                                                                    freeze_psqt ? " (frozen)" : "");
            if (x.pv      != 1.0f) fprintf(stderr, " PV=%.4g",      (double)x.pv);
            if (x.freeze_passthrough) fprintf(stderr, " PASSTHROUGH=frozen");
            fprintf(stderr, "\n");
        }
        return x;
    }();
    return m;
}

// ---------------------------------------------------------------------------
// nnue_mean_center_psqt_gradients — subtract the per-(slot, bucket) mean from
// PSQT gradients so PSQT learns only positional (per-square) corrections; the
// uniform slot-mean component is the material-scale shift and belongs to
// dense piece_val instead.
//
// Slots (from halfkav2_feature index structure):
//   fi % 704 / 64 → 0=own pawn, 1=opp pawn, 2=own knight, ...
//                    8=own queen, 9=opp queen, 10=king (both)
//
// Run before nnue_clip_gradients() so the about-to-be-removed slot-mean does
// not inflate the global L2 norm and throttle other layers' updates.  No-op
// when piece_val is not being trained or when PSQT gradients are absent.
// ---------------------------------------------------------------------------
void nnue_mean_center_psqt_gradients()
{
    if (TDLEAF_FREEZE_MATERIAL) return;  // no PSQT gradients exist to center
    if (!piece_val_active || !grad_psqt_w || !ft_dirty) return;

    // Per-slot aggregate (summed across all NNUE_PSQT_BKTS buckets): zeros only
    // the slot's total gradient, leaving per-bucket relative drift free.  This
    // is what lets the bucketed PSQT learn phase-dependent piece values
    // (e.g. "pawn is more valuable in deep endgame") — the absolute material
    // level (slot total) is anchored, but the distribution across buckets is
    // unconstrained.
    double slot_total[11] = {};
    int    slot_count[11] = {};
    for (int fi = 0; fi < NNUE_FT_INPUTS; fi++) {
        if (!ft_dirty[fi]) continue;
        int slot = (fi % 704) / 64;
        slot_count[slot]++;
        const float *gpw = grad_psqt_w + (size_t)fi * NNUE_PSQT_BKTS;
        for (int b = 0; b < NNUE_PSQT_BKTS; b++)
            slot_total[slot] += gpw[b];
    }
    for (int fi = 0; fi < NNUE_FT_INPUTS; fi++) {
        if (!ft_dirty[fi]) continue;
        int slot = (fi % 704) / 64;
        if (slot_count[slot] < 2) continue;
        float corr = (float)(slot_total[slot]
                             / ((double)slot_count[slot] * NNUE_PSQT_BKTS));
        float *gpw = grad_psqt_w + (size_t)fi * NNUE_PSQT_BKTS;
        for (int b = 0; b < NNUE_PSQT_BKTS; b++)
            gpw[b] -= corr;
    }
}

// ---------------------------------------------------------------------------
// nnue_compute_psqt_slot_means — fill means[11][8] with the average value of
// psqt_weights_f32 across all features in each (slot, bucket) cell.
// ---------------------------------------------------------------------------
static void nnue_compute_psqt_slot_means(float means[11][NNUE_PSQT_BKTS])
{
    if (!psqt_weights_f32) return;
    double  sum  [11][NNUE_PSQT_BKTS] = {};
    int     n    [11] = {};
    for (int fi = 0; fi < NNUE_FT_INPUTS; fi++) {
        int slot = (fi % 704) / 64;
        n[slot]++;
        const float *pw = psqt_weights_f32 + (size_t)fi * NNUE_PSQT_BKTS;
        for (int b = 0; b < NNUE_PSQT_BKTS; b++)
            sum[slot][b] += pw[b];
    }
    for (int slot = 0; slot < 11; slot++) {
        float inv_n = n[slot] > 0 ? 1.0f / (float)n[slot] : 0.0f;
        for (int b = 0; b < NNUE_PSQT_BKTS; b++)
            means[slot][b] = (float)(sum[slot][b] * (double)inv_n);
    }
}

// ---------------------------------------------------------------------------
// nnue_capture_psqt_init_slot_means — snapshot the current PSQT slot-means as
// the persisted re-centering target.  Called once at --init-nnue time, and as
// a fallback when loading a pre-v11 .tdleaf.bin file (which has no target).
// ---------------------------------------------------------------------------
void nnue_capture_psqt_init_slot_means()
{
    if (!psqt_weights_f32) return;
    nnue_compute_psqt_slot_means(psqt_init_slot_means);
    psqt_init_slot_means_valid = true;
}

// ---------------------------------------------------------------------------
// nnue_recenter_psqt_slot_means — pin the per-slot AGGREGATE PSQT mean (summed
// across all NNUE_PSQT_BKTS buckets) back to its persisted init target by
// subtracting the per-slot drift uniformly across every feature × bucket cell.
// Per-bucket distribution within a slot is left free, so the bucketed PSQT
// can learn phase-dependent piece values (e.g. pawn worth more in deep
// endgame) — only the absolute material level (slot total) is gauge-anchored.
//
// Touches psqt_weights_f32 only — NOT psqt_delta_f32.  The correction is a
// gauge restoration, not a user-applied delta: if we baked it into pd it would
// be re-applied at every subsequent merge cycle (since the merge protocol does
// pw = file_value + pd), causing geometric error growth.  pw is always set
// directly to the gauge-correct value; pd continues to track only the actual
// per-batch updates.  Also syncs the int32 inference array.
// ---------------------------------------------------------------------------
void nnue_recenter_psqt_slot_means()
{
    // Frozen material gauge: PSQT never moves, so there is no drift to snap
    // back.  Skipping also avoids touching the int32 inference array on load.
    if (TDLEAF_FREEZE_MATERIAL) return;
    if (!psqt_init_slot_means_valid || !psqt_weights_f32) return;

    float cur_means[11][NNUE_PSQT_BKTS];
    nnue_compute_psqt_slot_means(cur_means);

    // Per-slot mean across all buckets (= total / NNUE_PSQT_BKTS).
    float corr[11];
    bool  any_corr = false;
    for (int slot = 0; slot < 11; slot++) {
        double init_total = 0.0;
        double cur_total  = 0.0;
        for (int b = 0; b < NNUE_PSQT_BKTS; b++) {
            init_total += psqt_init_slot_means[slot][b];
            cur_total  += cur_means[slot][b];
        }
        // corr is added per cell; total drift per slot = corr × NNUE_PSQT_BKTS.
        corr[slot] = (float)((init_total - cur_total) / NNUE_PSQT_BKTS);
        if (corr[slot] != 0.0f) any_corr = true;
    }
    if (!any_corr) return;

    for (int fi = 0; fi < NNUE_FT_INPUTS; fi++) {
        int slot = (fi % 704) / 64;
        float c = corr[slot];
        if (c == 0.0f) continue;
        float *pw = psqt_weights_f32 + (size_t)fi * NNUE_PSQT_BKTS;
        for (int b = 0; b < NNUE_PSQT_BKTS; b++)
            pw[b] += c;
    }
    // Resync the int32 inference array.
    if (psqt_weights) {
        for (size_t i = 0; i < (size_t)NNUE_FT_INPUTS * NNUE_PSQT_BKTS; i++)
            psqt_weights[i] = (int32_t)roundf(psqt_weights_f32[i]);
    }
}

// ---------------------------------------------------------------------------
// Pre-clip gradient-norm telemetry.  Tracks call count, fire count, running
// min/mean/max, and a log-spaced histogram so we can answer "is the L2 clip
// guarding outliers, or just renormalising every batch?".
// Bin i covers norms in [EDGES[i-1], EDGES[i]); bin 0 is [0, EDGES[0]); the
// last bin is [EDGES[last], +inf).
// ---------------------------------------------------------------------------
static const float CLIP_NORM_HIST_EDGES[] = {
    0.1f, 0.3f, 1.0f, 3.0f, 10.0f, 30.0f, 100.0f, 300.0f, 1000.0f,
};
static constexpr int CLIP_NORM_HIST_NEDGES = (int)(sizeof(CLIP_NORM_HIST_EDGES) / sizeof(float));
static constexpr int CLIP_NORM_HIST_NBINS  = CLIP_NORM_HIST_NEDGES + 1;
static const int   CLIP_STATS_REPORT_EVERY = 50;  // print a summary every N calls
static uint64_t g_clip_call_count = 0;
static uint64_t g_clip_fire_count = 0;
static double   g_clip_norm_sum   = 0.0;
static float    g_clip_norm_min   = HUGE_VALF;
static float    g_clip_norm_max   = 0.0f;
static float    g_clip_last_thr   = 0.0f;
static uint64_t g_clip_hist[CLIP_NORM_HIST_NBINS] = {};

static void nnue_clip_stats_print()
{
    if (g_clip_call_count == 0) return;
    double mean = g_clip_norm_sum / (double)g_clip_call_count;
    double fire_pct = 100.0 * (double)g_clip_fire_count / (double)g_clip_call_count;
    fprintf(stderr,
            "TDLeaf clip stats: N=%llu fires=%llu (%.1f%%) thr=%.2f norm min=%.3f mean=%.3f max=%.3f  hist:",
            (unsigned long long)g_clip_call_count,
            (unsigned long long)g_clip_fire_count,
            fire_pct, (double)g_clip_last_thr,
            (double)g_clip_norm_min, mean, (double)g_clip_norm_max);
    // First bin label: "<EDGES[0]"; middle bins: "<EDGES[i]"; last bin: ">=EDGES[last]"
    for (int i = 0; i < CLIP_NORM_HIST_NBINS; i++) {
        if (i < CLIP_NORM_HIST_NEDGES)
            fprintf(stderr, " <%g:%llu", (double)CLIP_NORM_HIST_EDGES[i],
                    (unsigned long long)g_clip_hist[i]);
        else
            fprintf(stderr, " >=%g:%llu", (double)CLIP_NORM_HIST_EDGES[CLIP_NORM_HIST_NEDGES-1],
                    (unsigned long long)g_clip_hist[i]);
    }
    fprintf(stderr, "\n");
}

// Public: dump the running clip stats (e.g. at session end / flush).
// Non-destructive — does not reset counters.
void nnue_clip_gradient_stats_report()
{
    nnue_clip_stats_print();
}

// ---------------------------------------------------------------------------
// nnue_clip_gradients — compute global L2 norm of all gradient arrays and
// scale all gradients by max_norm/norm if the norm exceeds max_norm.
// Returns the pre-clip norm.  If max_norm <= 0, does nothing (returns 0).
// ---------------------------------------------------------------------------
float nnue_clip_gradients(float max_norm)
{
    // Freeze the FC0 passthrough row (output NNUE_L0_DIRECT = 15) if requested.
    // Zeroing here ensures: (a) passthrough gradients don't inflate the L2 norm
    // and falsely scale down other layers' updates; (b) the per-weight `if grad
    // != 0` guard in nnue_apply_gradients skips them, so Adam state (m, v) for
    // passthrough weights doesn't move and the weights stay at their init value.
    // Targets the runaway-bias pathology where TDLeaf concentrates "the eval is
    // off by a constant" corrections into row 15 (the only FC0 row with
    // strictly-positive activations from SqrCReLU, so its weight drift couples
    // to a non-zero expectation rather than cancelling).
    if (tdleaf_lr_multipliers().freeze_passthrough) {
        for (int s = 0; s < NNUE_LAYER_STACKS; s++) {
            for (int i = 0; i < NNUE_L0_INPUT; i++)
                grad_l0_w[s][NNUE_L0_DIRECT * NNUE_L0_INPUT + i] = 0.0f;
            grad_l0_b[s][NNUE_L0_DIRECT] = 0.0f;
        }
    }

    if (max_norm <= 0.0f) return 0.0f;

    // Compute global L2 norm across all gradient arrays (use double to avoid overflow).
    double sum_sq = 0.0;
    for (int s = 0; s < NNUE_LAYER_STACKS; s++) {
        for (int i = 0; i < NNUE_L0_SIZE * NNUE_L0_INPUT; i++)
            sum_sq += (double)grad_l0_w[s][i] * grad_l0_w[s][i];
        for (int i = 0; i < NNUE_L0_SIZE; i++)
            sum_sq += (double)grad_l0_b[s][i] * grad_l0_b[s][i];
        for (int i = 0; i < NNUE_L1_SIZE * NNUE_L1_PADDED; i++)
            sum_sq += (double)grad_l1_w[s][i] * grad_l1_w[s][i];
        for (int i = 0; i < NNUE_L1_SIZE; i++)
            sum_sq += (double)grad_l1_b[s][i] * grad_l1_b[s][i];
        for (int i = 0; i < NNUE_L2_PADDED; i++)
            sum_sq += (double)grad_l2_w[s][i] * grad_l2_w[s][i];
        sum_sq += (double)grad_l2_b[s] * grad_l2_b[s];
    }
    // FT weight gradients (only dirty rows).
    if (ft_dirty && grad_ft_w) {
        for (int fi = 0; fi < NNUE_FT_INPUTS; fi++) {
            if (!ft_dirty[fi]) continue;
            const float *gw = grad_ft_w + (size_t)fi * NNUE_HALF_DIMS;
            for (int d = 0; d < NNUE_HALF_DIMS; d++)
                sum_sq += (double)gw[d] * gw[d];
        }
    }
    // PSQT gradients (only dirty rows).
    if (ft_dirty && grad_psqt_w) {
        for (int fi = 0; fi < NNUE_FT_INPUTS; fi++) {
            if (!ft_dirty[fi]) continue;
            const float *gpw = grad_psqt_w + (size_t)fi * NNUE_PSQT_BKTS;
            for (int b = 0; b < NNUE_PSQT_BKTS; b++)
                sum_sq += (double)gpw[b] * gpw[b];
        }
    }
    // FT bias gradients.
    for (int d = 0; d < NNUE_HALF_DIMS; d++)
        sum_sq += (double)grad_ft_bias[d] * grad_ft_bias[d];
    // Dense piece value gradients.
    if (piece_val_active) {
        for (int pt = 0; pt < 6; pt++)
            sum_sq += (double)grad_piece_val[pt] * grad_piece_val[pt];
    }

    float norm = (float)sqrt(sum_sq);

    // Telemetry: tally call, fire, distribution.  Periodic summary every
    // CLIP_STATS_REPORT_EVERY calls; per-fire spam removed.
    g_clip_call_count++;
    g_clip_norm_sum += norm;
    if (norm < g_clip_norm_min) g_clip_norm_min = norm;
    if (norm > g_clip_norm_max) g_clip_norm_max = norm;
    g_clip_last_thr = max_norm;
    {
        int bin = 0;
        while (bin < CLIP_NORM_HIST_NEDGES && norm >= CLIP_NORM_HIST_EDGES[bin]) bin++;
        g_clip_hist[bin]++;
    }
    if (norm > max_norm) g_clip_fire_count++;
    if ((g_clip_call_count % CLIP_STATS_REPORT_EVERY) == 0) nnue_clip_stats_print();

    if (norm <= max_norm) return norm;

    // Scale all gradients by max_norm/norm.
    float scale = max_norm / norm;
    for (int s = 0; s < NNUE_LAYER_STACKS; s++) {
        for (int i = 0; i < NNUE_L0_SIZE * NNUE_L0_INPUT; i++) grad_l0_w[s][i] *= scale;
        for (int i = 0; i < NNUE_L0_SIZE; i++)                 grad_l0_b[s][i] *= scale;
        for (int i = 0; i < NNUE_L1_SIZE * NNUE_L1_PADDED; i++) grad_l1_w[s][i] *= scale;
        for (int i = 0; i < NNUE_L1_SIZE; i++)                 grad_l1_b[s][i] *= scale;
        for (int i = 0; i < NNUE_L2_PADDED; i++)               grad_l2_w[s][i] *= scale;
        grad_l2_b[s] *= scale;
    }
    if (ft_dirty && grad_ft_w) {
        for (int fi = 0; fi < NNUE_FT_INPUTS; fi++) {
            if (!ft_dirty[fi]) continue;
            float *gw = grad_ft_w + (size_t)fi * NNUE_HALF_DIMS;
            for (int d = 0; d < NNUE_HALF_DIMS; d++) gw[d] *= scale;
        }
    }
    if (ft_dirty && grad_psqt_w) {
        for (int fi = 0; fi < NNUE_FT_INPUTS; fi++) {
            if (!ft_dirty[fi]) continue;
            float *gpw = grad_psqt_w + (size_t)fi * NNUE_PSQT_BKTS;
            for (int b = 0; b < NNUE_PSQT_BKTS; b++) gpw[b] *= scale;
        }
    }
    for (int d = 0; d < NNUE_HALF_DIMS; d++) grad_ft_bias[d] *= scale;
    if (piece_val_active) {
        for (int pt = 0; pt < 6; pt++)
            grad_piece_val[pt] *= scale;
    }

    return norm;
}

// ---------------------------------------------------------------------------
// nnue_apply_gradients — update FP32 weights from accumulators, increment counts,
//                        then zero the accumulators.
// Only weights that received a non-zero gradient this game are updated / counted.
// ---------------------------------------------------------------------------
void nnue_apply_gradients(float lr_scale)
{
    t_adam++;
    t_ft_session++;

    // FT RMSProp bias-correction — two bc2 values to handle mixed warmed/fresh rows.
    //
    // "Cold" rows: v_ft_w was zeroed at startup (not loaded from disk).  Use
    // min(t_adam, t_ft_session) so bc2 tracks the actual sample count rather than
    // the persisted t_adam, preventing ~31× oversized first steps.
    //
    // "Warm" rows: v_ft_w was restored from disk (ft_v_warmed[fi] == true).  The
    // saved v is already a reliable estimate of E[g²] calibrated against t_adam
    // steps.  Using t_ft_session here would under-correct bc2 (bc2_small → v/bc2
    // >> E[g²] → step >> LR), so we use t_adam directly.
    const uint32_t ft_t        = std::min(t_adam, t_ft_session);
    const float    ft_bc2_cold  = 1.0f - powf(TDLEAF_ADAM_BETA2, (float)ft_t);
    const float    ft_bc2_warm  = 1.0f - powf(TDLEAF_ADAM_BETA2, (float)t_adam);

    // Linear LR warmup: ramp from 0 to full over the first WARMUP Adam steps.
    // Keyed on t_adam (persisted), so this only fires during the very first session.
    const float warmup_factor = (TDLEAF_ADAM_WARMUP > 0 && t_adam <= (uint32_t)TDLEAF_ADAM_WARMUP)
        ? (float)t_adam / (float)TDLEAF_ADAM_WARMUP
        : 1.0f;

    // Per-session FT LR warmup: ramp FT LR from 0→full over the first
    // TDLEAF_FT_SESSION_WARMUP Adam steps of each session.  Damps FT weight
    // updates during the v_ft_w accumulation phase at every restart, regardless
    // of whether v was loaded from disk.  Keyed on t_ft_session (not persisted).
    const float ft_session_factor =
        (TDLEAF_FT_SESSION_WARMUP > 0 && t_ft_session <= (uint32_t)TDLEAF_FT_SESSION_WARMUP)
        ? (float)t_ft_session / (float)TDLEAF_FT_SESSION_WARMUP
        : 1.0f;

    // Effective LRs.  lr_scale (<1.0 for replay) applied uniformly to all
    // categories to soften replay-pass updates; live path passes 1.0.  Env-var
    // multipliers (TDLEAF_LR_*) layered on top — read once at first call.
    const auto &lr_mul = tdleaf_lr_multipliers();
    // FC LRs are split four ways:
    //   fc_lr      — FC0/FC1 weights (int8 scale ~5)
    //   fc2_lr     — FC2 weights (int8 scale ~68; final 32→1 layer)
    //   fc_bias_lr — FC0/FC1/FC2 biases (int32 scale ~1500)
    // The split lets each section move at roughly the same fractional rate
    // per Adam step despite very different weight magnitudes.
    const float fc_lr      = lr_mul.fc      * lr_scale * warmup_factor * TDLEAF_ADAM_LR0;
    const float fc2_lr     = lr_mul.fc2     * lr_scale * warmup_factor * TDLEAF_ADAM_FC2_LR0;
    const float fc_bias_lr = lr_mul.fc_bias * lr_scale * warmup_factor * TDLEAF_ADAM_FC_BIAS_LR0;
    const float ft_lr      = lr_mul.ft      * lr_scale * warmup_factor * ft_session_factor * TDLEAF_ADAM_FT_LR0;
    const float ft_bias_lr = lr_mul.ft_bias * lr_scale * warmup_factor * TDLEAF_ADAM_FT_BIAS_LR0;
    const float psqt_lr    = lr_mul.psqt    * lr_scale * warmup_factor * TDLEAF_ADAM_PSQT_LR0;

    // Full Adam step for FC layers and FT biases — per-weight bias correction.
    // bc1 (beta1=0.9): skipped at cnt>=20 (0.9^20≈0.12 → bc1≈0.88, close to 1).
    // bc2 (beta2=0.999): ALWAYS applied (0.999^20≈0.98 → bc2=0.02; skipping gives ~7× oversized steps).
    // The unit-less step m_hat/sqrt(v_hat) is clipped to TDLEAF_ADAM_STEP_CLIP
    // before the LR multiply to bound the worst-case per-weight update.
    // lr is passed in so the same routine handles FC weights, FC2 weights,
    // FC biases, etc. — only the LR differs.
    auto do_step = [&](float g, float &m, float &v, uint32_t cnt, float lr) -> float {
        m = TDLEAF_ADAM_BETA1 * m + (1.0f - TDLEAF_ADAM_BETA1) * g;
        v = TDLEAF_ADAM_BETA2 * v + (1.0f - TDLEAF_ADAM_BETA2) * g * g;
        uint32_t eff_t = cnt + 1;
        float m_hat = (eff_t >= 20) ? m
            : m / (1.0f - powf(TDLEAF_ADAM_BETA1, (float)eff_t));
        float v_hat = v / (1.0f - powf(TDLEAF_ADAM_BETA2, (float)eff_t));
        float step  = m_hat / (sqrtf(v_hat) + TDLEAF_ADAM_EPS);
        step = clip_adam_step(step, step_max_fc, step_clips_fc);
        return lr * step;
    };
    // PSQT Adam step — same per-weight BC but uses TDLEAF_ADAM_PSQT_LR0.
    auto do_step_psqt = [&](float g, float &m, float &v, uint32_t cnt) -> float {
        m = TDLEAF_ADAM_BETA1 * m + (1.0f - TDLEAF_ADAM_BETA1) * g;
        v = TDLEAF_ADAM_BETA2 * v + (1.0f - TDLEAF_ADAM_BETA2) * g * g;
        uint32_t eff_t = cnt + 1;
        float m_hat = (eff_t >= 20) ? m
            : m / (1.0f - powf(TDLEAF_ADAM_BETA1, (float)eff_t));
        float v_hat = v / (1.0f - powf(TDLEAF_ADAM_BETA2, (float)eff_t));
        float step  = m_hat / (sqrtf(v_hat) + TDLEAF_ADAM_EPS);
        step = clip_adam_step(step, step_max_psqt, step_clips_psqt);
        return psqt_lr * step;
    };

    for (int s = 0; s < NNUE_LAYER_STACKS; s++) {
        // FC0 weights — fc_lr (int8 scale ~5)
        for (int i = 0; i < NNUE_L0_SIZE * NNUE_L0_INPUT; i++) {
            if (grad_l0_w[s][i] != 0.0f) {
                float dw = do_step(grad_l0_w[s][i], m_l0_w[s][i], v_l0_w[s][i], l0_weights_cnt[s][i], fc_lr);
                // AdamW: decoupled weight decay (weights only, not biases).
                // Scaled by the SAME LR used for the gradient step so the
                // relative pull-to-zero is constant across sections.
                float wd = TDLEAF_WEIGHT_DECAY * fc_lr * l0_weights_f32[s][i];
                l0_weights_f32[s][i] -= dw + wd;  delta_l0_w[s][i] -= dw + wd;
                // Clamp float shadow to int8 range: prevents zombie weights where the float
                // shadow drifts beyond ±127 while the requantised inference value is stuck.
                if (l0_weights_f32[s][i] >  127.0f) l0_weights_f32[s][i] =  127.0f;
                if (l0_weights_f32[s][i] < -127.0f) l0_weights_f32[s][i] = -127.0f;
                l0_weights_cnt[s][i]++;  delta_l0_w_cnt[s][i]++;
            }
        }
        // FC0 biases — fc_bias_lr (int32 scale ~2000)
        for (int i = 0; i < NNUE_L0_SIZE; i++) {
            if (grad_l0_b[s][i] != 0.0f) {
                float dw = do_step(grad_l0_b[s][i], m_l0_b[s][i], v_l0_b[s][i], l0_biases_cnt[s][i], fc_bias_lr);
                l0_biases_f32[s][i] -= dw;  delta_l0_b[s][i] -= dw;
                l0_biases_cnt[s][i]++;  delta_l0_b_cnt[s][i]++;
            }
        }
        // FC1 weights — fc_lr (int8 scale ~9)
        for (int i = 0; i < NNUE_L1_SIZE * NNUE_L1_PADDED; i++) {
            if (grad_l1_w[s][i] != 0.0f) {
                float dw = do_step(grad_l1_w[s][i], m_l1_w[s][i], v_l1_w[s][i], l1_weights_cnt[s][i], fc_lr);
                float wd = TDLEAF_WEIGHT_DECAY * fc_lr * l1_weights_f32[s][i];
                l1_weights_f32[s][i] -= dw + wd;  delta_l1_w[s][i] -= dw + wd;
                // Clamp float shadow to int8 range (same reason as FC0).
                if (l1_weights_f32[s][i] >  127.0f) l1_weights_f32[s][i] =  127.0f;
                if (l1_weights_f32[s][i] < -127.0f) l1_weights_f32[s][i] = -127.0f;
                l1_weights_cnt[s][i]++;  delta_l1_w_cnt[s][i]++;
            }
        }
        // FC1 biases — fc_bias_lr (int32 scale ~1500)
        for (int i = 0; i < NNUE_L1_SIZE; i++) {
            if (grad_l1_b[s][i] != 0.0f) {
                float dw = do_step(grad_l1_b[s][i], m_l1_b[s][i], v_l1_b[s][i], l1_biases_cnt[s][i], fc_bias_lr);
                l1_biases_f32[s][i] -= dw;  delta_l1_b[s][i] -= dw;
                l1_biases_cnt[s][i]++;  delta_l1_b_cnt[s][i]++;
            }
        }
        // FC2 weights — fc2_lr (int8 scale ~68; final 32→1 layer has higher
        // leverage per weight, so weights converge larger and need a larger LR).
        for (int i = 0; i < NNUE_L2_PADDED; i++) {
            if (grad_l2_w[s][i] != 0.0f) {
                float dw = do_step(grad_l2_w[s][i], m_l2_w[s][i], v_l2_w[s][i], l2_weights_cnt[s][i], fc2_lr);
                float wd = TDLEAF_WEIGHT_DECAY * fc2_lr * l2_weights_f32[s][i];
                l2_weights_f32[s][i] -= dw + wd;  delta_l2_w[s][i] -= dw + wd;
                // Clamp float shadow to int8 range (same reason as FC0/FC1).
                if (l2_weights_f32[s][i] >  127.0f) l2_weights_f32[s][i] =  127.0f;
                if (l2_weights_f32[s][i] < -127.0f) l2_weights_f32[s][i] = -127.0f;
                l2_weights_cnt[s][i]++;  delta_l2_w_cnt[s][i]++;
            }
        }
        // FC2 bias — fc_bias_lr (int32 scale ~860)
        if (grad_l2_b[s] != 0.0f) {
            float dw = do_step(grad_l2_b[s], m_l2_b[s], v_l2_b[s], l2_bias_cnt[s], fc_bias_lr);
            l2_bias_f32[s] -= dw;  delta_l2_b[s] -= dw;
            l2_bias_cnt[s]++;  delta_l2_b_cnt[s]++;
        }
    }
    memset(grad_l0_w, 0, sizeof(grad_l0_w));
    memset(grad_l0_b, 0, sizeof(grad_l0_b));
    memset(grad_l1_w, 0, sizeof(grad_l1_w));
    memset(grad_l1_b, 0, sizeof(grad_l1_b));
    memset(grad_l2_w, 0, sizeof(grad_l2_w));
    memset(grad_l2_b, 0, sizeof(grad_l2_b));

    // FT/PSQT: only iterate over dirty feature rows (sparse update).
    // PSQT mean-centering runs in nnue_mean_center_psqt_gradients() before
    // nnue_clip_gradients() so the slot-total does not enter the L2 norm.
    // That zeros the gradient slot-total but is NOT sufficient: Adam's
    // per-weight 1/sqrt(v_hat) normalisation makes the applied step (dw) not
    // zero-sum within a slot, since sparse features (low v) take
    // proportionally larger steps than dense ones.  Pass 2 below post-corrects
    // by spreading the per-slot mean dw back across every (dirty fi, bucket),
    // restoring the slot-total invariant exactly while leaving the per-bucket
    // distribution unconstrained (allows phase-dependent piece-value drift).
    // Without this, PSQT pawn slot drifted ~+13% over 50k games even with
    // piece_val[PAWN] hard-pinned at 0.
    if (ft_dirty) {
        double psqt_dw_total[11] = {};
        int    psqt_dirty_count[11] = {};
        // Pass 1: apply FT + PSQT Adam steps; accumulate per-slot PSQT dw.
        for (int fi = 0; fi < NNUE_FT_INPUTS; fi++) {
            if (!ft_dirty[fi]) continue;
            int slot = (fi % 704) / 64;
            // FT weights — per-weight RMSProp (no m; FT rows are too sparse for
            // first-moment to carry useful directional signal).
            float    *fw  = ft_weights_f32 + (size_t)fi * NNUE_HALF_DIMS;
            float    *gw  = grad_ft_w      + (size_t)fi * NNUE_HALF_DIMS;
            uint32_t *cnt = ft_weights_cnt + (size_t)fi * NNUE_HALF_DIMS;
            float    *fd  = ft_delta_f32 ? ft_delta_f32 + (size_t)fi * NNUE_HALF_DIMS : nullptr;
            if (v_ft_w) {
                // Select bias correction: warm rows (v loaded from disk) use t_adam;
                // cold rows (v=0 at startup) use min(t_adam, t_ft_session).
                const float ft_bc2 = (ft_v_warmed && ft_v_warmed[fi])
                                     ? ft_bc2_warm : ft_bc2_cold;
                float *vw = v_ft_w + (size_t)fi * NNUE_HALF_DIMS;
                for (int d = 0; d < NNUE_HALF_DIMS; d++) {
                    if (gw[d] != 0.0f) {
                        vw[d] = TDLEAF_ADAM_BETA2 * vw[d]
                               + (1.0f - TDLEAF_ADAM_BETA2) * gw[d] * gw[d];
                        float sv   = sqrtf(vw[d] / ft_bc2) + TDLEAF_ADAM_EPS;
                        float step = gw[d] / sv;
                        step = clip_adam_step(step, step_max_ft, step_clips_ft);
                        float dw = ft_lr * step;
                        // AdamW: decoupled weight decay (FT weights, not biases/PSQT)
                        float wd = TDLEAF_WEIGHT_DECAY * ft_lr * fw[d];
                        fw[d] -= dw + wd;  if (fd) fd[d] -= dw + wd;
                        cnt[d]++;
                        gw[d] = 0.0f;
                    }
                }
            }
            // PSQT weights — full Adam per-weight (only 8 buckets per row).
            // Skipped entirely under TDLEAF_FREEZE_MATERIAL (frozen prior).
            if (!TDLEAF_FREEZE_MATERIAL) {
                float    *pw   = psqt_weights_f32 + (size_t)fi * NNUE_PSQT_BKTS;
                float    *gpw  = grad_psqt_w      + (size_t)fi * NNUE_PSQT_BKTS;
                uint32_t *pcnt = psqt_weights_cnt + (size_t)fi * NNUE_PSQT_BKTS;
                float    *pd   = psqt_delta_f32 ? psqt_delta_f32 + (size_t)fi * NNUE_PSQT_BKTS : nullptr;
                for (int b = 0; b < NNUE_PSQT_BKTS; b++) {
                    if (gpw[b] != 0.0f && m_psqt_w && v_psqt_w) {
                        size_t vi = (size_t)fi * NNUE_PSQT_BKTS + b;
                        float dw = do_step_psqt(gpw[b], m_psqt_w[vi], v_psqt_w[vi], pcnt[b]);
                        pw[b] -= dw;  if (pd) pd[b] -= dw;
                        pcnt[b]++;
                        if (delta_psqt_cnt) delta_psqt_cnt[(size_t)fi * NNUE_PSQT_BKTS + b]++;
                        psqt_dw_total[slot] += dw;
                        gpw[b] = 0.0f;
                    }
                }
                psqt_dirty_count[slot]++;
            }
            // ft_dirty[fi] not cleared yet — Pass 2 needs to find dirty rows.
        }
        // Pass 2: post-Adam aggregate centering.  Spread the per-slot mean dw
        // (summed across all NNUE_PSQT_BKTS buckets) back to every dirty
        // feature × bucket cell so Σ_{fi dirty in slot S, b}(Δpw) = 0.  Buckets
        // can drift relative to each other (which is what lets PSQT learn
        // phase-dependent piece values via the HalfKAv2_hm bucket structure);
        // only the slot's absolute material level is anchored.  Slots with < 2
        // dirty features are skipped — their drift is bounded by O(1/2048).
        for (int fi = 0; fi < NNUE_FT_INPUTS; fi++) {
            if (!ft_dirty[fi]) continue;
            int slot = (fi % 704) / 64;
            if (!TDLEAF_FREEZE_MATERIAL && psqt_dirty_count[slot] >= 2) {
                float corr = (float)(psqt_dw_total[slot]
                                     / ((double)psqt_dirty_count[slot]
                                        * NNUE_PSQT_BKTS));
                float *pw = psqt_weights_f32 + (size_t)fi * NNUE_PSQT_BKTS;
                float *pd = psqt_delta_f32 ? psqt_delta_f32 + (size_t)fi * NNUE_PSQT_BKTS : nullptr;
                for (int b = 0; b < NNUE_PSQT_BKTS; b++) {
                    pw[b] += corr;
                    if (pd) pd[b] += corr;
                }
            }
            ft_dirty[fi] = false;
        }
    }

    // FT bias update — full Adam per-dimension, reduced LR to prevent dying-ReLU.
    // FT biases are dense (updated ~200×/game) while FT weights are sparse (~8/5000g).
    // Without a reduced LR, biases race ahead, suppressing dimensions before the
    // FT weights have learned useful features — the classic dying-ReLU problem.
    auto do_step_ft_bias = [&](float g, float &m, float &v, uint32_t cnt) -> float {
        m = TDLEAF_ADAM_BETA1 * m + (1.0f - TDLEAF_ADAM_BETA1) * g;
        v = TDLEAF_ADAM_BETA2 * v + (1.0f - TDLEAF_ADAM_BETA2) * g * g;
        uint32_t eff_t = cnt + 1;
        float m_hat = (eff_t >= 20) ? m
            : m / (1.0f - powf(TDLEAF_ADAM_BETA1, (float)eff_t));
        float v_hat = v / (1.0f - powf(TDLEAF_ADAM_BETA2, (float)eff_t));
        float step  = m_hat / (sqrtf(v_hat) + TDLEAF_ADAM_EPS);
        step = clip_adam_step(step, step_max_ft_bias, step_clips_ft_bias);
        return ft_bias_lr * step;
    };
    for (int d = 0; d < NNUE_HALF_DIMS; d++) {
        if (grad_ft_bias[d] == 0.0f) continue;
        float dw = do_step_ft_bias(grad_ft_bias[d], m_ft_bias[d], v_ft_bias[d], ft_bias_cnt[d]);
        ft_biases_f32[d] -= dw;
        ft_bias_delta[d] -= dw;
        ft_bias_cnt[d]++;  delta_ft_bias_cnt[d]++;
        grad_ft_bias[d] = 0.0f;
        ft_biases[d] = (int16_t)std::max(-32767.0f,
                                std::min( 32767.0f, roundf(ft_biases_f32[d])));
    }

    // Dense piece value update — full Adam, no weight decay.
    // Uses TDLEAF_ADAM_PV_LR0 (same scale as PSQT).
    // Skipped under TDLEAF_FREEZE_MATERIAL: piece_val stays at its init/loaded
    // value (0 on a fresh net — material lives in the frozen PSQT prior).
    if (piece_val_active && !TDLEAF_FREEZE_MATERIAL) {
        const float pv_lr = lr_mul.pv * lr_scale * warmup_factor * TDLEAF_ADAM_PV_LR0;
        auto do_step_pv = [&](float g, float &m, float &v, uint32_t cnt) -> float {
            m = TDLEAF_ADAM_BETA1 * m + (1.0f - TDLEAF_ADAM_BETA1) * g;
            v = TDLEAF_ADAM_BETA2 * v + (1.0f - TDLEAF_ADAM_BETA2) * g * g;
            uint32_t eff_t = cnt + 1;
            float m_hat = (eff_t >= 20) ? m
                : m / (1.0f - powf(TDLEAF_ADAM_BETA1, (float)eff_t));
            float v_hat = v / (1.0f - powf(TDLEAF_ADAM_BETA2, (float)eff_t));
            float step  = m_hat / (sqrtf(v_hat) + TDLEAF_ADAM_EPS);
            step = clip_adam_step(step, step_max_pv, step_clips_pv);
            return pv_lr * step;
        };
        bool pv_changed = false;
        for (int pt = 0; pt < 6; pt++) {
            // Gauge fix: piece_val[PAWN] is pinned at its init value (see
            // TDLEAF_PIN_PAWN_VALUE in tdleaf.h).  Discard the accumulated
            // pawn gradient so it doesn't leak into telemetry or the next
            // batch; leave moments (m, v) untouched.
            if (TDLEAF_PIN_PAWN_VALUE && pt == PAWN - 1) {
                grad_piece_val[pt] = 0.0f;
                continue;
            }
            if (grad_piece_val[pt] == 0.0f) continue;
            float dw = do_step_pv(grad_piece_val[pt],
                                  m_piece_val[pt], v_piece_val[pt],
                                  piece_val_cnt[pt]);
            float old_val = piece_val_f32[pt];
            // Clamp piece_val >= 0: negative piece values invert material
            // evaluation (engine plays to lose material), creating an
            // unrecoverable death spiral.  At zero, the engine has neutral
            // material knowledge (same as noprior), which FC/FT can recover
            // from.  The Adam moments continue to advance so the clamp is
            // lifted as soon as the gradient turns constructive.
            piece_val_f32[pt] = std::max(0.0f, old_val - dw);
            delta_piece_val[pt] += piece_val_f32[pt] - old_val;
            piece_val_cnt[pt]++;  delta_piece_val_cnt[pt]++;
            grad_piece_val[pt] = 0.0f;
            pv_changed = true;
        }
        if (pv_changed)
            nnue_extract_piece_values(false); // silent — called every batch during training
    }

    // Step-clip telemetry — write one line per batch to <engine_cfg.exec_path>tdleaf_telemetry.log.
    // Written to a file (not stderr) because cutechess-cli captures engine stderr
    // internally; our telemetry would otherwise be invisible.  Multiple concurrent
    // training engines share the same file path: each opens it in append mode, and
    // Linux guarantees atomic appends up to PIPE_BUF (4 KB), so line granularity
    // is preserved across processes without explicit locking.
    //
    // Single fprintf + fflush per line = one write() syscall.  File handle is
    // cached across calls (reopened lazily on first use).
    // Disabled at compile time with -D TDLEAF_LOG_STEP_CLIPS=0.
#if TDLEAF_LOG_STEP_CLIPS
    {
        // engine_cfg is declared in extern.h, already included via the unity build.
        static FILE *tele_fp = nullptr;
        if (tele_fp == nullptr) {
            char path[FILENAME_MAX + 64];
            snprintf(path, sizeof(path), "%stdleaf_telemetry.log", engine_cfg.exec_path);
            tele_fp = fopen(path, "a");
            // If fopen failed (e.g., read-only dir), silently skip — telemetry
            // is best-effort; the engine must not die over it.
        }
        if (tele_fp) {
            fprintf(tele_fp,
                    "[tdleaf step-clip] t_adam=%u  max|step| FC=%.2f FT=%.2f FTB=%.2f PSQT=%.2f PV=%.2f  "
                    "clips FC=%llu FT=%llu FTB=%llu PSQT=%llu PV=%llu  (clip=%.1f)\n",
                    (unsigned)t_adam,
                    step_max_fc, step_max_ft, step_max_ft_bias, step_max_psqt, step_max_pv,
                    (unsigned long long)step_clips_fc,
                    (unsigned long long)step_clips_ft,
                    (unsigned long long)step_clips_ft_bias,
                    (unsigned long long)step_clips_psqt,
                    (unsigned long long)step_clips_pv,
                    (double)TDLEAF_ADAM_STEP_CLIP);
            fflush(tele_fp);
        }
    }
#endif
    step_max_fc = step_max_ft = step_max_ft_bias = step_max_psqt = step_max_pv = 0.0f;
    step_clips_fc = step_clips_ft = step_clips_ft_bias = step_clips_psqt = step_clips_pv = 0;
}

// ---------------------------------------------------------------------------
// nnue_requantize_fc — FP32 → int8 for the live inference arrays
// ---------------------------------------------------------------------------
void nnue_requantize_fc()
{
    // Weights are stored at raw int8/int32 scale, so requantise by simply
    // rounding — no multiplication by q_scale or b_scale needed.
    for (int s = 0; s < NNUE_LAYER_STACKS; s++) {
        // FC0 weights: natural → vdotq layout
        for (int o = 0; o < NNUE_L0_SIZE; o++) {
            int ob = o / 4, k = o % 4;
            int64_t bq = (int64_t)roundf(l0_biases_f32[s][o]);
            l0_biases[s][o] = (int32_t)((bq < -2147483647) ? -2147483647 :
                                         (bq >  2147483647) ?  2147483647 : bq);
            for (int i = 0; i < NNUE_L0_INPUT; i++) {
                int ib = i / 4, j = i % 4;
                int q = (int)roundf(l0_weights_f32[s][o * NNUE_L0_INPUT + i]);
                if (q < -127) q = -127; if (q > 127) q = 127;
                l0_weights[s][ib * 64 + ob * 16 + k * 4 + j] = (int8_t)q;
            }
        }
        // FC1 weights: natural → vdotq layout
        for (int o = 0; o < NNUE_L1_SIZE; o++) {
            int ob = o / 4, k = o % 4;
            int64_t bq = (int64_t)roundf(l1_biases_f32[s][o]);
            l1_biases[s][o] = (int32_t)((bq < -2147483647) ? -2147483647 :
                                          (bq >  2147483647) ?  2147483647 : bq);
            for (int i = 0; i < NNUE_L1_PADDED; i++) {
                int ib = i / 4, j = i % 4;
                int q = (int)roundf(l1_weights_f32[s][o * NNUE_L1_PADDED + i]);
                if (q < -127) q = -127; if (q > 127) q = 127;
                l1_weights[s][ib * 128 + ob * 16 + k * 4 + j] = (int8_t)q;
            }
        }
        // FC2 weights: natural layout (no reorder needed)
        int64_t bq = (int64_t)roundf(l2_bias_f32[s]);
        out_biases[s] = (int32_t)((bq < -2147483647) ? -2147483647 :
                                   (bq >  2147483647) ?  2147483647 : bq);
        for (int i = 0; i < NNUE_L2_PADDED; i++) {
            int q = (int)roundf(l2_weights_f32[s][i]);
            if (q < -127) q = -127; if (q > 127) q = 127;
            out_weights[s][i] = (int8_t)q;
        }
    }
    // FT/PSQT weights: round float → int16/int32.
    // ft_weights_f32 tracks raw int16 scale (same units as ft_weights).
    if (ft_weights_f32) {
        for (int fi = 0; fi < NNUE_FT_INPUTS; fi++) {
            const float *fw = ft_weights_f32 + (size_t)fi * NNUE_HALF_DIMS;
            int16_t     *iw = ft_weights      + (size_t)fi * NNUE_HALF_DIMS;
            for (int d = 0; d < NNUE_HALF_DIMS; d++) {
                int v = (int)roundf(fw[d]);
                if (v < -32767) v = -32767;
                if (v >  32767) v =  32767;
                iw[d] = (int16_t)v;
            }
            const float *pw = psqt_weights_f32 + (size_t)fi * NNUE_PSQT_BKTS;
            int32_t     *ip = psqt_weights      + (size_t)fi * NNUE_PSQT_BKTS;
            for (int b = 0; b < NNUE_PSQT_BKTS; b++)
                ip[b] = (int32_t)roundf(pw[b]);
        }
    }

    // Clear score hash — cached evaluations are now stale.
    if (score_table && SCORE_SIZE > 0)
        memset(score_table, 0, SCORE_SIZE * sizeof(score_rec));
}

// ---------------------------------------------------------------------------
// nnue_save_fc_weights / nnue_load_fc_weights — companion .tdleaf.bin file
//
// Version 6 layout (current):
//   magic(4) + version(4)
//   8 stacks × FC block (same as v2):
//     FC0 biases:  float32[L0_SIZE]            × TDLEAF_SCALE  (64 B)
//                  uint32_t counts[L0_SIZE]                     (64 B)
//     FC0 weights: float32[L0_SIZE*L0_INPUT]   × TDLEAF_SCALE  (65536 B, natural layout)
//                  uint32_t counts[L0_SIZE*L0_INPUT]            (65536 B)
//     FC1 biases:  float32[L1_SIZE]            × TDLEAF_SCALE  (128 B)
//                  uint32_t counts[L1_SIZE]                     (128 B)
//     FC1 weights: float32[L1_SIZE*L1_PADDED]  × TDLEAF_SCALE  (4096 B, natural layout)
//                  uint32_t counts[L1_SIZE*L1_PADDED]           (4096 B)
//     FC2 bias:    float32[1]                  × TDLEAF_SCALE  (4 B)
//                  uint32_t count[1]                            (4 B)
//     FC2 weights: float32[L2_PADDED]          × TDLEAF_SCALE  (128 B)
//                  uint32_t counts[L2_PADDED]                   (128 B)
//   Sparse FT/PSQT section (new in v3):
//     n_ft_rows(4): number of feature rows with any update history
//     For each dirty row (8,260 bytes each):
//       fi(4): feature index in [0, FT_INPUTS)
//       float32[HALF_DIMS] × TDLEAF_SCALE: FT weights           (4096 B)
//       uint32_t[HALF_DIMS]: FT update counts                    (4096 B)
//       float32[PSQT_BKTS] × TDLEAF_SCALE: PSQT weights         (32 B)
//       uint32_t[PSQT_BKTS]: PSQT update counts                  (32 B)
//
// TDLEAF_SCALE = 128: stores w_f32 × 128 so sub-integer drift survives sessions.
// On load, divide by TDLEAF_SCALE to restore w_f32 exactly.
//
// Version 4 additions:
//   FT bias section (appended after sparse FT/PSQT rows):
//     float32[HALF_DIMS] × TDLEAF_SCALE: FT bias values         (4096 B)
//     uint32_t[HALF_DIMS]: FT bias update counts                 (4096 B)
//
// Version 5 additions:
//   Dense piece value section (appended after FT biases):
//     float32[6][PSQT_BKTS] × TDLEAF_SCALE: piece values        (192 B)
//     uint32_t[6][PSQT_BKTS]: update counts                      (192 B)
//
// Version 6 additions:
//   Adam v (second-moment) section — persists gradient scale across sessions.
//   Multi-writer merge uses max(v_file, v_local) per element — conservative
//   and safe for concurrent training instances.
//     uint32_t t_adam                                              (4 B)
//     8 stacks × FC v block (raw float32, no TDLEAF_SCALE):
//       float32[L0_SIZE]            v_l0_b                        (64 B)
//       float32[L0_SIZE*L0_INPUT]   v_l0_w                        (65536 B)
//       float32[L1_SIZE]            v_l1_b                        (128 B)
//       float32[L1_SIZE*L1_PADDED]  v_l1_w                        (4096 B)
//       float32[1]                  v_l2_b                        (4 B)
//       float32[L2_PADDED]          v_l2_w                        (128 B)
//     float32[HALF_DIMS]            v_ft_bias                     (4096 B)
//     float32[6][PSQT_BKTS]        v_piece_val                   (192 B)
//     uint32_t n_psqt_v_rows: count of dirty PSQT rows           (4 B)
//     For each dirty row:
//       uint32_t fi                                               (4 B)
//       float32[PSQT_BKTS]         v_psqt                        (32 B)
//   Total v section: ~563 KB (FC) + 4 KB (FT bias) + sparse PSQT
//   FT weight v (~92 MB) is NOT persisted — too large, and FT updates
//   are sparse enough that v barely converges before process restart.
//
// Version 7 additions:
//   Adam m (first-moment / momentum) section — persists gradient direction
//   across sessions.  Multi-writer merge uses element-wise average
//   (m_file + m_local) / 2.  FT weight m not applicable (RMSProp, no m).
//   8 stacks × FC m block (raw float32, signed):
//       float32[L0_SIZE]            m_l0_b                        (64 B)
//       float32[L0_SIZE*L0_INPUT]   m_l0_w                        (65536 B)
//       float32[L1_SIZE]            m_l1_b                        (128 B)
//       float32[L1_SIZE*L1_PADDED]  m_l1_w                        (4096 B)
//       float32[1]                  m_l2_b                        (4 B)
//       float32[L2_PADDED]          m_l2_w                        (128 B)
//     float32[HALF_DIMS]            m_ft_bias                     (4096 B)
//     float32[6][PSQT_BKTS]        m_piece_val                   (192 B)
//     uint32_t n_psqt_m_rows: count of dirty PSQT rows           (4 B)
//     For each dirty row:
//       uint32_t fi                                               (4 B)
//       float32[PSQT_BKTS]         m_psqt                        (32 B)
//   Total m section: ~563 KB (FC) + 4 KB (FT bias) + sparse PSQT (~1.5 MB)
//
// Version 8 additions (current):
//   Sparse FT v (second-moment) section — persists v_ft_w for feature rows
//   where v is non-zero (i.e. where gradient updates have actually occurred
//   in the current session).  CRITICAL: only non-zero rows are saved.  If a
//   dirty row has v=0 (e.g. first session from a v7 file, before any Adam
//   step touches that row), saving it as "warmed" would produce bc2_warm≈1
//   with v=0, giving sv≈ε and steps ~10,000× LR — catastrophic.
//   Multi-writer merge uses max(v_file, v_local) per element (same as FC v).
//   On load, ft_v_warmed[fi] is set to true for restored rows so the RMSProp
//   update uses t_adam (not t_ft_session) for bc2, since the saved v is
//   already calibrated against t_adam steps.
//     uint32_t n_ft_v_rows: count of rows with non-zero v (4 B)
//     For each row:
//       uint32_t fi                                               (4 B)
//       float32[HALF_DIMS]          v_ft                          (4096 B)
//   Total FT v section: 4 + n_dirty × 4100 B (e.g. 5 MB for 1250 dirty rows)
//
// Version 5 (legacy): FC + FT/PSQT + FT biases + piece values, no Adam v/m.
// Version 4 (legacy): FC + FT/PSQT + FT biases, no piece values.
// Version 3 (legacy): FC + sparse FT/PSQT, no FT bias section.
// Version 2 (legacy): FC block only, no FT/PSQT section.
// Version 1 (legacy): int32 biases + int8 weights, no counts.
//
// Version 10 additions (current):
//   Source-.nnue content fingerprint — prevents accidentally pairing a
//   .tdleaf.bin with a different .nnue than the one it was trained against.
//   Layout: header is now
//     magic(4) + version(4) + nnue_content_hash(4) + [rest unchanged]
//   nnue_content_hash is FNV-1a over the .nnue's FT weight bytes, computed
//   at load/init time (see nnue_update_content_hash() in nnue.cpp).
//   On load: reject the file if the stored hash does not match the loaded .nnue.
//   On save merge-read: abort the save rather than corrupting another worker's
//   weights written against a different .nnue.
//   v5–v9 files have no hash and are accepted without a check; saving promotes
//   them to v10 with the current .nnue's hash.
// ---------------------------------------------------------------------------
static const float    TDLEAF_SCALE   = 128.0f;
static const uint32_t TDLEAF_MAGIC   = 0x544D4C46u; // "TMLF"
static const uint32_t TDLEAF_VERSION = 11u;

// ---------------------------------------------------------------------------
// tdleaf_acquire_lock / tdleaf_release_lock
//
// Advisory file lock using a companion "<path>.lock" file so that locking
// survives atomic rename() of the main .tdleaf.bin.
//   how = LOCK_EX (exclusive write) or LOCK_SH (shared read)
// Returns the lock fd on success, -1 on failure.
// ---------------------------------------------------------------------------
// Returns lock fd on success, -2 if lock is busy (EWOULDBLOCK, caller should skip),
// or -1 on a real error.
static int tdleaf_acquire_lock(const char *path, int how)
{
    char lock_path[FILENAME_MAX];
    snprintf(lock_path, sizeof(lock_path), "%s.lock", path);
    int fd = open(lock_path, O_RDWR | O_CREAT, 0644);
    if (fd < 0) {
        fprintf(stderr, "TDLeaf: cannot open lock file %s\n", lock_path);
        return -1;
    }
    if (flock(fd, how) != 0) {
        int err = errno;
        close(fd);
        if (err == EWOULDBLOCK)
            return -2;  // lock held by another process — caller decides what to do
        fprintf(stderr, "TDLeaf: flock failed on %s\n", lock_path);
        return -1;
    }
    return fd;
}
static void tdleaf_release_lock(int fd) { if (fd >= 0) close(fd); }

// Counts how many saves were deferred because the lock was held by another process.
// Accumulated deltas are NOT lost — they are flushed on the next successful save.
static uint64_t tdleaf_save_skip_count = 0;

bool nnue_save_fc_weights(const char *path)
{
    // ---- Acquire exclusive lock (non-blocking) -------------------------
    // If another process holds the lock, skip this save rather than blocking.
    // Accumulated deltas are preserved in memory and flushed on the next
    // successful save, so no training signal is lost — just deferred.
    int lock_fd = tdleaf_acquire_lock(path, LOCK_EX | LOCK_NB);
    if (lock_fd == -2) {
        tdleaf_save_skip_count++;
        fprintf(stderr, "TDLeaf: lock busy — deferring save (deferred: %llu)\n",
                (unsigned long long)tdleaf_save_skip_count);
        return true;  // not an error; deltas will be written next time
    }
    if (lock_fd < 0) {
        fprintf(stderr, "TDLeaf: cannot acquire exclusive lock for %s\n", path);
        return false;
    }

    // Scratch buffers for section-level bulk I/O (sized to the largest FC
    // section, L0_SIZE × L0_INPUT floats).  Static: saves are serialized by
    // the exclusive lock, and within a process only one save runs at a time.
    static float    io_buf_f[NNUE_L0_SIZE * NNUE_L0_INPUT];
    static uint32_t io_buf_u[NNUE_L0_SIZE * NNUE_L0_INPUT];

    // ---- Re-read the current file and merge our deltas on top ----------
    // This picks up any changes written by other concurrent Leaf instances
    // since we last synced.  After merge, float shadows = file + our_delta.
    FILE *cur = fopen(path, "rb");
    if (cur) {
        setvbuf(cur, nullptr, _IOFBF, 4u << 20);
        uint32_t magic = 0, version = 0;
        bool ok = (fread(&magic, 4, 1, cur) == 1 &&
                   fread(&version, 4, 1, cur) == 1 &&
                   magic == TDLEAF_MAGIC &&
                   (version == TDLEAF_VERSION || version == 10u || version == 9u || version == 8u || version == 7u || version == 6u || version == 5u || version == 4u || version == 3u || version == 2u));
        if (ok && version >= 10u) {
            // v10+: header carries the source-.nnue content hash.  If another
            // worker wrote the file against a different .nnue, abort the save
            // rather than corrupt their weights with ours.
            uint32_t file_content_hash = 0;
            if (fread(&file_content_hash, 4, 1, cur) != 1) {
                ok = false;
            } else if (file_content_hash != nnue_content_hash) {
                fprintf(stderr,
                        "TDLeaf: %s was written against a different .nnue "
                        "(file hash=0x%08X, loaded .nnue hash=0x%08X).\n"
                        "        Aborting save to avoid corrupting the other worker's weights.\n",
                        path, file_content_hash, nnue_content_hash);
                fclose(cur);
                tdleaf_release_lock(lock_fd);
                return false;
            }
        }
        if (ok) {
            // FC section: float32 × TDLEAF_SCALE per weight, then uint32 counts.
            // Merge: shadow = file_value + our_delta; count = file_count + our_delta_count.
            for (int s = 0; s < NNUE_LAYER_STACKS && ok; s++) {
                // Bulk-read each section into the scratch buffer, then merge.
                // On a short read, merge the elements that were read (matching
                // the old per-element behavior) and report failure.
                auto merge_f = [&](float *shadow, float *delta, uint32_t *cnt, int n) -> bool {
                    (void)cnt;
                    size_t got = fread(io_buf_f, sizeof(float), n, cur);
                    for (size_t i = 0; i < got; i++) {
                        shadow[i] = io_buf_f[i] / TDLEAF_SCALE + delta[i];
                        delta[i]  = 0.0f;
                    }
                    return got == (size_t)n;
                };
                // Additive count merge: cnt = file_count + delta_count.
                // delta_cnt tracks only updates since last sync, so adding it to
                // the file's count correctly accumulates across concurrent instances.
                auto merge_cnt = [&](uint32_t *cnt, uint32_t *dcnt, int n) -> bool {
                    size_t got = fread(io_buf_u, sizeof(uint32_t), n, cur);
                    for (size_t i = 0; i < got; i++) {
                        cnt[i] = io_buf_u[i] + dcnt[i];
                        dcnt[i] = 0;
                    }
                    return got == (size_t)n;
                };
                ok = merge_f(l0_biases_f32[s],  delta_l0_b[s], l0_biases_cnt[s],  NNUE_L0_SIZE)
                  && merge_cnt(l0_biases_cnt[s],  delta_l0_b_cnt[s], NNUE_L0_SIZE)
                  && merge_f(l0_weights_f32[s], delta_l0_w[s], l0_weights_cnt[s], NNUE_L0_SIZE * NNUE_L0_INPUT)
                  && merge_cnt(l0_weights_cnt[s], delta_l0_w_cnt[s], NNUE_L0_SIZE * NNUE_L0_INPUT)
                  && merge_f(l1_biases_f32[s],  delta_l1_b[s], l1_biases_cnt[s],  NNUE_L1_SIZE)
                  && merge_cnt(l1_biases_cnt[s],  delta_l1_b_cnt[s], NNUE_L1_SIZE)
                  && merge_f(l1_weights_f32[s], delta_l1_w[s], l1_weights_cnt[s], NNUE_L1_SIZE * NNUE_L1_PADDED)
                  && merge_cnt(l1_weights_cnt[s], delta_l1_w_cnt[s], NNUE_L1_SIZE * NNUE_L1_PADDED)
                  && merge_f(&l2_bias_f32[s],    &delta_l2_b[s], &l2_bias_cnt[s],   1)
                  && merge_cnt(&l2_bias_cnt[s],   &delta_l2_b_cnt[s], 1)
                  && merge_f(l2_weights_f32[s], delta_l2_w[s], l2_weights_cnt[s], NNUE_L2_PADDED)
                  && merge_cnt(l2_weights_cnt[s], delta_l2_w_cnt[s], NNUE_L2_PADDED);
            }
            // FT/PSQT sparse section (v3+).
            if (ok && (version >= 3u) && ft_weights_f32) {
                uint32_t n_ft_rows = 0;
                if (fread(&n_ft_rows, sizeof(uint32_t), 1, cur) == 1) {
                    float tmp_w[NNUE_HALF_DIMS];
                    float tmp_p[NNUE_PSQT_BKTS];
                    uint32_t tmp_wc[NNUE_HALF_DIMS];
                    uint32_t tmp_pc[NNUE_PSQT_BKTS];
                    for (uint32_t k = 0; k < n_ft_rows; k++) {
                        uint32_t fi;
                        if (fread(&fi, sizeof(uint32_t), 1, cur) != 1 ||
                            fi >= (uint32_t)NNUE_FT_INPUTS) break;
                        if (fread(tmp_w,  sizeof(float),    NNUE_HALF_DIMS, cur) != (size_t)NNUE_HALF_DIMS ||
                            fread(tmp_wc, sizeof(uint32_t), NNUE_HALF_DIMS, cur) != (size_t)NNUE_HALF_DIMS ||
                            fread(tmp_p,  sizeof(float),    NNUE_PSQT_BKTS, cur) != (size_t)NNUE_PSQT_BKTS ||
                            fread(tmp_pc, sizeof(uint32_t), NNUE_PSQT_BKTS, cur) != (size_t)NNUE_PSQT_BKTS)
                            break;
                        // Merge FT: shadow = file_value + our_delta; count = max.
                        float    *fw  = ft_weights_f32   + (size_t)fi * NNUE_HALF_DIMS;
                        uint32_t *wc  = ft_weights_cnt   + (size_t)fi * NNUE_HALF_DIMS;
                        float    *fd  = ft_delta_f32     ? ft_delta_f32   + (size_t)fi * NNUE_HALF_DIMS : nullptr;
                        for (int d = 0; d < NNUE_HALF_DIMS; d++) {
                            fw[d] = tmp_w[d] / TDLEAF_SCALE + (fd ? fd[d] : 0.0f);
                            if (fd) fd[d] = 0.0f;
                            if (tmp_wc[d] > wc[d]) wc[d] = tmp_wc[d];
                        }
                        // Merge PSQT: additive count merge via delta_psqt_cnt.
                        float    *pw  = psqt_weights_f32 + (size_t)fi * NNUE_PSQT_BKTS;
                        uint32_t *pc  = psqt_weights_cnt + (size_t)fi * NNUE_PSQT_BKTS;
                        float    *pd  = psqt_delta_f32   ? psqt_delta_f32 + (size_t)fi * NNUE_PSQT_BKTS : nullptr;
                        uint32_t *pdc = delta_psqt_cnt   ? delta_psqt_cnt + (size_t)fi * NNUE_PSQT_BKTS : nullptr;
                        for (int b = 0; b < NNUE_PSQT_BKTS; b++) {
                            pw[b] = tmp_p[b] / TDLEAF_SCALE + (pd ? pd[b] : 0.0f);
                            if (pd) pd[b] = 0.0f;
                            if (pdc) { pc[b] = tmp_pc[b] + pdc[b]; pdc[b] = 0; }
                            else     { if (tmp_pc[b] > pc[b]) pc[b] = tmp_pc[b]; }
                        }
                    }
                }
                // FT bias section (v4+).
                if (version >= 4u) {
                    float tmp_b[NNUE_HALF_DIMS];
                    uint32_t tmp_bc[NNUE_HALF_DIMS];
                    if (fread(tmp_b,  sizeof(float),    NNUE_HALF_DIMS, cur) == (size_t)NNUE_HALF_DIMS &&
                        fread(tmp_bc, sizeof(uint32_t), NNUE_HALF_DIMS, cur) == (size_t)NNUE_HALF_DIMS) {
                        for (int d = 0; d < NNUE_HALF_DIMS; d++) {
                            ft_biases_f32[d] = tmp_b[d] / TDLEAF_SCALE + ft_bias_delta[d];
                            ft_bias_delta[d] = 0.0f;
                            ft_bias_cnt[d] = tmp_bc[d] + delta_ft_bias_cnt[d];
                            delta_ft_bias_cnt[d] = 0;
                        }
                    }
                }
                // Dense piece value section (v5+).
                // v9+: float32[6] + uint32[6] (one value per piece type).
                // v5-v8: float32[6][8] + uint32[6][8] (per-bucket; collapse by averaging).
                if (version >= 5u) {
                    if (version >= 9u) {
                        float tmp_pv[6]; uint32_t tmp_pvc[6];
                        if (fread(tmp_pv,  sizeof(float),    6, cur) == 6 &&
                            fread(tmp_pvc, sizeof(uint32_t), 6, cur) == 6) {
                            for (int pt = 0; pt < 6; pt++) {
                                // Clamp ≥ 0: a negative merge result inverts material
                                // evaluation (death spiral); see nnue_apply_gradients.
                                piece_val_f32[pt] = std::max(0.0f,
                                    tmp_pv[pt] / TDLEAF_SCALE + delta_piece_val[pt]);
                                delta_piece_val[pt] = 0.0f;
                                piece_val_cnt[pt] = tmp_pvc[pt] + delta_piece_val_cnt[pt];
                                delta_piece_val_cnt[pt] = 0;
                            }
                            piece_val_active = true;
                        }
                    } else {
                        // v5-v8: collapse [6][8] to [6] by averaging across buckets.
                        float tmp_pv[6][NNUE_PSQT_BKTS]; uint32_t tmp_pvc[6][NNUE_PSQT_BKTS];
                        if (fread(tmp_pv,  sizeof(float),    6*NNUE_PSQT_BKTS, cur) == (size_t)(6*NNUE_PSQT_BKTS) &&
                            fread(tmp_pvc, sizeof(uint32_t), 6*NNUE_PSQT_BKTS, cur) == (size_t)(6*NNUE_PSQT_BKTS)) {
                            for (int pt = 0; pt < 6; pt++) {
                                float sum = 0.0f; uint32_t cnt_sum = 0;
                                for (int b = 0; b < NNUE_PSQT_BKTS; b++) { sum += tmp_pv[pt][b]; cnt_sum += tmp_pvc[pt][b]; }
                                piece_val_f32[pt] = std::max(0.0f,
                                    sum / NNUE_PSQT_BKTS / TDLEAF_SCALE + delta_piece_val[pt]);
                                delta_piece_val[pt] = 0.0f;
                                piece_val_cnt[pt] = cnt_sum / NNUE_PSQT_BKTS + delta_piece_val_cnt[pt];
                                delta_piece_val_cnt[pt] = 0;
                            }
                            piece_val_active = true;
                        }
                    }
                }
                // Adam v section (v6+): max-merge per element.
                if (version >= 6u) {
                    uint32_t file_t;
                    if (fread(&file_t, sizeof(uint32_t), 1, cur) == 1) {
                        if (file_t > t_adam) t_adam = file_t;
                    }
                    // FC v arrays: max-merge each element (bulk read; on a short
                    // read merge what arrived, matching old per-element behavior).
                    auto merge_v = [&](float *v_local, int n) {
                        size_t got = fread(io_buf_f, sizeof(float), n, cur);
                        for (size_t i = 0; i < got; i++)
                            if (io_buf_f[i] > v_local[i]) v_local[i] = io_buf_f[i];
                    };
                    for (int s = 0; s < NNUE_LAYER_STACKS; s++) {
                        merge_v(v_l0_b[s], NNUE_L0_SIZE);
                        merge_v(v_l0_w[s], NNUE_L0_SIZE * NNUE_L0_INPUT);
                        merge_v(v_l1_b[s], NNUE_L1_SIZE);
                        merge_v(v_l1_w[s], NNUE_L1_SIZE * NNUE_L1_PADDED);
                        merge_v(&v_l2_b[s], 1);
                        merge_v(v_l2_w[s], NNUE_L2_PADDED);
                    }
                    merge_v(v_ft_bias, NNUE_HALF_DIMS);
                    if (version >= 9u)
                        merge_v(v_piece_val, 6);
                    else {
                        // v6-v8: 48 floats → take max per piece type across 8 buckets.
                        float tmp_v8[6][NNUE_PSQT_BKTS] = {};
                        for (int i = 0; i < 6 * NNUE_PSQT_BKTS; i++) {
                            float tmp; if (fread(&tmp, sizeof(float), 1, cur) != 1) break;
                            ((float*)tmp_v8)[i] = tmp;
                        }
                        for (int pt = 0; pt < 6; pt++)
                            for (int b = 0; b < NNUE_PSQT_BKTS; b++)
                                if (tmp_v8[pt][b] > v_piece_val[pt]) v_piece_val[pt] = tmp_v8[pt][b];
                    }
                    // Sparse PSQT v.
                    uint32_t n_pv_rows = 0;
                    if (fread(&n_pv_rows, sizeof(uint32_t), 1, cur) == 1 && v_psqt_w) {
                        for (uint32_t k = 0; k < n_pv_rows; k++) {
                            uint32_t fi;
                            float tmp_pv[NNUE_PSQT_BKTS];
                            if (fread(&fi, sizeof(uint32_t), 1, cur) != 1 ||
                                fi >= (uint32_t)NNUE_FT_INPUTS) break;
                            if (fread(tmp_pv, sizeof(float), NNUE_PSQT_BKTS, cur)
                                != (size_t)NNUE_PSQT_BKTS) break;
                            float *vp = v_psqt_w + (size_t)fi * NNUE_PSQT_BKTS;
                            for (int b = 0; b < NNUE_PSQT_BKTS; b++)
                                if (tmp_pv[b] > vp[b]) vp[b] = tmp_pv[b];
                        }
                    }
                }
                // Adam m section (v7+): average-merge per element.
                // Workers seeing the same gradient direction reinforce each other;
                // conflicting directions reduce toward zero (appropriate — uncertainty
                // about direction → smaller step, not a random-direction step).
                if (version >= 7u) {
                    auto merge_m = [&](float *m_local, int n) {
                        size_t got = fread(io_buf_f, sizeof(float), n, cur);
                        for (size_t i = 0; i < got; i++)
                            m_local[i] = 0.5f * (m_local[i] + io_buf_f[i]);
                    };
                    for (int s = 0; s < NNUE_LAYER_STACKS; s++) {
                        merge_m(m_l0_b[s], NNUE_L0_SIZE);
                        merge_m(m_l0_w[s], NNUE_L0_SIZE * NNUE_L0_INPUT);
                        merge_m(m_l1_b[s], NNUE_L1_SIZE);
                        merge_m(m_l1_w[s], NNUE_L1_SIZE * NNUE_L1_PADDED);
                        merge_m(&m_l2_b[s], 1);
                        merge_m(m_l2_w[s], NNUE_L2_PADDED);
                    }
                    merge_m(m_ft_bias, NNUE_HALF_DIMS);
                    if (version >= 9u)
                        merge_m(m_piece_val, 6);
                    else {
                        // v7-v8: 48 floats → average per piece type across 8 buckets.
                        float tmp_m8[6][NNUE_PSQT_BKTS] = {};
                        for (int i = 0; i < 6 * NNUE_PSQT_BKTS; i++) {
                            float tmp; if (fread(&tmp, sizeof(float), 1, cur) != 1) break;
                            ((float*)tmp_m8)[i] = tmp;
                        }
                        for (int pt = 0; pt < 6; pt++) {
                            float avg = 0.0f;
                            for (int b = 0; b < NNUE_PSQT_BKTS; b++) avg += tmp_m8[pt][b];
                            m_piece_val[pt] = 0.5f * (m_piece_val[pt] + avg / NNUE_PSQT_BKTS);
                        }
                    }
                    // Sparse PSQT m.
                    uint32_t n_pm_rows = 0;
                    if (fread(&n_pm_rows, sizeof(uint32_t), 1, cur) == 1 && m_psqt_w) {
                        for (uint32_t k = 0; k < n_pm_rows; k++) {
                            uint32_t fi;
                            float tmp_pm[NNUE_PSQT_BKTS];
                            if (fread(&fi, sizeof(uint32_t), 1, cur) != 1 ||
                                fi >= (uint32_t)NNUE_FT_INPUTS) break;
                            if (fread(tmp_pm, sizeof(float), NNUE_PSQT_BKTS, cur)
                                != (size_t)NNUE_PSQT_BKTS) break;
                            float *mp = m_psqt_w + (size_t)fi * NNUE_PSQT_BKTS;
                            for (int b = 0; b < NNUE_PSQT_BKTS; b++)
                                mp[b] = 0.5f * (mp[b] + tmp_pm[b]);
                        }
                    }
                }
                // Sparse FT v section (v8+): max-merge v_ft_w rows.
                if (version >= 8u && v_ft_w) {
                    uint32_t n_ftv_rows = 0;
                    if (fread(&n_ftv_rows, sizeof(uint32_t), 1, cur) == 1) {
                        for (uint32_t k = 0; k < n_ftv_rows; k++) {
                            uint32_t fi;
                            float tmp_v[NNUE_HALF_DIMS];
                            if (fread(&fi, sizeof(uint32_t), 1, cur) != 1 ||
                                fi >= (uint32_t)NNUE_FT_INPUTS) break;
                            if (fread(tmp_v, sizeof(float), NNUE_HALF_DIMS, cur)
                                != (size_t)NNUE_HALF_DIMS) break;
                            float *vw = v_ft_w + (size_t)fi * NNUE_HALF_DIMS;
                            for (int d = 0; d < NNUE_HALF_DIMS; d++)
                                if (tmp_v[d] > vw[d]) vw[d] = tmp_v[d];
                        }
                    }
                }
            }
        }
        fclose(cur);
        // If merge failed partway, clear remaining FC deltas so we don't double-count
        // them on the next write.  FT deltas for un-merged rows stay (non-zero) and
        // will be applied correctly on the next successful sync.
        if (!ok) {
            memset(delta_l0_w, 0, sizeof(delta_l0_w));
            memset(delta_l0_b, 0, sizeof(delta_l0_b));
            memset(delta_l1_w, 0, sizeof(delta_l1_w));
            memset(delta_l1_b, 0, sizeof(delta_l1_b));
            memset(delta_l2_w, 0, sizeof(delta_l2_w));
            memset(delta_l2_b, 0, sizeof(delta_l2_b));
            memset(delta_l0_w_cnt, 0, sizeof(delta_l0_w_cnt));
            memset(delta_l0_b_cnt, 0, sizeof(delta_l0_b_cnt));
            memset(delta_l1_w_cnt, 0, sizeof(delta_l1_w_cnt));
            memset(delta_l1_b_cnt, 0, sizeof(delta_l1_b_cnt));
            memset(delta_l2_w_cnt, 0, sizeof(delta_l2_w_cnt));
            memset(delta_l2_b_cnt, 0, sizeof(delta_l2_b_cnt));
            memset(delta_ft_bias_cnt,   0, sizeof(delta_ft_bias_cnt));
            memset(delta_piece_val_cnt, 0, sizeof(delta_piece_val_cnt));
            if (delta_psqt_cnt) memset(delta_psqt_cnt, 0, (size_t)NNUE_FT_INPUTS * NNUE_PSQT_BKTS * sizeof(uint32_t));
        }
    } else {
        // No existing file — first write.  Clear deltas (incorporated in shadow directly).
        memset(delta_l0_w, 0, sizeof(delta_l0_w));
        memset(delta_l0_b, 0, sizeof(delta_l0_b));
        memset(delta_l1_w, 0, sizeof(delta_l1_w));
        memset(delta_l1_b, 0, sizeof(delta_l1_b));
        memset(delta_l2_w, 0, sizeof(delta_l2_w));
        memset(delta_l2_b, 0, sizeof(delta_l2_b));
        memset(delta_l0_w_cnt, 0, sizeof(delta_l0_w_cnt));
        memset(delta_l0_b_cnt, 0, sizeof(delta_l0_b_cnt));
        memset(delta_l1_w_cnt, 0, sizeof(delta_l1_w_cnt));
        memset(delta_l1_b_cnt, 0, sizeof(delta_l1_b_cnt));
        memset(delta_l2_w_cnt, 0, sizeof(delta_l2_w_cnt));
        memset(delta_l2_b_cnt, 0, sizeof(delta_l2_b_cnt));
        memset(delta_ft_bias_cnt,   0, sizeof(delta_ft_bias_cnt));
        memset(delta_piece_val_cnt, 0, sizeof(delta_piece_val_cnt));
        if (delta_psqt_cnt) memset(delta_psqt_cnt, 0, (size_t)NNUE_FT_INPUTS * NNUE_PSQT_BKTS * sizeof(uint32_t));
        if (ft_delta_f32)   memset(ft_delta_f32,   0, (size_t)NNUE_FT_INPUTS * NNUE_HALF_DIMS  * sizeof(float));
        if (psqt_delta_f32) memset(psqt_delta_f32, 0, (size_t)NNUE_FT_INPUTS * NNUE_PSQT_BKTS * sizeof(float));
        memset(ft_bias_delta,   0, sizeof(ft_bias_delta));
        memset(delta_piece_val, 0, sizeof(delta_piece_val));
    }

    // ---- Restore PSQT slot-mean gauge after merge ----
    // The merge above replaced our in-memory shadow with (file_value + pd),
    // which doesn't preserve the per-slot-aggregate gauge invariant (each
    // worker's pd has zero slot-total per the per-batch centering, but the
    // file_value carries whatever gauge state the previous writers left it
    // in — possibly drifted from accumulated multi-worker merge artifacts).
    // Re-anchor the slot totals to the persisted init targets before writing,
    // so the file we emit has gauge-correct dirty rows.  Without this, drift
    // compounds geometrically across merge cycles and causes catastrophic
    // piece-value blow-ups (observed: piece_val[Bishop] +515 cp in 571 batches).
    nnue_recenter_psqt_slot_means();

    // Sync value[] with the (possibly co-worker-updated) piece_val.
    if (piece_val_active)
        nnue_extract_piece_values(false);

    // ---- Write merged content to a temp file, then atomically rename ----
    char tmp_path[FILENAME_MAX];
    snprintf(tmp_path, sizeof(tmp_path), "%s.tmp", path);
    FILE *f = fopen(tmp_path, "wb");
    if (!f) {
        fprintf(stderr, "TDLeaf: cannot write temp file %s\n", tmp_path);
        tdleaf_release_lock(lock_fd);
        return false;
    }
    setvbuf(f, nullptr, _IOFBF, 4u << 20);
    // Scale a section into the scratch buffer, then write it in one call.
    auto write_scaled = [&](const float *src, int n) {
        for (int i = 0; i < n; i++) io_buf_f[i] = src[i] * TDLEAF_SCALE;
        fwrite(io_buf_f, sizeof(float), n, f);
    };
    fwrite(&TDLEAF_MAGIC,   4, 1, f);
    fwrite(&TDLEAF_VERSION, 4, 1, f);
    // v10+: content fingerprint of the source .nnue, for load-time pairing check.
    fwrite(&nnue_content_hash, 4, 1, f);
    for (int s = 0; s < NNUE_LAYER_STACKS; s++) {
        // FC0 biases
        write_scaled(l0_biases_f32[s], NNUE_L0_SIZE);
        fwrite(l0_biases_cnt[s], sizeof(uint32_t), NNUE_L0_SIZE, f);
        // FC0 weights (natural output-major layout: o * L0_INPUT + i)
        write_scaled(l0_weights_f32[s], NNUE_L0_SIZE * NNUE_L0_INPUT);
        fwrite(l0_weights_cnt[s], sizeof(uint32_t), NNUE_L0_SIZE * NNUE_L0_INPUT, f);
        // FC1 biases
        write_scaled(l1_biases_f32[s], NNUE_L1_SIZE);
        fwrite(l1_biases_cnt[s], sizeof(uint32_t), NNUE_L1_SIZE, f);
        // FC1 weights (natural output-major layout: o * L1_PADDED + i)
        write_scaled(l1_weights_f32[s], NNUE_L1_SIZE * NNUE_L1_PADDED);
        fwrite(l1_weights_cnt[s], sizeof(uint32_t), NNUE_L1_SIZE * NNUE_L1_PADDED, f);
        // FC2 bias
        write_scaled(&l2_bias_f32[s], 1);
        fwrite(&l2_bias_cnt[s], sizeof(uint32_t), 1, f);
        // FC2 weights
        write_scaled(l2_weights_f32[s], NNUE_L2_PADDED);
        fwrite(l2_weights_cnt[s], sizeof(uint32_t), NNUE_L2_PADDED, f);
    }

    // Sparse FT/PSQT section (v3).
    // A feature row is "dirty" if any ft or psqt update count is non-zero.
    // Compute the flag once per row and reuse it in the three later loops
    // that previously rescanned the full count arrays (~92 MB per pass).
    static uint8_t row_dirty[NNUE_FT_INPUTS];
    uint32_t n_ft_rows = 0;
    if (ft_weights_f32) {
        for (int fi = 0; fi < NNUE_FT_INPUTS; fi++) {
            const uint32_t *wc = ft_weights_cnt   + (size_t)fi * NNUE_HALF_DIMS;
            const uint32_t *pc = psqt_weights_cnt + (size_t)fi * NNUE_PSQT_BKTS;
            bool dirty = false;
            for (int d = 0; d < NNUE_HALF_DIMS && !dirty; d++) dirty = (wc[d] != 0);
            for (int b = 0; b < NNUE_PSQT_BKTS && !dirty; b++) dirty = (pc[b] != 0);
            row_dirty[fi] = (uint8_t)dirty;
            if (dirty) n_ft_rows++;
        }
    } else {
        memset(row_dirty, 0, sizeof(row_dirty));
    }
    fwrite(&n_ft_rows, sizeof(uint32_t), 1, f);

    if (ft_weights_f32) {
        float tmp_w[NNUE_HALF_DIMS];
        float tmp_p[NNUE_PSQT_BKTS];
        for (int fi = 0; fi < NNUE_FT_INPUTS; fi++) {
            if (!row_dirty[fi]) continue;
            const uint32_t *wc = ft_weights_cnt   + (size_t)fi * NNUE_HALF_DIMS;
            const uint32_t *pc = psqt_weights_cnt + (size_t)fi * NNUE_PSQT_BKTS;

            uint32_t fi_u = (uint32_t)fi;
            fwrite(&fi_u, sizeof(uint32_t), 1, f);

            const float *fw = ft_weights_f32   + (size_t)fi * NNUE_HALF_DIMS;
            const float *pw = psqt_weights_f32 + (size_t)fi * NNUE_PSQT_BKTS;

            for (int d = 0; d < NNUE_HALF_DIMS;  d++) tmp_w[d] = fw[d] * TDLEAF_SCALE;
            fwrite(tmp_w, sizeof(float),    NNUE_HALF_DIMS,  f);
            fwrite(wc,    sizeof(uint32_t), NNUE_HALF_DIMS,  f);

            for (int b = 0; b < NNUE_PSQT_BKTS; b++) tmp_p[b] = pw[b] * TDLEAF_SCALE;
            fwrite(tmp_p, sizeof(float),    NNUE_PSQT_BKTS,  f);
            fwrite(pc,    sizeof(uint32_t), NNUE_PSQT_BKTS,  f);
        }
    }

    // FT bias section (v4+): float32[HALF_DIMS] × TDLEAF_SCALE + uint32_t[HALF_DIMS] counts.
    {
        float tmp_b[NNUE_HALF_DIMS];
        for (int d = 0; d < NNUE_HALF_DIMS; d++) tmp_b[d] = ft_biases_f32[d] * TDLEAF_SCALE;
        fwrite(tmp_b,       sizeof(float),    NNUE_HALF_DIMS, f);
        fwrite(ft_bias_cnt, sizeof(uint32_t), NNUE_HALF_DIMS, f);
    }

    // Dense piece value section (v9): float32[6] × TDLEAF_SCALE + uint32[6] counts.
    {
        float tmp_pv[6];
        for (int pt = 0; pt < 6; pt++)
            tmp_pv[pt] = piece_val_f32[pt] * TDLEAF_SCALE;
        fwrite(tmp_pv,        sizeof(float),    6, f);
        fwrite(piece_val_cnt, sizeof(uint32_t), 6, f);
    }

    // Adam v section (v6+): t_adam + FC v + FT bias v + piece_val v + sparse PSQT v.
    // Raw float32 (no TDLEAF_SCALE — v values are always non-negative and can be
    // large; scaling would lose precision without benefit).
    fwrite(&t_adam, sizeof(uint32_t), 1, f);
    for (int s = 0; s < NNUE_LAYER_STACKS; s++) {
        fwrite(v_l0_b[s], sizeof(float), NNUE_L0_SIZE, f);
        fwrite(v_l0_w[s], sizeof(float), NNUE_L0_SIZE * NNUE_L0_INPUT, f);
        fwrite(v_l1_b[s], sizeof(float), NNUE_L1_SIZE, f);
        fwrite(v_l1_w[s], sizeof(float), NNUE_L1_SIZE * NNUE_L1_PADDED, f);
        fwrite(&v_l2_b[s], sizeof(float), 1, f);
        fwrite(v_l2_w[s], sizeof(float), NNUE_L2_PADDED, f);
    }
    fwrite(v_ft_bias, sizeof(float), NNUE_HALF_DIMS, f);
    fwrite(v_piece_val, sizeof(float), 6, f);
    // Sparse PSQT v: write v for each dirty row (same dirty-row set as weights).
    {
        uint32_t n_pv_rows = v_psqt_w ? n_ft_rows : 0u;
        fwrite(&n_pv_rows, sizeof(uint32_t), 1, f);
        if (v_psqt_w) {
            for (int fi = 0; fi < NNUE_FT_INPUTS; fi++) {
                if (!row_dirty[fi]) continue;
                uint32_t fi_u = (uint32_t)fi;
                fwrite(&fi_u, sizeof(uint32_t), 1, f);
                fwrite(v_psqt_w + (size_t)fi * NNUE_PSQT_BKTS, sizeof(float), NNUE_PSQT_BKTS, f);
            }
        }
    }

    // Adam m section (v7): FC m + FT bias m + piece_val m + sparse PSQT m.
    // Raw float32 (signed; m can be positive or negative).
    for (int s = 0; s < NNUE_LAYER_STACKS; s++) {
        fwrite(m_l0_b[s], sizeof(float), NNUE_L0_SIZE, f);
        fwrite(m_l0_w[s], sizeof(float), NNUE_L0_SIZE * NNUE_L0_INPUT, f);
        fwrite(m_l1_b[s], sizeof(float), NNUE_L1_SIZE, f);
        fwrite(m_l1_w[s], sizeof(float), NNUE_L1_SIZE * NNUE_L1_PADDED, f);
        fwrite(&m_l2_b[s], sizeof(float), 1, f);
        fwrite(m_l2_w[s], sizeof(float), NNUE_L2_PADDED, f);
    }
    fwrite(m_ft_bias, sizeof(float), NNUE_HALF_DIMS, f);
    fwrite(m_piece_val, sizeof(float), 6, f);
    // Sparse PSQT m: same dirty rows as weights/v.
    {
        uint32_t n_pm_rows = m_psqt_w ? n_ft_rows : 0u;
        fwrite(&n_pm_rows, sizeof(uint32_t), 1, f);
        if (m_psqt_w) {
            for (int fi = 0; fi < NNUE_FT_INPUTS; fi++) {
                if (!row_dirty[fi]) continue;
                uint32_t fi_u = (uint32_t)fi;
                fwrite(&fi_u, sizeof(uint32_t), 1, f);
                fwrite(m_psqt_w + (size_t)fi * NNUE_PSQT_BKTS, sizeof(float), NNUE_PSQT_BKTS, f);
            }
        }
    }

    // Sparse FT v section (v8): v_ft_w rows where v is non-zero.
    // IMPORTANT: only rows with at least one non-zero v dimension are saved.
    // Dirty rows with v=0 (e.g. first session from a v7 file — v was never
    // accumulated) must NOT be saved: on reload they would be marked warmed
    // (ft_v_warmed[fi]=true) but with v=0 and bc2_warm≈1, giving sv≈ε and
    // steps ~10,000× the intended LR, catastrophically corrupting FT weights.
    // Merge strategy: max(v_file, v_local) — same as FC v.
    {
        // Compute the non-zero flag once per row and reuse it in the write
        // loop (previously both loops scanned the full ~92 MB v_ft_w array).
        static uint8_t ftv_nonzero[NNUE_FT_INPUTS];
        uint32_t n_ft_v_rows = 0;
        if (v_ft_w) {
            for (int fi = 0; fi < NNUE_FT_INPUTS; fi++) {
                const float *vw = v_ft_w + (size_t)fi * NNUE_HALF_DIMS;
                bool v_nonzero = false;
                for (int d = 0; d < NNUE_HALF_DIMS && !v_nonzero; d++)
                    v_nonzero = (vw[d] != 0.0f);
                ftv_nonzero[fi] = (uint8_t)v_nonzero;
                if (v_nonzero) n_ft_v_rows++;
            }
        }
        fwrite(&n_ft_v_rows, sizeof(uint32_t), 1, f);
        if (v_ft_w && n_ft_v_rows > 0) {
            for (int fi = 0; fi < NNUE_FT_INPUTS; fi++) {
                if (!ftv_nonzero[fi]) continue;
                uint32_t fi_u = (uint32_t)fi;
                fwrite(&fi_u, sizeof(uint32_t), 1, f);
                fwrite(v_ft_w + (size_t)fi * NNUE_HALF_DIMS, sizeof(float), NNUE_HALF_DIMS, f);
            }
        }
    }

    // v11: PSQT init slot-means (11 slots × NNUE_PSQT_BKTS).  Used by
    // nnue_recenter_psqt_slot_means() at load time to pin slot-means against
    // merge drift.  If the in-memory targets are not valid (extremely unusual:
    // a save called before any init or load populated them), fall back to the
    // current PSQT slot-means so the file is self-consistent.
    {
        float means[11][NNUE_PSQT_BKTS];
        if (psqt_init_slot_means_valid) {
            memcpy(means, psqt_init_slot_means, sizeof(means));
        } else {
            nnue_compute_psqt_slot_means(means);
        }
        fwrite(means, sizeof(float), 11 * NNUE_PSQT_BKTS, f);
    }

    fclose(f);

    // Atomic rename: temp → final (replaces the old file in one syscall).
    if (rename(tmp_path, path) != 0) {
        fprintf(stderr, "TDLeaf: rename %s → %s failed\n", tmp_path, path);
        tdleaf_release_lock(lock_fd);
        return false;
    }

    tdleaf_release_lock(lock_fd);
    return true;
}

bool nnue_load_fc_weights(const char *path)
{
    // Acquire shared lock — allows concurrent reads but blocks writes.
    int lock_fd = tdleaf_acquire_lock(path, LOCK_SH);
    // Non-fatal if locking fails (e.g. first run before lock file exists); proceed.

    FILE *f = fopen(path, "rb");
    if (!f) { tdleaf_release_lock(lock_fd); return false; }
    setvbuf(f, nullptr, _IOFBF, 4u << 20);
    uint32_t magic = 0, version = 0;
    if (fread(&magic, 4, 1, f) != 1 || fread(&version, 4, 1, f) != 1) {
        fclose(f); tdleaf_release_lock(lock_fd); return false;
    }
    if (magic != TDLEAF_MAGIC) {
        fprintf(stderr, "TDLeaf: bad magic in %s\n", path);
        fclose(f); tdleaf_release_lock(lock_fd); return false;
    }

    // ---- Version 1: legacy int8/int32 format, no counts ----
    if (version == 1) {
        for (int s = 0; s < NNUE_LAYER_STACKS; s++) {
            if (fread(l0_biases[s],  sizeof(int32_t), NNUE_L0_SIZE, f) != (size_t)NNUE_L0_SIZE ||
                fread(l0_weights[s], sizeof(int8_t),  NNUE_L0_SIZE * NNUE_L0_INPUT, f)
                    != (size_t)(NNUE_L0_SIZE * NNUE_L0_INPUT) ||
                fread(l1_biases[s],  sizeof(int32_t), NNUE_L1_SIZE, f) != (size_t)NNUE_L1_SIZE ||
                fread(l1_weights[s], sizeof(int8_t),  NNUE_L1_SIZE * NNUE_L1_PADDED, f)
                    != (size_t)(NNUE_L1_SIZE * NNUE_L1_PADDED) ||
                fread(&out_biases[s],  sizeof(int32_t), 1, f) != 1 ||
                fread(out_weights[s],  sizeof(int8_t),  NNUE_L2_PADDED, f) != (size_t)NNUE_L2_PADDED) {
                fprintf(stderr, "TDLeaf: read error in %s (stack %d)\n", path, s);
                fclose(f); return false;
            }
        }
        fclose(f);
        tdleaf_release_lock(lock_fd);
        // Sync FP32 copies and zero counts (v1 has no count data).
        nnue_init_fp32_weights();
        printf("TDLeaf: loaded v1 FC weights from %s (will upgrade to v2 on next save)\n", path);
        return true;
    }

    // ---- Version 2 / 3 / 4 / 5 / 6 / 7 / 8 / 9 / 10: float32 × TDLEAF_SCALE + uint32 counts ----
    if (version != 2u && version != 3u && version != 4u && version != 5u && version != 6u && version != 7u && version != 8u && version != 9u && version != 10u && version != TDLEAF_VERSION) {
        fprintf(stderr, "TDLeaf: unsupported version %u in %s\n", version, path);
        fclose(f); tdleaf_release_lock(lock_fd); return false;
    }

    // ---- v10+: verify source-.nnue content fingerprint matches loaded .nnue ----
    // Prevents silently pairing weight deltas with the wrong baseline network.
    // v5–v9 files predate this header and are accepted without a check; they
    // get promoted to v10 with the current .nnue's hash on the next save.
    if (version >= 10u) {
        uint32_t file_content_hash = 0;
        if (fread(&file_content_hash, 4, 1, f) != 1) {
            fprintf(stderr, "TDLeaf: short read on v%u content hash in %s\n", version, path);
            fclose(f); tdleaf_release_lock(lock_fd); return false;
        }
        if (file_content_hash != nnue_content_hash) {
            fprintf(stderr,
                    "TDLeaf: %s was trained against a different .nnue "
                    "(file hash=0x%08X, loaded .nnue hash=0x%08X).\n"
                    "        Refusing to load — move or rename the .tdleaf.bin, "
                    "or reload the matching .nnue.\n",
                    path, file_content_hash, nnue_content_hash);
            fclose(f); tdleaf_release_lock(lock_fd); return false;
        }
    } else {
        fprintf(stderr,
                "TDLeaf: %s is legacy v%u (no .nnue content hash); "
                "accepting — will upgrade to v%u on next save.\n",
                path, version, TDLEAF_VERSION);
    }

    bool ok = true;
    for (int s = 0; s < NNUE_LAYER_STACKS && ok; s++) {
        auto rf = [&](float *dst, int n) -> bool {
            float tmp;
            for (int i = 0; i < n; i++) {
                if (fread(&tmp, sizeof(float), 1, f) != 1) return false;
                dst[i] = tmp / TDLEAF_SCALE;
            }
            return true;
        };
        auto ru = [&](uint32_t *dst, int n) -> bool {
            return fread(dst, sizeof(uint32_t), n, f) == (size_t)n;
        };
        ok = rf(l0_biases_f32[s],  NNUE_L0_SIZE)                  &&
             ru(l0_biases_cnt[s],  NNUE_L0_SIZE)                  &&
             rf(l0_weights_f32[s], NNUE_L0_SIZE * NNUE_L0_INPUT)  &&
             ru(l0_weights_cnt[s], NNUE_L0_SIZE * NNUE_L0_INPUT)  &&
             rf(l1_biases_f32[s],  NNUE_L1_SIZE)                  &&
             ru(l1_biases_cnt[s],  NNUE_L1_SIZE)                  &&
             rf(l1_weights_f32[s], NNUE_L1_SIZE * NNUE_L1_PADDED) &&
             ru(l1_weights_cnt[s], NNUE_L1_SIZE * NNUE_L1_PADDED) &&
             rf(&l2_bias_f32[s],   1)                             &&
             ru(&l2_bias_cnt[s],   1)                             &&
             rf(l2_weights_f32[s], NNUE_L2_PADDED)                &&
             ru(l2_weights_cnt[s], NNUE_L2_PADDED);
    }

    // Sparse FT/PSQT section (v3+).
    int n_ft_loaded = 0;
    if (ok && (version >= 3u) && ft_weights_f32) {
        uint32_t n_ft_rows = 0;
        if (fread(&n_ft_rows, sizeof(uint32_t), 1, f) != 1) { ok = false; }
        float tmp_w[NNUE_HALF_DIMS];
        float tmp_p[NNUE_PSQT_BKTS];
        for (uint32_t k = 0; k < n_ft_rows && ok; k++) {
            uint32_t fi;
            if (fread(&fi, sizeof(uint32_t), 1, f) != 1 || fi >= (uint32_t)NNUE_FT_INPUTS)
                { ok = false; break; }

            float    *fw = ft_weights_f32   + (size_t)fi * NNUE_HALF_DIMS;
            uint32_t *wc = ft_weights_cnt   + (size_t)fi * NNUE_HALF_DIMS;
            float    *pw = psqt_weights_f32 + (size_t)fi * NNUE_PSQT_BKTS;
            uint32_t *pc = psqt_weights_cnt + (size_t)fi * NNUE_PSQT_BKTS;

            if (fread(tmp_w, sizeof(float), NNUE_HALF_DIMS, f) != (size_t)NNUE_HALF_DIMS)
                { ok = false; break; }
            for (int d = 0; d < NNUE_HALF_DIMS; d++) fw[d] = tmp_w[d] / TDLEAF_SCALE;
            if (fread(wc, sizeof(uint32_t), NNUE_HALF_DIMS, f) != (size_t)NNUE_HALF_DIMS)
                { ok = false; break; }

            if (fread(tmp_p, sizeof(float), NNUE_PSQT_BKTS, f) != (size_t)NNUE_PSQT_BKTS)
                { ok = false; break; }
            for (int b = 0; b < NNUE_PSQT_BKTS; b++) pw[b] = tmp_p[b] / TDLEAF_SCALE;
            if (fread(pc, sizeof(uint32_t), NNUE_PSQT_BKTS, f) != (size_t)NNUE_PSQT_BKTS)
                { ok = false; break; }

            n_ft_loaded++;
        }
    }

    // FT bias section (v4+).
    if (ok && version >= 4u) {
        float tmp_b[NNUE_HALF_DIMS];
        uint32_t tmp_bc[NNUE_HALF_DIMS];
        if (fread(tmp_b,  sizeof(float),    NNUE_HALF_DIMS, f) == (size_t)NNUE_HALF_DIMS &&
            fread(tmp_bc, sizeof(uint32_t), NNUE_HALF_DIMS, f) == (size_t)NNUE_HALF_DIMS) {
            for (int d = 0; d < NNUE_HALF_DIMS; d++) {
                ft_biases_f32[d] = tmp_b[d] / TDLEAF_SCALE;
                ft_biases[d]     = (int16_t)std::max(-32767.0f,
                                            std::min( 32767.0f, roundf(ft_biases_f32[d])));
                ft_bias_cnt[d]   = tmp_bc[d];
            }
        } else {
            ok = false;
        }
    }

    // Dense piece value section (v5+).
    // v9+: float32[6] + uint32[6].  v5-v8: float32[6][8] + uint32[6][8] → collapse by avg.
    if (ok && version >= 5u) {
        if (version >= 9u) {
            float tmp_pv[6]; uint32_t tmp_pvc[6];
            if (fread(tmp_pv,  sizeof(float),    6, f) == 6 &&
                fread(tmp_pvc, sizeof(uint32_t), 6, f) == 6) {
                for (int pt = 0; pt < 6; pt++) {
                    // Clamp ≥ 0: defensive against a corrupted file written
                    // before the merge-side clamp existed.
                    piece_val_f32[pt] = std::max(0.0f, tmp_pv[pt] / TDLEAF_SCALE);
                    piece_val_cnt[pt] = tmp_pvc[pt];
                }
                piece_val_active = true;
            } else { ok = false; }
        } else {
            // v5-v8: collapse [6][8] to [6] by averaging.
            float tmp_pv[6][NNUE_PSQT_BKTS]; uint32_t tmp_pvc[6][NNUE_PSQT_BKTS];
            if (fread(tmp_pv,  sizeof(float),    6*NNUE_PSQT_BKTS, f) == (size_t)(6*NNUE_PSQT_BKTS) &&
                fread(tmp_pvc, sizeof(uint32_t), 6*NNUE_PSQT_BKTS, f) == (size_t)(6*NNUE_PSQT_BKTS)) {
                for (int pt = 0; pt < 6; pt++) {
                    float sum = 0.0f; uint32_t cnt_sum = 0;
                    for (int b = 0; b < NNUE_PSQT_BKTS; b++) { sum += tmp_pv[pt][b]; cnt_sum += tmp_pvc[pt][b]; }
                    piece_val_f32[pt] = std::max(0.0f, sum / NNUE_PSQT_BKTS / TDLEAF_SCALE);
                    piece_val_cnt[pt] = cnt_sum / NNUE_PSQT_BKTS;
                }
                piece_val_active = true;
            } else { ok = false; }
        }
    }

    // Adam v section (v6+): restore gradient scale from file.
    bool adam_v_loaded = false;
    if (ok && version >= 6u) {
        uint32_t file_t;
        if (fread(&file_t, sizeof(uint32_t), 1, f) == 1) {
            t_adam = file_t;
            auto rf = [&](float *dst, int n) -> bool {
                return fread(dst, sizeof(float), n, f) == (size_t)n;
            };
            bool vok = true;
            for (int s = 0; s < NNUE_LAYER_STACKS && vok; s++) {
                vok = rf(v_l0_b[s], NNUE_L0_SIZE)
                   && rf(v_l0_w[s], NNUE_L0_SIZE * NNUE_L0_INPUT)
                   && rf(v_l1_b[s], NNUE_L1_SIZE)
                   && rf(v_l1_w[s], NNUE_L1_SIZE * NNUE_L1_PADDED)
                   && rf(&v_l2_b[s], 1)
                   && rf(v_l2_w[s], NNUE_L2_PADDED);
            }
            if (vok) vok = rf(v_ft_bias, NNUE_HALF_DIMS);
            if (vok) {
                if (version >= 9u) {
                    vok = rf(v_piece_val, 6);
                } else {
                    // v6-v8: 48 floats → take max per piece type.
                    float tmp_v8[6][NNUE_PSQT_BKTS];
                    vok = rf(&tmp_v8[0][0], 6 * NNUE_PSQT_BKTS);
                    if (vok) for (int pt = 0; pt < 6; pt++) {
                        float mx = 0.0f;
                        for (int b = 0; b < NNUE_PSQT_BKTS; b++) if (tmp_v8[pt][b] > mx) mx = tmp_v8[pt][b];
                        v_piece_val[pt] = mx;
                    }
                }
            }
            // Sparse PSQT v.
            uint32_t n_pv_rows = 0;
            if (vok && fread(&n_pv_rows, sizeof(uint32_t), 1, f) == 1 && v_psqt_w) {
                for (uint32_t k = 0; k < n_pv_rows; k++) {
                    uint32_t fi;
                    if (fread(&fi, sizeof(uint32_t), 1, f) != 1 ||
                        fi >= (uint32_t)NNUE_FT_INPUTS) break;
                    float *vp = v_psqt_w + (size_t)fi * NNUE_PSQT_BKTS;
                    if (fread(vp, sizeof(float), NNUE_PSQT_BKTS, f) != (size_t)NNUE_PSQT_BKTS)
                        break;
                }
            }
            if (vok) adam_v_loaded = true;
            // Non-fatal if v section is truncated — we just start with v=0 for
            // the missing entries (same as loading a v5/v6 file).
        }
    }

    // Adam m section (v7+): restore momentum from file.
    bool adam_m_loaded = false;
    if (ok && version >= 7u) {
        auto rf = [&](float *dst, int n) -> bool {
            return fread(dst, sizeof(float), n, f) == (size_t)n;
        };
        bool mok = true;
        for (int s = 0; s < NNUE_LAYER_STACKS && mok; s++) {
            mok = rf(m_l0_b[s], NNUE_L0_SIZE)
               && rf(m_l0_w[s], NNUE_L0_SIZE * NNUE_L0_INPUT)
               && rf(m_l1_b[s], NNUE_L1_SIZE)
               && rf(m_l1_w[s], NNUE_L1_SIZE * NNUE_L1_PADDED)
               && rf(&m_l2_b[s], 1)
               && rf(m_l2_w[s], NNUE_L2_PADDED);
        }
        if (mok) mok = rf(m_ft_bias, NNUE_HALF_DIMS);
        if (mok) {
            if (version >= 9u) {
                mok = rf(m_piece_val, 6);
            } else {
                // v7-v8: 48 floats → average per piece type.
                float tmp_m8[6][NNUE_PSQT_BKTS];
                mok = rf(&tmp_m8[0][0], 6 * NNUE_PSQT_BKTS);
                if (mok) for (int pt = 0; pt < 6; pt++) {
                    float avg = 0.0f;
                    for (int b = 0; b < NNUE_PSQT_BKTS; b++) avg += tmp_m8[pt][b];
                    m_piece_val[pt] = avg / NNUE_PSQT_BKTS;
                }
            }
        }
        // Sparse PSQT m.
        uint32_t n_pm_rows = 0;
        if (mok && fread(&n_pm_rows, sizeof(uint32_t), 1, f) == 1 && m_psqt_w) {
            for (uint32_t k = 0; k < n_pm_rows; k++) {
                uint32_t fi;
                if (fread(&fi, sizeof(uint32_t), 1, f) != 1 ||
                    fi >= (uint32_t)NNUE_FT_INPUTS) break;
                float *mp = m_psqt_w + (size_t)fi * NNUE_PSQT_BKTS;
                if (fread(mp, sizeof(float), NNUE_PSQT_BKTS, f) != (size_t)NNUE_PSQT_BKTS)
                    break;
            }
        }
        if (mok) adam_m_loaded = true;
        // Non-fatal if m section is truncated — missing entries stay at 0.
    }

    // Sparse FT v section (v8+): restore v_ft_w from disk and mark rows as warmed.
    int n_ft_v_loaded = 0;
    if (ok && version >= 8u && v_ft_w) {
        uint32_t n_ftv_rows = 0;
        if (fread(&n_ftv_rows, sizeof(uint32_t), 1, f) == 1) {
            for (uint32_t k = 0; k < n_ftv_rows; k++) {
                uint32_t fi;
                if (fread(&fi, sizeof(uint32_t), 1, f) != 1 ||
                    fi >= (uint32_t)NNUE_FT_INPUTS) break;
                float *vw = v_ft_w + (size_t)fi * NNUE_HALF_DIMS;
                if (fread(vw, sizeof(float), NNUE_HALF_DIMS, f)
                    != (size_t)NNUE_HALF_DIMS) break;
                if (ft_v_warmed) ft_v_warmed[fi] = true;
                n_ft_v_loaded++;
            }
        }
        // Non-fatal if FT v section is truncated; unloaded rows use t_ft_session bc2.
    }

    // v11: PSQT init slot-means used by nnue_recenter_psqt_slot_means().
    // Pre-v11 files have no targets stored — fall back to a snapshot of the
    // currently loaded PSQT slot-means (locks in whatever drift the older file
    // carried; subsequent training preserves that anchor and prevents further
    // drift).  Re-init via --init-nnue is the clean way to reset the targets.
    bool slot_means_loaded = false;
    if (ok && version >= 11u) {
        float means[11][NNUE_PSQT_BKTS];
        if (fread(means, sizeof(float), 11 * NNUE_PSQT_BKTS, f)
            == (size_t)(11 * NNUE_PSQT_BKTS)) {
            memcpy(psqt_init_slot_means, means, sizeof(means));
            psqt_init_slot_means_valid = true;
            slot_means_loaded = true;
        }
    }

    fclose(f);
    tdleaf_release_lock(lock_fd);
    if (!ok) {
        fprintf(stderr, "TDLeaf: read error in %s\n", path);
        return false;
    }
    memset(grad_l0_w, 0, sizeof(grad_l0_w));
    memset(grad_l0_b, 0, sizeof(grad_l0_b));
    memset(grad_l1_w, 0, sizeof(grad_l1_w));
    memset(grad_l1_b, 0, sizeof(grad_l1_b));
    memset(grad_l2_w, 0, sizeof(grad_l2_w));
    memset(grad_l2_b, 0, sizeof(grad_l2_b));
    // Delta accumulators start at zero after a fresh load.
    memset(delta_l0_w, 0, sizeof(delta_l0_w));
    memset(delta_l0_b, 0, sizeof(delta_l0_b));
    memset(delta_l1_w, 0, sizeof(delta_l1_w));
    memset(delta_l1_b, 0, sizeof(delta_l1_b));
    memset(delta_l2_w, 0, sizeof(delta_l2_w));
    memset(delta_l2_b, 0, sizeof(delta_l2_b));
    memset(delta_l0_w_cnt, 0, sizeof(delta_l0_w_cnt));
    memset(delta_l0_b_cnt, 0, sizeof(delta_l0_b_cnt));
    memset(delta_l1_w_cnt, 0, sizeof(delta_l1_w_cnt));
    memset(delta_l1_b_cnt, 0, sizeof(delta_l1_b_cnt));
    memset(delta_l2_w_cnt, 0, sizeof(delta_l2_w_cnt));
    memset(delta_l2_b_cnt, 0, sizeof(delta_l2_b_cnt));
    memset(delta_ft_bias_cnt,   0, sizeof(delta_ft_bias_cnt));
    memset(delta_piece_val_cnt, 0, sizeof(delta_piece_val_cnt));
    if (delta_psqt_cnt) memset(delta_psqt_cnt, 0, (size_t)NNUE_FT_INPUTS * NNUE_PSQT_BKTS * sizeof(uint32_t));
    if (ft_delta_f32)   memset(ft_delta_f32,   0, (size_t)NNUE_FT_INPUTS * NNUE_HALF_DIMS  * sizeof(float));
    if (psqt_delta_f32) memset(psqt_delta_f32, 0, (size_t)NNUE_FT_INPUTS * NNUE_PSQT_BKTS * sizeof(float));
    memset(grad_ft_bias,    0, sizeof(grad_ft_bias));
    memset(ft_bias_delta,   0, sizeof(ft_bias_delta));
    memset(grad_piece_val,  0, sizeof(grad_piece_val));
    memset(delta_piece_val, 0, sizeof(delta_piece_val));
    // ft_bias_cnt and ft_biases_f32 are populated from file in v4+; leave them.
    // For v2/v3 files, ft_biases_f32 was already initialised by nnue_init_fp32_weights.
    nnue_requantize_fc();

    // Pin PSQT slot-means to the persisted target after merging-in this file's
    // dirty rows.  For pre-v11 files (no targets persisted), snapshot the
    // current slot-means here as the going-forward anchor; this preserves the
    // loaded state but prevents future drift.
    if (!slot_means_loaded)
        nnue_capture_psqt_init_slot_means();
    nnue_recenter_psqt_slot_means();

    if (version == TDLEAF_VERSION)
        printf("TDLeaf: loaded v%u weights from %s (%d FT rows, %d FT-v rows, piece_val=%s, adam_v=%s, adam_m=%s, t_adam=%u, content_hash=0x%08X)\n",
               TDLEAF_VERSION, path, n_ft_loaded, n_ft_v_loaded, piece_val_active ? "yes" : "no",
               adam_v_loaded ? "yes" : "no", adam_m_loaded ? "yes" : "no", t_adam, nnue_content_hash);
    else if (version == 10u)
        printf("TDLeaf: loaded v10 weights from %s (%d FT rows, %d FT-v rows, piece_val=%s, adam_v=%s, adam_m=%s, t_adam=%u, will upgrade to v%u on next save — slot-mean targets snapshotted from loaded state)\n",
               path, n_ft_loaded, n_ft_v_loaded, piece_val_active ? "yes" : "no",
               adam_v_loaded ? "yes" : "no", adam_m_loaded ? "yes" : "no", t_adam, TDLEAF_VERSION);
    else if (version == 9u)
        printf("TDLeaf: loaded v9 weights from %s (%d FT rows, %d FT-v rows, piece_val=%s, adam_v=%s, adam_m=%s, t_adam=%u, will upgrade to v%u on next save)\n",
               path, n_ft_loaded, n_ft_v_loaded, piece_val_active ? "yes" : "no",
               adam_v_loaded ? "yes" : "no", adam_m_loaded ? "yes" : "no", t_adam, TDLEAF_VERSION);
    else if (version == 8u)
        printf("TDLeaf: loaded v8 weights from %s (%d FT rows, piece_val=%s, adam_v=%s, adam_m=%s, t_adam=%u, will upgrade to v%u on next save)\n",
               path, n_ft_loaded, piece_val_active ? "yes" : "no",
               adam_v_loaded ? "yes" : "no", adam_m_loaded ? "yes" : "no", t_adam, TDLEAF_VERSION);
    else if (version == 7u)
        printf("TDLeaf: loaded v7 weights from %s (%d FT rows, piece_val=%s, adam_v=%s, adam_m=%s, t_adam=%u, will upgrade to v%u on next save)\n",
               path, n_ft_loaded, piece_val_active ? "yes" : "no",
               adam_v_loaded ? "yes" : "no", adam_m_loaded ? "yes" : "no", t_adam, TDLEAF_VERSION);
    else if (version == 6u)
        printf("TDLeaf: loaded v6 weights from %s (%d FT rows, piece_val=%s, adam_v=%s, will upgrade to v%u on next save)\n",
               path, n_ft_loaded, piece_val_active ? "yes" : "no",
               adam_v_loaded ? "yes" : "no", TDLEAF_VERSION);
    else if (version == 5u)
        printf("TDLeaf: loaded v5 weights from %s (%d FT rows, piece_val=%s, will upgrade to v%u on next save)\n",
               path, n_ft_loaded, piece_val_active ? "yes" : "no", TDLEAF_VERSION);
    else if (version == 4u)
        printf("TDLeaf: loaded v4 weights from %s (%d FT rows, will upgrade to v%u on next save)\n",
               path, n_ft_loaded, TDLEAF_VERSION);
    else if (version == 3u)
        printf("TDLeaf: loaded v3 weights from %s (%d FT rows, will upgrade to v%u on next save)\n",
               path, n_ft_loaded, TDLEAF_VERSION);
    else
        printf("TDLeaf: loaded v2 FC weights from %s (will upgrade to v%u on next save)\n", path, TDLEAF_VERSION);
    return true;
}

// ---------------------------------------------------------------------------
// nnue_dense_piece_val — dense piece value contribution (centipawns, stm POV)
//
// Computes Σ_pt piece_val[pt] × (stm_count[pt] − opp_count[pt]),
// converted to centipawns via the same (pv_diff/2) × 100/5776 formula as PSQT.
// Returns 0 if piece_val is uninitialised (pre-existing .nnue without .tdleaf.bin).
// ---------------------------------------------------------------------------
int nnue_dense_piece_val(const position &pos, int stm, int piece_count)
{
    (void)piece_count;
    if (!piece_val_active) return 0;
    int32_t pv_diff = 0;
    for (int pt = PAWN; pt <= QUEEN; pt++) {
        int diff = pos.plist[stm][pt][0] - pos.plist[stm ^ 1][pt][0];
        if (diff != 0)
            pv_diff += (int32_t)roundf(piece_val_f32[pt - 1]) * diff;
    }
    return (int)((int64_t)(pv_diff / 2) * 100 / 5776);
}

// ---------------------------------------------------------------------------
// nnue_evaluate_acc_raw — evaluate from stored raw arrays (replay score refresh)
// ---------------------------------------------------------------------------
int nnue_evaluate_acc_raw(const int16_t acc[2][NNUE_HALF_DIMS],
                           const int32_t psqt[2][NNUE_PSQT_BKTS],
                           int stm, int piece_count)
{
    NNUEAccumulator tmp;
    memcpy(tmp.acc[0],  acc[0],  NNUE_HALF_DIMS  * sizeof(int16_t));
    memcpy(tmp.acc[1],  acc[1],  NNUE_HALF_DIMS  * sizeof(int16_t));
    memcpy(tmp.psqt[0], psqt[0], NNUE_PSQT_BKTS * sizeof(int32_t));
    memcpy(tmp.psqt[1], psqt[1], NNUE_PSQT_BKTS * sizeof(int32_t));
    tmp.computed = true;
    return nnue_evaluate(tmp, stm, piece_count);
}

#endif // TDLEAF
