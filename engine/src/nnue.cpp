// Leaf NNUE evaluation — HalfKAv2_hm format (Stockfish 15/16 era)
// Written from scratch; Stockfish source consulted only for file-format layout.
//
// File format confirmed from nn-ad9b42354671.nnue (Stockfish 15.1 exact release):
//   [Header] version(4) + hash(4) + desc_size(4) + desc(N)
//   [FT]     ft_hash(4) + LEB128(biases 1024 i16) + LEB128(weights 22528×1024 i16)
//             + LEB128(psqt 22528×8 i32)
//   [Stacks] 8 × [stack_hash(4) + FC0_bias(16×i32) + FC0_wt(16×1024×i8)
//                               + FC1_bias(32×i32) + FC1_wt(32×32×i8)
//                               + FC2_bias(1×i32)  + FC2_wt(32×i8)]
//   Each stack = 4+64+16384+128+1024+4+32 = 17640 bytes
//   (no separate net_hash; the FT section ends immediately before stack 0's hash)

#include <stdint.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <random>
#include "define.h"
#include "chess.h"
#include "nnue.h"

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

#if defined(__AVX2__)
#include <immintrin.h>
#define EPOCH_USE_AVX2 1
#endif

bool nnue_available = false;

// ---------------------------------------------------------------------------
// Weight storage
// ---------------------------------------------------------------------------
static int16_t *ft_biases    = nullptr; // [NNUE_HALF_DIMS]
static int16_t *ft_weights   = nullptr; // [NNUE_FT_INPUTS × NNUE_HALF_DIMS]
static int32_t *psqt_weights = nullptr; // [NNUE_FT_INPUTS × NNUE_PSQT_BKTS]

// FC0: output-major weight layout, 16 outputs × 3072 inputs (vdotq-reordered at load)
static int32_t l0_biases [NNUE_LAYER_STACKS][NNUE_L0_SIZE];
static int8_t  l0_weights[NNUE_LAYER_STACKS][NNUE_L0_SIZE * NNUE_L0_INPUT];

// FC1: output-major, 32 outputs × 32 padded-inputs
static int32_t l1_biases [NNUE_LAYER_STACKS][NNUE_L1_SIZE];
static int8_t  l1_weights[NNUE_LAYER_STACKS][NNUE_L1_SIZE * NNUE_L1_PADDED];

// FC2 (output): 1 output × 32 padded-inputs
static int32_t out_biases [NNUE_LAYER_STACKS];
static int8_t  out_weights[NNUE_LAYER_STACKS][NNUE_L2_PADDED];

// Path of the currently loaded .nnue file and per-stack hashes (for nnue_write_nnue).
static char    nnue_loaded_path[FILENAME_MAX] = "";
static uint32_t nnue_stack_hashes[NNUE_LAYER_STACKS] = {};

// Content fingerprint of the loaded .nnue, computed at load/init time over the
// FT weight bytes (FNV-1a 64 → 32).  Stored in .tdleaf.bin v10+ header so that a
// .tdleaf.bin can be rejected if it was trained against a different source .nnue.
// Zero before any net is loaded.
static uint32_t nnue_content_hash = 0;

// Compute FNV-1a 64-bit hash over a byte buffer and fold to 32 bits.
// Used to fingerprint the .nnue FT weights for .tdleaf.bin compatibility.
static uint32_t nnue_fnv1a_u32(const void *data, size_t len)
{
    const uint64_t FNV_OFFSET = 0xcbf29ce484222325ull;
    const uint64_t FNV_PRIME  = 0x100000001b3ull;
    uint64_t h = FNV_OFFSET;
    // Process 8 bytes at a time for speed; tail byte-by-byte.
    const uint8_t *p = (const uint8_t*)data;
    size_t n8 = len / 8;
    for (size_t i = 0; i < n8; i++) {
        uint64_t w;
        memcpy(&w, p + i * 8, 8);
        h ^= w;
        h *= FNV_PRIME;
    }
    for (size_t i = n8 * 8; i < len; i++) {
        h ^= p[i];
        h *= FNV_PRIME;
    }
    return (uint32_t)((h >> 32) ^ (uint32_t)h);
}

// Recompute nnue_content_hash from the currently-allocated ft_weights buffer.
// Call once at load/init time (after FT weights are populated); do NOT call
// again after training modifies in-memory FT, since the stored hash must
// continue to identify the original source .nnue.
static void nnue_update_content_hash()
{
    if (!ft_weights) { nnue_content_hash = 0; return; }
    size_t ft_bytes = (size_t)NNUE_FT_INPUTS * NNUE_HALF_DIMS * sizeof(int16_t);
    nnue_content_hash = nnue_fnv1a_u32(ft_weights, ft_bytes);
}

// Load-time diagnostics (see nnue.h) — populated silently, printed on demand
// by the `netinfo` command via nnue_print_diag_info().
NNUEDiagInfo nnue_diag;

const char* nnue_get_loaded_path() { return nnue_loaded_path; }
uint32_t    nnue_get_content_hash() { return nnue_content_hash; }

// Forward declaration for the PSQT FP32 shadow weights (the sole trainable
// material channel under pure-PSQT).  Defined as static ~1300 lines below.
static float *psqt_weights_f32 = nullptr;

// ---------------------------------------------------------------------------
// HalfKAv2_hm lookup tables
// ---------------------------------------------------------------------------
static const int PS_NB = 704;  // 11 piece-type slots × 64 squares

static const int KingBuckets[64] = {
    28*PS_NB, 29*PS_NB, 30*PS_NB, 31*PS_NB, 31*PS_NB, 30*PS_NB, 29*PS_NB, 28*PS_NB,
    24*PS_NB, 25*PS_NB, 26*PS_NB, 27*PS_NB, 27*PS_NB, 26*PS_NB, 25*PS_NB, 24*PS_NB,
    20*PS_NB, 21*PS_NB, 22*PS_NB, 23*PS_NB, 23*PS_NB, 22*PS_NB, 21*PS_NB, 20*PS_NB,
    16*PS_NB, 17*PS_NB, 18*PS_NB, 19*PS_NB, 19*PS_NB, 18*PS_NB, 17*PS_NB, 16*PS_NB,
    12*PS_NB, 13*PS_NB, 14*PS_NB, 15*PS_NB, 15*PS_NB, 14*PS_NB, 13*PS_NB, 12*PS_NB,
     8*PS_NB,  9*PS_NB, 10*PS_NB, 11*PS_NB, 11*PS_NB, 10*PS_NB,  9*PS_NB,  8*PS_NB,
     4*PS_NB,  5*PS_NB,  6*PS_NB,  7*PS_NB,  7*PS_NB,  6*PS_NB,  5*PS_NB,  4*PS_NB,
     0*PS_NB,  1*PS_NB,  2*PS_NB,  3*PS_NB,  3*PS_NB,  2*PS_NB,  1*PS_NB,  0*PS_NB,
};

// ---------------------------------------------------------------------------
// HalfKAv2_hm feature index
// ---------------------------------------------------------------------------
static inline int halfkav2_feature(int persp, int ksq,
                                   int psq,  int ptype, int pside)
{
    // Note: own king IS included as a feature (same PS_KING slot as enemy king).
    // Both kings map to PS_KING=640; the own king's feature index is valid and
    // contributes to the accumulator.  Only excluded: no-piece (ptype == 0).

    int flip   = (persp == BLACK) ? 56 : 0;
    int ksq_f  = ksq ^ flip;
    // Horizontal mirror: normalize so the own king is always on the right half.
    // Flip file when king is on files 0-3 (a-d = queen side) for BOTH perspectives.
    // This matches Stockfish HalfKAv2_hm: orient = s ^ (king_file<4 ? 7 : 0) ^ rank_flip.
    // File is unchanged by the rank-flip, so (ksq_f & 7) == (ksq & 7).
    int orient = ((ksq_f & 7) < 4) ? 7 : 0;
    int psq_o  = (psq ^ flip) ^ orient;
    int bucket = KingBuckets[ksq_f];

    int ps;
    if (ptype == KING) {
        ps = 640;
    } else {
        bool is_own = (pside == persp);
        ps = (ptype - 1) * 128 + (is_own ? 0 : 64);
    }

    return bucket + ps + psq_o;
}

// ---------------------------------------------------------------------------
// Accumulator add/sub helpers
// ---------------------------------------------------------------------------
static inline void add_feat(int16_t *half_acc, int32_t *half_psqt, int fidx)
{
    if (fidx < 0) return;
    const int16_t *w  = ft_weights   + (size_t)fidx * NNUE_HALF_DIMS;
    const int32_t *pw = psqt_weights + (size_t)fidx * NNUE_PSQT_BKTS;
    for (int j = 0; j < NNUE_HALF_DIMS; j++) half_acc[j]  += w[j];
    for (int b = 0; b < NNUE_PSQT_BKTS; b++) half_psqt[b] += pw[b];
}
static inline void sub_feat(int16_t *half_acc, int32_t *half_psqt, int fidx)
{
    if (fidx < 0) return;
    const int16_t *w  = ft_weights   + (size_t)fidx * NNUE_HALF_DIMS;
    const int32_t *pw = psqt_weights + (size_t)fidx * NNUE_PSQT_BKTS;
    for (int j = 0; j < NNUE_HALF_DIMS; j++) half_acc[j]  -= w[j];
    for (int b = 0; b < NNUE_PSQT_BKTS; b++) half_psqt[b] -= pw[b];
}

// Set to true by nnue_init_zero_weights(); controls description in nnue_write_nnue().
static bool nnue_zero_initialized = false;
// Records the prior mode passed to nnue_init_zero_weights() (one of NNUE_PRIOR_*).
// Used by nnue_write_nnue() to write an accurate architecture description.
static int  nnue_init_prior_mode = 0;  // = NNUE_PRIOR_MATERIAL by default


// ---------------------------------------------------------------------------
// nnue_init_accumulator
// ---------------------------------------------------------------------------
void nnue_init_accumulator(NNUEAccumulator &acc, const position &pos)
{
    for (int persp = 0; persp < 2; persp++) {
        memcpy(acc.acc[persp], ft_biases, NNUE_HALF_DIMS * sizeof(int16_t));
        memset(acc.psqt[persp], 0, NNUE_PSQT_BKTS * sizeof(int32_t));

        int ksq = pos.plist[persp][KING][1];

        for (int side = 0; side < 2; side++) {
            for (int ptype = PAWN; ptype <= KING; ptype++) {
                for (int i = 1; i <= pos.plist[side][ptype][0]; i++) {
                    int psq  = pos.plist[side][ptype][i];
                    int fidx = halfkav2_feature(persp, ksq, psq, ptype, side);
                    add_feat(acc.acc[persp], acc.psqt[persp], fidx);
                }
            }
        }

        acc.dirty[persp] = false;
    }
    acc.computed = true;
}

// ---------------------------------------------------------------------------
// nnue_update_accumulator
// ---------------------------------------------------------------------------
void nnue_update_accumulator(NNUEAccumulator &acc,
                             const position  &before,
                             const position  &after,
                             move mv)
{
    int from     = mv.b.from;
    int to       = mv.b.to;
    int mtype    = mv.b.type;
    int mover_pt = PTYPE(before.sq[from]);
    int mover_sd = PSIDE(before.sq[from]);
    int capt_pt  = PTYPE(before.sq[to]);

    int wksq = after.plist[WHITE][KING][1];
    int bksq = after.plist[BLACK][KING][1];

    if (mover_pt == KING) {
        acc.dirty[mover_sd] = true;

        int opp_sd   = mover_sd ^ 1;
        int opp_king = (opp_sd == WHITE) ? wksq : bksq;

        if (!acc.dirty[opp_sd]) {
            sub_feat(acc.acc[opp_sd], acc.psqt[opp_sd],
                     halfkav2_feature(opp_sd, opp_king, from, KING, mover_sd));
            add_feat(acc.acc[opp_sd], acc.psqt[opp_sd],
                     halfkav2_feature(opp_sd, opp_king, to,   KING, mover_sd));

            // King capture: remove the captured piece from the opponent's accumulator.
            if (capt_pt) {
                sub_feat(acc.acc[opp_sd], acc.psqt[opp_sd],
                         halfkav2_feature(opp_sd, opp_king, to, capt_pt, mover_sd ^ 1));
            }

            if (mtype & CASTLE) {
                int rook_from, rook_to;
                if (mover_sd == WHITE) {
                    if (to == 6) { rook_from = before.Krook[WHITE]; rook_to = 5; }
                    else         { rook_from = before.Qrook[WHITE]; rook_to = 3; }
                } else {
                    if (to == 62) { rook_from = before.Krook[BLACK]; rook_to = 61; }
                    else          { rook_from = before.Qrook[BLACK]; rook_to = 59; }
                }
                sub_feat(acc.acc[opp_sd], acc.psqt[opp_sd],
                         halfkav2_feature(opp_sd, opp_king, rook_from, ROOK, mover_sd));
                add_feat(acc.acc[opp_sd], acc.psqt[opp_sd],
                         halfkav2_feature(opp_sd, opp_king, rook_to,   ROOK, mover_sd));
            }
        }
        return;
    }

    for (int persp = 0; persp < 2; persp++) {
        if (acc.dirty[persp]) continue;

        int our_king = (persp == WHITE) ? wksq : bksq;

        sub_feat(acc.acc[persp], acc.psqt[persp],
                 halfkav2_feature(persp, our_king, from, mover_pt, mover_sd));

        if (capt_pt && !(mtype & EP)) {
            sub_feat(acc.acc[persp], acc.psqt[persp],
                     halfkav2_feature(persp, our_king, to, capt_pt, mover_sd ^ 1));
        }

        if (mtype & EP) {
            int ep_sq = mover_sd ? (to - 8) : (to + 8);
            sub_feat(acc.acc[persp], acc.psqt[persp],
                     halfkav2_feature(persp, our_king, ep_sq, PAWN, mover_sd ^ 1));
        }

        if (mtype & PROMOTE) {
            add_feat(acc.acc[persp], acc.psqt[persp],
                     halfkav2_feature(persp, our_king, to, mv.b.promote, mover_sd));
        } else {
            add_feat(acc.acc[persp], acc.psqt[persp],
                     halfkav2_feature(persp, our_king, to, mover_pt, mover_sd));
        }
    }
}

// ---------------------------------------------------------------------------
// nnue_record_delta — record feature-index changes for a move (no ft_weights access)
// ---------------------------------------------------------------------------
void nnue_record_delta(NNUEAccumulator &acc,
                       const position  &before,
                       const position  &after,
                       move mv)
{
    acc.computed        = false;
    acc.n_add[0]        = acc.n_add[1] = 0;
    acc.n_sub[0]        = acc.n_sub[1] = 0;
    acc.need_refresh[0] = acc.need_refresh[1] = false;

    int from     = mv.b.from;
    int to       = mv.b.to;
    int mtype    = mv.b.type;
    int mover_pt = PTYPE(before.sq[from]);
    int mover_sd = PSIDE(before.sq[from]);
    int capt_pt  = PTYPE(before.sq[to]);

    int wksq = after.plist[WHITE][KING][1];
    int bksq = after.plist[BLACK][KING][1];

    if (mover_pt == KING) {
        // Moving king's own perspective must be fully rebuilt.
        acc.need_refresh[mover_sd] = true;

        // Opponent's perspective: incremental update for king relocation.
        int opp_sd   = mover_sd ^ 1;
        int opp_king = (opp_sd == WHITE) ? wksq : bksq;

        int fi;
        fi = halfkav2_feature(opp_sd, opp_king, from, KING, mover_sd);
        if (fi >= 0) acc.sub[opp_sd][acc.n_sub[opp_sd]++] = fi;
        fi = halfkav2_feature(opp_sd, opp_king, to,   KING, mover_sd);
        if (fi >= 0) acc.add[opp_sd][acc.n_add[opp_sd]++] = fi;

        // King capture: remove the captured piece from the opponent's accumulator.
        if (capt_pt) {
            fi = halfkav2_feature(opp_sd, opp_king, to, capt_pt, mover_sd ^ 1);
            if (fi >= 0) acc.sub[opp_sd][acc.n_sub[opp_sd]++] = fi;
        }

        if (mtype & CASTLE) {
            int rook_from, rook_to;
            if (mover_sd == WHITE) {
                if (to == 6) { rook_from = before.Krook[WHITE]; rook_to = 5; }
                else         { rook_from = before.Qrook[WHITE]; rook_to = 3; }
            } else {
                if (to == 62) { rook_from = before.Krook[BLACK]; rook_to = 61; }
                else          { rook_from = before.Qrook[BLACK]; rook_to = 59; }
            }
            fi = halfkav2_feature(opp_sd, opp_king, rook_from, ROOK, mover_sd);
            if (fi >= 0) acc.sub[opp_sd][acc.n_sub[opp_sd]++] = fi;
            fi = halfkav2_feature(opp_sd, opp_king, rook_to,   ROOK, mover_sd);
            if (fi >= 0) acc.add[opp_sd][acc.n_add[opp_sd]++] = fi;
        }
        return;
    }

    // Non-king move: both perspectives updated incrementally.
    for (int persp = 0; persp < 2; persp++) {
        int our_king = (persp == WHITE) ? wksq : bksq;
        int fi;

        fi = halfkav2_feature(persp, our_king, from, mover_pt, mover_sd);
        if (fi >= 0) acc.sub[persp][acc.n_sub[persp]++] = fi;

        if (capt_pt && !(mtype & EP)) {
            fi = halfkav2_feature(persp, our_king, to, capt_pt, mover_sd ^ 1);
            if (fi >= 0) acc.sub[persp][acc.n_sub[persp]++] = fi;
        }

        if (mtype & EP) {
            int ep_sq = mover_sd ? (to - 8) : (to + 8);
            fi = halfkav2_feature(persp, our_king, ep_sq, PAWN, mover_sd ^ 1);
            if (fi >= 0) acc.sub[persp][acc.n_sub[persp]++] = fi;
        }

        if (mtype & PROMOTE) {
            fi = halfkav2_feature(persp, our_king, to, mv.b.promote, mover_sd);
            if (fi >= 0) acc.add[persp][acc.n_add[persp]++] = fi;
        } else {
            fi = halfkav2_feature(persp, our_king, to, mover_pt, mover_sd);
            if (fi >= 0) acc.add[persp][acc.n_add[persp]++] = fi;
        }
    }
}

// ---------------------------------------------------------------------------
// nnue_apply_delta — materialise a lazy accumulator from its parent
// ---------------------------------------------------------------------------
void nnue_apply_delta(NNUEAccumulator &acc,
                      const NNUEAccumulator &parent_acc,
                      const position &pos)
{
    for (int p = 0; p < 2; p++) {
        if (acc.need_refresh[p]) {
            // King moved for this perspective: full rebuild from position.
            int ksq = pos.plist[p][KING][1];
            memcpy(acc.acc[p], ft_biases, NNUE_HALF_DIMS * sizeof(int16_t));
            memset(acc.psqt[p], 0, NNUE_PSQT_BKTS * sizeof(int32_t));
            for (int side = 0; side < 2; side++) {
                for (int ptype = PAWN; ptype <= KING; ptype++) {
                    for (int i = 1; i <= pos.plist[side][ptype][0]; i++) {
                        int psq = pos.plist[side][ptype][i];
                        add_feat(acc.acc[p], acc.psqt[p],
                                 halfkav2_feature(p, ksq, psq, ptype, side));
                    }
                }
            }
        } else {
            // Incremental: copy parent's accumulator and apply the stored deltas.
            memcpy(acc.acc[p],  parent_acc.acc[p],  NNUE_HALF_DIMS  * sizeof(int16_t));
            memcpy(acc.psqt[p], parent_acc.psqt[p], NNUE_PSQT_BKTS * sizeof(int32_t));
            for (int k = 0; k < acc.n_sub[p]; k++)
                sub_feat(acc.acc[p], acc.psqt[p], acc.sub[p][k]);
            for (int k = 0; k < acc.n_add[p]; k++)
                add_feat(acc.acc[p], acc.psqt[p], acc.add[p][k]);
        }
    }
    acc.dirty[0] = acc.dirty[1] = false;
    acc.computed = true;

#ifdef NNUE_CHECK_LAZY
    // Diagnostic: verify lazy result against full-refresh
    {
        NNUEAccumulator ref;
        ref.dirty[0] = ref.dirty[1] = true;
        nnue_init_accumulator(ref, pos);
        for (int p = 0; p < 2; p++) {
            for (int j = 0; j < NNUE_HALF_DIMS; j++) {
                if (acc.acc[p][j] != ref.acc[p][j]) {
                    fprintf(stderr, "LAZY_MISMATCH persp=%d j=%d lazy=%d ref=%d parent=%d "
                            "n_sub=%d n_add=%d need_refresh=%d\n",
                            p, j, (int)acc.acc[p][j], (int)ref.acc[p][j],
                            (int)parent_acc.acc[p][j],
                            (int)acc.n_sub[p], (int)acc.n_add[p], (int)acc.need_refresh[p]);
                    // Print sub features and their weight at dim j
                    for (int k = 0; k < acc.n_sub[p]; k++) {
                        int fi = acc.sub[p][k];
                        fprintf(stderr, "  SUB[%d] fi=%d w[j]=%d\n",
                                k, fi, (int)ft_weights[(int64_t)fi * NNUE_HALF_DIMS + j]);
                    }
                    for (int k = 0; k < acc.n_add[p]; k++) {
                        int fi = acc.add[p][k];
                        fprintf(stderr, "  ADD[%d] fi=%d w[j]=%d\n",
                                k, fi, (int)ft_weights[(int64_t)fi * NNUE_HALF_DIMS + j]);
                    }
                    // Scan ref features for this perspective and print those whose
                    // weight at dim j is non-zero (to identify the missing/extra feature).
                    {
                        int ksq = pos.plist[p][KING][1];
                        fprintf(stderr, "  Ref features (persp=%d ksq=%d):\n", p, ksq);
                        for (int side = 0; side < 2; side++) {
                            for (int ptype = PAWN; ptype <= KING; ptype++) {
                                for (int i = 1; i <= pos.plist[side][ptype][0]; i++) {
                                    int psq = pos.plist[side][ptype][i];
                                    int fi = halfkav2_feature(p, ksq, psq, ptype, side);
                                    if (fi >= 0) {
                                        int w = (int)ft_weights[(int64_t)fi * NNUE_HALF_DIMS + j];
                                        if (w != 0)
                                            fprintf(stderr, "  feat fi=%d side=%d ptype=%d psq=%d w[j]=%d\n",
                                                    fi, side, ptype, psq, w);
                                    }
                                }
                            }
                        }
                    }
                    goto done_check;
                }
            }
        }
        done_check:;
    }
#endif
}

// ---------------------------------------------------------------------------
// nnue_evaluate — forward pass → score from stm's perspective
//
// Input preparation (SqrCReLU — Stockfish 15.1 exact):
//   For each perspective (stm, opp), from NNUE_HALF_DIMS=1024 int16 accumulator:
//     - Clamp acc[i] and acc[i+512] directly to [0, 127] (NO right-shift)
//     - SqrCReLU on pairs for i=0..511:
//         a = clamp(acc[i],       0, 127)
//         b = clamp(acc[i+512],   0, 127)
//         out[i] = (a*b) >> SQR_SHIFT   (≤ 127, no overflow)
//     - No CReLU part — only 512 values per perspective
//   Total FC0 input: [stm_512 | opp_512] = 1024 int8 values
//
// FC0 (16 outputs, output-major weights 16×1024):
//   raw[o] = bias[o] + dot(l0_in, weights[o*1024 .. (o+1)*1024-1])
//
// Dual activation of FC0 outputs 0..14:
//   v = clamp(raw[o]>>WEIGHT_SHIFT, 0, 127)
//   fc1_in[o]    = clamp((v*v)>>SQR_SHIFT, 0, 127)   // SqrCReLU
//   fc1_in[15+o] = v                                  // CReLU
// FC0 output 15 contributes directly to the final output.
//
// FC1 (32 outputs, 32-padded input; fc1_in[30..31] = 0):
//   raw1[o] = bias[o] + dot(fc1_in[0..31], weights[o*32 .. o*32+31])
//   fc2_in[o] = clamp(raw1[o]>>WEIGHT_SHIFT, 0, 127)
//
// FC2 output + FC0 direct + PSQT → centipawns
// ---------------------------------------------------------------------------
int nnue_evaluate(const NNUEAccumulator &acc, int stm, int piece_count)
{
    if (piece_count < 1)  piece_count = 1;
    if (piece_count > 32) piece_count = 32;
    int stack = (piece_count - 1) / 4;  // 0..7

    // 1. SqrCReLU activation: produce 1024 int8 values
    //    Layout: [stm_sqr(0..511) | opp_sqr(0..511)]
    //    Accumulator int16 clamped directly to [0,127] — no right-shift.
    int8_t l0_in[NNUE_L0_INPUT];
    const int16_t *persp_acc[2] = { acc.acc[stm], acc.acc[stm ^ 1] };

#ifdef __ARM_NEON
    {
        const int16x8_t zero16    = vdupq_n_s16(0);
        const int16x8_t max127_16 = vdupq_n_s16(127);
        for (int p = 0; p < 2; p++) {
            const int16_t *a = persp_acc[p];
            int8_t *out = l0_in + p * 512;
            for (int i = 0; i < 512; i += 8) {
                // Clamp directly to [0,127] — no >> 6 shift
                int16x8_t va_c = vminq_s16(vmaxq_s16(vld1q_s16(a + i),       zero16), max127_16);
                int16x8_t vb_c = vminq_s16(vmaxq_s16(vld1q_s16(a + i + 512), zero16), max127_16);
                int8x8_t  va8  = vmovn_s16(va_c);
                int8x8_t  vb8  = vmovn_s16(vb_c);
                // SqrCReLU: max product = 127*127=16129, >>7 = 126 ≤ 127, no clamp needed
                int8x8_t  sq8  = vmovn_s16(vshrq_n_s16(vmull_s8(va8, vb8), 7)); // SQR_SHIFT=7
                vst1_s8(out + i, sq8); // SqrCReLU → [0..511]
            }
        }
    }
#elif defined(EPOCH_USE_AVX2)
    {
        const __m256i zero   = _mm256_setzero_si256();
        const __m256i max127 = _mm256_set1_epi16(127);
        for (int p = 0; p < 2; p++) {
            const int16_t *a = persp_acc[p];
            int8_t *out = l0_in + p * 512;
            for (int i = 0; i < 512; i += 32) {
                __m256i va0 = _mm256_loadu_si256((const __m256i*)(a + i));
                __m256i vb0 = _mm256_loadu_si256((const __m256i*)(a + i + 512));
                va0 = _mm256_min_epi16(_mm256_max_epi16(va0, zero), max127);
                vb0 = _mm256_min_epi16(_mm256_max_epi16(vb0, zero), max127);
                __m256i prod0 = _mm256_srai_epi16(_mm256_mullo_epi16(va0, vb0), NNUE_SQR_SHIFT);

                __m256i va1 = _mm256_loadu_si256((const __m256i*)(a + i + 16));
                __m256i vb1 = _mm256_loadu_si256((const __m256i*)(a + i + 512 + 16));
                va1 = _mm256_min_epi16(_mm256_max_epi16(va1, zero), max127);
                vb1 = _mm256_min_epi16(_mm256_max_epi16(vb1, zero), max127);
                __m256i prod1 = _mm256_srai_epi16(_mm256_mullo_epi16(va1, vb1), NNUE_SQR_SHIFT);

                // Pack int16→int8 and fix AVX2 lane interleaving
                __m256i packed = _mm256_permute4x64_epi64(
                    _mm256_packs_epi16(prod0, prod1), _MM_SHUFFLE(3,1,2,0));
                _mm256_storeu_si256((__m256i*)(out + i), packed);
            }
        }
    }
#else
    for (int p = 0; p < 2; p++) {
        const int16_t *a = persp_acc[p];
        int8_t *out = l0_in + p * 512;

        // SqrCReLU on pairs (i, i+512) — clamp directly, no shift
        for (int i = 0; i < 512; i++) {
            int va = (int)a[i];
            int vb = (int)a[i + 512];
            if (va < 0) va = 0; else if (va > 127) va = 127;
            if (vb < 0) vb = 0; else if (vb > 127) vb = 127;
            int sq = (va * vb) >> NNUE_SQR_SHIFT;
            out[i] = (int8_t)(sq > 127 ? 127 : sq);
        }
    }
#endif

    // 2. FC0: 1024 → 16
    //    Weights are stored in vdotq-friendly layout (set at load time):
    //      wt[ib*64 + ob*16 + k*4 + j]  where ib=i/4, j=i%4, ob=o/4, k=o%4
    //    Each loop iteration processes 4 consecutive inputs against all 16 outputs
    //    via 4 vdotq_s32 calls — 256 iterations total (vs 16×256 per-output).
    int32_t fc0_raw[NNUE_L0_SIZE];
    {
        const int32_t *bias = l0_biases[stack];
        const int8_t  *wt   = l0_weights[stack];
#ifdef __ARM_FEATURE_DOTPROD
        // vdotq_s32(acc, b, c): for k=0..3, acc[k] += b[4k+j]*c[4k+j] for j=0..3
        // b = [in[i+0], in[i+1], in[i+2], in[i+3]] repeated 4 times (one per output lane)
        // c (per output block ob) = [w(i+j, o=ob*4+k)] arranged to match b's lane pattern
        int32x4_t acc0 = vld1q_s32(bias);
        int32x4_t acc1 = vld1q_s32(bias + 4);
        int32x4_t acc2 = vld1q_s32(bias + 8);
        int32x4_t acc3 = vld1q_s32(bias + 12);
        for (int ib = 0; ib < NNUE_L0_INPUT / 4; ib++) {
            // Replicate the 4-byte input chunk across all 4 int32 lanes.
            int8x16_t b = vreinterpretq_s8_s32(vdupq_n_s32(*(const int32_t *)(l0_in + ib * 4)));
            acc0 = vdotq_s32(acc0, b, vld1q_s8(wt + ib * 64));      // out block 0 (o 0-3)
            acc1 = vdotq_s32(acc1, b, vld1q_s8(wt + ib * 64 + 16)); // out block 1 (o 4-7)
            acc2 = vdotq_s32(acc2, b, vld1q_s8(wt + ib * 64 + 32)); // out block 2 (o 8-11)
            acc3 = vdotq_s32(acc3, b, vld1q_s8(wt + ib * 64 + 48)); // out block 3 (o 12-15)
        }
        vst1q_s32(fc0_raw,      acc0);
        vst1q_s32(fc0_raw + 4,  acc1);
        vst1q_s32(fc0_raw + 8,  acc2);
        vst1q_s32(fc0_raw + 12, acc3);
#elif defined(EPOCH_USE_AVX2)
        {
            // AVX2: process all 16 outputs in two __m256i accumulators (8 int32 each).
            // Uses maddubs(uint8,int8)+madd idiom to emulate vdotq_s32 on x86.
            // l0_in values are in [0,127] from SqrCReLU so uint8 interpretation is safe.
            const __m256i ones = _mm256_set1_epi16(1);
            __m256i acc0 = _mm256_loadu_si256((const __m256i*)bias);
            __m256i acc1 = _mm256_loadu_si256((const __m256i*)(bias + 8));
            for (int ib = 0; ib < NNUE_L0_INPUT / 4; ib++) {
                __m256i b = _mm256_set1_epi32(*(const int32_t*)(l0_in + ib * 4));
                const int8_t *wts = wt + ib * 64;
                acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(
                    _mm256_maddubs_epi16(b, _mm256_loadu_si256((const __m256i*)wts)), ones));
                acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(
                    _mm256_maddubs_epi16(b, _mm256_loadu_si256((const __m256i*)(wts + 32))), ones));
            }
            _mm256_storeu_si256((__m256i*)fc0_raw,      acc0);
            _mm256_storeu_si256((__m256i*)(fc0_raw + 8), acc1);
        }
#else
        // Scalar fallback (same vdotq weight layout, computed element-wise).
        for (int o = 0; o < NNUE_L0_SIZE; o++) fc0_raw[o] = bias[o];
        for (int ib = 0; ib < NNUE_L0_INPUT / 4; ib++) {
            for (int j = 0; j < 4; j++) {
                int32_t v = (int32_t)l0_in[ib * 4 + j];
                for (int ob = 0; ob < NNUE_L0_SIZE / 4; ob++)
                    for (int k = 0; k < 4; k++)
                        fc0_raw[ob * 4 + k] += v * (int32_t)wt[ib * 64 + ob * 16 + k * 4 + j];
            }
        }
#endif
    }

    // 3. Dual activation of FC0 outputs 0..L0_DIRECT-1 → 2×L0_DIRECT values for FC1
    //    fc1_in[0..14]  = SqrCReLU(fc0_raw[0..14])
    //    fc1_in[15..29] = CReLU(fc0_raw[0..14])
    //    fc1_in[30..31] = 0 (padding)
    //
    //    Stockfish SqrClippedReLU (sqr_clipped_relu.h):
    //      output = clamp(0, 127, (input * input) >> (2*WeightScaleBits + SqrShift))
    //             = clamp(0, 127, fc0^2 >> 19)
    //    Squaring the raw int32 value before clamping is essential — negative inputs
    //    produce non-zero squared outputs.  Clamping BEFORE squaring (old code) was
    //    wrong because it zeroed all negative fc0 values.
    int8_t fc1_in[NNUE_L1_PADDED];
    memset(fc1_in, 0, sizeof(fc1_in));
    for (int o = 0; o < NNUE_L0_DIRECT; o++) {
        // SqrCReLU: square raw value (works for negative), shift by 2*6+7=19, clamp
        int64_t sq64 = (int64_t)fc0_raw[o] * fc0_raw[o] >>
                       (2 * NNUE_WEIGHT_SHIFT + NNUE_SQR_SHIFT);
        fc1_in[o] = (int8_t)(sq64 > 127 ? 127 : sq64);
        // CReLU: shift and clamp to [0, 127]
        int v = fc0_raw[o] >> NNUE_WEIGHT_SHIFT;
        fc1_in[NNUE_L0_DIRECT + o] = (int8_t)(v < 0 ? 0 : v > 127 ? 127 : v);
    }

    // 4. FC1: 30(→32 padded) → 32
    int8_t fc2_in[NNUE_L1_SIZE];
    {
        const int32_t *bias = l1_biases[stack];
        const int8_t  *wt   = l1_weights[stack];
#ifdef __ARM_FEATURE_DOTPROD
        // vdotq layout: wt[ib*128 + ob*16 + k*4 + j] (8 input blocks × 8 output blocks)
        int32x4_t a0 = vld1q_s32(bias),      a1 = vld1q_s32(bias + 4);
        int32x4_t a2 = vld1q_s32(bias + 8),  a3 = vld1q_s32(bias + 12);
        int32x4_t a4 = vld1q_s32(bias + 16), a5 = vld1q_s32(bias + 20);
        int32x4_t a6 = vld1q_s32(bias + 24), a7 = vld1q_s32(bias + 28);
        for (int ib = 0; ib < NNUE_L1_PADDED / 4; ib++) {  // 8 iterations
            int8x16_t b = vreinterpretq_s8_s32(vdupq_n_s32(*(const int32_t *)(fc1_in + ib * 4)));
            a0 = vdotq_s32(a0, b, vld1q_s8(wt + ib * 128));
            a1 = vdotq_s32(a1, b, vld1q_s8(wt + ib * 128 + 16));
            a2 = vdotq_s32(a2, b, vld1q_s8(wt + ib * 128 + 32));
            a3 = vdotq_s32(a3, b, vld1q_s8(wt + ib * 128 + 48));
            a4 = vdotq_s32(a4, b, vld1q_s8(wt + ib * 128 + 64));
            a5 = vdotq_s32(a5, b, vld1q_s8(wt + ib * 128 + 80));
            a6 = vdotq_s32(a6, b, vld1q_s8(wt + ib * 128 + 96));
            a7 = vdotq_s32(a7, b, vld1q_s8(wt + ib * 128 + 112));
        }
        // Shift by WEIGHT_SHIFT=6, clamp to [0,127], narrow int32→int16→int8
        const int32x4_t zero32   = vdupq_n_s32(0);
        const int32x4_t max127_32 = vdupq_n_s32(127);
#define SHR6CLAMP(v) vminq_s32(vmaxq_s32(vshrq_n_s32(v, 6), zero32), max127_32)
        int32x4_t c0=SHR6CLAMP(a0), c1=SHR6CLAMP(a1), c2=SHR6CLAMP(a2), c3=SHR6CLAMP(a3);
        int32x4_t c4=SHR6CLAMP(a4), c5=SHR6CLAMP(a5), c6=SHR6CLAMP(a6), c7=SHR6CLAMP(a7);
#undef SHR6CLAMP
        int8x8_t b01 = vmovn_s16(vcombine_s16(vmovn_s32(c0), vmovn_s32(c1)));
        int8x8_t b23 = vmovn_s16(vcombine_s16(vmovn_s32(c2), vmovn_s32(c3)));
        int8x8_t b45 = vmovn_s16(vcombine_s16(vmovn_s32(c4), vmovn_s32(c5)));
        int8x8_t b67 = vmovn_s16(vcombine_s16(vmovn_s32(c6), vmovn_s32(c7)));
        vst1q_s8(fc2_in,      vcombine_s8(b01, b23));
        vst1q_s8(fc2_in + 16, vcombine_s8(b45, b67));
#elif defined(EPOCH_USE_AVX2)
        {
            // AVX2: 32 inputs × 32 outputs using vdotq weight layout.
            // fc1_in values are in [0,127] so uint8 interpretation is safe for maddubs.
            const __m128i ones = _mm_set1_epi16(1);
            __m128i a0 = _mm_loadu_si128((const __m128i*)bias);
            __m128i a1 = _mm_loadu_si128((const __m128i*)(bias + 4));
            __m128i a2 = _mm_loadu_si128((const __m128i*)(bias + 8));
            __m128i a3 = _mm_loadu_si128((const __m128i*)(bias + 12));
            __m128i a4 = _mm_loadu_si128((const __m128i*)(bias + 16));
            __m128i a5 = _mm_loadu_si128((const __m128i*)(bias + 20));
            __m128i a6 = _mm_loadu_si128((const __m128i*)(bias + 24));
            __m128i a7 = _mm_loadu_si128((const __m128i*)(bias + 28));
            for (int ib = 0; ib < NNUE_L1_PADDED / 4; ib++) {
                __m128i b = _mm_set1_epi32(*(const int32_t*)(fc1_in + ib * 4));
                const int8_t *wts = wt + ib * 128;
#define MADD(acc, off) acc = _mm_add_epi32(acc, _mm_madd_epi16( \
    _mm_maddubs_epi16(b, _mm_loadu_si128((const __m128i*)(wts + off))), ones))
                MADD(a0,   0); MADD(a1,  16); MADD(a2,  32); MADD(a3,  48);
                MADD(a4,  64); MADD(a5,  80); MADD(a6,  96); MADD(a7, 112);
#undef MADD
            }
            // Shift by WEIGHT_SHIFT, clamp to [0,127], narrow int32→int16→int8
            const __m128i zero128    = _mm_setzero_si128();
            const __m128i max127_128 = _mm_set1_epi32(127);
#define SHR6CLAMP(v) _mm_min_epi32(_mm_max_epi32(_mm_srai_epi32(v, NNUE_WEIGHT_SHIFT), zero128), max127_128)
            a0=SHR6CLAMP(a0); a1=SHR6CLAMP(a1); a2=SHR6CLAMP(a2); a3=SHR6CLAMP(a3);
            a4=SHR6CLAMP(a4); a5=SHR6CLAMP(a5); a6=SHR6CLAMP(a6); a7=SHR6CLAMP(a7);
#undef SHR6CLAMP
            _mm_storeu_si128((__m128i*)fc2_in,
                _mm_packs_epi16(_mm_packs_epi32(a0, a1), _mm_packs_epi32(a2, a3)));
            _mm_storeu_si128((__m128i*)(fc2_in + 16),
                _mm_packs_epi16(_mm_packs_epi32(a4, a5), _mm_packs_epi32(a6, a7)));
        }
#else
        for (int o = 0; o < NNUE_L1_SIZE; o++) {
            int32_t sum = bias[o];
            const int8_t *row = wt + o * NNUE_L1_PADDED;
            for (int i = 0; i < NNUE_L1_PADDED; i++)
                sum += (int32_t)fc1_in[i] * (int32_t)row[i];
            sum >>= NNUE_WEIGHT_SHIFT;
            fc2_in[o] = (int8_t)(sum < 0 ? 0 : sum > 127 ? 127 : sum);
        }
#endif
    }

    // 5. FC2 (output): 32 → 1
    int32_t fc2_out = out_biases[stack];
    for (int i = 0; i < NNUE_L2_PADDED; i++)
        fc2_out += (int32_t)fc2_in[i] * (int32_t)out_weights[stack][i];

    // 6. Passthrough: FC0 output 15 is scaled and added to FC2 result.
    //    Stockfish formula: fwdOut = fc0_raw[15] * (600*OutputScale) / (127*WeightScaleBits)
    //                              = fc0_raw[15] * 9600 / 8128
    int32_t fwdOut = (int32_t)((int64_t)fc0_raw[NNUE_L0_DIRECT] * 9600 / 8128);
    int32_t positional = fc2_out + fwdOut;

    // 7. PSQT + final score
    //    Stockfish: value = (psqt_diff/2 + positional) / OutputScale(16)  [internal Value units]
    //    To convert to centipawns: * 100 / NormalizeToPawnValue(361)
    //    Combined: * 100 / (16 * 361) = * 100 / 5776
    int32_t psqt_diff = acc.psqt[stm][stack] - acc.psqt[stm ^ 1][stack];
#ifdef NNUE_PSQT_ONLY
    int score = (int32_t)((int64_t)(psqt_diff / 2) * 100 / 5776);
#else
    int score = (int32_t)((int64_t)(psqt_diff / 2 + positional) * 100 / 5776);
#endif

    if (getenv("NNUE_DEBUG"))
        fprintf(stderr, "NNUE_DEBUG: stack=%d fc2=%d fwd=%d positional=%d psqt_diff=%d total=%d\n",
                stack, fc2_out, fwdOut, positional, psqt_diff, score);

    if (getenv("NNUE_DEBUG_VERBOSE")) {
        fprintf(stderr, "NNUE_VERBOSE: stack=%d stm=%d pc=%d\n", stack, stm, piece_count);
        // Raw accumulator (first 16 of each perspective)
        fprintf(stderr, "  acc[stm][0..15]:");
        for (int i = 0; i < 16; i++) fprintf(stderr, " %d", (int)acc.acc[stm][i]);
        fprintf(stderr, "\n  acc[opp][0..15]:");
        for (int i = 0; i < 16; i++) fprintf(stderr, " %d", (int)acc.acc[stm^1][i]);
        fprintf(stderr, "\n");
        // FT activation output — print all non-zero values
        fprintf(stderr, "  l0_in nonzero:");
        for (int i = 0; i < NNUE_L0_INPUT; i++)
            if (l0_in[i]) fprintf(stderr, " [%d]=%d", i, (int)l0_in[i]);
        fprintf(stderr, "\n");
        // FC0 raw outputs
        fprintf(stderr, "  fc0_raw[0..15]:");
        for (int i = 0; i < NNUE_L0_SIZE; i++) fprintf(stderr, " %d", fc0_raw[i]);
        fprintf(stderr, "\n");
        // FC1 inputs (all 30 active + 2 padding)
        fprintf(stderr, "  fc1_in[0..31] (sqr|clip|pad):");
        for (int i = 0; i < NNUE_L1_PADDED; i++) fprintf(stderr, " %d", (int)fc1_in[i]);
        fprintf(stderr, "\n");
        // FC2 inputs (FC1 CReLU outputs)
        fprintf(stderr, "  fc2_in[0..31]:");
        for (int i = 0; i < NNUE_L1_SIZE; i++) fprintf(stderr, " %d", (int)fc2_in[i]);
        fprintf(stderr, "\n");
        fprintf(stderr, "  fc2_out=%d fwdOut=%d positional=%d psqt_diff=%d score=%d\n",
                fc2_out, fwdOut, positional, psqt_diff, score);
    }
    return score;
}
// ---------------------------------------------------------------------------
// nnue_extract_piece_values — derive cp values from loaded PSQT and write
// into value[1..5] (score.h global), replacing the hardcoded constants.
//
// Averages own-perspective PSQT weights across all squares and all 8 material
// buckets for each piece type.  In a well-trained symmetric network the enemy
// features are ≈ −own, so avg_own × 100/5776 gives the correct cp contribution.
// Under NNUE_FIXED_PIECE_VALUES the result is report-only (value[] is not
// overwritten); it is the drift canary for the pure-PSQT material scale.
//
// value[0] is unused; value[6] (king sentinel = 10000) is not touched.
// ---------------------------------------------------------------------------
void nnue_extract_piece_values(bool verbose)
{
    // value[] is defined in score.h, included earlier in the unity build.
    extern int value[7];
    if (!nnue_available) return;

    const int PS_NB = 704;   // HalfKAv2_hm: 11 piece-sq slots × 64 squares
    double sum[5] = {};
    int    cnt[5] = {};

    for (int fi = 0; fi < NNUE_FT_INPUTS; fi++) {
        int fi_in_bkt = fi % PS_NB;
        int ps_slot   = fi_in_bkt / 128;      // 0=pawn … 5=king
        bool is_own   = (fi_in_bkt % 128) < 64;
        if (!is_own || ps_slot >= 5) continue; // skip enemy features and king

#if TDLEAF
        if (psqt_weights_f32) {
            // Use the FP32 PSQT shadow for accuracy when it is allocated.
            const float *pf = psqt_weights_f32 + (size_t)fi * NNUE_PSQT_BKTS;
            for (int b = 0; b < NNUE_PSQT_BKTS; b++)
                sum[ps_slot] += pf[b];
            cnt[ps_slot] += NNUE_PSQT_BKTS;
            continue;
        }
#endif
        // Non-TDLEAF (or before fp32 shadow is allocated): use int32 array.
        const int32_t *pw = psqt_weights + (size_t)fi * NNUE_PSQT_BKTS;
        for (int b = 0; b < NNUE_PSQT_BKTS; b++)
            sum[ps_slot] += pw[b];
        cnt[ps_slot] += NNUE_PSQT_BKTS;
    }

    int ext[5] = {};                      // extracted cp per piece (diagnostic)
    for (int pt = 0; pt < 5; pt++) {      // pt 0-4 → PAWN-QUEEN → value[1-5]
        if (cnt[pt] == 0) continue;
        int cp = (int)round(sum[pt] / cnt[pt] * 100.0 / 5776.0);
        if (cp < 1) cp = 1;              // never allow non-positive piece values in search
        ext[pt] = cp;
#if !NNUE_FIXED_PIECE_VALUES
        value[pt + 1] = cp;
#endif
    }

    if (verbose)
        printf("NNUE: piece values from PSQT%s: P=%d N=%d B=%d R=%d Q=%d cp\n",
               NNUE_FIXED_PIECE_VALUES ? " (report only; search uses classical)" : "",
               ext[0], ext[1], ext[2], ext[3], ext[4]);
}

// ---------------------------------------------------------------------------
// nnue_print_diag_info — the `netinfo` command's handler.  Prints the load
// diagnostics captured silently in nnue_diag at startup, plus a live
// (re-extracted) piece-value report so it reflects any drift since load.
// ---------------------------------------------------------------------------
void nnue_print_diag_info()
{
    if (!nnue_available) { printf("NNUE: not loaded (classical evaluation)\n"); return; }

    printf("NNUE: %s (version=0x%08X hash=0x%08X ft_hash=0x%08X content_hash=0x%08X)\n",
           nnue_get_loaded_path(), nnue_diag.version, nnue_diag.file_hash,
           nnue_diag.ft_hash, nnue_get_content_hash());
    if (nnue_diag.arch_desc[0])
        printf("NNUE: architecture: %s\n", nnue_diag.arch_desc);

#if TDLEAF
    if (nnue_diag.tdleaf_loaded)
        printf("%s\n", nnue_diag.tdleaf_summary);
    else
        printf("TDLeaf: no weights file found — using pretrained .nnue weights.\n"
               "TDLeaf: run with --init-nnue --write-nnue <file> to create a fresh net.\n");
#endif

    nnue_extract_piece_values(true);
}
