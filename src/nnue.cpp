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

// ---------------------------------------------------------------------------
// LEB128 decompressor (used for FT biases, weights, and PSQT weights)
// ---------------------------------------------------------------------------
static bool read_leb128_i16(FILE *f, int16_t *buf, size_t count)
{
    char magic[18] = {};
    if (fread(magic, 1, 17, f) != 17) return false;

    if (memcmp(magic, "COMPRESSED_LEB128", 17) != 0) {
        fseek(f, -17, SEEK_CUR);
        return fread(buf, sizeof(int16_t), count, f) == count;
    }

    uint32_t nbytes;
    if (fread(&nbytes, 4, 1, f) != 1) return false;

    unsigned char *cbuf = (unsigned char*)malloc(nbytes);
    if (!cbuf) return false;
    if (fread(cbuf, 1, nbytes, f) != nbytes) { free(cbuf); return false; }

    size_t pos = 0;
    for (size_t i = 0; i < count; i++) {
        int32_t val = 0; int shift = 0;
        unsigned char byte;
        do {
            if (pos >= nbytes) { free(cbuf); return false; }
            byte = cbuf[pos++];
            val |= (int32_t)(byte & 0x7F) << shift;
            shift += 7;
        } while (byte & 0x80);
        if (shift < 32 && (byte & 0x40))
            val |= ~((int32_t)0) << shift;
        buf[i] = (int16_t)val;
    }
    free(cbuf);
    return true;
}

static bool read_leb128_i32(FILE *f, int32_t *buf, size_t count)
{
    char magic[18] = {};
    if (fread(magic, 1, 17, f) != 17) return false;

    if (memcmp(magic, "COMPRESSED_LEB128", 17) != 0) {
        fseek(f, -17, SEEK_CUR);
        return fread(buf, sizeof(int32_t), count, f) == count;
    }

    uint32_t nbytes;
    if (fread(&nbytes, 4, 1, f) != 1) return false;

    unsigned char *cbuf = (unsigned char*)malloc(nbytes);
    if (!cbuf) return false;
    if (fread(cbuf, 1, nbytes, f) != nbytes) { free(cbuf); return false; }

    size_t pos = 0;
    for (size_t i = 0; i < count; i++) {
        int64_t val = 0; int shift = 0;
        unsigned char byte;
        do {
            if (pos >= nbytes) { free(cbuf); return false; }
            byte = cbuf[pos++];
            val |= (int64_t)(byte & 0x7F) << shift;
            shift += 7;
        } while (byte & 0x80);
        if (shift < 64 && (byte & 0x40))
            val |= ~((int64_t)0) << shift;
        buf[i] = (int32_t)val;
    }
    free(cbuf);
    return true;
}

static uint32_t read_u32(FILE *f) {
    uint32_t v = 0;
    (void)fread(&v, 4, 1, f);
    return v;
}

// ---------------------------------------------------------------------------
// nnue_alloc_arrays — allocate heap FT arrays without loading a file.
// Called by nnue_load() and by --init-nnue mode before nnue_init_zero_weights().
// ---------------------------------------------------------------------------
void nnue_alloc_arrays()
{
    if (!ft_biases)    ft_biases    = new int16_t[NNUE_HALF_DIMS];
    if (!ft_weights)   ft_weights   = new int16_t[(size_t)NNUE_FT_INPUTS * NNUE_HALF_DIMS];
    if (!psqt_weights) psqt_weights = new int32_t[(size_t)NNUE_FT_INPUTS * NNUE_PSQT_BKTS];
    nnue_available = true;
}

// ---------------------------------------------------------------------------
// nnue_load
// ---------------------------------------------------------------------------
bool nnue_load(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        printf("NNUE: could not open %s\n", path);
        return false;
    }

    // Header
    uint32_t version   = read_u32(f);
    uint32_t file_hash = read_u32(f);
    uint32_t desc_size = read_u32(f);
    printf("NNUE: version=0x%08X  hash=0x%08X  desc_size=%u\n",
           version, file_hash, desc_size);

    if (desc_size > 0) {
        char *desc = new char[desc_size + 1];
        size_t nr  = fread(desc, 1, desc_size, f);
        desc[nr]   = '\0';
        printf("NNUE: architecture: %s\n", desc);
        delete[] desc;
        if (nr != desc_size) { fclose(f); return false; }
    }

    // Feature Transformer
    uint32_t ft_hash = read_u32(f);
    printf("NNUE: ft_hash=0x%08X\n", ft_hash);

    nnue_alloc_arrays();  // no-op if already allocated

    printf("NNUE: reading FT biases [%d int16] ...\n", NNUE_HALF_DIMS);
    if (!read_leb128_i16(f, ft_biases, NNUE_HALF_DIMS)) {
        printf("NNUE: FT bias read failed\n"); fclose(f); return false;
    }

    size_t ft_w = (size_t)NNUE_FT_INPUTS * NNUE_HALF_DIMS;
    printf("NNUE: reading FT weights [%zu int16] ...\n", ft_w);
    if (!read_leb128_i16(f, ft_weights, ft_w)) {
        printf("NNUE: FT weight read failed\n"); fclose(f); return false;
    }

    size_t psqt_w = (size_t)NNUE_FT_INPUTS * NNUE_PSQT_BKTS;
    printf("NNUE: reading PSQT weights [%zu int32] ...\n", psqt_w);
    if (!read_leb128_i32(f, psqt_weights, psqt_w)) {
        printf("NNUE: PSQT weight read failed\n"); fclose(f); return false;
    }
    printf("NNUE: FT + PSQT loaded OK\n");

    // Network: 8 layer stacks.
    // Each stack begins with a 4-byte hash (no separate net_hash before the stacks).
    // Layout per stack:
    //   stack_hash(4) + FC0_bias(16*4) + FC0_wt(16*3072) +
    //   FC1_bias(32*4) + FC1_wt(32*32) + FC2_bias(4) + FC2_wt(32)
    //   = 4 + 64 + 49152 + 128 + 1024 + 4 + 32 = 50408 bytes
    printf("NNUE: reading %d layer stacks ...\n", NNUE_LAYER_STACKS);

    for (int s = 0; s < NNUE_LAYER_STACKS; s++) {
        uint32_t stack_hash = read_u32(f);
        nnue_stack_hashes[s] = stack_hash;  // saved for nnue_write_nnue()

        // FC0
        if (fread(l0_biases[s], sizeof(int32_t), NNUE_L0_SIZE, f) != (size_t)NNUE_L0_SIZE)
            { printf("NNUE: stack %d FC0 bias read failed\n", s); fclose(f); return false; }
        size_t fc0_w = (size_t)NNUE_L0_SIZE * NNUE_L0_INPUT;
        {
            // Read output-major [o * NNUE_L0_INPUT + i] into a temp buffer,
            // then rearrange into the vdotq-friendly layout:
            //   For each 4-input block ib (0..255) and 4-output block ob (0..3),
            //   store 16 bytes as [w(i0,o0),w(i1,o0),w(i2,o0),w(i3,o0),
            //                      w(i0,o1),w(i1,o1),w(i2,o1),w(i3,o1), ...]
            //   i.e. l0_weights[s][ib*64 + ob*16 + k*4 + j]
            //      = original[o = ob*4+k][i = ib*4+j]
            // This lets a single vdotq_s32 call accumulate 4 inputs into 4 outputs.
            int8_t *tmp = new int8_t[fc0_w];
            if (fread(tmp, sizeof(int8_t), fc0_w, f) != fc0_w)
                { delete[] tmp; printf("NNUE: stack %d FC0 weight read failed\n", s); fclose(f); return false; }
            for (int o = 0; o < NNUE_L0_SIZE; o++) {
                int ob = o / 4, k = o % 4;
                for (int i = 0; i < NNUE_L0_INPUT; i++) {
                    int ib = i / 4, j = i % 4;
                    l0_weights[s][ib * 64 + ob * 16 + k * 4 + j] =
                        tmp[o * NNUE_L0_INPUT + i];
                }
            }
            delete[] tmp;
        }

        // FC1
        if (fread(l1_biases[s], sizeof(int32_t), NNUE_L1_SIZE, f) != (size_t)NNUE_L1_SIZE)
            { printf("NNUE: stack %d FC1 bias read failed\n", s); fclose(f); return false; }
        size_t fc1_w = (size_t)NNUE_L1_SIZE * NNUE_L1_PADDED;
        {
            // Read output-major [o * NNUE_L1_PADDED + i] into a temp buffer,
            // then rearrange into the vdotq-friendly layout (same scheme as FC0):
            //   l1_weights[s][ib*128 + ob*16 + k*4 + j]
            //     = original[o = ob*4+k][i = ib*4+j]
            //   ib=i/4, j=i%4, ob=o/4, k=o%4  (i∈[0,31], o∈[0,31])
            int8_t tmp[NNUE_L1_SIZE * NNUE_L1_PADDED];
            if (fread(tmp, sizeof(int8_t), fc1_w, f) != fc1_w)
                { printf("NNUE: stack %d FC1 weight read failed\n", s); fclose(f); return false; }
#if defined(__ARM_FEATURE_DOTPROD) || defined(EPOCH_USE_AVX2)
            for (int o = 0; o < NNUE_L1_SIZE; o++) {
                int ob = o / 4, k = o % 4;
                for (int i = 0; i < NNUE_L1_PADDED; i++) {
                    int ib = i / 4, j = i % 4;
                    l1_weights[s][ib * 128 + ob * 16 + k * 4 + j] = tmp[o * NNUE_L1_PADDED + i];
                }
            }
#else
            memcpy(l1_weights[s], tmp, fc1_w);
#endif
        }

        // FC2 (output layer)
        if (fread(&out_biases[s], sizeof(int32_t), 1, f) != 1)
            { printf("NNUE: stack %d FC2 bias read failed\n", s); fclose(f); return false; }
        if (fread(out_weights[s], sizeof(int8_t), NNUE_L2_PADDED, f) != (size_t)NNUE_L2_PADDED)
            { printf("NNUE: stack %d FC2 weight read failed\n", s); fclose(f); return false; }
    }

    fclose(f);
    nnue_available = true;
    strncpy(nnue_loaded_path, path, FILENAME_MAX - 1);
    printf("NNUE: all %d stacks loaded OK\n", NNUE_LAYER_STACKS);
#if TDLEAF
    nnue_init_fp32_weights();
#endif
    return true;
}

// Set to true by nnue_init_zero_weights(); controls description in nnue_write_nnue().
static bool nnue_zero_initialized = false;

// ---------------------------------------------------------------------------
// nnue_backup_file — rename path → path.bak if path exists.
// ---------------------------------------------------------------------------
void nnue_backup_file(const char *path)
{
    FILE *probe = fopen(path, "rb");
    if (!probe) return;  // nothing to back up
    fclose(probe);
    char bak[FILENAME_MAX + 4];
    snprintf(bak, sizeof(bak), "%s.bak", path);
    if (rename(path, bak) == 0)
        fprintf(stderr, "NNUE: backed up '%s' -> '%s'\n", path, bak);
    else
        fprintf(stderr, "NNUE: warning: could not back up '%s'\n", path);
}

// ---------------------------------------------------------------------------
// SLEB128 writers — mirror the readers above; used by nnue_write_nnue().
// Format: "COMPRESSED_LEB128"(17) + nbytes(4) + LEB128 data.
// ---------------------------------------------------------------------------
static bool write_leb128_i16(FILE *f, const int16_t *buf, size_t count)
{
    fwrite("COMPRESSED_LEB128", 1, 17, f);
    long cnt_pos   = ftell(f);
    uint32_t zero  = 0;
    fwrite(&zero, 4, 1, f);            // placeholder for byte count
    long data_start = ftell(f);

    // Buffered writes to avoid per-value fwrite overhead.
    const size_t WBUF = 1 << 21;      // 2 MB
    unsigned char *wb = new unsigned char[WBUF];
    size_t wp = 0;

    for (size_t i = 0; i < count; i++) {
        int32_t val = (int32_t)buf[i];
        for (;;) {
            unsigned char byte = (unsigned char)(val & 0x7F);
            val >>= 7;                 // arithmetic shift
            bool more = !((val == 0 && !(byte & 0x40)) || (val == -1 && (byte & 0x40)));
            if (more) byte |= 0x80;
            wb[wp++] = byte;
            if (wp == WBUF) { fwrite(wb, 1, wp, f); wp = 0; }
            if (!more) break;
        }
    }
    if (wp) fwrite(wb, 1, wp, f);
    delete[] wb;

    long data_end  = ftell(f);
    uint32_t nbytes = (uint32_t)(data_end - data_start);
    fseek(f, cnt_pos, SEEK_SET);
    fwrite(&nbytes, 4, 1, f);
    fseek(f, 0, SEEK_END);
    return true;
}

static bool write_leb128_i32(FILE *f, const int32_t *buf, size_t count)
{
    fwrite("COMPRESSED_LEB128", 1, 17, f);
    long cnt_pos   = ftell(f);
    uint32_t zero  = 0;
    fwrite(&zero, 4, 1, f);
    long data_start = ftell(f);

    const size_t WBUF = 1 << 21;
    unsigned char *wb = new unsigned char[WBUF];
    size_t wp = 0;

    for (size_t i = 0; i < count; i++) {
        int64_t val = (int64_t)buf[i];
        for (;;) {
            unsigned char byte = (unsigned char)(val & 0x7F);
            val >>= 7;
            bool more = !((val == 0 && !(byte & 0x40)) || (val == -1 && (byte & 0x40)));
            if (more) byte |= 0x80;
            wb[wp++] = byte;
            if (wp == WBUF) { fwrite(wb, 1, wp, f); wp = 0; }
            if (!more) break;
        }
    }
    if (wp) fwrite(wb, 1, wp, f);
    delete[] wb;

    long data_end  = ftell(f);
    uint32_t nbytes = (uint32_t)(data_end - data_start);
    fseek(f, cnt_pos, SEEK_SET);
    fwrite(&nbytes, 4, 1, f);
    fseek(f, 0, SEEK_END);
    return true;
}

// ---------------------------------------------------------------------------
// nnue_write_nnue — write current NNUE weights into a complete .nnue file.
// FT biases, FT weights, and PSQT are written from the current in-memory
// arrays (reflecting any zeroing or TDLeaf training), encoded as SLEB128.
// The architecture description is updated:
//   - Normal (trained): original description + " Trained by Leaf TDLeaf"
//   - Zero-initialized: "Random init + basic piece values" (replaces original)
// ---------------------------------------------------------------------------
bool nnue_write_nnue(const char *dst_path)
{
    if (!nnue_available) {
        fprintf(stderr, "nnue_write_nnue: no net loaded\n");
        return false;
    }

    // Header: read from source file if one was loaded; otherwise use SF15.1 constants.
    uint32_t version, file_hash, ft_hash;
    char orig_desc[4096] = {};
    if (nnue_loaded_path[0]) {
        FILE *src = fopen(nnue_loaded_path, "rb");
        if (!src) {
            fprintf(stderr, "nnue_write_nnue: cannot open source '%s'\n", nnue_loaded_path);
            return false;
        }
        uint32_t orig_desc_size;
        (void)fread(&version,        sizeof(uint32_t), 1, src);
        (void)fread(&file_hash,      sizeof(uint32_t), 1, src);
        (void)fread(&orig_desc_size, sizeof(uint32_t), 1, src);
        if (orig_desc_size > 0 && orig_desc_size < sizeof(orig_desc))
            (void)fread(orig_desc, 1, orig_desc_size, src);
        (void)fread(&ft_hash, sizeof(uint32_t), 1, src);
        fclose(src);
    } else {
        // No source file (--init-nnue mode): use nn-ad9b42354671.nnue header constants.
        version   = 0x7AF32F20u;
        file_hash = 0x1C102EF2u;
        ft_hash   = 0x7F2344B8u;
        // orig_desc stays empty; description set below
    }

    // Build new description.
    char new_desc[4096];
    if (nnue_zero_initialized || !orig_desc[0])
        snprintf(new_desc, sizeof(new_desc), "Random init + classical material+PSQ by game stage");
    else
        snprintf(new_desc, sizeof(new_desc), "%s Trained by Leaf TDLeaf", orig_desc);
    uint32_t new_desc_size = (uint32_t)strlen(new_desc);

    nnue_backup_file(dst_path);

    FILE *dst = fopen(dst_path, "wb");
    if (!dst) {
        fprintf(stderr, "nnue_write_nnue: cannot create '%s'\n", dst_path);
        return false;
    }

    // Write header.
    fwrite(&version,       sizeof(uint32_t), 1, dst);
    fwrite(&file_hash,     sizeof(uint32_t), 1, dst);
    fwrite(&new_desc_size, sizeof(uint32_t), 1, dst);
    fwrite(new_desc,       1, new_desc_size, dst);
    fwrite(&ft_hash,       sizeof(uint32_t), 1, dst);

    // Write FT section from current in-memory arrays (LEB128-encoded).
    // This correctly reflects zeroing or TDLeaf training applied in memory.
    printf("NNUE: writing FT biases...\n");
    write_leb128_i16(dst, ft_biases, NNUE_HALF_DIMS);
    printf("NNUE: writing FT weights (23M values, may take a moment)...\n");
    write_leb128_i16(dst, ft_weights, (size_t)NNUE_FT_INPUTS * NNUE_HALF_DIMS);
    printf("NNUE: writing PSQT weights...\n");
    write_leb128_i32(dst, psqt_weights, (size_t)NNUE_FT_INPUTS * NNUE_PSQT_BKTS);

    // Write each FC stack with current weights, reversing the vdotq reordering.
    const size_t fc0_w = (size_t)NNUE_L0_SIZE * NNUE_L0_INPUT;
    const size_t fc1_w = (size_t)NNUE_L1_SIZE * NNUE_L1_PADDED;
    int8_t *tmp = new int8_t[fc0_w > fc1_w ? fc0_w : fc1_w];

    for (int s = 0; s < NNUE_LAYER_STACKS; s++) {
        fwrite(&nnue_stack_hashes[s], 4, 1, dst);

        // FC0 biases
        fwrite(l0_biases[s], sizeof(int32_t), NNUE_L0_SIZE, dst);
        // FC0 weights: un-reorder from vdotq layout back to output-major [o * INPUT + i]
        for (int o = 0; o < NNUE_L0_SIZE; o++) {
            int ob = o / 4, k = o % 4;
            for (int i = 0; i < NNUE_L0_INPUT; i++) {
                int ib = i / 4, j = i % 4;
                tmp[o * NNUE_L0_INPUT + i] = l0_weights[s][ib * 64 + ob * 16 + k * 4 + j];
            }
        }
        fwrite(tmp, 1, fc0_w, dst);

        // FC1 biases
        fwrite(l1_biases[s], sizeof(int32_t), NNUE_L1_SIZE, dst);
        // FC1 weights: un-reorder back to output-major [o * PADDED + i]
#if defined(__ARM_FEATURE_DOTPROD) || defined(EPOCH_USE_AVX2)
        for (int o = 0; o < NNUE_L1_SIZE; o++) {
            int ob = o / 4, k = o % 4;
            for (int i = 0; i < NNUE_L1_PADDED; i++) {
                int ib = i / 4, j = i % 4;
                tmp[o * NNUE_L1_PADDED + i] = l1_weights[s][ib * 128 + ob * 16 + k * 4 + j];
            }
        }
#else
        memcpy(tmp, l1_weights[s], fc1_w);
#endif
        fwrite(tmp, 1, fc1_w, dst);

        // FC2 bias + weights
        fwrite(&out_biases[s], sizeof(int32_t), 1, dst);
        fwrite(out_weights[s], sizeof(int8_t), NNUE_L2_PADDED, dst);
    }

    delete[] tmp;
    fclose(dst);
    printf("NNUE: wrote %s\n", dst_path);
    return true;
}

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

// ===========================================================================
// TDLeaf(λ) support — compiled only when TDLEAF=1
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

// FT/PSQT float shadow arrays (heap — OS lazy-paged, physical use ∝ active features)
// ft_weights_f32 / grad_ft_w: [FT_INPUTS × HALF_DIMS]  ~92 MB each
// psqt_weights_f32 / grad_psqt_w: [FT_INPUTS × PSQT_BKTS] ~720 KB each
// ft_dirty: [FT_INPUTS] — which feature rows received gradient this game
// ft_delta_f32 / psqt_delta_f32: accumulated FT/PSQT deltas since last file sync
static float    *ft_weights_f32   = nullptr;
static float    *psqt_weights_f32 = nullptr;
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
// Adam moment arrays — session-local (process memory only; not saved to .tdleaf.bin).
// All zeroed at session start in nnue_init_fp32_weights / nnue_init_zero_weights.
//
// FC layers + FT biases: true per-weight m and v (static, ~1.1 MB total).
// FT weights:            per-row v only (RMSProp; m omitted — per-dim m would
//                        require 92 MB heap and the per-row mean is too coarse
//                        to be directionally useful).  Heap, ~88 KB.
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

static float    *v_ft_row  = nullptr;  // [NNUE_FT_INPUTS] — FT per-row second moment (~88 KB)
static float    *v_psqt_w  = nullptr;  // [NNUE_FT_INPUTS × PSQT_BKTS] — PSQT per-weight v (~720 KB)
static float    *m_psqt_w  = nullptr;  // [NNUE_FT_INPUTS × PSQT_BKTS] — PSQT per-weight m (~720 KB)

// Session-local Adam step counter.  Incremented once per nnue_apply_gradients() call.
// Reset alongside the v/m arrays so bias correction v/(1-β²^t) is always valid.
static uint32_t  t_adam    = 0;

// ---------------------------------------------------------------------------
// nnue_init_zero_weights — fresh-start FC/FT initialisation + classical PSQT
//
// === Weight means: all zero ===
// Non-zero means inherited from a trained network (Stockfish 15.1) are the
// endpoint of a fully-converged training run, not a useful prior for TDLeaf.
// They create a systematic directional bias in every neuron from game 1 that
// TDLeaf must first cancel before it can learn signal.  Zero mean (the core
// He/Kaiming principle) is the correct starting point.
//
// === FC0 std: He-adjusted for 1024 inputs at int8 scale ===
// FC0 receives 1024 SqrCReLU outputs in [0,127].  With ~30 active FT features
// per position and FT weights N(0, 44.4), accumulator dimensions sit near
// N(0, 243).  After the SqrCReLU pairing the FC0 inputs have:
//   E[x] ≈ 20,  E[x²] ≈ 2200  (mean of clamped(N(0,243), 0,127) squared pairs)
// With n_in=1024 inputs and zero-mean int8 weights of std σ_w:
//   std(raw >> 6) ≈ (1/64) × sqrt(1024 × σ_w² × E[x²]) ≈ 23.5 × σ_w
// Targeting saturation < 5% of active neurons (matching He's design intent):
//   σ_w = 3.0  →  std(raw>>6) ≈ 70,  P(sat above 127) ≈ 3%  ✓
//   σ_w = 8.4  →  std(raw>>6) ≈ 197, P(sat above 127) ≈ 24% ✗
// At σ_w=3 the He formula is effectively inapplicable (sqrt(2/1024)≈0.044 in
// FP32 maps to <0.1 int8 units at this quantization scale) — 3.0 is the
// smallest int8-representable std that achieves the He saturation target.
//
// === FC1 / FC2 / FT stds: SF15.1 measurements retained ===
// FC1 (30 inputs): low fan-in greatly reduces saturation sensitivity; sf15.1
//   std 18.3 remains appropriate, only the mean is zeroed.
// FC2 (32 inputs, output layer): no activation clamping; std 30 (already
//   reduced from measured 76.38 in prior code) retained; mean zeroed.
// FT (int16, 22528×1024): std 44.4 is calibrated to the ~30-feature summation
//   that fills the accumulator; unchanged.
//
// === Rejection sampling ===
// Int8 weights use truncated-Gaussian rejection sampling (discard and redraw
// if |w| > 127) rather than clipping, to avoid density spikes at the int8
// boundaries.  With the current stds the rejection rate is negligible.
//
// === PSQT ===
// Initialised from classical evaluator piece values + piece-square tables
// (score.h), with each of the 8 PSQT buckets mapped to an interpolated
// classical game stage.  See the PSQT block below for details.
//
// Both ft_weights (int16, inference) and ft_weights_f32 (float, backprop)
// are initialised together; nnue_apply_gradients keeps them in sync for
// dirty feature rows.
// ---------------------------------------------------------------------------
#define INIT_FT_W_MEAN    0.0f       // was -0.7119; zero mean (He principle)
#define INIT_FT_W_STD    44.4149f    // unchanged — int16, ~30 features/position
#define INIT_FC0_W_MEAN   0.0f       // was +0.2368; zero mean (He principle)
#define INIT_FC0_W_STD    3.0f       // was 8.4252; He-adjusted: ~3% sat vs ~24%
#define INIT_FC1_W_MEAN   0.0f       // was -1.0989; zero mean (He principle)
#define INIT_FC1_W_STD   18.3019f    // unchanged — 30 inputs, low sat risk
#define INIT_FC2_W_MEAN   0.0f       // was +1.0977; zero mean (He principle)
#define INIT_FC2_W_STD   30.0000f    // unchanged — output layer, no clamping

void nnue_init_zero_weights()
{
    // ---- FC layers: zero-mean weights (He principle); FC0 std He-adjusted; biases zero ----
    std::mt19937 rng(42);  // fixed seed for reproducibility
    // Truncated normal for int8 weights: reject samples outside [-127, 127]
    // rather than clipping, to avoid artificial density spikes at the boundaries.
    auto rnd_w = [&](float mean, float std) -> float {
        std::normal_distribution<float> d(mean, std);
        float v;
        do { v = d(rng); } while (v < -127.f || v > 127.f);
        return v;
    };

    for (int s = 0; s < NNUE_LAYER_STACKS; s++) {
        for (int i = 0; i < NNUE_L0_SIZE * NNUE_L0_INPUT; i++)
            l0_weights_f32[s][i] = rnd_w(INIT_FC0_W_MEAN, INIT_FC0_W_STD);
        for (int i = 0; i < NNUE_L0_SIZE; i++)
            l0_biases_f32[s][i]  = 0.0f;
        for (int i = 0; i < NNUE_L1_SIZE * NNUE_L1_PADDED; i++)
            l1_weights_f32[s][i] = rnd_w(INIT_FC1_W_MEAN, INIT_FC1_W_STD);
        for (int i = 0; i < NNUE_L1_SIZE; i++)
            l1_biases_f32[s][i]  = 0.0f;
        for (int i = 0; i < NNUE_L2_PADDED; i++)
            l2_weights_f32[s][i] = rnd_w(INIT_FC2_W_MEAN, INIT_FC2_W_STD);
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

    // ---- FT biases: zero init ----
    // FT weights already break symmetry across dimensions, so zero FT biases
    // are sufficient to get varied SqrCReLU activations from game 1.
    if (ft_biases)
        memset(ft_biases, 0, NNUE_HALF_DIMS * sizeof(int16_t));

    // ---- FT weights: random; PSQT: classical material + piece-square by game stage ----
    // Conversion: V = cp * 5776 / 100.  PSQT is side-to-move based:
    //   own pieces (pside==persp)  → +V  (positive contribution to stm score)
    //   opp pieces (pside!=persp)  → -V  (negative contribution to stm score)
    // Material values from score.h: value[] = {0,100,377,399,596,1197,0}.
    // King material = 0 (mate sentinel); king piece_sq positional bonuses ARE included.
    // Internal units: cp * 5776 / 100.
    //
    // The 8 PSQT buckets map to the 4 classical game stages (2 buckets per stage):
    //   bucket b → effective gstage = 2b → psq_stage = b/2 (capped at 3), rem = (2b)%4
    //   This matches the classical interpolation: ((4-rem)*stage[s] + rem*stage[s+1]) / 4
    //   Bucket 0: pure opening (stage 0)
    //   Buckets 1-2: opening→early MG transition
    //   Buckets 2-3: early MG
    //   Buckets 3-4: early→late MG transition
    //   Buckets 4-5: late MG
    //   Buckets 5-6: late MG→endgame transition
    //   Buckets 6-7: endgame (stage 3)
    //
    // Square indexing: piece_sq tables are indexed from black's POV (sq 0=a1).
    //   For white pieces, whitef[psq] rank-flips the index (same as score.cpp).
    //   Mirror-symmetric (ksq) pairs that share the same feature index fi always
    //   produce the same PSQ value because the piece_sq tables are file-symmetric.
    static const float PSQT_VAL[7] = {
        0.f,                        // EMPTY
        100.f * 5776.f / 100.f,     // PAWN   = 5776.0
        377.f * 5776.f / 100.f,     // KNIGHT = 21775.5
        399.f * 5776.f / 100.f,     // BISHOP = 23046.2
        596.f * 5776.f / 100.f,     // ROOK   = 34425.0
        1197.f * 5776.f / 100.f,    // QUEEN  = 69143.7
        0.f,                        // KING   = 0 (no material value; PSQ still applied)
    };
    static const float PSQ_SCALE = 5776.f / 100.f;  // cp → internal units

    if (ft_weights_f32) {
        size_t ft_sz   = (size_t)NNUE_FT_INPUTS * NNUE_HALF_DIMS;
        size_t psqt_sz = (size_t)NNUE_FT_INPUTS * NNUE_PSQT_BKTS;
        {
            std::normal_distribution<float> ft_w_dist(INIT_FT_W_MEAN, INIT_FT_W_STD);
            for (size_t i = 0; i < ft_sz; i++) {
                float v = ft_w_dist(rng);
                if (v < -32767.f) v = -32767.f;
                if (v >  32767.f) v =  32767.f;
                ft_weights_f32[i] = v;
                ft_weights[i]     = (int16_t)v;
            }
        }
        memset(psqt_weights_f32, 0, psqt_sz * sizeof(float));
        memset(ft_weights_cnt,   0, ft_sz   * sizeof(uint32_t));
        memset(psqt_weights_cnt, 0, psqt_sz * sizeof(uint32_t));
        memset(grad_ft_w,        0, ft_sz   * sizeof(float));
        memset(grad_psqt_w,      0, psqt_sz * sizeof(float));
        memset(ft_dirty,         0, NNUE_FT_INPUTS * sizeof(bool));

        // Adam heap arrays — allocate on first use; zero for fresh network.
        if (!v_ft_row)  v_ft_row  = new float[NNUE_FT_INPUTS]();
        if (!v_psqt_w)  v_psqt_w  = new float[psqt_sz]();
        if (!m_psqt_w)  m_psqt_w  = new float[psqt_sz]();
        memset(v_ft_row,  0, NNUE_FT_INPUTS * sizeof(float));
        memset(v_psqt_w,  0, psqt_sz * sizeof(float));
        memset(m_psqt_w,  0, psqt_sz * sizeof(float));

        // Enumerate all possible features and assign classical material + PSQ values.
        // Each of the 8 PSQT buckets gets a stage-interpolated positional bonus.
        // Conflicts between mirror-symmetric (ksq) pairs always write the same value
        // because piece_sq tables are symmetric across the file-mirror axis.
        for (int persp = 0; persp < 2; persp++) {
            for (int ksq = 0; ksq < 64; ksq++) {
                for (int psq = 0; psq < 64; psq++) {
                    for (int ptype = PAWN; ptype <= KING; ptype++) {
                        for (int pside = 0; pside < 2; pside++) {
                            int fi = halfkav2_feature(persp, ksq, psq, ptype, pside);
                            if (fi < 0 || fi >= NNUE_FT_INPUTS) continue;

                            float sign = (pside == persp) ? 1.f : -1.f;
                            float mat  = PSQT_VAL[ptype];

                            // Classical square: black indexes directly, white uses whitef[].
                            int sq_cl = (pside == WHITE) ? whitef[psq] : psq;

                            for (int b = 0; b < NNUE_PSQT_BKTS; b++) {
                                // Map bucket b to classical stage via effective gstage = 2b.
                                int g     = 2 * b;
                                int stage = g / 4;           // 0..3
                                int rem   = g % 4;           // 0..3 within-stage progress
                                int next  = (stage < 3) ? stage + 1 : 3;

                                // Interpolate between adjacent classical stages.
                                float psq_cp =
                                    ((4 - rem) * (float)piece_sq[stage][ptype][sq_cl]
                                   +      rem  * (float)piece_sq[next ][ptype][sq_cl]) / 4.f;

                                psqt_weights_f32[fi * NNUE_PSQT_BKTS + b] =
                                    sign * (mat + psq_cp * PSQ_SCALE);
                            }
                        }
                    }
                }
            }
        }
    }

    // Sync all int8/int16/int32 inference arrays from the zeroed float shadows.
    nnue_requantize_fc();
    nnue_zero_initialized = true;
    printf("NNUE TDLeaf: FC weights=N(0,sigma) He-adjusted (FC0 sigma=3); "
           "FT weights=N(0,44.4); biases=zero; "
           "PSQT=classical material+PSQ mapped to 8 game-stage buckets\n");
}

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
        v_ft_row  = new float[NNUE_FT_INPUTS]();
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
    // Zero delta accumulators — fresh session, no pending changes yet.
    memset(delta_l0_w,     0, sizeof(delta_l0_w));
    memset(delta_l0_b,     0, sizeof(delta_l0_b));
    memset(delta_l1_w,     0, sizeof(delta_l1_w));
    memset(delta_l1_b,     0, sizeof(delta_l1_b));
    memset(delta_l2_w,     0, sizeof(delta_l2_w));
    memset(delta_l2_b,     0, sizeof(delta_l2_b));
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
    if (v_ft_row)  memset(v_ft_row,  0, NNUE_FT_INPUTS * sizeof(float));
    if (v_psqt_w)  memset(v_psqt_w,  0, psqt_sz * sizeof(float));
    if (m_psqt_w)  memset(m_psqt_w,  0, psqt_sz * sizeof(float));
    t_adam = 0;

    printf("NNUE TDLeaf: FP32 weights initialised (%d stacks + FT/PSQT + FT biases)\n", NNUE_LAYER_STACKS);
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
void nnue_accumulate_gradients(const NNUEActivations &act, float grad_scale)
{
    int s = act.stack;

    // 7. ∂loss/∂positional = grad_scale  (d_score_cp/d_positional = 100/5776 absorbed)
    float g_pos = grad_scale;

    // 6. FC2 backward: g_pos → grad of fc2_raw = g_pos
    float g_fc2_raw = g_pos;
    // grad w.r.t. FC2 weights and bias
    for (int i = 0; i < NNUE_L2_PADDED; i++)
        grad_l2_w[s][i] += g_fc2_raw * act.fc2_in[i];
    grad_l2_b[s] += NNUE_FC_BIAS_LR_SCALE * g_fc2_raw;
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
        grad_l1_b[s][o] += NNUE_FC_BIAS_LR_SCALE * g_fc1_raw[o];
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
        grad_l0_b[s][o] += NNUE_FC_BIAS_LR_SCALE * g_fc0_raw[o];
    }

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
    //   g_acc[persp][j]     += g_l0_in[p*512+j] * clamp(a[j+512],0,127) / 128  (if a[j]>0)
    //   g_acc[persp][j+512] += g_l0_in[p*512+j] * clamp(a[j],    0,127) / 128  (if a[j+512]>0)
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
            if (clo > 0.0f) g_a[j]       += g * chi;
            if (chi > 0.0f) g_a[j + 512] += g * clo;
        }
    }

    // FT weight gradient: for each active feature fi in perspective p:
    //   grad_ft_w[fi × HALF_DIMS + d] += NNUE_FT_LR_SCALE × g_acc[p][d]
    // PSQT gradient (only bucket `stack` is used by this position):
    //   psqt_diff = psqt[stm][stack] - psqt[opp][stack]
    //   ∂score_cp/∂psqt_diff = cp_factor/2 (half of the cp_factor used for positional)
    //   g_psqt_diff = grad_scale × 0.5  (grad_scale already includes cp_factor for positional)
    //   grad_psqt_w[fi × PSQT_BKTS + stack] += NNUE_PSQT_LR_SCALE × g_psqt_diff × (+1 for stm, -1 for opp)
    //   NNUE_PSQT_LR_SCALE is large (~10000) because grad_scale is tiny (|grad_scale| ≈ 2e-4)
    //   due to the 1/K and cp_factor terms in sig_grad, while psqt_f32 is at int32 scale (~5776/pawn).
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
                gfw[d] += NNUE_FT_LR_SCALE * g_a[d];
            grad_psqt_w[fi * NNUE_PSQT_BKTS + s] +=
                NNUE_PSQT_LR_SCALE * g_psqt_diff * psqt_sign;
        }
    }

    // FT bias gradient: ∂loss/∂ft_biases[d] = Σ_persp g_acc[persp][d]
    // Both perspectives share the same bias vector, so gradients sum across them.
    for (int d = 0; d < NNUE_HALF_DIMS; d++)
        grad_ft_bias[d] += NNUE_FT_BIAS_LR_SCALE * (g_acc[0][d] + g_acc[1][d]);
}

// ---------------------------------------------------------------------------
// nnue_apply_gradients — update FP32 weights from accumulators, increment counts,
//                        then zero the accumulators.
// Only weights that received a non-zero gradient this game are updated / counted.
// Each update is clamped to ±TDLEAF_MAX_UPDATE_FRAC × max(|w|, 1.0) so no single
// game can move any weight by more than that fraction of its current magnitude.
// ---------------------------------------------------------------------------
void nnue_apply_gradients()
{
    // Adam with per-weight LR decay.
    // Set TDLEAF_ADAM_LR0 = 0.0 to fall back to plain gradient descent.
    const bool use_adam = (TDLEAF_ADAM_LR0 > 0.0f);

    if (use_adam) t_adam++;

    // Bias-correction denominators — hoisted outside all per-weight loops.
    const float bc1 = use_adam ? (1.0f - powf(TDLEAF_ADAM_BETA1, (float)t_adam)) : 1.0f;
    const float bc2 = use_adam ? (1.0f - powf(TDLEAF_ADAM_BETA2, (float)t_adam)) : 1.0f;

    // Full Adam step for FC layers and FT biases.
    // When use_adam=false this degenerates to plain gradient descent (returns g).
    // LR decays as lr(cnt) = LR0 / (1 + cnt/C), so converged weights move less each game.
    auto do_step = [&](float g, float &m, float &v, uint32_t cnt) -> float {
        if (!use_adam) return g;
        m = TDLEAF_ADAM_BETA1 * m + (1.0f - TDLEAF_ADAM_BETA1) * g;
        v = TDLEAF_ADAM_BETA2 * v + (1.0f - TDLEAF_ADAM_BETA2) * g * g;
        float m_hat = m / bc1;
        float v_hat = v / bc2;
        float lr    = TDLEAF_ADAM_LR0 / (1.0f + (float)cnt / TDLEAF_ADAM_C);
        return lr * m_hat / (sqrtf(v_hat) + TDLEAF_ADAM_EPS);
    };
    // PSQT Adam step — same algorithm but uses TDLEAF_ADAM_PSQT_LR0.
    // PSQT weights are at int32 scale (std ~36,000) while FC weights are int8 scale (std ~3-50).
    // Adam normalises gradient magnitude, so only LR0 sets the per-step size in weight-space.
    // PSQT_LR0 must be ~1000× larger than FC LR0 to achieve comparable fractional updates.
    auto do_step_psqt = [&](float g, float &m, float &v, uint32_t cnt) -> float {
        if (!use_adam) return g;
        m = TDLEAF_ADAM_BETA1 * m + (1.0f - TDLEAF_ADAM_BETA1) * g;
        v = TDLEAF_ADAM_BETA2 * v + (1.0f - TDLEAF_ADAM_BETA2) * g * g;
        float m_hat = m / bc1;
        float v_hat = v / bc2;
        float lr    = TDLEAF_ADAM_PSQT_LR0 / (1.0f + (float)cnt / TDLEAF_ADAM_C);
        return lr * m_hat / (sqrtf(v_hat) + TDLEAF_ADAM_EPS);
    };

    for (int s = 0; s < NNUE_LAYER_STACKS; s++) {
        for (int i = 0; i < NNUE_L0_SIZE * NNUE_L0_INPUT; i++) {
            if (grad_l0_w[s][i] != 0.0f) {
                float dw = do_step(grad_l0_w[s][i], m_l0_w[s][i], v_l0_w[s][i], l0_weights_cnt[s][i]);
                l0_weights_f32[s][i] -= dw;  delta_l0_w[s][i] -= dw;
                // Clamp float shadow to int8 range: prevents zombie weights where the float
                // shadow drifts beyond ±127 while the requantised inference value is stuck.
                if (l0_weights_f32[s][i] >  127.0f) l0_weights_f32[s][i] =  127.0f;
                if (l0_weights_f32[s][i] < -127.0f) l0_weights_f32[s][i] = -127.0f;
                l0_weights_cnt[s][i]++;
            }
        }
        for (int i = 0; i < NNUE_L0_SIZE; i++) {
            if (grad_l0_b[s][i] != 0.0f) {
                float dw = do_step(grad_l0_b[s][i], m_l0_b[s][i], v_l0_b[s][i], l0_biases_cnt[s][i]);
                l0_biases_f32[s][i] -= dw;  delta_l0_b[s][i] -= dw;
                l0_biases_cnt[s][i]++;
            }
        }
        for (int i = 0; i < NNUE_L1_SIZE * NNUE_L1_PADDED; i++) {
            if (grad_l1_w[s][i] != 0.0f) {
                float dw = do_step(grad_l1_w[s][i], m_l1_w[s][i], v_l1_w[s][i], l1_weights_cnt[s][i]);
                l1_weights_f32[s][i] -= dw;  delta_l1_w[s][i] -= dw;
                // Clamp float shadow to int8 range (same reason as FC0).
                if (l1_weights_f32[s][i] >  127.0f) l1_weights_f32[s][i] =  127.0f;
                if (l1_weights_f32[s][i] < -127.0f) l1_weights_f32[s][i] = -127.0f;
                l1_weights_cnt[s][i]++;
            }
        }
        for (int i = 0; i < NNUE_L1_SIZE; i++) {
            if (grad_l1_b[s][i] != 0.0f) {
                float dw = do_step(grad_l1_b[s][i], m_l1_b[s][i], v_l1_b[s][i], l1_biases_cnt[s][i]);
                l1_biases_f32[s][i] -= dw;  delta_l1_b[s][i] -= dw;
                l1_biases_cnt[s][i]++;
            }
        }
        for (int i = 0; i < NNUE_L2_PADDED; i++) {
            if (grad_l2_w[s][i] != 0.0f) {
                float dw = do_step(grad_l2_w[s][i], m_l2_w[s][i], v_l2_w[s][i], l2_weights_cnt[s][i]);
                l2_weights_f32[s][i] -= dw;  delta_l2_w[s][i] -= dw;
                l2_weights_cnt[s][i]++;
            }
        }
        if (grad_l2_b[s] != 0.0f) {
            float dw = do_step(grad_l2_b[s], m_l2_b[s], v_l2_b[s], l2_bias_cnt[s]);
            l2_bias_f32[s] -= dw;  delta_l2_b[s] -= dw;
            l2_bias_cnt[s]++;
        }
    }
    memset(grad_l0_w, 0, sizeof(grad_l0_w));
    memset(grad_l0_b, 0, sizeof(grad_l0_b));
    memset(grad_l1_w, 0, sizeof(grad_l1_w));
    memset(grad_l1_b, 0, sizeof(grad_l1_b));
    memset(grad_l2_w, 0, sizeof(grad_l2_w));
    memset(grad_l2_b, 0, sizeof(grad_l2_b));

    // FT/PSQT: only iterate over dirty feature rows (sparse update).
    if (ft_dirty) {
        for (int fi = 0; fi < NNUE_FT_INPUTS; fi++) {
            if (!ft_dirty[fi]) continue;
            // FT weights — RMSProp with per-row v (no m; per-row mean is too coarse
            // to carry directional first-moment signal).  Per-weight LR decay via cnt.
            float    *fw  = ft_weights_f32 + (size_t)fi * NNUE_HALF_DIMS;
            float    *gw  = grad_ft_w      + (size_t)fi * NNUE_HALF_DIMS;
            uint32_t *cnt = ft_weights_cnt + (size_t)fi * NNUE_HALF_DIMS;
            float    *fd  = ft_delta_f32 ? ft_delta_f32 + (size_t)fi * NNUE_HALF_DIMS : nullptr;
            if (use_adam && v_ft_row) {
                // Accumulate row mean g² over non-zero dims; update per-row v.
                float g2sum = 0.0f;
                int   n_nz  = 0;
                for (int d = 0; d < NNUE_HALF_DIMS; d++) {
                    if (gw[d] != 0.0f) { g2sum += gw[d] * gw[d]; n_nz++; }
                }
                if (n_nz > 0) {
                    v_ft_row[fi] = TDLEAF_ADAM_BETA2 * v_ft_row[fi]
                                 + (1.0f - TDLEAF_ADAM_BETA2) * (g2sum / (float)n_nz);
                    float sv = sqrtf(v_ft_row[fi] / bc2) + TDLEAF_ADAM_EPS;
                    for (int d = 0; d < NNUE_HALF_DIMS; d++) {
                        if (gw[d] != 0.0f) {
                            float lr = TDLEAF_ADAM_LR0 / (1.0f + (float)cnt[d] / TDLEAF_ADAM_C);
                            float dw = lr * gw[d] / sv;
                            fw[d] -= dw;  if (fd) fd[d] -= dw;
                            cnt[d]++;
                            gw[d] = 0.0f;
                        }
                    }
                }
            } else {
                for (int d = 0; d < NNUE_HALF_DIMS; d++) {
                    if (gw[d] != 0.0f) {
                        fw[d] -= gw[d];  if (fd) fd[d] -= gw[d];
                        cnt[d]++;
                        gw[d] = 0.0f;
                    }
                }
            }
            // PSQT weights — full Adam per-weight (only 8 buckets per row).
            float    *pw   = psqt_weights_f32 + (size_t)fi * NNUE_PSQT_BKTS;
            float    *gpw  = grad_psqt_w      + (size_t)fi * NNUE_PSQT_BKTS;
            uint32_t *pcnt = psqt_weights_cnt + (size_t)fi * NNUE_PSQT_BKTS;
            float    *pd   = psqt_delta_f32 ? psqt_delta_f32 + (size_t)fi * NNUE_PSQT_BKTS : nullptr;
            for (int b = 0; b < NNUE_PSQT_BKTS; b++) {
                if (gpw[b] != 0.0f) {
                    float dw;
                    if (use_adam && m_psqt_w && v_psqt_w) {
                        size_t vi = (size_t)fi * NNUE_PSQT_BKTS + b;
                        dw = do_step_psqt(gpw[b], m_psqt_w[vi], v_psqt_w[vi], pcnt[b]);
                    } else {
                        dw = gpw[b];
                    }
                    pw[b] -= dw;  if (pd) pd[b] -= dw;
                    pcnt[b]++;
                    gpw[b] = 0.0f;
                }
            }
            ft_dirty[fi] = false;
        }
    }

    // FT bias update — full Adam per-dimension.
    for (int d = 0; d < NNUE_HALF_DIMS; d++) {
        if (grad_ft_bias[d] == 0.0f) continue;
        float dw = do_step(grad_ft_bias[d], m_ft_bias[d], v_ft_bias[d], ft_bias_cnt[d]);
        ft_biases_f32[d] -= dw;
        ft_bias_delta[d] -= dw;
        ft_bias_cnt[d]++;
        grad_ft_bias[d] = 0.0f;
        ft_biases[d] = (int16_t)std::max(-32767.0f,
                                std::min( 32767.0f, roundf(ft_biases_f32[d])));
    }
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
// Version 4 layout (current):
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
// Version 4 additions (current):
//   FT bias section (appended after sparse FT/PSQT rows):
//     float32[HALF_DIMS] × TDLEAF_SCALE: FT bias values         (4096 B)
//     uint32_t[HALF_DIMS]: FT bias update counts                 (4096 B)
//
// Version 3 (legacy): FC + sparse FT/PSQT, no FT bias section.  Still readable.
// Version 2 (legacy): FC block only, no FT/PSQT section.  Still readable.
// Version 1 (legacy): int32 biases + int8 weights, no counts.  Still readable.
// ---------------------------------------------------------------------------
static const float    TDLEAF_SCALE   = 128.0f;
static const uint32_t TDLEAF_MAGIC   = 0x544D4C46u; // "TMLF"
static const uint32_t TDLEAF_VERSION = 4u;

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

    // ---- Re-read the current file and merge our deltas on top ----------
    // This picks up any changes written by other concurrent Leaf instances
    // since we last synced.  After merge, float shadows = file + our_delta.
    FILE *cur = fopen(path, "rb");
    if (cur) {
        uint32_t magic = 0, version = 0;
        bool ok = (fread(&magic, 4, 1, cur) == 1 &&
                   fread(&version, 4, 1, cur) == 1 &&
                   magic == TDLEAF_MAGIC &&
                   (version == TDLEAF_VERSION || version == 3u || version == 2u));
        if (ok) {
            // FC section: float32 × TDLEAF_SCALE per weight, then uint32 counts.
            // Merge: shadow = file_value + our_delta; count = max(file, ours).
            for (int s = 0; s < NNUE_LAYER_STACKS && ok; s++) {
                auto merge_f = [&](float *shadow, float *delta, uint32_t *cnt, int n) -> bool {
                    for (int i = 0; i < n; i++) {
                        float raw;
                        if (fread(&raw, sizeof(float), 1, cur) != 1) return false;
                        shadow[i] = raw / TDLEAF_SCALE + delta[i];
                        delta[i]  = 0.0f;
                    }
                    return true;
                };
                auto merge_cnt = [&](uint32_t *cnt, int n) -> bool {
                    for (int i = 0; i < n; i++) {
                        uint32_t fc;
                        if (fread(&fc, sizeof(uint32_t), 1, cur) != 1) return false;
                        if (fc > cnt[i]) cnt[i] = fc;
                    }
                    return true;
                };
                ok = merge_f(l0_biases_f32[s],  delta_l0_b[s], l0_biases_cnt[s],  NNUE_L0_SIZE)
                  && merge_cnt(l0_biases_cnt[s],  NNUE_L0_SIZE)
                  && merge_f(l0_weights_f32[s], delta_l0_w[s], l0_weights_cnt[s], NNUE_L0_SIZE * NNUE_L0_INPUT)
                  && merge_cnt(l0_weights_cnt[s], NNUE_L0_SIZE * NNUE_L0_INPUT)
                  && merge_f(l1_biases_f32[s],  delta_l1_b[s], l1_biases_cnt[s],  NNUE_L1_SIZE)
                  && merge_cnt(l1_biases_cnt[s],  NNUE_L1_SIZE)
                  && merge_f(l1_weights_f32[s], delta_l1_w[s], l1_weights_cnt[s], NNUE_L1_SIZE * NNUE_L1_PADDED)
                  && merge_cnt(l1_weights_cnt[s], NNUE_L1_SIZE * NNUE_L1_PADDED)
                  && merge_f(&l2_bias_f32[s],    &delta_l2_b[s], &l2_bias_cnt[s],   1)
                  && merge_cnt(&l2_bias_cnt[s],   1)
                  && merge_f(l2_weights_f32[s], delta_l2_w[s], l2_weights_cnt[s], NNUE_L2_PADDED)
                  && merge_cnt(l2_weights_cnt[s], NNUE_L2_PADDED);
            }
            // FT/PSQT sparse section (v3/v4).
            if (ok && (version == TDLEAF_VERSION || version == 3u) && ft_weights_f32) {
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
                        // Merge PSQT.
                        float    *pw  = psqt_weights_f32 + (size_t)fi * NNUE_PSQT_BKTS;
                        uint32_t *pc  = psqt_weights_cnt + (size_t)fi * NNUE_PSQT_BKTS;
                        float    *pd  = psqt_delta_f32   ? psqt_delta_f32 + (size_t)fi * NNUE_PSQT_BKTS : nullptr;
                        for (int b = 0; b < NNUE_PSQT_BKTS; b++) {
                            pw[b] = tmp_p[b] / TDLEAF_SCALE + (pd ? pd[b] : 0.0f);
                            if (pd) pd[b] = 0.0f;
                            if (tmp_pc[b] > pc[b]) pc[b] = tmp_pc[b];
                        }
                    }
                }
                // FT bias section (v4 only).
                if (version == TDLEAF_VERSION) {
                    float tmp_b[NNUE_HALF_DIMS];
                    uint32_t tmp_bc[NNUE_HALF_DIMS];
                    if (fread(tmp_b,  sizeof(float),    NNUE_HALF_DIMS, cur) == (size_t)NNUE_HALF_DIMS &&
                        fread(tmp_bc, sizeof(uint32_t), NNUE_HALF_DIMS, cur) == (size_t)NNUE_HALF_DIMS) {
                        for (int d = 0; d < NNUE_HALF_DIMS; d++) {
                            ft_biases_f32[d] = tmp_b[d] / TDLEAF_SCALE + ft_bias_delta[d];
                            ft_bias_delta[d] = 0.0f;
                            if (tmp_bc[d] > ft_bias_cnt[d]) ft_bias_cnt[d] = tmp_bc[d];
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
        }
    } else {
        // No existing file — first write.  Clear deltas (incorporated in shadow directly).
        memset(delta_l0_w, 0, sizeof(delta_l0_w));
        memset(delta_l0_b, 0, sizeof(delta_l0_b));
        memset(delta_l1_w, 0, sizeof(delta_l1_w));
        memset(delta_l1_b, 0, sizeof(delta_l1_b));
        memset(delta_l2_w, 0, sizeof(delta_l2_w));
        memset(delta_l2_b, 0, sizeof(delta_l2_b));
        if (ft_delta_f32)   memset(ft_delta_f32,   0, (size_t)NNUE_FT_INPUTS * NNUE_HALF_DIMS  * sizeof(float));
        if (psqt_delta_f32) memset(psqt_delta_f32, 0, (size_t)NNUE_FT_INPUTS * NNUE_PSQT_BKTS * sizeof(float));
        memset(ft_bias_delta, 0, sizeof(ft_bias_delta));
    }

    // ---- Write merged content to a temp file, then atomically rename ----
    char tmp_path[FILENAME_MAX];
    snprintf(tmp_path, sizeof(tmp_path), "%s.tmp", path);
    FILE *f = fopen(tmp_path, "wb");
    if (!f) {
        fprintf(stderr, "TDLeaf: cannot write temp file %s\n", tmp_path);
        tdleaf_release_lock(lock_fd);
        return false;
    }
    fwrite(&TDLEAF_MAGIC,   4, 1, f);
    fwrite(&TDLEAF_VERSION, 4, 1, f);
    for (int s = 0; s < NNUE_LAYER_STACKS; s++) {
        // FC0 biases
        for (int i = 0; i < NNUE_L0_SIZE; i++) {
            float v = l0_biases_f32[s][i] * TDLEAF_SCALE;
            fwrite(&v, sizeof(float), 1, f);
        }
        fwrite(l0_biases_cnt[s], sizeof(uint32_t), NNUE_L0_SIZE, f);
        // FC0 weights (natural output-major layout: o * L0_INPUT + i)
        for (int i = 0; i < NNUE_L0_SIZE * NNUE_L0_INPUT; i++) {
            float v = l0_weights_f32[s][i] * TDLEAF_SCALE;
            fwrite(&v, sizeof(float), 1, f);
        }
        fwrite(l0_weights_cnt[s], sizeof(uint32_t), NNUE_L0_SIZE * NNUE_L0_INPUT, f);
        // FC1 biases
        for (int i = 0; i < NNUE_L1_SIZE; i++) {
            float v = l1_biases_f32[s][i] * TDLEAF_SCALE;
            fwrite(&v, sizeof(float), 1, f);
        }
        fwrite(l1_biases_cnt[s], sizeof(uint32_t), NNUE_L1_SIZE, f);
        // FC1 weights (natural output-major layout: o * L1_PADDED + i)
        for (int i = 0; i < NNUE_L1_SIZE * NNUE_L1_PADDED; i++) {
            float v = l1_weights_f32[s][i] * TDLEAF_SCALE;
            fwrite(&v, sizeof(float), 1, f);
        }
        fwrite(l1_weights_cnt[s], sizeof(uint32_t), NNUE_L1_SIZE * NNUE_L1_PADDED, f);
        // FC2 bias
        { float v = l2_bias_f32[s] * TDLEAF_SCALE; fwrite(&v, sizeof(float), 1, f); }
        fwrite(&l2_bias_cnt[s], sizeof(uint32_t), 1, f);
        // FC2 weights
        for (int i = 0; i < NNUE_L2_PADDED; i++) {
            float v = l2_weights_f32[s][i] * TDLEAF_SCALE;
            fwrite(&v, sizeof(float), 1, f);
        }
        fwrite(l2_weights_cnt[s], sizeof(uint32_t), NNUE_L2_PADDED, f);
    }

    // Sparse FT/PSQT section (v3).
    // A feature row is "dirty" if any ft or psqt update count is non-zero.
    uint32_t n_ft_rows = 0;
    if (ft_weights_f32) {
        for (int fi = 0; fi < NNUE_FT_INPUTS; fi++) {
            const uint32_t *wc = ft_weights_cnt   + (size_t)fi * NNUE_HALF_DIMS;
            const uint32_t *pc = psqt_weights_cnt + (size_t)fi * NNUE_PSQT_BKTS;
            bool dirty = false;
            for (int d = 0; d < NNUE_HALF_DIMS && !dirty; d++) dirty = (wc[d] != 0);
            for (int b = 0; b < NNUE_PSQT_BKTS && !dirty; b++) dirty = (pc[b] != 0);
            if (dirty) n_ft_rows++;
        }
    }
    fwrite(&n_ft_rows, sizeof(uint32_t), 1, f);

    if (ft_weights_f32) {
        float tmp_w[NNUE_HALF_DIMS];
        float tmp_p[NNUE_PSQT_BKTS];
        for (int fi = 0; fi < NNUE_FT_INPUTS; fi++) {
            const uint32_t *wc = ft_weights_cnt   + (size_t)fi * NNUE_HALF_DIMS;
            const uint32_t *pc = psqt_weights_cnt + (size_t)fi * NNUE_PSQT_BKTS;
            bool dirty = false;
            for (int d = 0; d < NNUE_HALF_DIMS && !dirty; d++) dirty = (wc[d] != 0);
            for (int b = 0; b < NNUE_PSQT_BKTS && !dirty; b++) dirty = (pc[b] != 0);
            if (!dirty) continue;

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

    // FT bias section (v4): float32[HALF_DIMS] × TDLEAF_SCALE + uint32_t[HALF_DIMS] counts.
    {
        float tmp_b[NNUE_HALF_DIMS];
        for (int d = 0; d < NNUE_HALF_DIMS; d++) tmp_b[d] = ft_biases_f32[d] * TDLEAF_SCALE;
        fwrite(tmp_b,       sizeof(float),    NNUE_HALF_DIMS, f);
        fwrite(ft_bias_cnt, sizeof(uint32_t), NNUE_HALF_DIMS, f);
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

    // ---- Version 2 / 3 / 4: float32 × TDLEAF_SCALE + uint32 counts ----
    if (version != 2u && version != 3u && version != TDLEAF_VERSION) {
        fprintf(stderr, "TDLeaf: unsupported version %u in %s\n", version, path);
        fclose(f); tdleaf_release_lock(lock_fd); return false;
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

    // Sparse FT/PSQT section (v3/v4).
    int n_ft_loaded = 0;
    if (ok && (version == 3u || version == TDLEAF_VERSION) && ft_weights_f32) {
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

    // FT bias section (v4 only).
    if (ok && version == TDLEAF_VERSION) {
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
    if (ft_delta_f32)   memset(ft_delta_f32,   0, (size_t)NNUE_FT_INPUTS * NNUE_HALF_DIMS  * sizeof(float));
    if (psqt_delta_f32) memset(psqt_delta_f32, 0, (size_t)NNUE_FT_INPUTS * NNUE_PSQT_BKTS * sizeof(float));
    memset(grad_ft_bias,  0, sizeof(grad_ft_bias));
    memset(ft_bias_delta, 0, sizeof(ft_bias_delta));
    // ft_bias_cnt and ft_biases_f32 are populated from file in v4; leave them.
    // For v2/v3 files, ft_biases_f32 was already initialised by nnue_init_fp32_weights.
    nnue_requantize_fc();
    if (version == TDLEAF_VERSION)
        printf("TDLeaf: loaded v4 weights from %s (%d FT rows)\n", path, n_ft_loaded);
    else if (version == 3u)
        printf("TDLeaf: loaded v3 weights from %s (%d FT rows, will upgrade to v4 on next save)\n",
               path, n_ft_loaded);
    else
        printf("TDLeaf: loaded v2 FC weights from %s (will upgrade to v4 on next save)\n", path);
    return true;
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
