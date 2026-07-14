// nnue_io.cpp — NNUE file I/O: .nnue loading (file and embedded), LEB128
// compression, and nnue_write_nnue for exporting trained weights.
//
// Unity-build position: after nnue.cpp (needs weight arrays and nnue_available).
// ---------------------------------------------------------------------------
// MemStream — thin read abstraction over FILE* or in-memory buffer.
// Allows nnue_load and nnue_load_from_memory to share the same parser.
// ---------------------------------------------------------------------------
struct MemStream {
    FILE *fp;
    const uint8_t *mem;
    size_t mem_size;
    size_t mem_pos;
};

static size_t ms_read(MemStream *s, void *dst, size_t n) {
    if (s->fp) return fread(dst, 1, n, s->fp);
    size_t avail = (s->mem_pos < s->mem_size) ? s->mem_size - s->mem_pos : 0;
    size_t to_read = (n < avail) ? n : avail;
    memcpy(dst, s->mem + s->mem_pos, to_read);
    s->mem_pos += to_read;
    return to_read;
}

static bool ms_seek_cur(MemStream *s, long offset) {
    if (s->fp) return fseek(s->fp, offset, SEEK_CUR) == 0;
    long new_pos = (long)s->mem_pos + offset;
    if (new_pos < 0 || (size_t)new_pos > s->mem_size) return false;
    s->mem_pos = (size_t)new_pos;
    return true;
}

static void ms_close(MemStream *s) {
    if (s->fp) fclose(s->fp);
}

// ---------------------------------------------------------------------------
// LEB128 decompressor (used for FT biases, weights, and PSQT weights)
// ---------------------------------------------------------------------------
static bool read_leb128_i16(MemStream *s, int16_t *buf, size_t count)
{
    char magic[18] = {};
    if (ms_read(s, magic, 17) != 17) return false;

    if (memcmp(magic, "COMPRESSED_LEB128", 17) != 0) {
        ms_seek_cur(s, -17);
        return ms_read(s, buf, sizeof(int16_t) * count) == sizeof(int16_t) * count;
    }

    uint32_t nbytes;
    if (ms_read(s, &nbytes, 4) != 4) return false;

    unsigned char *cbuf = (unsigned char*)malloc(nbytes);
    if (!cbuf) return false;
    if (ms_read(s, cbuf, nbytes) != nbytes) { free(cbuf); return false; }

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

static bool read_leb128_i32(MemStream *s, int32_t *buf, size_t count)
{
    char magic[18] = {};
    if (ms_read(s, magic, 17) != 17) return false;

    if (memcmp(magic, "COMPRESSED_LEB128", 17) != 0) {
        ms_seek_cur(s, -17);
        return ms_read(s, buf, sizeof(int32_t) * count) == sizeof(int32_t) * count;
    }

    uint32_t nbytes;
    if (ms_read(s, &nbytes, 4) != 4) return false;

    unsigned char *cbuf = (unsigned char*)malloc(nbytes);
    if (!cbuf) return false;
    if (ms_read(s, cbuf, nbytes) != nbytes) { free(cbuf); return false; }

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

static uint32_t read_u32(MemStream *s) {
    uint32_t v = 0;
    (void)ms_read(s, &v, 4);
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
// nnue_load_stream — shared parser for file and memory paths
// ---------------------------------------------------------------------------
static bool nnue_load_stream(MemStream *s)
{
    // Header
    uint32_t version   = read_u32(s);
    uint32_t file_hash = read_u32(s);
    uint32_t desc_size = read_u32(s);
    printf("NNUE: version=0x%08X  hash=0x%08X  desc_size=%u\n",
           version, file_hash, desc_size);

    if (desc_size > 0) {
        char *desc = new char[desc_size + 1];
        size_t nr  = ms_read(s, desc, desc_size);
        desc[nr]   = '\0';
        printf("NNUE: architecture: %s\n", desc);
        delete[] desc;
        if (nr != desc_size) { ms_close(s); return false; }
    }

    // Feature Transformer
    uint32_t ft_hash = read_u32(s);
    printf("NNUE: ft_hash=0x%08X\n", ft_hash);

    nnue_alloc_arrays();  // no-op if already allocated

    printf("NNUE: reading FT biases [%d int16] ...\n", NNUE_HALF_DIMS);
    if (!read_leb128_i16(s, ft_biases, NNUE_HALF_DIMS)) {
        printf("NNUE: FT bias read failed\n"); ms_close(s); return false;
    }

    size_t ft_w = (size_t)NNUE_FT_INPUTS * NNUE_HALF_DIMS;
    printf("NNUE: reading FT weights [%zu int16] ...\n", ft_w);
    if (!read_leb128_i16(s, ft_weights, ft_w)) {
        printf("NNUE: FT weight read failed\n"); ms_close(s); return false;
    }

    // Fingerprint the loaded FT weights now, before any training modifies them.
    nnue_update_content_hash();
    printf("NNUE: content hash=0x%08X\n", nnue_content_hash);

    size_t psqt_w = (size_t)NNUE_FT_INPUTS * NNUE_PSQT_BKTS;
    printf("NNUE: reading PSQT weights [%zu int32] ...\n", psqt_w);
    if (!read_leb128_i32(s, psqt_weights, psqt_w)) {
        printf("NNUE: PSQT weight read failed\n"); ms_close(s); return false;
    }
    printf("NNUE: FT + PSQT loaded OK\n");

    // Network: 8 layer stacks.
    // Each stack begins with a 4-byte hash (no separate net_hash before the stacks).
    // Layout per stack:
    //   stack_hash(4) + FC0_bias(16*4) + FC0_wt(16*3072) +
    //   FC1_bias(32*4) + FC1_wt(32*32) + FC2_bias(4) + FC2_wt(32)
    //   = 4 + 64 + 49152 + 128 + 1024 + 4 + 32 = 50408 bytes
    printf("NNUE: reading %d layer stacks ...\n", NNUE_LAYER_STACKS);

    for (int st = 0; st < NNUE_LAYER_STACKS; st++) {
        uint32_t stack_hash = read_u32(s);
        nnue_stack_hashes[st] = stack_hash;  // saved for nnue_write_nnue()

        // FC0
        if (ms_read(s, l0_biases[st], sizeof(int32_t) * NNUE_L0_SIZE) != sizeof(int32_t) * NNUE_L0_SIZE)
            { printf("NNUE: stack %d FC0 bias read failed\n", st); ms_close(s); return false; }
        size_t fc0_w = (size_t)NNUE_L0_SIZE * NNUE_L0_INPUT;
        {
            // Read output-major [o * NNUE_L0_INPUT + i] into a temp buffer,
            // then rearrange into the vdotq-friendly layout.
            int8_t *tmp = new int8_t[fc0_w];
            if (ms_read(s, tmp, fc0_w) != fc0_w)
                { delete[] tmp; printf("NNUE: stack %d FC0 weight read failed\n", st); ms_close(s); return false; }
            for (int o = 0; o < NNUE_L0_SIZE; o++) {
                int ob = o / 4, k = o % 4;
                for (int i = 0; i < NNUE_L0_INPUT; i++) {
                    int ib = i / 4, j = i % 4;
                    l0_weights[st][ib * 64 + ob * 16 + k * 4 + j] =
                        tmp[o * NNUE_L0_INPUT + i];
                }
            }
            delete[] tmp;
        }

        // FC1
        if (ms_read(s, l1_biases[st], sizeof(int32_t) * NNUE_L1_SIZE) != sizeof(int32_t) * NNUE_L1_SIZE)
            { printf("NNUE: stack %d FC1 bias read failed\n", st); ms_close(s); return false; }
        size_t fc1_w = (size_t)NNUE_L1_SIZE * NNUE_L1_PADDED;
        {
            int8_t tmp[NNUE_L1_SIZE * NNUE_L1_PADDED];
            if (ms_read(s, tmp, fc1_w) != fc1_w)
                { printf("NNUE: stack %d FC1 weight read failed\n", st); ms_close(s); return false; }
#if defined(__ARM_FEATURE_DOTPROD) || defined(EPOCH_USE_AVX2)
            for (int o = 0; o < NNUE_L1_SIZE; o++) {
                int ob = o / 4, k = o % 4;
                for (int i = 0; i < NNUE_L1_PADDED; i++) {
                    int ib = i / 4, j = i % 4;
                    l1_weights[st][ib * 128 + ob * 16 + k * 4 + j] = tmp[o * NNUE_L1_PADDED + i];
                }
            }
#else
            memcpy(l1_weights[st], tmp, fc1_w);
#endif
        }

        // FC2 (output layer)
        if (ms_read(s, &out_biases[st], sizeof(int32_t)) != sizeof(int32_t))
            { printf("NNUE: stack %d FC2 bias read failed\n", st); ms_close(s); return false; }
        if (ms_read(s, out_weights[st], NNUE_L2_PADDED) != (size_t)NNUE_L2_PADDED)
            { printf("NNUE: stack %d FC2 weight read failed\n", st); ms_close(s); return false; }
    }

    ms_close(s);
    nnue_available = true;
    printf("NNUE: all %d stacks loaded OK\n", NNUE_LAYER_STACKS);
#if TDLEAF
    nnue_init_fp32_weights();
#endif
    return true;
}

// ---------------------------------------------------------------------------
// nnue_load — load from a file path
// ---------------------------------------------------------------------------
bool nnue_load(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        printf("NNUE: could not open %s\n", path);
        return false;
    }
    MemStream s = {};
    s.fp = f;
    bool ok = nnue_load_stream(&s);
    if (ok) strncpy(nnue_loaded_path, path, FILENAME_MAX - 1);
    return ok;
}

// ---------------------------------------------------------------------------
// nnue_load_from_memory — load from an in-memory buffer (for embedded nets)
// ---------------------------------------------------------------------------
bool nnue_load_from_memory(const uint8_t *data, size_t size)
{
    MemStream s = {};
    s.mem = data;
    s.mem_size = size;
    bool ok = nnue_load_stream(&s);
    if (ok) strncpy(nnue_loaded_path, "(embedded)", FILENAME_MAX - 1);
    return ok;
}

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
    if (nnue_zero_initialized || !orig_desc[0]) {
        const char *prior_desc;
        switch (nnue_init_prior_mode) {
            case NNUE_PRIOR_NOPRIOR:
                prior_desc = "PSQT=symmetric uniform 100 cp (own=+V,enemy=-V; "
                             "P=N=B=R=Q=100 cp; noprior)";
                break;
            case NNUE_PRIOR_CLASSICAL:
                prior_desc = "PSQT=classical material + 4-stage piece-square tables "
                             "(gstage-interpolated across 8 buckets)";
                break;
            case NNUE_PRIOR_MATERIAL:
            default:
                prior_desc = "PSQT=symmetric classical material (own=+V,enemy=-V; "
                             "P=100 N=380 B=400 R=600 Q=1200 cp)";
                break;
        }
        snprintf(new_desc, sizeof(new_desc), "Random init; %s", prior_desc);
    }
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
    // Write PSQT.  Under pure-PSQT the FP32 shadow IS the trained material
    // channel (all material lives in PSQT), so it is written directly with no
    // piece_val baking — the exported .nnue is fully self-contained.
    printf("NNUE: writing PSQT weights...\n");
    if (psqt_weights_f32 == nullptr) {
        // Non-TDLEAF build: float shadow never allocated.  Write the int32
        // PSQT array directly (always allocated by nnue_alloc_arrays).
        write_leb128_i32(dst, psqt_weights, (size_t)NNUE_FT_INPUTS * NNUE_PSQT_BKTS);
    } else {
        const size_t n_psqt = (size_t)NNUE_FT_INPUTS * NNUE_PSQT_BKTS;
        int32_t *psqt_i32 = new int32_t[n_psqt];
        for (size_t i = 0; i < n_psqt; i++)
            psqt_i32[i] = (int32_t)roundf(psqt_weights_f32[i]);
        write_leb128_i32(dst, psqt_i32, n_psqt);
        delete[] psqt_i32;
    }

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
