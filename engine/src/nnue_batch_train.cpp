// Leaf offline batch trainer — supervised training on quiet-position sets.
// Compiled only when NNUE=1 && TDLEAF=1 (included by Leaf.cc after tdleaf.cpp).
//
// Consumes TSV files produced by scripts/extract_quiet_positions.py
// (columns: fen  cp  result  ply  depth  gid; cp and result are WHITE POV)
// and trains all layers (FT / PSQT / FC / piece_val) with the same FP32
// gradient machinery, per-section Adam LRs, and gauge anchors (PSQT
// mean-centering, PAWN pin) as online TDLeaf.  The per-position target is
// the lambda blend
//
//     p_target = lambda * result + (1 - lambda) * sigmoid(cp_label / K)
//
// and the loss is squared error in probability space, (p_target - d)^2,
// matching the TD update form so the existing LR calibration carries over.
// Records with depth == 0 (leaf-dump rows — their cp is the net's own static
// eval at dump time, acting as a magnitude anchor) use their own outcome
// weight, --bt-leaf-lambda (default: same as --bt-lambda; 1.0 = outcome-only).
//
// Invocation (requires a NNUE=1 TDLEAF=1 build; loads the .nnue and
// .tdleaf.bin next to the binary exactly like a normal training session,
// trains, writes per-epoch snapshots, and exits):
//
//   ./Leaf_vbt --batch-train quiet_a.tsv[,quiet_b.tsv...] --bt-out prefix
//              [--bt-epochs N]   epochs over the training split   (default 3)
//              [--bt-lambda L]   outcome weight in the blend      (default 0.7)
//              [--bt-leaf-lambda L]  outcome weight for depth-0 (leaf) rows
//                                (default: same as --bt-lambda)
//              [--bt-K cp]       sigmoid temperature              (default 220)
//              [--bt-lr S]       LR scale on all category LRs     (default 0.25)
//              [--bt-batch N]    positions per Adam step          (default 512)
//              [--bt-val F]      validation fraction, BY GAME     (default 0.05)
//              [--bt-seed N]     shuffle/split seed               (default 42)
//              [--bt-max N]      cap on loaded positions, 0 = all (default 0)
//              [--bt-sync F]     shared .tdleaf.bin for multi-process
//                                data-parallel training (see below)
//              [--bt-sync-every N]  batches between syncs        (default 256)
//
// Per epoch: <prefix>_ep<N>.nnue and <prefix>_ep<N>.tdleaf.bin are written.
//
// Multi-process data parallelism (--bt-sync): shard the TSV across N trainer
// processes and point them all at the same --bt-sync file.  Every
// --bt-sync-every batches each process calls nnue_save_fc_weights() on it —
// the same POSIX-locked delta-merge protocol used by concurrent online
// training — then requantizes, pulling co-workers' accumulated updates into
// its own inference arrays.  Busy locks defer the sync (deltas are retained),
// exactly as online.  Give each process a distinct --bt-out prefix; the
// per-epoch .tdleaf.bin/.nnue snapshots are per-process views, while the
// --bt-sync file holds the merged state (a final sync runs at exit).

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>

// ---------------------------------------------------------------------------
// Packed training record — 32 bytes.  Board is stored as an occupancy
// bitboard (a1=0 … h8=63, same SQR() indexing as position.sq) plus one
// nibble per occupied square in ascending-square order, using the same
// piece encoding as position.sq: (side << 3) | ptype, side WHITE=1/BLACK=0.
// ---------------------------------------------------------------------------
struct BTRecord {
    uint64_t occ;
    uint8_t  nib[16];
    int16_t  cp;        // eval label, WHITE POV, centipawns
    uint8_t  result2;   // 2 × white result: 0 / 1 / 2
    uint8_t  wtm;       // 1 = White to move
    uint32_t gid;       // game id (validation split key)
    uint8_t  depth;     // search depth of the cp label; 0 = no search label
                        // (e.g. leaf-dump rows) → trained outcome-only
    uint8_t  pad[3];
};

// ---------------------------------------------------------------------------
// FEN board-field + stm parser → BTRecord board part.
// Returns false on malformed rows or plist-overflow positions (>9 of one
// piece type per side — position.plist rows hold 9 squares max).
// ---------------------------------------------------------------------------
static bool bt_parse_fen(const char *fen, BTRecord &r)
{
    int counts[2][7] = {{0}};
    r.occ = 0;
    memset(r.nib, 0, sizeof(r.nib));
    uint8_t codes[64];
    int rx = 0, ry = 7;
    const char *c = fen;
    for (; *c && *c != ' '; c++) {
        if (*c == '/') { ry--; rx = 0; continue; }
        if (*c >= '1' && *c <= '8') { rx += *c - '0'; continue; }
        int pt, side;
        switch (*c | 32) {   // lowercase
            case 'p': pt = PAWN;   break;
            case 'n': pt = KNIGHT; break;
            case 'b': pt = BISHOP; break;
            case 'r': pt = ROOK;   break;
            case 'q': pt = QUEEN;  break;
            case 'k': pt = KING;   break;
            default: return false;
        }
        side = (*c >= 'A' && *c <= 'Z') ? WHITE : BLACK;
        if (rx > 7 || ry < 0) return false;
        if (++counts[side][pt] > 9) return false;
        int s = SQR(rx, ry);
        codes[s] = (uint8_t)((side << 3) | pt);
        r.occ |= (1ULL << s);
        rx++;
    }
    if (counts[WHITE][KING] != 1 || counts[BLACK][KING] != 1) return false;
    // Nibbles in ascending-square order.
    uint64_t occ = r.occ;
    int k = 0;
    while (occ) {
        int s = __builtin_ctzll(occ);
        occ &= occ - 1;
        r.nib[k >> 1] |= (uint8_t)(codes[s] << ((k & 1) * 4));
        k++;
    }
    if (*c != ' ') return false;
    c++;
    if (*c == 'w') r.wtm = 1;
    else if (*c == 'b') r.wtm = 0;
    else return false;
    return true;
}

// Rebuild the minimal position fields the NNUE paths need (plist + wtm;
// nnue_init_accumulator, nnue_dense_piece_val, and halfkav2_feature all
// read only plist).
static void bt_decode(const BTRecord &r, position &pos)
{
    memset(pos.plist, 0, sizeof(pos.plist));
    pos.wtm = r.wtm;
    uint64_t occ = r.occ;
    int k = 0;
    while (occ) {
        int s = __builtin_ctzll(occ);
        occ &= occ - 1;
        int code = (r.nib[k >> 1] >> ((k & 1) * 4)) & 0xF;
        k++;
        int side = code >> 3, pt = code & 7;
        int cnt = ++pos.plist[side][pt][0];
        pos.plist[side][pt][cnt] = (int8_t)s;
    }
}

// ---------------------------------------------------------------------------
// TSV loader
// ---------------------------------------------------------------------------
static bool bt_load_file(const char *path, std::vector<BTRecord> &out,
                         uint32_t gid_base, uint32_t &gid_max, size_t max_records)
{
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "batch-train: cannot open %s\n", path); return false; }
    setvbuf(f, nullptr, _IOFBF, 4u << 20);
    char line[512];
    size_t rows = 0, skipped = 0;
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '#' || strncmp(line, "fen\t", 4) == 0) continue;
        if (max_records && out.size() >= max_records) break;
        // fen \t cp \t result \t ply \t depth \t gid
        char *tab1 = strchr(line, '\t');
        if (!tab1) { skipped++; continue; }
        *tab1 = '\0';
        char *p = tab1 + 1;
        long cp = strtol(p, &p, 10);
        if (*p != '\t') { skipped++; continue; }
        p++;
        uint8_t result2;
        if      (p[0] == '1')               { result2 = 2; }
        else if (p[0] == '0' && p[1] == '.'){ result2 = 1; }
        else if (p[0] == '0')               { result2 = 0; }
        else { skipped++; continue; }
        // skip to ply, then read depth and gid
        p = strchr(p, '\t'); if (!p) { skipped++; continue; }
        p = strchr(p + 1, '\t'); if (!p) { skipped++; continue; }   // past ply
        long depth = strtol(p + 1, &p, 10);
        if (*p != '\t') { skipped++; continue; }
        unsigned long gid = strtoul(p + 1, nullptr, 10);

        BTRecord r;
        if (!bt_parse_fen(line, r)) { skipped++; continue; }
        if (cp > 32000) cp = 32000;
        if (cp < -32000) cp = -32000;
        r.cp      = (int16_t)cp;
        r.result2 = result2;
        r.depth   = (uint8_t)((depth < 0) ? 0 : (depth > 255) ? 255 : depth);
        r.gid     = gid_base + (uint32_t)gid;
        if (r.gid > gid_max) gid_max = r.gid;
        out.push_back(r);
        rows++;
    }
    fclose(f);
    fprintf(stderr, "batch-train: %s — %zu positions loaded (%zu skipped)\n",
            path, rows, skipped);
    return true;
}

// ---------------------------------------------------------------------------
// Evaluate one record with the current weights.
// Fills act (forward activations + backprop fields) when train != nullptr.
// Returns d = sigmoid(white-POV score / K).
// ---------------------------------------------------------------------------
static float bt_eval_record(const BTRecord &r, float K, NNUEActivations *act_out)
{
    static position pos;   // single-threaded trainer; plist/wtm reset per call
    bt_decode(r, pos);

    static NNUEAccumulator acc;
    nnue_init_accumulator(acc, pos);

    int pc = 0;
    for (int sd = 0; sd < 2; sd++)
        for (int pt = PAWN; pt <= KING; pt++)
            pc += pos.plist[sd][pt][0];
    pc = (pc < 1) ? 1 : (pc > 32) ? 32 : pc;

    int score_stm = nnue_evaluate_acc_raw(acc.acc, acc.psqt, (int)pos.wtm, pc)
                  + nnue_dense_piece_val(pos, (int)pos.wtm, pc);
    float score_w = pos.wtm ? (float)score_stm : -(float)score_stm;
    float d = 1.0f / (1.0f + expf(-score_w / K));

    if (act_out) {
        NNUEActivations &act = *act_out;
        act.stack = (pc - 1) / 4;
        nnue_forward_fp32(acc.acc, acc.psqt, (bool)pos.wtm, act);
        memcpy(act.acc_raw[0], acc.acc[0], NNUE_HALF_DIMS * sizeof(int16_t));
        memcpy(act.acc_raw[1], acc.acc[1], NNUE_HALF_DIMS * sizeof(int16_t));
        // Active feature indices, both perspectives (mirrors tdleaf_record_ply).
        for (int persp = 0; persp < 2; persp++) {
            int ksq = pos.plist[persp][KING][1];
            act.n_ft[persp] = 0;
            for (int sd = 0; sd < 2; sd++)
                for (int pt = PAWN; pt <= KING; pt++)
                    for (int i = 1; i <= pos.plist[sd][pt][0]; i++) {
                        if (act.n_ft[persp] >= NNUE_MAX_FT_PER_PERSP) goto ft_done;
                        int fi = halfkav2_feature(persp, ksq, pos.plist[sd][pt][i], pt, sd);
                        if (fi >= 0) act.ft_idx[persp][act.n_ft[persp]++] = fi;
                    }
            ft_done:;
        }
        int stm_p = pos.wtm ? 1 : 0;
        for (int pt = PAWN; pt <= KING; pt++)
            act.piece_count_diff[pt - 1] = (int8_t)(pos.plist[stm_p][pt][0]
                                                   - pos.plist[stm_p ^ 1][pt][0]);
    }
    return d;
}

static inline float bt_target(const BTRecord &r, float lambda,
                              float leaf_lambda, float K)
{
    // depth 0 = no search label (leaf-dump rows: cp is the net's own static
    // eval at dump time, a magnitude anchor) → these get their own outcome
    // weight, --bt-leaf-lambda (default: --bt-lambda; 1.0 = outcome-only).
    float lam     = (r.depth == 0) ? leaf_lambda : lambda;
    float outcome = 0.5f * (float)r.result2;
    float ev = 1.0f / (1.0f + expf(-(float)r.cp / K));
    return lam * outcome + (1.0f - lam) * ev;
}

// ---------------------------------------------------------------------------
// Entry point — called from main() when --batch-train is present.
// ---------------------------------------------------------------------------
int nnue_batch_train(int argc, char *argv[])
{
    // ---- Options --------------------------------------------------------
    const char *files   = nullptr;
    const char *out_pfx = "bt";
    int   epochs   = 3;
    float lambda   = 0.7f;
    float K        = 220.0f;
    float lr_scale = 0.25f;
    int   batch    = 512;
    float val_frac = 0.05f;
    unsigned seed  = 42;
    size_t max_rec = 0;
    const char *sync_path = nullptr;
    int   sync_every = 256;
    float leaf_lambda = -1.0f;   // < 0 → follow --bt-lambda

    for (int i = 1; i < argc; i++) {
        auto next = [&](const char *flag) -> const char* {
            if (strcmp(argv[i], flag) == 0 && i + 1 < argc) return argv[++i];
            return nullptr;
        };
        const char *v;
        if ((v = next("--batch-train"))) files    = v;
        else if ((v = next("--bt-out")))     out_pfx  = v;
        else if ((v = next("--bt-epochs")))  epochs   = atoi(v);
        else if ((v = next("--bt-lambda")))  lambda   = (float)atof(v);
        else if ((v = next("--bt-K")))       K        = (float)atof(v);
        else if ((v = next("--bt-lr")))      lr_scale = (float)atof(v);
        else if ((v = next("--bt-batch")))   batch    = atoi(v);
        else if ((v = next("--bt-val")))     val_frac = (float)atof(v);
        else if ((v = next("--bt-seed")))    seed     = (unsigned)atoi(v);
        else if ((v = next("--bt-max")))     max_rec  = (size_t)atoll(v);
        else if ((v = next("--bt-sync")))    sync_path = v;
        else if ((v = next("--bt-sync-every"))) sync_every = atoi(v);
        else if ((v = next("--bt-leaf-lambda"))) leaf_lambda = (float)atof(v);
    }
    if (!files) { fprintf(stderr, "batch-train: no input files\n"); return 1; }
    if (batch < 1) batch = 1;
    if (leaf_lambda < 0.0f) leaf_lambda = lambda;

    fprintf(stderr, "batch-train: lambda=%.2f leaf_lambda=%.2f K=%.0f "
                    "lr_scale=%.3f batch=%d epochs=%d val=%.3f seed=%u\n",
            (double)lambda, (double)leaf_lambda, (double)K, (double)lr_scale,
            batch, epochs, (double)val_frac, seed);
    if (sync_path)
        fprintf(stderr, "batch-train: multi-process sync → %s every %d batches\n",
                sync_path, sync_every);

    // ---- Load data ------------------------------------------------------
    std::vector<BTRecord> recs;
    recs.reserve(max_rec ? max_rec : (1u << 25));
    {
        char buf[4096];
        strncpy(buf, files, sizeof(buf) - 1);
        buf[sizeof(buf) - 1] = '\0';
        uint32_t gid_base = 0, gid_max = 0;
        for (char *tok = strtok(buf, ","); tok; tok = strtok(nullptr, ",")) {
            if (!bt_load_file(tok, recs, gid_base, gid_max, max_rec)) return 1;
            gid_base = gid_max + 1;   // keep gids unique across files
        }
    }
    if (recs.empty()) { fprintf(stderr, "batch-train: no records\n"); return 1; }

    // ---- Train/validation split BY GAME ---------------------------------
    // Knuth-hash the gid so the split is stable across runs with the same data.
    auto is_val = [&](uint32_t gid) {
        return ((gid * 2654435761u) >> 8) % 10000 < (uint32_t)(val_frac * 10000.0f);
    };
    std::vector<uint32_t> train_idx, val_idx;
    train_idx.reserve(recs.size());
    for (uint32_t i = 0; i < (uint32_t)recs.size(); i++)
        (is_val(recs[i].gid) ? val_idx : train_idx).push_back(i);
    fprintf(stderr, "batch-train: %zu positions (%zu train / %zu val, split by game)\n",
            recs.size(), train_idx.size(), val_idx.size());

    // ---- Validation-loss helper -----------------------------------------
    auto val_loss = [&]() {
        double se_blend = 0.0, se_outcome = 0.0;
        for (uint32_t i : val_idx) {
            float d  = bt_eval_record(recs[i], K, nullptr);
            float tb = bt_target(recs[i], lambda, leaf_lambda, K);
            float to = 0.5f * (float)recs[i].result2;
            se_blend   += (double)(tb - d) * (tb - d);
            se_outcome += (double)(to - d) * (to - d);
        }
        double n = (double)std::max<size_t>(val_idx.size(), 1);
        fprintf(stderr, "  val MSE(blend)=%.6f  MSE(outcome)=%.6f  (n=%zu)\n",
                se_blend / n, se_outcome / n, val_idx.size());
        return se_blend / n;
    };

    fprintf(stderr, "batch-train: baseline (epoch 0)\n");
    val_loss();

    const float cp_factor = 100.0f / 5776.0f;
    std::mt19937 rng(seed);
    static NNUEActivations act;   // ~12 KB; single-threaded

    // Multi-process sync: merge our deltas into the shared file (or defer if
    // the lock is busy — deltas are retained, same as online training), then
    // requantize so co-workers' merged updates reach our inference arrays.
    int batches_since_sync = 0;
    auto do_sync = [&](const char *when) {
        if (!sync_path) return;
        if (!nnue_save_fc_weights(sync_path))
            fprintf(stderr, "batch-train: sync (%s) to %s FAILED\n", when, sync_path);
        nnue_requantize_fc();
        batches_since_sync = 0;
    };

    for (int ep = 1; ep <= epochs; ep++) {
        std::shuffle(train_idx.begin(), train_idx.end(), rng);
        double se = 0.0;
        int in_batch = 0;
        auto t0 = std::chrono::steady_clock::now();

        for (size_t n = 0; n < train_idx.size(); n++) {
            const BTRecord &r = recs[train_idx[n]];
            float d      = bt_eval_record(r, K, &act);
            float target = bt_target(r, lambda, leaf_lambda, K);
            float e      = target - d;
            se += (double)e * e;

            // Same descent-form gradient scale as tdleaf_accumulate_game:
            // wtm here is the position's stm (the "leaf" of a 0-ply PV).
            float sig_grad   = d * (1.0f - d) / K;
            float wtm_sign   = r.wtm ? -1.0f : 1.0f;
            float grad_scale = e * sig_grad * cp_factor * wtm_sign;
            if (grad_scale != 0.0f)
                nnue_accumulate_gradients(act, grad_scale, false);

            if (++in_batch >= batch) {
                nnue_mean_center_psqt_gradients();
                nnue_clip_gradients(TDLEAF_GRAD_CLIP_NORM);
                nnue_apply_gradients(lr_scale);
                nnue_requantize_fc();
                in_batch = 0;
                if (sync_path && ++batches_since_sync >= sync_every)
                    do_sync("periodic");
            }
            if ((n + 1) % 1000000 == 0) {
                auto el = std::chrono::duration<double>(
                              std::chrono::steady_clock::now() - t0).count();
                fprintf(stderr, "  epoch %d: %zuM positions  train MSE=%.6f  (%.0f pos/s)\n",
                        ep, (n + 1) / 1000000, se / (double)(n + 1), (n + 1) / el);
            }
        }
        if (in_batch > 0) {   // flush the tail batch
            nnue_mean_center_psqt_gradients();
            nnue_clip_gradients(TDLEAF_GRAD_CLIP_NORM);
            nnue_apply_gradients(lr_scale);
            nnue_requantize_fc();
        }

        // Merge before validation/snapshot so both reflect co-workers' work.
        do_sync("epoch-end");

        auto el = std::chrono::duration<double>(
                      std::chrono::steady_clock::now() - t0).count();
        fprintf(stderr, "batch-train: epoch %d done — train MSE=%.6f  (%.0fs, %.0f pos/s)\n",
                ep, se / (double)std::max<size_t>(train_idx.size(), 1), el,
                train_idx.size() / el);
        val_loss();

        // Per-epoch snapshots: .nnue for gauntlet binaries + .tdleaf.bin to
        // resume/continue (fresh path → clean standalone write, no merge).
        char path[FILENAME_MAX];
        snprintf(path, sizeof(path), "%s_ep%d.nnue", out_pfx, ep);
        if (!nnue_write_nnue(path))
            fprintf(stderr, "batch-train: failed to write %s\n", path);
        else
            fprintf(stderr, "batch-train: wrote %s\n", path);
        snprintf(path, sizeof(path), "%s_ep%d.tdleaf.bin", out_pfx, ep);
        if (!nnue_save_fc_weights(path))
            fprintf(stderr, "batch-train: failed to write %s\n", path);
        else
            fprintf(stderr, "batch-train: wrote %s\n", path);
    }

    return 0;
}
