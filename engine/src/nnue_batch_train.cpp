// Leaf offline batch trainer — supervised training on quiet-position sets.
// Compiled only when NNUE=1 && TDLEAF=1 (included by Leaf.cc after tdleaf.cpp).
//
// Consumes TSV files produced by scripts/extract_quiet_positions.py
// (columns: fen  cp  result  ply  depth  gid  [endply]; cp and result are
// WHITE POV; endply — the game's true final ply — is optional)
// and trains all layers (FT / PSQT / FC) with the same FP32 gradient
// machinery and per-section Adam LRs as online TDLeaf (pure-PSQT: the
// bucketed PSQT is the sole material channel, no gauge machinery).  The
// per-position target is the distance-decayed blend
//
//     p_target = w * result + (1 - w) * sigmoid(cp_label / K)
//     w        = lambda_eff * td_lambda^(N_game - ply)
//
// i.e. the outcome's credibility decays with distance from the game end
// (the TD(lambda) forward view: the terminal outcome carries weight
// lambda^(N-t) in the lambda-return, with the remainder on bootstrapped
// values — here approximated by the position's own eval).  The two weights
// always sum to 1, so targets stay calibrated at any decay.  N_game comes
// from the corpus's optional 7th column (endply, exact) or the per-gid max
// ply (fallback; short by the quiet-filtered game tail).  --bt-td-lambda
// defaults to TDLEAF_LAMBDA (tdleaf.h) so offline targets match the
// lambda-returns the online games were trained on; 1.0 = flat blend (the
// pre-decay behaviour).  The settled recipe (2026-07-05) is the pure
// lambda-return: --bt-lambda 1.0 (the default) so td_lambda supplies all
// the moderation and is the single knob of record; the lambda ceilings
// remain as dormant scale knobs (decoupled from decay shape, e.g. to
// renormalize across corpora with different ply-gap distributions).
// The loss is squared error in probability space,
// (p_target - d)^2, matching the TD update form so the existing LR
// calibration carries over.  Records with depth == 0 (leaf-dump rows —
// their cp is the net's own static eval at dump time, acting as a magnitude
// anchor) use their own ceiling, --bt-leaf-lambda (default: --bt-lambda).
//
// Invocation (requires a NNUE=1 TDLEAF=1 build; loads the .nnue and
// .tdleaf.bin next to the binary exactly like a normal training session,
// trains, writes per-epoch snapshots, and exits):
//
//   ./Leaf_vbt --batch-train quiet_a.tsv[,quiet_b.tsv...] --bt-out prefix
//              [--bt-epochs N]   epochs over the training split   (default 3)
//              [--bt-lambda L]   outcome-weight ceiling (root rows) (default 1.0)
//              [--bt-leaf-lambda L]  outcome-weight ceiling for depth-0 (leaf)
//                                rows              (default: same as --bt-lambda)
//              [--bt-td-lambda L]  result decay per ply from the game end
//                                (default TDLEAF_LAMBDA; 1.0 = flat blend)
//              [--bt-K cp]       sigmoid temperature              (default 220)
//              [--bt-lr S]       LR scale on all category LRs     (default 0.25)
//              [--bt-batch N]    positions per Adam step          (default 512)
//              [--bt-val F]      validation fraction, BY GAME     (default 0.05)
//              [--bt-seed N]     shuffle/split seed               (default 42)
//              [--bt-max N]      cap on loaded positions, 0 = all (default 0)
//              [--bt-rows R]     leaf | root | both: which corpus rows to
//                                train on, by the depth column (0 = leaf,
//                                >0 = root)                     (default both)
//              [--bt-threads N]  worker threads for gradient compute (default 1)
//              [--bt-clip-every N]  compute the L2 clip norm on every Nth batch
//                                only (default 64; 1 = every batch).  The norm
//                                scan is serial and touches every dirty FT/PSQT
//                                row; measured batch norms sit 12-19x under
//                                TDLEAF_GRAD_CLIP_NORM on self-play corpora, so
//                                sampling is outcome-identical in practice.
//                                Safety: if a sampled norm ever exceeds half
//                                the threshold, per-batch scanning is restored
//                                for the rest of the run.  Skipped batches
//                                still run the freeze-passthrough housekeeping.
//
// Per epoch: <prefix>_ep<N>.nnue and <prefix>_ep<N>.tdleaf.bin are written.
//
// Within-batch thread parallelism (--bt-threads): synchronous data parallelism
// in ONE process — mathematically identical to single-threaded training, up to
// float summation order.  Each batch's positions are split contiguously across
// N worker threads; every thread computes forward+backprop against the SAME
// frozen weights into its own gradient buffer (nnue_gradbuf_alloc).  The workers
// are then summed into the global buffer in thread-index order (deterministic
// for a fixed thread count) and a single Adam step + requantize runs on the
// merged gradient.  No optimizer sees a stale weight, so — unlike the removed
// multi-process --bt-sync sharding — there is no gradient staleness; the ONLY
// deviation from --bt-threads 1 is reduction summation order.  The per-batch
// Adam step (the bottleneck) is parallelized separately via
// nnue_apply_gradients_parallel.  Measured ~2.85x on 8 cores.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>

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
    uint32_t gid;       // game id (validation split key + per-game N lookup)
    uint8_t  depth;     // search depth of the cp label; 0 = no search label
                        // (leaf-dump rows) → weighted by --bt-leaf-lambda
    uint8_t  pad;
    uint16_t ply;       // 1-based ply of the position (result-decay distance)
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
// Row-type selection (--bt-rows): 0 = leaf rows only (depth == 0), 1 = root
// rows only (depth > 0), 2 = both (default).  Filtered at load so one
// archived corpus supports leaf/root/both sweeps without reassembly.  Note:
// on legacy corpora with no endply column, filtering slightly shortens the
// per-gid max-ply fallback for the result-decay distance (skipped rows don't
// register their ply); corpora with endply (all current ones) are exact.
static int    bt_rows_mode     = 2;
static size_t bt_rows_filtered = 0;

static bool bt_load_file(const char *path, std::vector<BTRecord> &out,
                         uint32_t gid_base, uint32_t &gid_max, size_t max_records,
                         std::vector<uint16_t> &gid_N, bool &out_game_ply_axis)
{
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "batch-train: cannot open %s\n", path); return false; }
    setvbuf(f, nullptr, _IOFBF, 4u << 20);
    char line[512];
    size_t rows = 0, skipped = 0;
    out_game_ply_axis = false;
    while (fgets(line, sizeof(line), f)) {
        // Corpus axis marker (game-ply λ^Δ era): the ply/endply columns are true
        // game-ply, so the result-decay is per game-ply.  Legacy corpora lack it
        // and use the old per-record-index axis.
        if (line[0] == '#') {
            if (strstr(line, "axis=game-ply")) out_game_ply_axis = true;
            continue;
        }
        if (strncmp(line, "fen\t", 4) == 0) continue;
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
        // read ply, depth, gid (+ optional 7th column: true final ply of the
        // game — corpora that carry it get an exact result-decay distance;
        // older corpora fall back to the per-gid max ply seen, see below)
        p = strchr(p, '\t'); if (!p) { skipped++; continue; }
        long ply = strtol(p + 1, &p, 10);
        if (*p != '\t') { skipped++; continue; }
        long depth = strtol(p + 1, &p, 10);
        if (*p != '\t') { skipped++; continue; }
        if (bt_rows_mode == 0 && depth != 0) { bt_rows_filtered++; continue; }
        if (bt_rows_mode == 1 && depth == 0) { bt_rows_filtered++; continue; }
        unsigned long gid = strtoul(p + 1, &p, 10);
        long endply = (*p == '\t') ? strtol(p + 1, nullptr, 10) : 0;

        BTRecord r;
        if (!bt_parse_fen(line, r)) { skipped++; continue; }
        if (cp > 32000) cp = 32000;
        if (cp < -32000) cp = -32000;
        r.cp      = (int16_t)cp;
        r.result2 = result2;
        r.depth   = (uint8_t)((depth < 0) ? 0 : (depth > 255) ? 255 : depth);
        r.gid     = gid_base + (uint32_t)gid;
        if (r.gid > gid_max) gid_max = r.gid;
        r.ply     = (uint16_t)((ply < 0) ? 0 : (ply > 65535) ? 65535 : ply);
        // Per-game final ply: exact from the endply column when present, else
        // the max ply seen in the corpus (short by the quiet-filtered tail).
        uint16_t np = (uint16_t)((endply > 65535) ? 65535
                                : (endply > r.ply) ? endply : r.ply);
        if (r.gid >= gid_N.size()) gid_N.resize(r.gid + 1, 0);
        if (np > gid_N[r.gid]) gid_N[r.gid] = np;
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
// pos/acc are caller-owned scratch (per worker thread — no statics, so the
// trainer is thread-safe: the weight arrays are read-only within a batch).
// Fills act_out (forward activations + backprop fields) when non-null.
// Returns d = sigmoid(white-POV score / K).
// ---------------------------------------------------------------------------
static float bt_eval_record(const BTRecord &r, float K, position &pos,
                            NNUEAccumulator &acc, NNUEActivations *act_out)
{
    bt_decode(r, pos);
    nnue_init_accumulator(acc, pos);

    int pc = 0;
    for (int sd = 0; sd < 2; sd++)
        for (int pt = PAWN; pt <= KING; pt++)
            pc += pos.plist[sd][pt][0];
    pc = (pc < 1) ? 1 : (pc > 32) ? 32 : pc;

    int score_stm = nnue_evaluate_acc_raw(acc.acc, acc.psqt, (int)pos.wtm, pc);
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
                              float leaf_lambda, float K, float decay)
{
    // Result weight w = lambda_eff * td_lambda^(N - ply): the outcome's
    // credibility decays with distance from the game end (the TD(lambda)
    // forward view); the freed weight returns to the eval bootstrap so the
    // two weights always sum to 1.  `decay` = td_lambda^(N - ply), 1.0 at
    // the final ply (and everywhere when --bt-td-lambda 1.0 = flat blend).
    // depth 0 = no search label (leaf-dump rows: cp is the net's own static
    // eval at dump time, a magnitude anchor) → these get their own ceiling,
    // --bt-leaf-lambda (default: --bt-lambda; 1.0 at td_lambda 1.0 =
    // outcome-only).
    float w       = ((r.depth == 0) ? leaf_lambda : lambda) * decay;
    float outcome = 0.5f * (float)r.result2;
    float ev = 1.0f / (1.0f + expf(-(float)r.cp / K));
    return w * outcome + (1.0f - w) * ev;
}

// ---------------------------------------------------------------------------
// Minimal fixed-size worker pool.  run(fn) invokes fn(tid) on T worker threads
// and blocks until all return — a barrier per phase.  No per-batch thread spawn
// (a full run is hundreds of thousands of batches).  T==1 runs fn(0) inline on
// the calling thread, so the single-thread path has zero threading overhead.
// ---------------------------------------------------------------------------
struct BTPool {
    int T;
    std::vector<std::thread> threads;
    std::mutex m;
    std::condition_variable cv_go, cv_done;
    std::function<void(int)> fn;
    int generation = 0;   // incremented per dispatch
    int done = 0;         // workers finished this generation
    bool stop = false;

    explicit BTPool(int t) : T(t) {
        for (int i = 1; i < T; i++)
            threads.emplace_back([this, i] { worker(i); });
    }
    ~BTPool() {
        {
            std::unique_lock<std::mutex> lk(m);
            stop = true;
            generation++;
            cv_go.notify_all();
        }
        for (auto &th : threads) th.join();
    }
    void worker(int tid) {
        int seen = 0;
        for (;;) {
            std::unique_lock<std::mutex> lk(m);
            cv_go.wait(lk, [&] { return generation != seen; });
            seen = generation;
            if (stop) return;
            lk.unlock();
            fn(tid);
            lk.lock();
            if (++done == T - 1) cv_done.notify_one();
        }
    }
    // Run fn(tid) for tid in [0, T); returns when all have finished.
    void run(const std::function<void(int)> &f) {
        if (T == 1) { f(0); return; }
        {
            std::unique_lock<std::mutex> lk(m);
            fn = f;
            done = 0;
            generation++;
            cv_go.notify_all();
        }
        fn(0);   // the calling thread is worker 0
        {
            std::unique_lock<std::mutex> lk(m);
            cv_done.wait(lk, [&] { return done == T - 1; });
        }
    }
};

// Per-thread scratch: private position/accumulator/activations + gradient buffer.
// se/se2 accumulate this thread's squared-error partials (se2 = outcome-MSE, val
// only).  Heap-allocated (NNUEGradBuf owns ~93 MB of FT/PSQT gradient arrays).
struct BTWorker {
    position         pos;
    NNUEAccumulator  acc;
    NNUEActivations  act;
    NNUEGradBuf     *gb = nullptr;
    double           se = 0.0, se2 = 0.0, nll = 0.0;
};

// ---------------------------------------------------------------------------
// Entry point — called from main() when --batch-train is present.
// ---------------------------------------------------------------------------
int nnue_batch_train(int argc, char *argv[])
{
    // ---- Options --------------------------------------------------------
    const char *files   = nullptr;
    const char *out_pfx = "bt";
    int   epochs   = 3;
    float lambda   = 1.0f;   // full lambda-return; td_lambda decay is the knob of record
    float K        = 220.0f;
    float lr_scale = 0.25f;
    int   batch    = 512;
    float val_frac = 0.05f;
    unsigned seed  = 42;
    size_t max_rec = 0;
    int   threads  = 1;          // worker threads for within-batch gradient compute
    int   clip_every = 64;       // L2 clip-norm scan stride (1 = every batch)
    float loss_gamma = 1.0f;     // sig_grad = (d(1-d))^gamma / K.  1=MSE, 0=CE, 0.5=focal
    float leaf_lambda = -1.0f;   // < 0 → follow --bt-lambda
    float td_lambda   = TDLEAF_LAMBDA;   // result-decay per game-ply (default sqrt(0.98))
    bool  td_lambda_explicit = false;    // true if --bt-td-lambda was passed

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
        else if ((v = next("--bt-threads"))) threads  = atoi(v);
        else if ((v = next("--bt-clip-every"))) clip_every = atoi(v);
        else if ((v = next("--bt-loss-gamma"))) loss_gamma = (float)atof(v);
        else if ((v = next("--bt-leaf-lambda"))) leaf_lambda = (float)atof(v);
        else if ((v = next("--bt-td-lambda")))   { td_lambda = (float)atof(v); td_lambda_explicit = true; }
        else if ((v = next("--bt-rows"))) {
            bt_rows_mode = (strcmp(v, "leaf") == 0) ? 0
                         : (strcmp(v, "root") == 0) ? 1
                         : (strcmp(v, "both") == 0) ? 2 : -1;
            if (bt_rows_mode < 0) {
                fprintf(stderr, "batch-train: --bt-rows must be leaf|root|both\n");
                return 1;
            }
        }
    }
    if (!files) { fprintf(stderr, "batch-train: no input files\n"); return 1; }
    if (batch < 1) batch = 1;
    if (threads < 1) threads = 1;
    if (threads > 16) threads = 16;
    if (clip_every < 1) clip_every = 1;
    if (loss_gamma < 0.0f) loss_gamma = 0.0f;
    if (leaf_lambda < 0.0f) leaf_lambda = lambda;
    if (td_lambda < 0.0f || td_lambda > 1.0f) {
        fprintf(stderr, "batch-train: --bt-td-lambda must be in [0,1]\n");
        return 1;
    }

    // ---- Load data ------------------------------------------------------
    std::vector<BTRecord> recs;
    std::vector<uint16_t> gid_N;   // per-game final ply (see bt_load_file)
    recs.reserve(max_rec ? max_rec : (1u << 25));
    int n_game_ply = 0, n_legacy = 0;   // corpus-axis tally across input files
    {
        char buf[4096];
        strncpy(buf, files, sizeof(buf) - 1);
        buf[sizeof(buf) - 1] = '\0';
        uint32_t gid_base = 0, gid_max = 0;
        for (char *tok = strtok(buf, ","); tok; tok = strtok(nullptr, ",")) {
            bool file_game_ply = false;
            if (!bt_load_file(tok, recs, gid_base, gid_max, max_rec, gid_N, file_game_ply))
                return 1;
            (file_game_ply ? n_game_ply : n_legacy)++;
            gid_base = gid_max + 1;   // keep gids unique across files
        }
    }
    if (recs.empty()) { fprintf(stderr, "batch-train: no records\n"); return 1; }

    // Result-decay axis.  New (game-ply) corpora carry the axis marker and use
    // td_lambda per game-ply.  Legacy corpora index ply by record and need the
    // squared decay (TDLEAF_LAMBDA² = 0.98) to reproduce the historical
    // per-own-move rate.  Refuse a mix — the two axes are not interchangeable.
    if (n_game_ply > 0 && n_legacy > 0) {
        fprintf(stderr, "batch-train: refusing to mix game-ply-axis corpora (marked "
                        "'# tdleaf-corpus axis=game-ply') with legacy record-index "
                        "corpora — regenerate the legacy set or train them separately\n");
        return 1;
    }
    bool corpus_game_ply = (n_game_ply > 0);
    if (!td_lambda_explicit && !corpus_game_ply)
        td_lambda = TDLEAF_LAMBDA * TDLEAF_LAMBDA;   // legacy record-index axis → 0.98/record

    fprintf(stderr, "batch-train: lambda=%.2f leaf_lambda=%.2f td_lambda=%.5f "
                    "K=%.0f lr_scale=%.3f batch=%d epochs=%d val=%.3f seed=%u "
                    "threads=%d clip_every=%d loss_gamma=%.2f  axis=%s  rows=%s\n",
            (double)lambda, (double)leaf_lambda, (double)td_lambda, (double)K,
            (double)lr_scale, batch, epochs, (double)val_frac, seed, threads,
            clip_every, (double)loss_gamma,
            corpus_game_ply ? "game-ply" : "record-index(legacy)",
            (bt_rows_mode == 0) ? "leaf" : (bt_rows_mode == 1) ? "root" : "both");
    if (bt_rows_mode != 2)
        fprintf(stderr, "batch-train: --bt-rows %s — %zu %s rows filtered at load\n",
                (bt_rows_mode == 0) ? "leaf" : "root", bt_rows_filtered,
                (bt_rows_mode == 0) ? "root" : "leaf");

    // ---- Result-decay lookup: decay(r) = td_lambda^(N_game - ply) --------
    // Precomputed per integer gap (gaps span 0..max game length, a few
    // hundred) so the epoch loop pays one table load, not a powf.
    int max_gap = 0;
    for (const BTRecord &r : recs) {
        int g = (int)gid_N[r.gid] - (int)r.ply;
        if (g > max_gap) max_gap = g;
    }
    std::vector<float> powtab(max_gap + 1);
    for (int g = 0; g <= max_gap; g++)
        powtab[g] = powf(td_lambda, (float)g);
    auto decay = [&](const BTRecord &r) {
        int g = (int)gid_N[r.gid] - (int)r.ply;
        return powtab[g < 0 ? 0 : g];
    };
    {
        double dsum = 0.0;
        for (const BTRecord &r : recs) dsum += (double)decay(r);
        fprintf(stderr, "batch-train: result decay td_lambda=%.3f — mean "
                        "decay %.3f over %zu positions (max gap %d)\n",
                (double)td_lambda, dsum / (double)recs.size(), recs.size(),
                max_gap);
    }

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

    const float cp_factor = 100.0f / 5776.0f;

    // ---- Worker pool + per-thread scratch/gradient buffers --------------
    // Workers accumulate into private NNUEGradBuf; the reduce phase sums them
    // into the global buffer (nnue_global_gradbuf) in thread order.
    BTPool pool(threads);
    std::vector<BTWorker*> workers(threads);
    for (int t = 0; t < threads; t++) {
        workers[t] = new BTWorker();
        workers[t]->gb = nnue_gradbuf_alloc();
    }

    // ---- Validation-loss helper (read-only → parallel) ------------------
    // Contiguous per-worker slices, partials summed in thread order.
    auto val_loss = [&]() {
        pool.run([&](int tid) {
            BTWorker &w = *workers[tid];
            w.se = w.se2 = w.nll = 0.0;
            size_t lo = (size_t)tid * val_idx.size() / threads;
            size_t hi = (size_t)(tid + 1) * val_idx.size() / threads;
            for (size_t k = lo; k < hi; k++) {
                uint32_t i = val_idx[k];
                float d  = bt_eval_record(recs[i], K, w.pos, w.acc, nullptr);
                float tb = bt_target(recs[i], lambda, leaf_lambda, K, decay(recs[i]));
                float to = 0.5f * (float)recs[i].result2;
                w.se  += (double)(tb - d) * (tb - d);
                w.se2 += (double)(to - d) * (to - d);
                // Soft-label cross-entropy against the blend target — reported
                // as a calibration diagnostic only (never fed to training).
                // Clamp d to [0.05,0.95] so log() can't overflow at the tails;
                // this touches the METRIC only, not the gradient.
                float dc = d < 0.05f ? 0.05f : (d > 0.95f ? 0.95f : d);
                w.nll += -((double)tb * log(dc) + (1.0 - (double)tb) * log(1.0 - dc));
            }
        });
        double se_blend = 0.0, se_outcome = 0.0, nll = 0.0;
        for (int t = 0; t < threads; t++) {
            se_blend += workers[t]->se; se_outcome += workers[t]->se2; nll += workers[t]->nll;
        }
        double n = (double)std::max<size_t>(val_idx.size(), 1);
        fprintf(stderr, "  val MSE(blend)=%.6f  MSE(outcome)=%.6f  NLL(blend)=%.6f  (n=%zu)\n",
                se_blend / n, se_outcome / n, nll / n, val_idx.size());
        return se_blend / n;
    };

    fprintf(stderr, "batch-train: baseline (epoch 0)\n");
    val_loss();

    std::mt19937 rng(seed);

    // ---- One batch: parallel compute → deterministic reduce → clear → serial
    // Adam step.  se_batch collects the batch's squared-error partials (thread
    // order).  Phase timers accumulate per epoch.
    double t_compute = 0, t_reduce = 0, t_clear = 0, t_tail = 0;
    size_t batch_no = 0;   // global batch counter (clip-norm sampling stride)

    auto run_batch = [&](size_t base, int cur, double &se_batch) {
        auto ta = std::chrono::steady_clock::now();
        // Phase 1: compute (parallel over contiguous position slices).
        pool.run([&](int tid) {
            BTWorker &w = *workers[tid];
            w.se = 0.0;
            int lo = (int)((long)tid * cur / threads);
            int hi = (int)((long)(tid + 1) * cur / threads);
            for (int j = lo; j < hi; j++) {
                const BTRecord &r = recs[train_idx[base + j]];
                float d      = bt_eval_record(r, K, w.pos, w.acc, &w.act);
                float target = bt_target(r, lambda, leaf_lambda, K, decay(r));
                float e      = target - d;
                w.se += (double)e * e;
                // Same descent-form gradient scale as tdleaf_accumulate_game;
                // wtm here is the position's stm (the "leaf" of a 0-ply PV).
                // Focal-γ loss family: sig_grad = (d(1-d))^γ / K.  γ=1 is MSE
                // (the sigmoid Jacobian in full, kept bit-identical via the
                // branch); γ=0 drops it → soft-label cross-entropy gradient
                // (target-d)/K, which keeps full strength at the confident
                // tails; γ=0.5 sits between.
                float sig_grad   = (loss_gamma == 1.0f)
                                     ? d * (1.0f - d) / K
                                     : powf(d * (1.0f - d), loss_gamma) / K;
                float wtm_sign   = r.wtm ? -1.0f : 1.0f;
                float grad_scale = e * sig_grad * cp_factor * wtm_sign;
                if (grad_scale != 0.0f)
                    nnue_accumulate_gradients(w.act, grad_scale, false, w.gb);
            }
        });
        auto tb = std::chrono::steady_clock::now();
        // Phase 2: reduce into g_grad.  Each thread owns a disjoint FT/PSQT row
        // range and sums all workers (index order) into it; thread 0 also sums
        // the dense FC + FT-bias grads.  Disjoint rows → no races; fixed order
        // → deterministic.
        pool.run([&](int tid) {
            int lo = (int)((long)tid * NNUE_FT_INPUTS / threads);
            int hi = (int)((long)(tid + 1) * NNUE_FT_INPUTS / threads);
            for (int t = 0; t < threads; t++)
                nnue_gradbuf_merge_ft_rows(workers[t]->gb, lo, hi);
            if (tid == 0)
                for (int t = 0; t < threads; t++)
                    nnue_gradbuf_merge_dense(workers[t]->gb);
        });
        auto tc = std::chrono::steady_clock::now();
        // Phase 3: clear each worker buffer for next batch (dense FC grads +
        // dirty cursor only — FT/PSQT rows were zeroed by the merge phase).
        pool.run([&](int tid) { nnue_gradbuf_clear(workers[tid]->gb); });
        auto td = std::chrono::steady_clock::now();
        // Phase 4: Adam step on the merged gradient, then targeted requantize
        // (only the rows apply touched — O(dirty), not a full 23M-weight FT
        // requantize).  apply re-zeroes g_grad for the next batch.  At >1 thread
        // the FC-stack + FT-row Adam (the bottleneck) is parallelized; threads==1
        // uses the untouched serial path (bit-identical to the pre-threading
        // trainer).  The L2 clip-norm scan (serial, touches every dirty FT/PSQT
        // row) runs on every clip_every-th batch; skipped batches still call
        // nnue_clip_gradients(0) for the freeze-passthrough housekeeping.  If a
        // sampled norm ever reaches half the threshold, per-batch scanning is
        // restored — measured self-play corpora sit 12-19x under it, so the
        // sampled scan is outcome-identical in practice.
        if (clip_every == 1 || batch_no % (size_t)clip_every == 0) {
            float norm = nnue_clip_gradients(TDLEAF_GRAD_CLIP_NORM);
            if (clip_every > 1 && norm > 0.5f * TDLEAF_GRAD_CLIP_NORM) {
                fprintf(stderr, "batch-train: sampled grad norm %.3f above half "
                                "the clip threshold %.2f — restoring per-batch "
                                "clip scans\n",
                        (double)norm, (double)TDLEAF_GRAD_CLIP_NORM);
                clip_every = 1;
            }
        } else {
            nnue_clip_gradients(0.0f);
        }
        batch_no++;
        if (threads == 1)
            nnue_apply_gradients(lr_scale);
        else
            nnue_apply_gradients_parallel(lr_scale, threads,
                [&](const std::function<void(int)> &fn) { pool.run(fn); });
        nnue_requantize_fc_applied();
        auto te = std::chrono::steady_clock::now();

        for (int t = 0; t < threads; t++) se_batch += workers[t]->se;
        t_compute += std::chrono::duration<double>(tb - ta).count();
        t_reduce  += std::chrono::duration<double>(tc - tb).count();
        t_clear   += std::chrono::duration<double>(td - tc).count();
        t_tail    += std::chrono::duration<double>(te - td).count();
    };

    for (int ep = 1; ep <= epochs; ep++) {
        std::shuffle(train_idx.begin(), train_idx.end(), rng);
        double se = 0.0;
        t_compute = t_reduce = t_clear = t_tail = 0.0;
        size_t next_report = 1000000;
        auto t0 = std::chrono::steady_clock::now();

        for (size_t base = 0; base < train_idx.size(); base += batch) {
            int cur = (int)std::min<size_t>(batch, train_idx.size() - base);
            run_batch(base, cur, se);
            size_t done = base + cur;
            if (done >= next_report) {
                auto el = std::chrono::duration<double>(
                              std::chrono::steady_clock::now() - t0).count();
                fprintf(stderr, "  epoch %d: %zuM positions  train MSE=%.6f  (%.0f pos/s)\n",
                        ep, done / 1000000, se / (double)done, done / el);
                next_report += 1000000;
            }
        }

        auto el = std::chrono::duration<double>(
                      std::chrono::steady_clock::now() - t0).count();
        fprintf(stderr, "batch-train: epoch %d done — train MSE=%.6f  (%.0fs, %.0f pos/s)\n",
                ep, se / (double)std::max<size_t>(train_idx.size(), 1), el,
                train_idx.size() / el);
        fprintf(stderr, "  phase timing: compute=%.1fs reduce=%.1fs clear=%.1fs "
                        "serial-tail=%.1fs (tail %.0f%% of batch time)\n",
                t_compute, t_reduce, t_clear, t_tail,
                100.0 * t_tail / std::max(1e-9, t_compute + t_reduce + t_clear + t_tail));
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

    for (int t = 0; t < threads; t++) {
        nnue_gradbuf_free(workers[t]->gb);
        delete workers[t];
    }
    return 0;
}
