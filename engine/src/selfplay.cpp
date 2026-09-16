// Leaf internal self-play driver (--selfplay)
//
// Plays whole games inside one process: both sides are searched by the same
// net, tdleaf_record_ply() runs after EVERY search (so records alternate root
// STM and the game-ply gap between consecutive records is 1), and the TDLeaf
// update runs at game end with exact in-engine result detection — no UCI
// harness, no self-adjudication ambiguity.
//
// Openings come from a plain EPD file (one "board stm castle ep" line per
// position; X-FEN / Shredder-FEN castling fields are handled by setboard, so
// FRC openings work).  Striping (--epd-offset/--epd-stride) lets N concurrent
// processes split one file without duplicating openings.
//
// Learning configuration stays in the environment exactly as in harness play:
// TDLEAF_FREEZE=1 + TDLEAF_DUMP_TSV=<prefix> gives frozen corpus generation,
// unset gives live online learning.
//
// Usage (must be the last engine args; `hash`/`cores` args must come first):
//   Leaf_vX --selfplay --epd FILE [--games N] [--depth D] [--tdleaf-out PATH]
//           [--epd-offset K] [--epd-stride S] [--epd-shuffle SEED]
//           [--max-ply P] [--no-adjudication] [--verbose]
//           [--pgn-out FILE] [--pgn-name TAG]

#if TDLEAF

#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <ctime>
#include "selfplay_traj.h"

struct SelfplayEpdLine {
    char board[128];
    char ms;
    char castle[8];
    char ep[4];
};

struct SelfplayConfig {
    const char *epd_path;
    int      games;        // 0 = one pass over this process's EPD slice
    int      depth;
    uint64_t nodes;         // --nodes: node budget per move (0 = fixed depth)
    int      max_ply;      // draw adjudication cap (game plies)
    int      epd_offset;
    int      epd_stride;
    unsigned shuffle_seed;
    bool     shuffle;
    bool     adjudicate;
    bool     verbose;
    char     tdleaf_out[FILENAME_MAX];
    const char *traj_dir;     // Stage 1: emit per-game .tdg files here (NULL = off)
    int      traj_max_pending; // backpressure: sleep while this many .tdg await the learner
    const char *pgn_out;      // --pgn-out: append played games here (NULL = off)
    const char *pgn_name;     // --pgn-name: White/Black tag text
};

struct SelfplayStats {
    int played, white_wins, black_wins, draws;
    int skipped_short;                    // games skipped by TDLeaf min-ply gates
    int term_mate, term_stale, term_fifty, term_rep, term_material;
    int term_maxply, term_resign, term_drawadj, term_error;
};

enum SelfplayTerm {
    SP_TERM_ERROR = 0, SP_TERM_MATE, SP_TERM_STALEMATE, SP_TERM_FIFTY,
    SP_TERM_REP, SP_TERM_MATERIAL, SP_TERM_MAXPLY, SP_TERM_RESIGN, SP_TERM_DRAWADJ
};

// ---------------------------------------------------------------------------
// EPD loading: plain 4-field lines ("board stm castle ep").  Extra fields on
// a line are ignored.  Not the test_suite reader — that one expects bm-move
// lists terminated by ';' and would desync on plain opening files.
// ---------------------------------------------------------------------------
static bool selfplay_load_epd(const char *path, std::vector<SelfplayEpdLine> &out)
{
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "selfplay: cannot open EPD file %s\n", path); return false; }
    char line[512];
    while (fgets(line, sizeof(line), f)) {
        SelfplayEpdLine e;
        char ms[4];
        int n = sscanf(line, "%127s %3s %7s %3s", e.board, ms, e.castle, e.ep);
        if (n < 4 || (ms[0] != 'w' && ms[0] != 'b')) continue;   // blank/comment/garbage
        e.ms = ms[0];
        out.push_back(e);
    }
    fclose(f);
    if (out.empty()) fprintf(stderr, "selfplay: no usable positions in %s\n", path);
    return !out.empty();
}

// ---------------------------------------------------------------------------
// Per-game engine state reset — mirrors the UCI `ucinewgame` handler so a
// selfplay game starts from the same clean state as a harness game: the hash
// tables are wiped (TT probes match on key only, so stale entries from the
// previous game WOULD be probed without this), h_id/depth trackers, and
// history/reply tables.
//
// The hash tables are NOT wiped when the weights are frozen.  Probes match on
// the full 64-bit Zobrist key, so an entry surviving from an earlier game is a
// genuinely identical position and its score is still valid -- a frozen actor
// evaluates with the same weights all run, so cross-game reuse is free search.
//
// It is only safe *because* the weights are frozen.  The score hash caches NNUE
// evaluations and the TT stores search scores; both are weight-dependent, so a
// process that LEARNS between games would be probing entries from an older net
// -- exactly the label-vintage error the refresh machinery exists to remove.
// train.py's actors always run TDLEAF_FREEZE=1, so the recipe takes the fast
// path; a standalone learning --selfplay run still clears.
//
// (clear_hash(), not set_hash_size(): the geometry never changes between games,
// so the old free/aligned_alloc round trip bought nothing and cost a full
// page-fault-in of the table every game -- ~20% of game time at depth 6 with
// the default 128 MB hash, paid 500,000 times in a single training iteration.)
// ---------------------------------------------------------------------------
static void selfplay_new_game_reset()
{
    if (!tdleaf_frozen()) clear_hash();
    game.ts.last_ponder = 0;
    game.ts.last_depth  = 1;
    game.ts.singular_response.t = NOMOVE;
    game.ts.h_id = 0;
    for (int ti = 0; ti < thread_cfg.threads; ti++) {
        for (int i = 0; i < 15; i++)
            for (int j = 0; j < 64; j++) {
                game.ts.tdata[ti].history[i][j] = 0;
                game.ts.tdata[ti].reply[i][j]   = 0;
            }
        game.ts.tdata[ti].pc[0][0].t = NOMOVE;
    }
}

// ---------------------------------------------------------------------------
// Score-history adjudication on the alternating-STM record stream.
// fastchess-faithful semantics, shared constants with tdleaf_self_adjudicate:
//   resign — the last TDLEAF_RESIGN_PLIES records where side X was to move at
//            the root all have score_root_stm <= -TDLEAF_RESIGN_CP (the score
//            is X's own POV at those records, so no sign conversion);
//   draw   — past move TDLEAF_DRAW_MOVE_NUMBER, the last TDLEAF_DRAW_PLIES
//            records (both sides) have |score| <= TDLEAF_DRAW_CP (|STM POV| ==
//            |white POV|, so again sign-free).
// Do NOT reuse tdleaf_self_adjudicate here: its score section assumes a
// single-color record stream (harness mode).
// ---------------------------------------------------------------------------
static bool selfplay_adjudicate(const TDGameRecord &rec, int game_T, float &result_w)
{
    int n = rec.n_plies;

    for (int side = 0; side <= 1; side++) {
        int seen = 0; bool all_lost = true;
        for (int t = n - 1; t >= 0 && seen < TDLEAF_RESIGN_PLIES; t--) {
            if ((int)rec.plies[t].root_wtm != side) continue;
            seen++;
            if (rec.plies[t].score_root_stm > -TDLEAF_RESIGN_CP) { all_lost = false; break; }
        }
        if (seen >= TDLEAF_RESIGN_PLIES && all_lost) {
            result_w = side ? 0.0f : 1.0f;   // side resigns -> side loses
            return true;
        }
    }

    int move_number = (game_T - 1) / 2 + 1;
    if (move_number >= TDLEAF_DRAW_MOVE_NUMBER && n >= TDLEAF_DRAW_PLIES) {
        bool drawish = true;
        for (int t = n - TDLEAF_DRAW_PLIES; t < n; t++) {
            int s = rec.plies[t].score_root_stm;
            if (s > TDLEAF_DRAW_CP || s < -TDLEAF_DRAW_CP) { drawish = false; break; }
        }
        if (drawish) { result_w = 0.5f; return true; }
    }
    return false;
}

// ---------------------------------------------------------------------------
// Trajectory emission (Stage 1 actor side): one .tdg file per completed game,
// written as <name>.tdg.tmp then rename()d so the learner never sees partials.
// Ships only positions + search outputs + POV/gate metadata; the learner
// rebuilds accumulators/features/stack via tdleaf_rebuild_record.
// ---------------------------------------------------------------------------
static int selfplay_count_tdg(const char *dir)
{
    DIR *d = opendir(dir);
    if (!d) return 0;
    int n = 0;
    struct dirent *e;
    while ((e = readdir(d)) != nullptr) {
        size_t l = strlen(e->d_name);
        if (l > 4 && strcmp(e->d_name + l - 4, ".tdg") == 0) n++;
    }
    closedir(d);
    return n;
}

static bool selfplay_write_traj(const SelfplayConfig &cfg, const TDGameRecord &rec,
                                float result_w, uint32_t seq)
{
    // Backpressure: don't let the emit directory grow unboundedly if the
    // learner falls behind (each file is ~100 KB).
    while (selfplay_count_tdg(cfg.traj_dir) >= cfg.traj_max_pending)
        usleep(500000);

    char fin[FILENAME_MAX], tmp[FILENAME_MAX];
    snprintf(fin, sizeof(fin), "%s/g-%05d-%08u.tdg",
             cfg.traj_dir, (int)(getpid() % 100000), seq);
    snprintf(tmp, sizeof(tmp), "%s.tmp", fin);
    FILE *f = fopen(tmp, "wb");
    if (!f) {
        fprintf(stderr, "selfplay: cannot write trajectory %s\n", tmp);
        return false;
    }
    TDTrajHeader h;
    h.magic             = TDTRAJ_MAGIC;
    h.version           = TDTRAJ_VERSION;
    h.nnue_content_hash = nnue_get_content_hash();
    h.result_white_pov  = result_w;
    h.n_records         = rec.n_plies;
    // Informational only: no consumer reads h.gid (the learner assigns its own
    // corpus gid in tdleaf_dump_game).  Kept for offline .tdg inspection.
    h.gid               = (((uint32_t)getpid() & 0xFFF) << 20) | (seq & 0xFFFFF);
    bool ok = fwrite(&h, sizeof(h), 1, f) == 1;
    for (int t = 0; ok && t < rec.n_plies; t++) {
        const TDRecord &r = rec.plies[t];
        TDTrajRecord tr;
        memset(&tr, 0, sizeof(tr));
        tr.pos               = r.pos;
        tr.root_pos          = r.root_pos;
        tr.score_stm         = r.score_stm;
        tr.score_root_stm    = r.score_root_stm;
        tr.root_static       = r.root_static;
        tr.game_ply          = r.game_ply;
        tr.id_score_variance = r.id_score_variance;
        tr.id_depth          = r.id_depth;
        tr.wtm               = (uint8_t)r.wtm;
        tr.root_wtm          = (uint8_t)r.root_wtm;
        ok = fwrite(&tr, sizeof(tr), 1, f) == 1;
    }
    fclose(f);
    if (!ok || rename(tmp, fin) != 0) {
        fprintf(stderr, "selfplay: trajectory write failed for %s\n", fin);
        remove(tmp);
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// PGN export (--pgn-out).  One file per actor process, appended; the comment on
// each half-move is the fastchess shape `{<score>/<depth> <time>s}`, so the
// existing readers (extract_positions.py, pgn_winrate.py, bayeselo_ratings.py)
// take these files unchanged.
//
// The ACTOR writes it because it is the only party that sees a whole game: a
// .tdg trajectory carries positions but no move order, no clock, and only the
// plies that pass the TDLeaf gates.  What lands here is the complete record --
// including the games the learner never sees (early 3-rep draws), openings that
// are already terminal, and aborted games (Result "*").
//
// Cost: +0.23% of actor wall clock at depth 6 / 800 nodes over four paired
// 200-game runs -- inside the run-to-run spread, so read it as "under 0.5%".
// print_move only ever touches copies of the position, so the games played are
// bit-identical with and without --pgn-out (verified over a fixed 150-game
// slice: same W/D/L, same termination histogram, same FEN headers).
// ---------------------------------------------------------------------------
struct SelfplayPgn {
    FILE       *f;           // NULL = export off
    const char *name;        // White/Black tag text
    const char *budget;      // SearchBudget tag text
    char        date[16];    // "YYYY.MM.DD", stamped once at open
    char        fen[160];    // start position of the current game
    std::string moves;       // SAN + comments accumulated for the current game
    int         col;         // output column, for 80-column wrapping
    int         plies;       // half-moves written this game
};

// Shredder-FEN castling field: the rook's FILE as a letter, uppercase for
// White.  pos.Krook/Qrook hold absolute squares once setboard has resolved
// them, and are meaningful only when the matching pos.castle bit is set
// (1/2 = White king/queen side, 4/8 = Black king/queen side).  Shredder rather
// than the EPD's X-FEN spelling so the FRC openings survive a round trip
// through tools that do not implement X-FEN disambiguation.
static void selfplay_shredder_castle(const position &p, char *out)
{
    // A right is emitted only if its rook is really there.  setboard leaves the
    // resolver sentinels in place (-1 / 64 / -100) when a castling letter names
    // a rook the board does not have, and FILE() of a sentinel is a plausible-
    // looking wrong letter, so an inconsistent EPD line would otherwise produce
    // a FEN asserting rights that cannot exist ("HHha" for one rook on h1).
    auto emit = [&](int sq, int side, char base) -> int {
        if (sq < 0 || sq > 63) return 0;
        if (PTYPE(p.sq[sq]) != ROOK || PSIDE(p.sq[sq]) != side) return 0;
        return (int)(base + FILE(sq));
    };
    // ...and the two rooks of a side are two different rooks.  Same cause: an
    // EPD claiming both rights when the back rank holds one rook resolves both
    // squares to it, which would spell a duplicate letter ("HH").
    int n = 0, c;
    if ((p.castle & 1) && (c = emit(p.Krook[WHITE], WHITE, 'A'))) out[n++] = (char)c;
    if ((p.castle & 2) && p.Qrook[WHITE] != p.Krook[WHITE] &&
        (c = emit(p.Qrook[WHITE], WHITE, 'A'))) out[n++] = (char)c;
    if ((p.castle & 4) && (c = emit(p.Krook[BLACK], BLACK, 'a'))) out[n++] = (char)c;
    if ((p.castle & 8) && p.Qrook[BLACK] != p.Krook[BLACK] &&
        (c = emit(p.Qrook[BLACK], BLACK, 'a'))) out[n++] = (char)c;
    if (!n) out[n++] = '-';
    out[n] = '\0';
}

static bool selfplay_pgn_open(SelfplayPgn &pgn, const SelfplayConfig &cfg,
                              const char *budget)
{
    pgn.f       = nullptr;
    pgn.name    = "Leaf";
    pgn.budget  = budget;
    pgn.col     = 0;
    pgn.plies   = 0;
    pgn.fen[0]  = '\0';
    pgn.date[0] = '\0';
    if (!cfg.pgn_out) return true;
    pgn.f = fopen(cfg.pgn_out, "a");
    if (!pgn.f) {
        fprintf(stderr, "selfplay: cannot open PGN %s\n", cfg.pgn_out);
        return false;
    }
    // Buffer a whole game rather than syscalling per move: an unbuffered
    // per-move write (what --verbose does) is most of the measured overhead.
    static char pgn_buf[1 << 16];
    setvbuf(pgn.f, pgn_buf, _IOFBF, sizeof(pgn_buf));
    pgn.name = cfg.pgn_name ? cfg.pgn_name : "Leaf";
    time_t now = time(nullptr);
    struct tm tmv;
    localtime_r(&now, &tmv);
    strftime(pgn.date, sizeof(pgn.date), "%Y.%m.%d", &tmv);
    return true;
}

static void selfplay_pgn_new_game(SelfplayPgn &pgn, const SelfplayEpdLine &op,
                                  const position &start)
{
    if (!pgn.f) return;
    char castle[8];
    selfplay_shredder_castle(start, castle);
    // An opening always starts a game, so the halfmove clock is 0 and the move
    // number 1 (setboard sets T = 1 white to move, 2 black to move).
    snprintf(pgn.fen, sizeof(pgn.fen), "%s %c %s %s 0 1",
             op.board, op.ms, castle, op.ep[0] ? op.ep : "-");
    pgn.moves.clear();
    pgn.col   = 0;
    pgn.plies = 0;
}

// Append one whitespace-separated token, wrapping at 80 columns.  A comment is
// one token even though it contains a space -- same as fastchess, and it keeps
// `{score/depth time}` readable by line-oriented greps.
static void selfplay_pgn_token(SelfplayPgn &pgn, const char *tok)
{
    int len = (int)strlen(tok);
    if (pgn.col && pgn.col + 1 + len > 80) { pgn.moves += '\n'; pgn.col = 0; }
    else if (pgn.col)                      { pgn.moves += ' ';  pgn.col++; }
    pgn.moves += tok;
    pgn.col   += len;
}

// Record the move about to be played.  `before` is the position it is played
// from -- print_move needs it to build SAN, and must be called before exec_move.
static void selfplay_pgn_move(SelfplayPgn &pgn, position &before, move m, int T,
                              int score, int depth, double secs)
{
    if (!pgn.f) return;
    char tok[64], san[10], sc[16];

    if (T & 1) {                        // white to move: "12."
        snprintf(tok, sizeof(tok), "%d.", (T + 1) / 2);
        selfplay_pgn_token(pgn, tok);
    } else if (!pgn.plies) {            // game opens with Black: "12..."
        snprintf(tok, sizeof(tok), "%d...", (T + 1) / 2);
        selfplay_pgn_token(pgn, tok);
    }

    before.print_move(m, san, &game.ts.tdata[0]);
    selfplay_pgn_token(pgn, san);

    // Score is root-STM POV, i.e. from the point of view of the side that just
    // moved -- already the convention fastchess writes, so no flip.
    if (score > MATE / 2)        snprintf(sc, sizeof(sc), "+M%d", (MATE - score + 1) / 2);
    else if (score < -MATE / 2)  snprintf(sc, sizeof(sc), "-M%d", (MATE + score + 1) / 2);
    else {
        int cp = (value[PAWN] > 0) ? (score * 100) / value[PAWN] : score;
        snprintf(sc, sizeof(sc), "%+.2f", cp / 100.0);
    }
    snprintf(tok, sizeof(tok), "{%s/%d %.3fs}", sc, depth, secs);
    selfplay_pgn_token(pgn, tok);
    pgn.plies++;
}

static void selfplay_pgn_finish(SelfplayPgn &pgn, SelfplayTerm term,
                                float result_w, int round)
{
    if (!pgn.f) return;
    const char *res = (term == SP_TERM_ERROR) ? "*"
                    : (result_w > 0.75f)      ? "1-0"
                    : (result_w < 0.25f)      ? "0-1" : "1/2-1/2";
    const char *why;
    switch (term) {
        case SP_TERM_ERROR:   why = "abandoned";    break;
        case SP_TERM_MAXPLY:
        case SP_TERM_RESIGN:
        case SP_TERM_DRAWADJ: why = "adjudication"; break;
        default:              why = "normal";       break;
    }
    // Variant is stamped Chess960 for every game: the training book mixes FRC
    // and standard openings, standard chess is a well-formed Chess960 position,
    // and one tag for the whole file keeps the reader's job simple.
    fprintf(pgn.f,
            "[Event \"Leaf self-play\"]\n"
            "[Site \"?\"]\n"
            "[Date \"%s\"]\n"
            "[Round \"%d\"]\n"
            "[White \"%s\"]\n"
            "[Black \"%s\"]\n"
            "[Result \"%s\"]\n"
            "[SetUp \"1\"]\n"
            "[FEN \"%s\"]\n"
            "[Variant \"Chess960\"]\n"
            "[PlyCount \"%d\"]\n"
            "[Termination \"%s\"]\n"
            "[TimeControl \"-\"]\n"
            "[SearchBudget \"%s\"]\n"
            "[NetHash \"%08x\"]\n\n",
            pgn.date, round, pgn.name, pgn.name, res, pgn.fen,
            pgn.plies, why, pgn.budget, nnue_get_content_hash());
    selfplay_pgn_token(pgn, res);
    fputs(pgn.moves.c_str(), pgn.f);
    fputs("\n\n", pgn.f);
    // Flush per game, not per buffer: actors are SIGTERMed on driver shutdown,
    // and a 64 KB buffer would take ~13 finished games with it.  One write
    // syscall per game is nothing against a game's worth of search.
    fflush(pgn.f);
}

static void selfplay_pgn_close(SelfplayPgn &pgn)
{
    if (!pgn.f) return;
    fclose(pgn.f);
    pgn.f = nullptr;
}

// ---------------------------------------------------------------------------
// Play one game from an opening.  Returns the termination reason; result_w is
// the white-POV outcome (undefined for SP_TERM_ERROR).
// ---------------------------------------------------------------------------
static SelfplayTerm selfplay_play_game(const SelfplayEpdLine &op, const SelfplayConfig &cfg,
                                       float &result_w, SelfplayPgn &pgn)
{
    selfplay_new_game_reset();
    game.setboard(op.board, op.ms, op.castle, op.ep);
    // After setboard: Krook/Qrook are resolved, so the Shredder castling field
    // for the PGN header can be read off the start position.
    selfplay_pgn_new_game(pgn, op, game.pos);
    game.book = 0;                       // never probe the opening book
    game.over = 0;
    game.mttc = 0;
    // --depth is the guaranteed depth in BOTH modes: without --nodes it is the
    // fixed depth, with --nodes it is the FLOOR and the budget buys whatever
    // extra the position affords.  One reading either way: "at least this deep".
    game.ts.max_search_depth  = cfg.nodes ? MAXD : cfg.depth;
    game.ts.min_search_depth  = cfg.nodes ? cfg.depth : 0;
    game.ts.analysis_mode = 0;
    // --nodes: budget in nodes instead of depth.  Fixed depth spends the same
    // effort on a forced recapture and on a critical fail-low; a node budget
    // with the extend/reduce logic reproduces what a clock gives without the
    // timing noise, and stays deterministic single-threaded.  max_search_depth
    // stays as the ceiling so the ID loop still terminates.
    game.ts.max_nodes = cfg.nodes;
    // Fixed-depth (or fixed-node) search: unlimited clock, depth/nodes stop ID.
    game.timeleft[0] = game.timeleft[1] = (float)MAXT;

    // An opening that is already mate/stalemate produces no game.
    {
        position scratch = game.pos;
        int mate = scratch.in_check_mate();
        if (mate == 1) { result_w = game.pos.wtm ? 0.0f : 1.0f; return SP_TERM_MATE; }
        if (mate == 2) { result_w = 0.5f;                       return SP_TERM_STALEMATE; }
    }

    char mstring[10];
    int plies_played = 0;
    while (1) {
        game.p_side = game.pos.wtm ^ 1;  // engine is always the side to move
        // GetTime() is centiseconds, and a move here costs single-digit
        // milliseconds -- every PGN time would read 0.00s.  gettimeofday is a
        // vDSO read (~25 ns), so it is cheaper than branching around it.
        struct timeval mv_t0, mv_t1;
        gettimeofday(&mv_t0, nullptr);
        game.best = game.ts.search(game.pos, MAXT, game.T, &game);
        gettimeofday(&mv_t1, nullptr);

#if !TDLEAF_READONLY
        if (nnue_available) {
            tdleaf_record_ply(game.td_game,
                              game.pos,
                              game.ts.tdata[0].n[0].acc,
                              game.ts.tdata[0].pc[0],
                              game.ts.g_last,
                              game.ts.id_scores,
                              game.ts.id_score_count,
                              game.ts.last_depth,
                              game.T);
        }
#endif

        if (cfg.verbose) {
            game.pos.print_move(game.best, mstring, &game.ts.tdata[0]);
            fprintf(stderr, "%s%s ", game.pos.wtm ? "" : "..", mstring);
        }

        game.temp = game.pos;
        if (!game.temp.exec_move(game.best, 0)) {
            fprintf(stderr, "selfplay: illegal best move at ply %d — aborting game\n", game.T);
            return SP_TERM_ERROR;
        }

        // After the legality check so an illegal best move cannot reach the PGN
        // as garbage SAN, but before game.pos advances -- SAN is built against
        // the position the move is played FROM.
        selfplay_pgn_move(pgn, game.pos, game.best, game.T,
                          game.ts.g_last, game.ts.last_depth,
                          (double)(mv_t1.tv_sec - mv_t0.tv_sec) +
                          (double)(mv_t1.tv_usec - mv_t0.tv_usec) / 1e6);

        game.last = game.pos;
        game.pos  = game.temp;

        game.game_history[game.T - 1] = game.best;
        for (int ti = 0; ti < MAX_THREADS; ti++)
            game.ts.tdata[ti].plist[game.T] = game.pos.hcode;

        // Terminal detection on the new position (same order as make_move()).
        int mate = game.pos.in_check_mate();
        if (mate == 1) {
            result_w = game.pos.wtm ? 0.0f : 1.0f;   // side to move is mated
            game.T++;  return SP_TERM_MATE;
        }
        if (mate == 2) { result_w = 0.5f; game.T++; return SP_TERM_STALEMATE; }
        if (game.pos.fifty >= 100) { result_w = 0.5f; game.T++; return SP_TERM_FIFTY; }
        {
            // Scan back at stride 2 (same-STM) no further than the last
            // irreversible move.  setboard() starts every game at T >= 1 with
            // fifty == 0 and fifty can only grow in step with T, so the floor is
            // currently >= 1 on its own — but clamping makes that explicit
            // rather than depending on an invariant established in game_rec.cpp,
            // and matches tdleaf_self_adjudicate's floor.  Without it, any future
            // change that seeds pos.fifty from the opening (e.g. an EPD reader
            // that parses the halfmove clock) turns this into a negative index.
            int rep_floor = game.T - game.pos.fifty;
            if (rep_floor < 0) rep_floor = 0;
            int rep_count = 0;
            for (int ri = game.T - 2; ri >= rep_floor && rep_count < 2; ri -= 2)
                if (game.ts.tdata[0].plist[ri] == game.pos.hcode) rep_count++;
            if (rep_count >= 2) { result_w = 0.5f; game.T++; return SP_TERM_REP; }
        }
        if (tdleaf_insufficient_material(game.pos)) {
            result_w = 0.5f; game.T++; return SP_TERM_MATERIAL;
        }
        game.T++;
        plies_played++;
        if (plies_played >= cfg.max_ply) { result_w = 0.5f; return SP_TERM_MAXPLY; }

        if (cfg.adjudicate && selfplay_adjudicate(game.td_game, game.T, result_w))
            return (result_w == 0.5f) ? SP_TERM_DRAWADJ : SP_TERM_RESIGN;
    }
}

// ---------------------------------------------------------------------------
// Entry point, dispatched from main() when --selfplay is on the command line.
// ---------------------------------------------------------------------------
int selfplay_main(int argc, char *argv[])
{
    SelfplayConfig cfg;
    cfg.epd_path   = nullptr;
    cfg.games      = 0;
    cfg.depth      = 8;
    cfg.nodes      = 0ULL;   // 0 = fixed depth (the historical mode)
    cfg.max_ply    = 500;
    cfg.epd_offset = 0;
    cfg.epd_stride = 1;
    cfg.shuffle_seed = 0;
    cfg.shuffle    = false;
    cfg.adjudicate = true;
    cfg.verbose    = false;
    cfg.traj_dir   = nullptr;
    cfg.traj_max_pending = 500;
    cfg.pgn_out    = nullptr;
    cfg.pgn_name   = nullptr;
    snprintf(cfg.tdleaf_out, sizeof(cfg.tdleaf_out), "%s%s",
             engine_cfg.exec_path, NNUE_TDLEAF_BIN);

    for (int ai = 1; ai < argc; ai++) {
        if (!strcmp(argv[ai], "--epd")        && ai + 1 < argc) cfg.epd_path   = argv[++ai];
        else if (!strcmp(argv[ai], "--games") && ai + 1 < argc) cfg.games      = atoi(argv[++ai]);
        else if (!strcmp(argv[ai], "--depth") && ai + 1 < argc) cfg.depth      = atoi(argv[++ai]);
        else if (!strcmp(argv[ai], "--nodes") && ai + 1 < argc) cfg.nodes      = strtoull(argv[++ai], nullptr, 10);
        else if (!strcmp(argv[ai], "--max-ply") && ai + 1 < argc) cfg.max_ply  = atoi(argv[++ai]);
        else if (!strcmp(argv[ai], "--epd-offset") && ai + 1 < argc) cfg.epd_offset = atoi(argv[++ai]);
        else if (!strcmp(argv[ai], "--epd-stride") && ai + 1 < argc) cfg.epd_stride = atoi(argv[++ai]);
        else if (!strcmp(argv[ai], "--epd-shuffle") && ai + 1 < argc) {
            cfg.shuffle = true; cfg.shuffle_seed = (unsigned)atol(argv[++ai]);
        }
        else if (!strcmp(argv[ai], "--tdleaf-out") && ai + 1 < argc) {
            snprintf(cfg.tdleaf_out, sizeof(cfg.tdleaf_out), "%s", argv[++ai]);
        }
        else if (!strcmp(argv[ai], "--traj-out") && ai + 1 < argc) cfg.traj_dir = argv[++ai];
        else if (!strcmp(argv[ai], "--traj-max-pending") && ai + 1 < argc)
            cfg.traj_max_pending = atoi(argv[++ai]);
        else if (!strcmp(argv[ai], "--pgn-out")  && ai + 1 < argc) cfg.pgn_out  = argv[++ai];
        else if (!strcmp(argv[ai], "--pgn-name") && ai + 1 < argc) cfg.pgn_name = argv[++ai];
        else if (!strcmp(argv[ai], "--no-adjudication")) cfg.adjudicate = false;
        else if (!strcmp(argv[ai], "--verbose"))         cfg.verbose    = true;
    }
    if (cfg.traj_dir) {
        struct stat stbuf;
        if (stat(cfg.traj_dir, &stbuf) != 0 || !S_ISDIR(stbuf.st_mode)) {
            fprintf(stderr, "selfplay: --traj-out %s is not a directory\n", cfg.traj_dir);
            return 1;
        }
        tdleaf_capture_root = true;   // .tdg ships root_pos/root_static
    }

    if (!cfg.epd_path) {
        fprintf(stderr, "selfplay: --epd <file> is required\n");
        return 1;
    }
    if (!nnue_available) {
        fprintf(stderr, "selfplay: requires a loaded NNUE network\n");
        return 1;
    }
    if (cfg.depth < 1)    cfg.depth = 1;
    if (cfg.depth > MAXD) cfg.depth = MAXD;
    if (cfg.max_ply > MAX_GAME_PLY - 2) cfg.max_ply = MAX_GAME_PLY - 2;
    if (cfg.epd_stride < 1) cfg.epd_stride = 1;

    std::vector<SelfplayEpdLine> openings;
    if (!selfplay_load_epd(cfg.epd_path, openings)) return 1;
    if (cfg.shuffle) {
        std::mt19937 rng(cfg.shuffle_seed);
        std::shuffle(openings.begin(), openings.end(), rng);
    }

    // This process's opening slice: offset, offset+stride, ... (wrapping when
    // --games asks for more than one pass over the slice).
    int slice_count = 0;
    for (size_t k = cfg.epd_offset; k < openings.size(); k += cfg.epd_stride) slice_count++;
    if (slice_count == 0) {
        fprintf(stderr, "selfplay: EPD slice is empty (offset %d, stride %d, %zu openings)\n",
                cfg.epd_offset, cfg.epd_stride, openings.size());
        return 1;
    }
    int total_games = cfg.games > 0 ? cfg.games : slice_count;

    proto.post = 0;   // no per-iteration search output
    char budget[64];
    if (cfg.nodes) snprintf(budget, sizeof(budget), "depth>=%d then up to %llu nodes/move",
                            cfg.depth, (unsigned long long)cfg.nodes);
    else           snprintf(budget, sizeof(budget), "depth %d", cfg.depth);
    fprintf(stderr, "selfplay: %d games, %s, %zu openings (slice %d: offset %d stride %d)%s%s%s\n",
            total_games, budget, openings.size(), slice_count,
            cfg.epd_offset, cfg.epd_stride,
            tdleaf_frozen() ? ", weights FROZEN" : "",
            getenv("TDLEAF_DUMP_TSV") ? ", dumping TSV" : "",
            cfg.pgn_out ? ", writing PGN" : "");

    SelfplayPgn pgn;
    if (!selfplay_pgn_open(pgn, cfg, budget)) return 1;

    SelfplayStats st;
    memset(&st, 0, sizeof(st));
    int start_time = GetTime();

    for (int g = 0; g < total_games; g++) {
        size_t idx = (size_t)cfg.epd_offset +
                     (size_t)((g % slice_count)) * (size_t)cfg.epd_stride;
        float result_w = 0.5f;
        SelfplayTerm term = selfplay_play_game(openings[idx], cfg, result_w, pgn);

        // Every game reaches the PGN, including the ones dropped below: an
        // already-terminal opening, an early 3-rep the learner never sees, and
        // an aborted game (Result "*") are all part of a complete record.
        selfplay_pgn_finish(pgn, term, result_w, g + 1);

        if (cfg.verbose) fprintf(stderr, "\n");

        switch (term) {
            case SP_TERM_ERROR:     st.term_error++;    break;
            case SP_TERM_MATE:      st.term_mate++;     break;
            case SP_TERM_STALEMATE: st.term_stale++;    break;
            case SP_TERM_FIFTY:     st.term_fifty++;    break;
            case SP_TERM_REP:       st.term_rep++;      break;
            case SP_TERM_MATERIAL:  st.term_material++; break;
            case SP_TERM_MAXPLY:    st.term_maxply++;   break;
            case SP_TERM_RESIGN:    st.term_resign++;   break;
            case SP_TERM_DRAWADJ:   st.term_drawadj++;  break;
        }

        if (term == SP_TERM_ERROR || game.td_game.n_plies == 0) {
            // Aborted game or nothing recorded: drop it.
            game.td_game.n_plies      = 0;
            game.td_game.engine_color = -1;
            continue;
        }

        st.played++;
        if (result_w > 0.75f)      st.white_wins++;
        else if (result_w < 0.25f) st.black_wins++;
        else                       st.draws++;

#if !TDLEAF_READONLY
        // Same game-end handling as make_move(): skip degenerate early
        // repetition draws, otherwise run the TDLeaf update.
        if (nnue_available) {
            if (term == SP_TERM_REP && game.td_game.n_plies < TDLEAF_MIN_PLIES_REP) {
                fprintf(stderr, "[TDLeaf] Skipping early 3-rep draw (%d plies < %d)\n",
                        game.td_game.n_plies, TDLEAF_MIN_PLIES_REP);
                st.skipped_short++;
            } else {
                // Emit the trajectory for exactly the games that reach the
                // update (same skip gates), so a learner replaying the .tdg
                // stream sees the identical game sequence — the basis of the
                // bit-exactness equivalence with online learning.
                if (cfg.traj_dir)
                    selfplay_write_traj(cfg, game.td_game, result_w,
                                        (uint32_t)(st.played - 1));
                tdleaf_update_after_game(game.td_game, result_w, cfg.tdleaf_out);
            }
        }
#endif
        game.td_game.n_plies      = 0;
        game.td_game.engine_color = -1;

        if ((g + 1) % 100 == 0 || g + 1 == total_games) {
            float el = (GetTime() - start_time) / 100.0f;
            fprintf(stderr, "selfplay: %d/%d games  +%d =%d -%d  (%.2f games/s)\n",
                    g + 1, total_games, st.white_wins, st.draws, st.black_wins,
                    el > 0 ? st.played / el : 0.0f);
        }
    }

#if !TDLEAF_READONLY
    tdleaf_flush_batch(cfg.tdleaf_out);
#endif

    selfplay_pgn_close(pgn);

    if (cfg.nodes)
        fprintf(stderr, "selfplay: node budget %llu/move — %llu extends, "
                        "%llu reductions over %d games\n",
                (unsigned long long)cfg.nodes,
                (unsigned long long)game.ts.node_extend_count,
                (unsigned long long)game.ts.node_reduce_count, st.played);
    fprintf(stderr,
            "selfplay: done — %d played (+%d =%d -%d white POV), %d skipped\n"
            "selfplay: terminations: mate %d, stalemate %d, 50-move %d, 3-rep %d, "
            "material %d, max-ply %d, resign %d, draw-adj %d, error %d\n",
            st.played, st.white_wins, st.draws, st.black_wins, st.skipped_short,
            st.term_mate, st.term_stale, st.term_fifty, st.term_rep,
            st.term_material, st.term_maxply, st.term_resign, st.term_drawadj,
            st.term_error);
#if !TDLEAF_READONLY
    tdleaf_report_pv_stats(nullptr);
#endif
#if PVTRUNC_DIAG
    {   extern unsigned long long pvt_fifty, pvt_rep, pvt_kk, pvt_tt;
        fprintf(stderr, "PV-node early returns WITHOUT pc: repetition=%llu "
                "fifty=%llu KKdraw=%llu TTcutoff=%llu\n",
                pvt_rep, pvt_fifty, pvt_kk, pvt_tt);
        extern unsigned long long pvt_fh_stub, pvt_resolved, pvt_searches;
        fprintf(stderr, "ROOT PV provenance: searches=%llu | from FAIL-HIGH stub "
                "(pc cut to 2 by construction)=%llu (%.1f%%) | from resolved "
                "pc_update=%llu (%.1f%%)\n", pvt_searches,
                pvt_fh_stub, 100.0*pvt_fh_stub/(double)pvt_searches,
                pvt_resolved, 100.0*pvt_resolved/(double)pvt_searches);   }
#endif
    return 0;
}

// ===========================================================================
// Stage 1 learner (--learn-stream): consume actor-emitted .tdg trajectories
// in arrival order and run the EXACT online update with one optimizer — the
// single-writer replacement for N processes merging into one .tdleaf.bin.
// ===========================================================================

struct LearnerConfig {
    const char *dir;
    char  tdleaf_out[FILENAME_MAX];
    long  total_games;      // 0 = run until a <dir>/STOP sentinel appears
    bool  delete_consumed;  // default: archive consumed files to <dir>/done/
    const char *publish;    // optional: bake current weights to this .nnue
    int   publish_every;    // games between bakes
    bool  publish_stamp;    // name each bake <base>-<games>g.nnue and keep it
    bool  refresh_scores;   // Flavor A: re-eval leaf statics with current weights
};

// Bake the learner's current weights to a .nnue.
//
// stamp=false (default): always the same path, overwritten each time -- a
// "latest weights" file.  stamp=true (--publish-stamped): "<base>-<games>g.nnue"
// so each bake is kept, giving a rateable ladder of the online trajectory.  The
// checkpoint ladder is the only way to tell a handoff overshoot (drop then
// partial recovery) from equilibration (monotone approach); endpoint
// measurements cannot distinguish them.  See Online_Learning_Investigation 7.10.
static void learner_publish(const char *path, long games, bool stamp)
{
    char stamped[FILENAME_MAX];
    const char *out = path;
    if (stamp) {
        size_t n = strlen(path);
        const char *ext = (n > 5 && !strcmp(path + n - 5, ".nnue")) ? path + n - 5 : nullptr;
        if (ext) snprintf(stamped, sizeof(stamped), "%.*s-%ldg.nnue",
                          (int)(n - 5), path, games);
        else     snprintf(stamped, sizeof(stamped), "%s-%ldg.nnue", path, games);
        out = stamped;
    }
    char tmp[FILENAME_MAX];
    snprintf(tmp, sizeof(tmp), "%s.tmp", out);
    if (nnue_write_nnue(tmp) && rename(tmp, out) == 0)
        fprintf(stderr, "learner: published %s\n", out);
    else
        fprintf(stderr, "learner: failed to publish %s\n", out);
}

// Process one .tdg file: validate, rebuild records, run the online update.
// Returns false for malformed/mismatched files (logged; caller still archives
// them so they are never rescanned).
static bool learner_process_file(const char *path, TDGameRecord *grec,
                                 const LearnerConfig &cfg)
{
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "learner: cannot open %s\n", path); return false; }
    TDTrajHeader h;
    if (fread(&h, sizeof(h), 1, f) != 1 ||
        h.magic != TDTRAJ_MAGIC || h.version != TDTRAJ_VERSION ||
        h.n_records < 1 || h.n_records > MAX_GAME_PLY) {
        fprintf(stderr, "learner: bad header in %s — skipping\n", path);
        fclose(f);
        return false;
    }
    if (h.nnue_content_hash != nnue_get_content_hash()) {
        fprintf(stderr, "learner: %s was generated against a different base net "
                        "(hash 0x%08X != 0x%08X) — skipping\n",
                path, h.nnue_content_hash, nnue_get_content_hash());
        fclose(f);
        return false;
    }

    for (int t = 0; t < h.n_records; t++) {
        TDTrajRecord tr;
        if (fread(&tr, sizeof(tr), 1, f) != 1) {
            fprintf(stderr, "learner: truncated %s at record %d — skipping\n", path, t);
            fclose(f);
            return false;
        }
        TDRecord &r = grec->plies[t];
        r.pos               = tr.pos;
        r.root_pos          = tr.root_pos;
        r.score_stm         = tr.score_stm;
        r.score_root_stm    = tr.score_root_stm;
        r.root_static       = tr.root_static;
        r.game_ply          = tr.game_ply;
        r.id_score_variance = tr.id_score_variance;
        r.id_depth          = tr.id_depth;
        r.wtm               = (bool)tr.wtm;
        r.root_wtm          = (bool)tr.root_wtm;
        tdleaf_rebuild_record(r, cfg.refresh_scores);
    }
    fclose(f);
    grec->n_plies      = h.n_records;
    grec->engine_color = -1;   // meaningless for alternating-STM trajectories

    tdleaf_update_after_game(*grec, h.result_white_pov, cfg.tdleaf_out);
    grec->n_plies = 0;
    return true;
}

int learner_main(int argc, char *argv[])
{
    bool opt_reset = false;
    LearnerConfig cfg;
    cfg.dir             = nullptr;
    cfg.total_games     = 0;
    cfg.delete_consumed = false;
    cfg.publish         = nullptr;
    cfg.publish_every   = 512;
    cfg.publish_stamp   = false;
    cfg.refresh_scores  = false;
    snprintf(cfg.tdleaf_out, sizeof(cfg.tdleaf_out), "%s%s",
             engine_cfg.exec_path, NNUE_TDLEAF_BIN);

    for (int ai = 1; ai < argc; ai++) {
        if (!strcmp(argv[ai], "--learn-stream") && ai + 1 < argc) cfg.dir = argv[++ai];
        else if (!strcmp(argv[ai], "--tdleaf-out") && ai + 1 < argc)
            snprintf(cfg.tdleaf_out, sizeof(cfg.tdleaf_out), "%s", argv[++ai]);
        else if (!strcmp(argv[ai], "--total-games") && ai + 1 < argc)
            cfg.total_games = atol(argv[++ai]);
        else if (!strcmp(argv[ai], "--publish") && ai + 1 < argc) cfg.publish = argv[++ai];
        else if (!strcmp(argv[ai], "--publish-every") && ai + 1 < argc)
            cfg.publish_every = atoi(argv[++ai]);
        else if (!strcmp(argv[ai], "--publish-stamped")) cfg.publish_stamp = true;
        else if (!strcmp(argv[ai], "--delete"))         cfg.delete_consumed = true;
        else if (!strcmp(argv[ai], "--refresh-scores")) cfg.refresh_scores  = true;
        else if (!strcmp(argv[ai], "--lr-scale") && ai + 1 < argc)
            tdleaf_lr_scale = (float)atof(argv[++ai]);
        else if (!strcmp(argv[ai], "--opt-reset"))  opt_reset  = true;
        else if (!strcmp(argv[ai], "--grad-norm"))  nnue_grad_normalize = true;
    }

    if (!cfg.dir) { fprintf(stderr, "learner: --learn-stream <dir> is required\n"); return 1; }
    // Guard the sign and a sane ceiling: a negative scale would ascend the loss,
    // and anything above ~4 blows past the step clip on every category.
    if (!(tdleaf_lr_scale >= 0.0f) || tdleaf_lr_scale > 4.0f) {
        fprintf(stderr, "learner: --lr-scale %g out of range (expected 0..4)\n",
                (double)tdleaf_lr_scale);
        return 1;
    }
    if (tdleaf_lr_scale != 1.0f)
        fprintf(stderr, "learner: online LR scaled by %.4g (all categories)\n",
                (double)tdleaf_lr_scale);
    // Must run AFTER the .tdleaf.bin has been loaded (weights + moments come in
    // together at NNUE init), and before the first trajectory is consumed.
    if (opt_reset) nnue_reset_optimizer_state();
    if (nnue_grad_normalize)
        fprintf(stderr, "learner: gradients normalised per position "
                        "(mean, not batch sum)\n");
    if (!nnue_available) { fprintf(stderr, "learner: requires a loaded NNUE network\n"); return 1; }
    struct stat sb;
    if (stat(cfg.dir, &sb) != 0 || !S_ISDIR(sb.st_mode)) {
        fprintf(stderr, "learner: %s is not a directory\n", cfg.dir);
        return 1;
    }
    char done_dir[FILENAME_MAX], stop_path[FILENAME_MAX];
    snprintf(done_dir, sizeof(done_dir), "%s/done", cfg.dir);
    snprintf(stop_path, sizeof(stop_path), "%s/STOP", cfg.dir);
    if (!cfg.delete_consumed) mkdir(done_dir, 0755);

    TDGameRecord *grec = new TDGameRecord;   // ~10 MB — heap, reused per game
    grec->n_plies = 0;

    fprintf(stderr, "learner: consuming %s -> %s (%s consumed files; "
                    "stop: %s%s)\n",
            cfg.dir, cfg.tdleaf_out,
            cfg.delete_consumed ? "deleting" : "archiving",
            cfg.total_games ? "game budget or " : "", "STOP sentinel");

    long consumed = 0, rejected = 0;
    int  since_publish = 0;
    bool stopping = false;
    while (!stopping) {
        // Collect pending .tdg files, oldest first (mtime, then name).
        std::vector<std::pair<std::pair<long, std::string>, std::string> > files;
        DIR *d = opendir(cfg.dir);
        if (d) {
            struct dirent *e;
            while ((e = readdir(d)) != nullptr) {
                size_t l = strlen(e->d_name);
                if (l <= 4 || strcmp(e->d_name + l - 4, ".tdg") != 0) continue;
                char p[FILENAME_MAX];
                snprintf(p, sizeof(p), "%s/%s", cfg.dir, e->d_name);
                struct stat fs;
                if (stat(p, &fs) != 0) continue;
                files.push_back({{(long)fs.st_mtime, std::string(e->d_name)},
                                 std::string(e->d_name)});
            }
            closedir(d);
        }
        std::sort(files.begin(), files.end());

        if (files.empty()) {
            if (stat(stop_path, &sb) == 0) break;
            usleep(300000);
            continue;
        }

        for (auto &fe : files) {
            char path[FILENAME_MAX];
            snprintf(path, sizeof(path), "%s/%s", cfg.dir, fe.second.c_str());
            bool ok = learner_process_file(path, grec, cfg);
            if (cfg.delete_consumed) {
                remove(path);
            } else {
                char dst[FILENAME_MAX];
                snprintf(dst, sizeof(dst), "%s/%s", done_dir, fe.second.c_str());
                rename(path, dst);
            }
            if (ok) {
                consumed++;
                since_publish++;
                if (consumed % 100 == 0)
                    fprintf(stderr, "learner: %ld games consumed\n", consumed);
                if (cfg.publish && since_publish >= cfg.publish_every) {
                    learner_publish(cfg.publish, consumed, cfg.publish_stamp);
                    since_publish = 0;
                }
                if (cfg.total_games && consumed >= cfg.total_games) { stopping = true; break; }
            } else {
                rejected++;
            }
            if (stat(stop_path, &sb) == 0) { stopping = true; break; }
        }
    }

    tdleaf_flush_batch(cfg.tdleaf_out);
    if (cfg.publish) learner_publish(cfg.publish, consumed, cfg.publish_stamp);
    fprintf(stderr, "learner: done — %ld games consumed, %ld rejected\n",
            consumed, rejected);
    delete grec;
    return 0;
}

#endif // TDLEAF
