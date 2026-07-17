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

#if TDLEAF

#include <vector>
#include <random>
#include <algorithm>

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
    int      max_ply;      // draw adjudication cap (game plies)
    int      epd_offset;
    int      epd_stride;
    unsigned shuffle_seed;
    bool     shuffle;
    bool     adjudicate;
    bool     verbose;
    char     tdleaf_out[FILENAME_MAX];
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
// selfplay game starts from the same clean state as a harness game: full
// hash-table realloc (TT probes match on key only, so stale entries from the
// previous game WOULD be probed without this), h_id/depth trackers, and
// history/reply tables.
// ---------------------------------------------------------------------------
static void selfplay_new_game_reset()
{
    set_hash_size(engine_cfg.hash_size);
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
// Play one game from an opening.  Returns the termination reason; result_w is
// the white-POV outcome (undefined for SP_TERM_ERROR).
// ---------------------------------------------------------------------------
static SelfplayTerm selfplay_play_game(const SelfplayEpdLine &op, const SelfplayConfig &cfg,
                                       float &result_w)
{
    selfplay_new_game_reset();
    game.setboard(op.board, op.ms, op.castle, op.ep);
    game.book = 0;                       // never probe the opening book
    game.over = 0;
    game.mttc = 0;
    game.ts.max_search_depth = cfg.depth;
    game.ts.analysis_mode = 0;
    // Fixed-depth search: give it effectively unlimited clock, depth stops ID.
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
        game.best = game.ts.search(game.pos, MAXT, game.T, &game);

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
            int rep_count = 0;
            for (int ri = game.T - 2; ri >= game.T - game.pos.fifty && rep_count < 2; ri -= 2)
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
    cfg.max_ply    = 500;
    cfg.epd_offset = 0;
    cfg.epd_stride = 1;
    cfg.shuffle_seed = 0;
    cfg.shuffle    = false;
    cfg.adjudicate = true;
    cfg.verbose    = false;
    snprintf(cfg.tdleaf_out, sizeof(cfg.tdleaf_out), "%s%s",
             engine_cfg.exec_path, NNUE_TDLEAF_BIN);

    for (int ai = 1; ai < argc; ai++) {
        if (!strcmp(argv[ai], "--epd")        && ai + 1 < argc) cfg.epd_path   = argv[++ai];
        else if (!strcmp(argv[ai], "--games") && ai + 1 < argc) cfg.games      = atoi(argv[++ai]);
        else if (!strcmp(argv[ai], "--depth") && ai + 1 < argc) cfg.depth      = atoi(argv[++ai]);
        else if (!strcmp(argv[ai], "--max-ply") && ai + 1 < argc) cfg.max_ply  = atoi(argv[++ai]);
        else if (!strcmp(argv[ai], "--epd-offset") && ai + 1 < argc) cfg.epd_offset = atoi(argv[++ai]);
        else if (!strcmp(argv[ai], "--epd-stride") && ai + 1 < argc) cfg.epd_stride = atoi(argv[++ai]);
        else if (!strcmp(argv[ai], "--epd-shuffle") && ai + 1 < argc) {
            cfg.shuffle = true; cfg.shuffle_seed = (unsigned)atol(argv[++ai]);
        }
        else if (!strcmp(argv[ai], "--tdleaf-out") && ai + 1 < argc) {
            snprintf(cfg.tdleaf_out, sizeof(cfg.tdleaf_out), "%s", argv[++ai]);
        }
        else if (!strcmp(argv[ai], "--no-adjudication")) cfg.adjudicate = false;
        else if (!strcmp(argv[ai], "--verbose"))         cfg.verbose    = true;
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
    fprintf(stderr, "selfplay: %d games, depth %d, %zu openings (slice %d: offset %d stride %d)%s%s\n",
            total_games, cfg.depth, openings.size(), slice_count,
            cfg.epd_offset, cfg.epd_stride,
            tdleaf_frozen() ? ", weights FROZEN" : "",
            getenv("TDLEAF_DUMP_TSV") ? ", dumping TSV" : "");

    SelfplayStats st;
    memset(&st, 0, sizeof(st));
    int start_time = GetTime();

    for (int g = 0; g < total_games; g++) {
        size_t idx = (size_t)cfg.epd_offset +
                     (size_t)((g % slice_count)) * (size_t)cfg.epd_stride;
        float result_w = 0.5f;
        SelfplayTerm term = selfplay_play_game(openings[idx], cfg, result_w);

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
        // repetition draws, otherwise run the TDLeaf update + replay.
        if (nnue_available) {
            if (term == SP_TERM_REP && game.td_game.n_plies < TDLEAF_MIN_PLIES_REP) {
                fprintf(stderr, "[TDLeaf] Skipping early 3-rep draw (%d plies < %d)\n",
                        game.td_game.n_plies, TDLEAF_MIN_PLIES_REP);
                st.skipped_short++;
            } else {
                tdleaf_update_after_game(game.td_game, result_w, cfg.tdleaf_out);
                tdleaf_replay(game.td_game, result_w, cfg.tdleaf_out);
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

    fprintf(stderr,
            "selfplay: done — %d played (+%d =%d -%d white POV), %d skipped\n"
            "selfplay: terminations: mate %d, stalemate %d, 50-move %d, 3-rep %d, "
            "material %d, max-ply %d, resign %d, draw-adj %d, error %d\n",
            st.played, st.white_wins, st.draws, st.black_wins, st.skipped_short,
            st.term_mate, st.term_stale, st.term_fifty, st.term_rep,
            st.term_material, st.term_maxply, st.term_resign, st.term_drawadj,
            st.term_error);
    return 0;
}

#endif // TDLEAF
