// uci.cpp — UCI protocol implementation for Leaf chess engine
// Included as part of the unity build via Leaf.cc (after main.cpp)

#include <string>
#include <sstream>
#include <deque>
#include <pthread.h>

//----------------------------------------------------------------------
// UCI command queue (I/O thread -> main/search thread)
//----------------------------------------------------------------------

static std::deque<std::string> uci_queue;
static pthread_mutex_t         uci_queue_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t          uci_queue_cond  = PTHREAD_COND_INITIALIZER;
static pthread_t               uci_io_thread_id;
static int                     uci_stop_flag = 0;   // set by stop/quit during search

// Stored go parameters for ponderhit time calculation
static int uci_go_wtime     = -1;
static int uci_go_btime     = -1;
static int uci_go_winc      = 0;
static int uci_go_binc      = 0;
static int uci_go_movestogo = 0;
static int uci_go_movetime  = -1;
static int uci_go_depth     = 0;
static int uci_go_infinite  = 0;
static int uci_chess960     = 0;   // UCI_Chess960 option

//----------------------------------------------------------------------
// I/O reader thread
//----------------------------------------------------------------------

static void *uci_io_reader(void *)
{
    std::string line;
    while (std::getline(std::cin, line)) {
        pthread_mutex_lock(&uci_queue_mutex);
        uci_queue.push_back(line);
        pthread_cond_signal(&uci_queue_cond);
        pthread_mutex_unlock(&uci_queue_mutex);
        // If we just pushed "quit", stop reading
        if (line.size() >= 4 && line.substr(0, 4) == "quit") break;
    }
    // EOF: push synthetic quit if queue is empty or last entry isn't quit
    pthread_mutex_lock(&uci_queue_mutex);
    if (uci_queue.empty() || uci_queue.back().substr(0,4) != "quit") {
        uci_queue.push_back("quit");
        pthread_cond_signal(&uci_queue_cond);
    }
    pthread_mutex_unlock(&uci_queue_mutex);
    return nullptr;
}

// Blocking dequeue (used by main loop when idle)
static std::string uci_dequeue_blocking()
{
    pthread_mutex_lock(&uci_queue_mutex);
    while (uci_queue.empty()) {
        pthread_cond_wait(&uci_queue_cond, &uci_queue_mutex);
    }
    std::string line = uci_queue.front();
    uci_queue.pop_front();
    pthread_mutex_unlock(&uci_queue_mutex);
    return line;
}

//----------------------------------------------------------------------
// Move format: Leaf internal <-> UCI long algebraic
//----------------------------------------------------------------------

// Convert a Leaf move to UCI string (e.g. "e2e4", "e7e8q")
static void uci_move_str(move m, char out[6], const position *pos = nullptr)
{
    if (!m.t) { strcpy(out, "0000"); return; }
    int from = m.b.from;
    int to   = m.b.to;

    // UCI_Chess960: castling is encoded as king-captures-own-rook
    if (uci_chess960 && (m.b.type & CASTLE) && pos) {
        // to is the king's destination (g1/c1/g8/c8 internally);
        // output the rook's starting square instead.
        int side = (from < 8) ? WHITE : BLACK;
        int rook_sq;
        if (to == 6 || to == 62)     // king-side
            rook_sq = pos->Krook[side];
        else                         // queen-side
            rook_sq = pos->Qrook[side];
        out[0] = 'a' + (from    & 7);  out[1] = '1' + (from    >> 3);
        out[2] = 'a' + (rook_sq & 7);  out[3] = '1' + (rook_sq >> 3);
        out[4] = '\0';
        return;
    }

    out[0] = 'a' + (from & 7);  out[1] = '1' + (from >> 3);
    out[2] = 'a' + (to   & 7);  out[3] = '1' + (to   >> 3);
    out[4] = '\0';
    if (m.b.type & PROMOTE) {
        // Leaf piece encoding: KNIGHT=2, BISHOP=3, ROOK=4, QUEEN=5
        static const char pc[] = { '?', '?', 'n', 'b', 'r', 'q' };
        int pidx = m.b.promote;
        if (pidx >= 2 && pidx <= 5) { out[4] = pc[pidx]; out[5] = '\0'; }
    }
}

// Parse a UCI move string and find the matching legal move.
// Returns a move with t==0 if not found.
static move uci_parse_move(const char *s, position &p, ts_thread_data *tdata_ptr)
{
    move nomove; nomove.t = 0;
    if (!s || strlen(s) < 4) return nomove;

    int fc = s[0] - 'a'; int fr = s[1] - '1';
    int tc = s[2] - 'a'; int tr = s[3] - '1';
    if (fc<0||fc>7||fr<0||fr>7||tc<0||tc>7||tr<0||tr>7) return nomove;
    int from = fr*8 + fc;
    int to   = tr*8 + tc;

    int promo = 0;
    if (s[4]) {
        switch(s[4]) {
            case 'q': promo = QUEEN;  break;
            case 'r': promo = ROOK;   break;
            case 'b': promo = BISHOP; break;
            case 'n': promo = KNIGHT; break;
        }
    }

    // UCI_Chess960: castling is king-captures-own-rook.
    // Detect king moving to own rook square and try as castling first.
    int castle_to = -1;
    if (uci_chess960) {
        int piece = p.sq[from] & 7;  // strip color bits
        if (piece == KING) {
            int side = (fr == 0) ? WHITE : BLACK;
            if (to == p.Krook[side]) {
                castle_to = (side == WHITE) ? 6 : 62;   // g1 or g8
            } else if (to == p.Qrook[side]) {
                castle_to = (side == WHITE) ? 2 : 58;   // c1 or c8
            }
        }
    }

    move_list ml;
    p.allmoves(&ml, tdata_ptr);

    // If this could be castling, try that first.
    if (castle_to >= 0) {
        for (int i = 0; i < ml.count; i++) {
            move m = ml.mv[i].m;
            if (m.b.from != from || m.b.to != castle_to) continue;
            if (!(m.b.type & CASTLE)) continue;
            position tmp = p;
            if (tmp.exec_move(m, 0)) return m;
        }
    }

    // Standard move matching (also handles non-castling king moves to rook square).
    for (int i = 0; i < ml.count; i++) {
        move m = ml.mv[i].m;
        if (m.b.from != from || m.b.to != to) continue;
        if (promo && !(m.b.type & PROMOTE)) continue;
        if (promo && m.b.promote != promo) continue;
        if (!promo && (m.b.type & PROMOTE)) continue;
        position tmp = p;
        if (tmp.exec_move(m, 0)) return m;
    }
    return nomove;
}

//----------------------------------------------------------------------
// uci_check_interrupt() -- called from inter() during search
// Returns 1 if search should stop, 0 to continue
//----------------------------------------------------------------------

int uci_check_interrupt()
{
    // Non-blocking check of the command queue
    pthread_mutex_lock(&uci_queue_mutex);
    while (!uci_queue.empty()) {
        std::string cmd = uci_queue.front();
        uci_queue.pop_front();
        pthread_mutex_unlock(&uci_queue_mutex);

        // Get first token
        std::istringstream iss(cmd);
        std::string token;
        if (!(iss >> token)) { pthread_mutex_lock(&uci_queue_mutex); continue; }

        if (token == "stop") {
            uci_stop_flag = 1;
            game.terminate_search = 1;
            return 1;
        } else if (token == "quit") {
            game.program_run = 0;
            game.terminate_search = 1;
            return 1;
        } else if (token == "ponderhit") {
            // Transition from ponder to real search: lift the analysis_mode lock
            uci_in_ponder = 0;
            game.ts.analysis_mode = 0;
            // Recompute time limit from stored go parameters
            int stm = game.pos.wtm;
            int tl_ms  = stm ? uci_go_wtime : uci_go_btime;
            int inc_ms = stm ? uci_go_winc  : uci_go_binc;
            if (tl_ms > 0) {
                game.timeleft[stm] = tl_ms / 10.0f;
                game.inc = inc_ms / 1000.0f;
                game.mttc = uci_go_movestogo;
                int projected = (int)game.timeleft[stm];
                int moves_rem = game.mttc + 1;
                if (!game.mttc || moves_rem > 40) moves_rem = 40;
                projected += moves_rem * (int)(game.inc * 100);
                projected = MAX(projected, (int)(game.timeleft[stm] / 2));
                int new_limit = 75 * projected / (100 * moves_rem);
                if (ponder_flag) new_limit = (115 * new_limit) / 100;
                new_limit = MIN(new_limit, (int)(game.timeleft[stm] / 2));
                if (game.inc > 0.0f && game.inc < 0.10f) {
                    new_limit = MAX(1, new_limit - MOVE_OVERHEAD_CS);
                }
                game.ts.limit     = new_limit;
                game.ts.max_limit = MIN((int)(8.0 * new_limit),
                                        MAX(new_limit, (int)(game.timeleft[stm] / 4.0)));
            }
            // Don't stop the search -- let it continue with new time limits
            return 0;
        } else if (token == "isready") {
            // Must respond even during search
            printf("readyok\n"); fflush(stdout);
        }
        // Other commands during search are ignored
        pthread_mutex_lock(&uci_queue_mutex);
    }
    pthread_mutex_unlock(&uci_queue_mutex);
    return 0;
}

//----------------------------------------------------------------------
// Send UCI options
//----------------------------------------------------------------------

static void uci_send_options()
{
    printf("option name Hash type spin default 128 min 1 max 4096\n");
    printf("option name Threads type spin default 1 min 1 max %d\n", MAX_THREADS);
    printf("option name Ponder type check default false\n");
    printf("option name UCI_AnalyseMode type check default false\n");
    printf("option name UCI_Chess960 type check default false\n");
    printf("option name Skill type spin default %d min 1 max 100\n", game.knowledge_scale);
    fflush(stdout);
}

//----------------------------------------------------------------------
// Handle setoption
//----------------------------------------------------------------------

static void uci_setoption(const std::string &line)
{
    // Format: "setoption name <name> value <value>"
    std::istringstream iss(line);
    std::string tok, name_str, val_str;
    iss >> tok; // "setoption"
    iss >> tok; // "name"
    // name may be multi-word up to "value"
    while (iss >> tok && tok != "value") {
        if (!name_str.empty()) name_str += ' ';
        name_str += tok;
    }
    // rest is value
    std::getline(iss, val_str);
    // trim leading space
    if (!val_str.empty() && val_str[0] == ' ') val_str = val_str.substr(1);

    if (name_str == "Hash") {
        int mb = atoi(val_str.c_str());
        if (mb > 0) set_hash_size(mb);
    } else if (name_str == "Threads") {
        int n = atoi(val_str.c_str());
        THREADS = (unsigned int)MIN(MAX(1, n), MAX_THREADS);
        game.ts.initialize_extra_threads();
    } else if (name_str == "Ponder") {
        ponder_flag = (val_str == "true") ? 1 : 0;
    } else if (name_str == "UCI_Chess960") {
        uci_chess960 = (val_str == "true") ? 1 : 0;
    } else if (name_str == "Skill") {
        int v = atoi(val_str.c_str());
        if (v < 1) v = 1;
        if (v > 100) v = 100;
        game.knowledge_scale = v;
    }
    // UCI_AnalyseMode handled per "go infinite"
}

//----------------------------------------------------------------------
// Handle position command
//----------------------------------------------------------------------

static void uci_set_position(const std::string &line)
{
    std::istringstream iss(line);
    std::string tok;
    iss >> tok; // "position"
    iss >> tok; // "startpos" or "fen"

    if (tok == "startpos") {
        game.setboard((char*)i_pos, 'w', (char*)"KQkq", (char*)"-");
        game.T = 1;
    } else if (tok == "fen") {
        // Read FEN fields
        std::string pieces, color, castling, ep, halfmove, fullmove;
        iss >> pieces >> color >> castling >> ep >> halfmove >> fullmove;
        char fen_pieces[128], fen_castling[8], fen_ep[8];
        strncpy(fen_pieces,   pieces.c_str(),   sizeof(fen_pieces)-1);   fen_pieces[sizeof(fen_pieces)-1] = '\0';
        strncpy(fen_castling, castling.c_str(), sizeof(fen_castling)-1); fen_castling[sizeof(fen_castling)-1] = '\0';
        strncpy(fen_ep,       ep.c_str(),       sizeof(fen_ep)-1);       fen_ep[sizeof(fen_ep)-1] = '\0';
        char color_char = color.empty() ? 'w' : color[0];
        game.setboard(fen_pieces, color_char, fen_castling, fen_ep);
        game.T = (color_char == 'w') ? 1 : 2;
        // Set fifty-move counter from FEN half-move clock
        if (!halfmove.empty()) {
            game.pos.fifty = atoi(halfmove.c_str()) * 2;
        }
    } else {
        return; // unknown format
    }

    // Reset singular response for new position
    game.ts.singular_response.t = NOMOVE;

    // Replay the move list after "moves" keyword
    while (iss >> tok) {
        if (tok == "moves") continue;
        move m = uci_parse_move(tok.c_str(), game.pos, &game.ts.tdata[0]);
        if (!m.t) break;
        // Record hash in tdata position lists for all threads (for repetition detection)
        for (int ti = 0; ti < MAX_THREADS; ti++) {
            game.ts.tdata[ti].plist[game.T] = game.pos.hcode;
        }
        if (!game.pos.exec_move(m, 0)) break;
        game.T++;
    }
    // Sync p_side
    game.p_side = game.pos.wtm;
    // Clear first PV entry to avoid stale data
    game.ts.last_depth = 1;
    game.ts.last_ponder = 0;
    for (int ti = 0; ti < THREADS; ti++) {
        game.ts.tdata[ti].pc[0][0].t = NOMOVE;
    }
}

//----------------------------------------------------------------------
// Handle go command -- compute time limit and start search
//----------------------------------------------------------------------

static void uci_dispatch_go(const std::string &line)
{
    std::istringstream iss(line);
    std::string tok;
    iss >> tok; // "go"

    // Parse all go parameters
    uci_go_wtime     = -1; uci_go_btime    = -1;
    uci_go_winc      = 0;  uci_go_binc     = 0;
    uci_go_movestogo = 0;  uci_go_movetime = -1;
    uci_go_depth     = 0;  uci_go_infinite = 0;
    int ponder_flag_go = 0;

    while (iss >> tok) {
        if      (tok == "wtime")     { iss >> uci_go_wtime; }
        else if (tok == "btime")     { iss >> uci_go_btime; }
        else if (tok == "winc")      { iss >> uci_go_winc; }
        else if (tok == "binc")      { iss >> uci_go_binc; }
        else if (tok == "movestogo") { iss >> uci_go_movestogo; }
        else if (tok == "movetime")  { iss >> uci_go_movetime; }
        else if (tok == "depth")     { iss >> uci_go_depth; }
        else if (tok == "infinite")  { uci_go_infinite = 1; }
        else if (tok == "ponder")    { ponder_flag_go = 1; }
        // "nodes", "searchmoves" not yet implemented
    }

    int stm = game.pos.wtm;
    uci_stop_flag = 0;
    game.terminate_search = 0;

    if (ponder_flag_go) {
        // UCI ponder: use analysis_mode + uci_in_ponder so search doesn't time out.
        // Loop the search because the ID loop only runs to MAX_MAIN_TREE (depth 79)
        // and would return naturally on simple positions before stop/ponderhit arrives.
        uci_in_ponder = 1;
        game.ts.analysis_mode = 1;
        game.ts.max_search_depth = MAXD;
        game.ts.max_search_time  = MAXT;
        game.timeleft[stm] = float(MAXT * 100);
        game.inc  = 0.0f;
        game.mttc = 0;
        while (uci_in_ponder && !uci_stop_flag && game.program_run) {
            game.best = game.ts.search(game.pos, MAXT, game.T, &game);
        }
        uci_in_ponder = 0;
        game.ts.analysis_mode = 0;
    } else {
        // Normal search
        game.ts.analysis_mode    = uci_go_infinite ? 1 : 0;
        game.ts.max_search_depth = (uci_go_depth > 0) ? uci_go_depth : MAXD;
        game.ts.max_search_time  = MAXT;

        // Set clock state from UCI go parameters
        if (uci_go_wtime >= 0) game.timeleft[1] = uci_go_wtime / 10.0f;
        if (uci_go_btime >= 0) game.timeleft[0] = uci_go_btime / 10.0f;
        game.inc  = stm ? uci_go_winc / 1000.0f : uci_go_binc / 1000.0f;
        game.mttc = uci_go_movestogo;

        int time_limit;
        if (uci_go_movetime > 0) {
            time_limit = uci_go_movetime / 10;
            game.mttc  = 1; // treat as one move remaining
        } else if (uci_go_infinite || uci_go_depth > 0) {
            time_limit = MAXT;
        } else {
            // Mirror make_move() GUI time computation
            int projected  = (int)game.timeleft[stm];
            int moves_rem  = game.mttc + 1;
            if (!game.mttc || moves_rem > 40) moves_rem = 40;
            projected += moves_rem * (int)(game.inc * 100);
            if (interface_lag_count >= LAG_COUNT) {
                projected -= moves_rem * average_lag;
            }
            projected  = MAX(projected, (int)(game.timeleft[stm] / 2));
            time_limit = 75 * projected / (100 * moves_rem);
            if (ponder_flag) time_limit = (115 * time_limit) / 100;
            time_limit = MIN(time_limit, (int)(game.timeleft[stm] / 2));
            if (game.inc > 0.0f && game.inc < 0.10f) {
                time_limit = MAX(1, time_limit - MOVE_OVERHEAD_CS);
            }
        }

        int search_start = GetTime();
        if (uci_go_infinite) {
            // Loop until stop received — same reason as ponder: ID exits naturally at depth 79
            while (game.ts.analysis_mode && !uci_stop_flag && game.program_run) {
                game.best = game.ts.search(game.pos, time_limit, game.T, &game);
            }
            game.ts.analysis_mode = 0;
        } else {
            game.best = game.ts.search(game.pos, time_limit, game.T, &game);
        }
        int elapsed_cs = GetTime() - search_start;

        // Update time tracking
        game.timeleft[stm] -= float(elapsed_cs);
        game.timeleft[stm] += float(game.inc * 100);
        if (game.timeleft[stm] < 0.0f) game.timeleft[stm] = 0.0f;

        // Decrement mttc if using time control
        if (game.mttc > 0 && !uci_go_movetime) {
            if (!(game.T & 1)) {
                game.mttc--;
                if (!game.mttc) {
                    game.timeleft[0] += game.base * 100;
                    game.timeleft[1] += game.base * 100;
                    game.mttc = game.omttc;
                }
            }
        }
    }

    // If search returned no move (e.g. immediate stop before first iteration),
    // find any legal move as fallback
    if (!game.best.t) {
        move_list ml;
        game.pos.allmoves(&ml, &game.ts.tdata[0]);
        for (int i = 0; i < ml.count; i++) {
            position tmp = game.pos;
            if (tmp.exec_move(ml.mv[i].m, 0)) {
                game.best = ml.mv[i].m;
                break;
            }
        }
    }

    char best_str[6] = "0000";
    uci_move_str(game.best, best_str, &game.pos);

    // Ponder move is PV[1] if ponder option is on
    move ponder_mv; ponder_mv.t = 0;
    if (ponder_flag && !uci_go_infinite) {
        ponder_mv = game.ts.tdata[0].pc[0][1];
    }

    if (ponder_mv.t) {
        // Verify the ponder move is legal in the position after best move
        position tmp = game.pos;
        if (tmp.exec_move(game.best, 0)) {
            char ponder_str[6] = "0000";
            uci_move_str(ponder_mv, ponder_str, &tmp);
            printf("bestmove %s ponder %s\n", best_str, ponder_str);
        } else {
            printf("bestmove %s\n", best_str);
        }
    } else {
        printf("bestmove %s\n", best_str);
    }
    fflush(stdout);
}

//----------------------------------------------------------------------
// UCI info output -- called from log_search() in support.cpp
//----------------------------------------------------------------------

void uci_send_info(int score, int depth, int elapsed_cs, unsigned long long nodes, tree_search *ts)
{
    if (!uci_mode) return;

    int elapsed_ms = elapsed_cs * 10;
    long long nps  = (elapsed_cs > 0) ? (long long)(100LL * (long long)nodes / elapsed_cs) : 0LL;

    if (score > MATE/2) {
        int mate_in = (MATE - score + 1) / 2;
        printf("info depth %d score mate %d time %d nodes %llu nps %lld",
               depth, mate_in, elapsed_ms, nodes, nps);
    } else if (score < -MATE/2) {
        int mate_in = (MATE + score + 1) / 2;
        printf("info depth %d score mate -%d time %d nodes %llu nps %lld",
               depth, mate_in, elapsed_ms, nodes, nps);
    } else {
        int cp = (value[PAWN] > 0) ? (score * 100) / value[PAWN] : score;
        printf("info depth %d score cp %d time %d nodes %llu nps %lld",
               depth, cp, elapsed_ms, nodes, nps);
    }

    // PV
    printf(" pv");
    position p = ts->tdata[0].n[0].pos;
    for (int i = 0; i < MAXD; i++) {
        move m = ts->tdata[0].pc[0][i];
        if (!m.t) break;
        char ms[6];
        uci_move_str(m, ms, &p);
        printf(" %s", ms);
        if (!p.exec_move(m, 0)) break;
    }
    printf("\n");
    fflush(stdout);
}

//----------------------------------------------------------------------
// Main UCI loop
//----------------------------------------------------------------------

void uci_loop(game_rec *gr)
{
    // Send identification and options
    printf("id name Leaf v%s%s\n", VERS, VERS2);
    printf("id author Daniel C. Homan\n");
    uci_send_options();
    printf("uciok\n");
    fflush(stdout);

    // Start I/O reader thread
    pthread_create(&uci_io_thread_id, nullptr, uci_io_reader, nullptr);

    while (gr->program_run) {
        std::string line = uci_dequeue_blocking();

        // Get first token
        std::istringstream iss(line);
        std::string tok;
        if (!(iss >> tok)) continue;

        if (tok == "isready") {
            printf("readyok\n"); fflush(stdout);

        } else if (tok == "uci") {
            // GUI re-sent uci (valid, re-identify)
            printf("id name Leaf v%s%s\n", VERS, VERS2);
            printf("id author Daniel C. Homan\n");
            uci_send_options();
            printf("uciok\n"); fflush(stdout);

        } else if (tok == "ucinewgame") {
            // Reset engine state for a new game
            set_hash_size(HASH_SIZE);
            game.setboard((char*)i_pos, 'w', (char*)"KQkq", (char*)"-");
            game.T = 1;
            game.ts.last_ponder = 0;
            game.ts.last_depth  = 1;
            game.ts.singular_response.t = NOMOVE;
            game.ts.h_id = 0;
            for (int ti = 0; ti < THREADS; ti++) {
                for (int i = 0; i < 15; i++) {
                    for (int j = 0; j < 64; j++) {
                        game.ts.tdata[ti].history[i][j] = 0;
                        game.ts.tdata[ti].reply[i][j]   = 0;
                    }
                }
                game.ts.tdata[ti].pc[0][0].t = NOMOVE;
            }

        } else if (tok == "position") {
            uci_set_position(line);

        } else if (tok == "go") {
            uci_dispatch_go(line);

        } else if (tok == "stop") {
            // stop without an active search -- no-op
            uci_stop_flag = 0;
            game.terminate_search = 0;

        } else if (tok == "setoption") {
            uci_setoption(line);

        } else if (tok == "debug") {
            // optional: ignore

        } else if (tok == "quit") {
            gr->program_run = 0;
            break;
        }
        // ponderhit outside of search: handled inside uci_check_interrupt
        // if it arrives here the search already ended (ponder miss); ignore it
    }

    pthread_cancel(uci_io_thread_id);
    pthread_join(uci_io_thread_id, nullptr);
}
