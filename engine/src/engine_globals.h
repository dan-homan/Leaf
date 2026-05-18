// engine_globals.h — grouped global state for the Leaf chess engine.
// Each struct bundles related globals that were previously scattered across
// multiple files.  Global instances are defined in main.cpp / smp.cpp.

#ifndef ENGINE_GLOBALS_H
#define ENGINE_GLOBALS_H

#include <fstream>
#include <pthread.h>

using std::ofstream;

// ---------------------------------------------------------------------------
// Protocol state — interface mode flags and logging
// ---------------------------------------------------------------------------
struct ProtocolState {
    int xboard         = 0;
    int post           = 0;
    int uci_mode       = 0;
    int interface_mode  = 0;
    int uci_in_ponder  = 0;
    int logging        = 0;
    ofstream logfile;
};

// ---------------------------------------------------------------------------
// Search tuning parameters — loaded at startup, adjustable via UCI/CLI
// ---------------------------------------------------------------------------
struct SearchConfig {
    int null_move          = 1;
    int verify_margin      = 200;
    int draw_score         = -20;
    int no_root_lmr_score  = 85;
    int extend_time_score  = 15;
    int move_overhead_cs   = 2;
    int check_inter        = 4095;
    float var1 = 4.95f;
    float var2 = 2.46f;
    float var3 = 20.1f;
    float var4 = 2.685f;
    int abort_search_fraction[8] = { 0, 128, 406, 625, 717, 808, 900, 1100 };
};

// ---------------------------------------------------------------------------
// Engine configuration — paths, limits, feature toggles
// ---------------------------------------------------------------------------
struct EngineConfig {
    char exec_path[FILENAME_MAX]  = {};
    char book_file[FILENAME_MAX]  = {};
    char start_book[FILENAME_MAX] = {};
    int max_logs       = 100;
    int gambit_score   = 80;
    int book_learning  = 0;
    int chess_skill    = KNOWLEDGE;  // compile-time default; see define.h
    int hash_size      = 128;
};

// ---------------------------------------------------------------------------
// Thread configuration
// ---------------------------------------------------------------------------
struct ThreadConfig {
    unsigned int threads = 1;
    pthread_mutex_t log_lock;
};

#endif // ENGINE_GLOBALS_H
