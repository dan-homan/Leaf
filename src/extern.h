/* Global constants used all over the program */

#ifndef EXTERN_H
#define EXTERN_H

#include <cstdio>
#include <fstream>

/* setup.cpp */
extern int taxi_cab[64][64];
//
// Simple tables for quick in_check? tests 
//
extern uint64_t check_table[64];
extern uint64_t rook_check_table[64];
extern uint64_t bishop_check_table[64];
extern uint64_t knight_check_table[64];
extern uint64_t slide_check_table[64];

extern int logging;                      // flag for logging  

/* smp.cpp */
extern unsigned int THREADS, LIMIT_THREADS;                   

/* score.h */
extern int value[7];                              // piece values, indexed by type
extern int BAD_BISHOP, WEAK_PAWN_EARLY, WEAK_PAWN_LATE, BACKWARD_PAWN_EARLY, BACKWARD_PAWN_LATE,
           PAWN_ISLAND_EARLY, PAWN_ISLAND_LATE, PASSED_PAWN, BISHOP_PAIR,
           CON_PASSED_PAWNS, TRADES_EARLY, TRADES_LATE,
           HALF_FILE_BONUS, CASTLED, NO_POSSIBLE_CASTLE,
           ROOK_KING_FILE, ROOK_MOBILITY, QUEEN_MOBILITY, KNIGHT_MOBILITY,
           PAWN_DUO, BOXED_IN_ROOK, MINOR_OUTPOST, MINOR_OUTP_GUARD, ROOK_OPEN_FILE, 
           DOUBLED_PAWN_EARLY, DOUBLED_PAWN_LATE, //CONNECTED_ROOKS,// ROOK_ON_7TH,
           ROOK_HALF_OPEN_FILE, SIDE_ON_MOVE_EARLY, SIDE_ON_MOVE_LATE, MINOR_BLOCKER,
           BMINOR_OUTPOST, BMINOR_OUTP_GUARD, BMINOR_BLOCKER,
           PAWN_THREAT_MINOR, PAWN_THREAT_MAJOR, SPACE;

/* search.h -- extensions */
extern int CHESS_SKILL, NULL_MOVE, VERIFY_MARGIN;
extern int NO_ROOT_LMR_SCORE, DRAW_SCORE;
extern float VAR1, VAR2, VAR3, VAR4;

/* other external variables */
extern char exec_path[FILENAME_MAX], BOOK_FILE[FILENAME_MAX], START_BOOK[FILENAME_MAX];
extern int MAX_LOGS, GAMBIT_SCORE, BOOK_LEARNING, SCORE_LEARNING;
extern int SCORE_LEARNING;

/* hash.h */
extern const h_code h_pv, hstm, hval[13][64];
extern const h_code castle_code[16], ep_code[8];
extern unsigned int TAB_SIZE, PAWN_SIZE, SCORE_SIZE, CMOVE_SIZE;
extern unsigned int phash_count;
extern pawn_rec *pawn_table;
extern int HASH_SIZE;

/* main.cpp */
extern int xboard, post;           // xboard flag, posting flag
extern int ALLEG;                  // flags from main.cpp to control the book
                                   // and playing modes
extern ofstream logfile;           // logfile
extern int average_lag;
extern int uci_mode;               // 1 when running UCI protocol
extern int interface_mode;         // 1 when any GUI protocol is active (xboard or UCI)
extern int uci_in_ponder;          // 1 when UCI ponder search is running

/* book.cpp */
extern int learn_count;

/* smp.cpp */
extern pthread_mutex_t log_lock;
extern pthread_mutex_t egtb_lock;

/* fltk_gui.cpp */
extern int FLTK_post;
extern int abortflag;
#if FLTK_GUI
#include <FL/Fl_Text_Buffer.H>
#include <FL/Fl_Text_Display.H>
extern Fl_Text_Buffer *searchout_buffer;
extern Fl_Text_Display *searchout;
#endif

#if TABLEBASES
extern char EGTB_PATH[FILENAME_MAX];
extern float CACHE_SIZE;
#endif


#endif  /* EXTERN_H */

