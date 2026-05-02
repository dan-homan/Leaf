/* Global constants and struct instances used across the engine */

#ifndef EXTERN_H
#define EXTERN_H

#include <cstdio>
#include "engine_globals.h"

// ---------------------------------------------------------------------------
// Grouped state instances (defined in main.cpp)
// ---------------------------------------------------------------------------
extern ProtocolState proto;
extern EngineConfig  engine_cfg;
extern SearchConfig  search_cfg;
extern ThreadConfig  thread_cfg;

// ---------------------------------------------------------------------------
// Setup tables (defined in setup.cpp)
// ---------------------------------------------------------------------------
extern int taxi_cab[64][64];
extern uint64_t check_table[64];
extern uint64_t rook_check_table[64];
extern uint64_t bishop_check_table[64];
extern uint64_t knight_check_table[64];
extern uint64_t slide_check_table[64];
// proto.logging replaces the old int logging (now a ProtocolState member)

// ---------------------------------------------------------------------------
// Eval weights — piece values and hand-crafted scoring (defined in score.h)
// ---------------------------------------------------------------------------
extern int value[7];                              // piece values, indexed by type
extern int BAD_BISHOP, WEAK_PAWN_EARLY, WEAK_PAWN_LATE, BACKWARD_PAWN_EARLY, BACKWARD_PAWN_LATE,
           PAWN_ISLAND_EARLY, PAWN_ISLAND_LATE, PASSED_PAWN, BISHOP_PAIR,
           CON_PASSED_PAWNS, TRADES_EARLY, TRADES_LATE,
           HALF_FILE_BONUS, CASTLED, NO_POSSIBLE_CASTLE,
           ROOK_KING_FILE, ROOK_MOBILITY, QUEEN_MOBILITY, KNIGHT_MOBILITY,
           PAWN_DUO, BOXED_IN_ROOK, MINOR_OUTPOST, MINOR_OUTP_GUARD, ROOK_OPEN_FILE,
           DOUBLED_PAWN_EARLY, DOUBLED_PAWN_LATE,
           ROOK_HALF_OPEN_FILE, SIDE_ON_MOVE_EARLY, SIDE_ON_MOVE_LATE, MINOR_BLOCKER,
           BMINOR_OUTPOST, BMINOR_OUTP_GUARD, BMINOR_BLOCKER,
           PAWN_THREAT_MINOR, PAWN_THREAT_MAJOR, SPACE;

// ---------------------------------------------------------------------------
// Hash tables and Zobrist keys (defined in hash.cpp / setup.cpp)
// ---------------------------------------------------------------------------
extern const h_code h_pv, hstm, hval[13][64];
extern const h_code castle_code[16], ep_code[8];
extern unsigned int TAB_SIZE, PAWN_SIZE, SCORE_SIZE, CMOVE_SIZE;
extern unsigned int phash_count;
extern pawn_rec *pawn_table;

// ---------------------------------------------------------------------------
// Misc globals not yet grouped
// ---------------------------------------------------------------------------
extern int ALLEG;

// ---------------------------------------------------------------------------
// fltk_gui.cpp (optional GUI)
// ---------------------------------------------------------------------------
extern int FLTK_post;
extern int abortflag;
#if FLTK_GUI
#include <FL/Fl_Text_Buffer.H>
#include <FL/Fl_Text_Display.H>
extern Fl_Text_Buffer *searchout_buffer;
extern Fl_Text_Display *searchout;
#endif

#endif  /* EXTERN_H */
