// EXchess source code, (c) Daniel C. Homan  1997-2017
// Released under the GNU public license, see file license.txt

/*--------------------- setup.cpp ------------------------*/
//
//  This file contains setup routines for EXchess.  These
//  routines are run when EXchess first starts up.
//

#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstdlib>
#include <cstring>

#include "define.h"
#include "chess.h"
#include "const.h"
#include "funct.h"
#include "hash.h"
#include "extern.h"

/* Simple tables for quick in_check? tests */
uint64_t check_table[64];
uint64_t rook_check_table[64];
uint64_t bishop_check_table[64];
uint64_t knight_check_table[64];
uint64_t slide_check_table[64];

/* flag for logging */
int logging = 0;

/* taxi-cab distance between squares */
int taxi_cab[64][64];

char SCORES_FILE[100] = "score.par";

/*
// variables for dynamic piece square allocation
//  -- see score.h for how these are supposed to look

float OFFSET_PSQ[2][7] = {
 { 0, 0, 0, 0, 0, 0, 0 },
 { 0, 0, 0, 0, 0, 0, 0 }
};
float SCALE_PSQ[2][7] = { 
 {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 },
 {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 }
};
char piece_sq_save[4][7][64];
*/

/*-------- Set search and scoring parameters ------------*/
//
// This function sets certain search and scoring parameters.
// It is part of the initialization startup of EXchess.
//
//

void set_search_param()
{

 char dummy[50], line[200], parameter_file[100];

 // look for the search parameter file in the
 // executable directory and in the current directory

 strcpy(parameter_file, exec_path);
 strcat(parameter_file, "search.par");

/*
 // save piece-sq information
 for(int i = 0; i < 4; i++)
   for(int j = 0; j < 7; j++)
     for(int k = 0; k < 64; k++)  piece_sq_save[i][j][k] = piece_sq[i][j][k];

 fill_psq(PAWN);
 fill_psq(KNIGHT);
 fill_psq(BISHOP);
 fill_psq(ROOK);
 fill_psq(QUEEN);
 fill_psq(KING);
*/
 
 ifstream parfile(parameter_file, IOS_IN_TEXT);
 if(!parfile) parfile.open("search.par", IOS_IN_TEXT);

 if(parfile) {
   // Read through search parameter file, setting
   // values as appropriate.
   
   while(!parfile.eof()) {
     parfile >> dummy;
     if(dummy[0] == '#') {
       parfile.getline(line, 199);
       continue;
     }
     
     if(!strcmp(dummy, "NULL_MOVE")) {
       parfile >> NULL_MOVE;
//     } else if(!strcmp(dummy, "NO_ROOT_LMR_SCORE")) {
//       parfile >> NO_ROOT_LMR_SCORE;
#if TABLEBASES     
     } else if(!strcmp(dummy, "EGTB_PATH")) {
       parfile >> EGTB_PATH;
     } else if(!strcmp(dummy, "EGTB_CACHE_SIZE")) {
       parfile >> CACHE_SIZE;
#endif
     } else if(!strcmp(dummy, "LOG")) {
       parfile >> logging;
     } else if(!strcmp(dummy, "MAX_LOGS")) {
       parfile >> MAX_LOGS;
     } else if(!strcmp(dummy, "BOOK_FILE")) {
       parfile >> BOOK_FILE;
     } else if(!strcmp(dummy, "START_BOOK")) {
       parfile >> START_BOOK;
     } else if(!strcmp(dummy, "HASH")) {
       parfile >> HASH_SIZE;
     } else if(!strcmp(dummy, "XBOARD")) {
       parfile >> xboard; 
     } else if(!strcmp(dummy, "THREADS")) {
       parfile >> THREADS; 
       if(THREADS > 0) { 
	 THREADS = MIN(THREADS,MAX_THREADS);
       } else {
	 cout << "ERROR(parfile has invalid THREADS value)\n";
       }
     } else if(!strcmp(dummy, "CHESS_SKILL")) {
       parfile >> CHESS_SKILL; 
     } else parfile.getline(line, 199);
     
   }

 } else { cout << "Error(NoParFile--Using defaults)"; }

 //---------------------------- 
 // Apply some of these values 
 //---------------------------- 
 // Do we learn? 
 game.learn_bk = BOOK_LEARNING;
 // Set chess skill
 game.knowledge_scale = CHESS_SKILL;
 if(game.knowledge_scale < 1) game.knowledge_scale = 1;
 if(game.knowledge_scale > 100) game.knowledge_scale = 100;

}

/* Function to generate check tables */
//
//  Simply sets a "1" if a given FROM-TO attack
//  combination is at all possible.  Sets a "0" if
//  such a combination is not possible.
//
//  These tables help to quickly rule out certain
//  kinds of attacks which could generate a check.
//
//  Function also generates the 'taxi-cab' distance
//  table used in the scoring functions

void gen_check_table()
{

  int i, j;
  // initialize these all to zero
  for(i = 0; i < 64; i++) {
    check_table[i] = ZERO;
    rook_check_table[i] = ZERO;
    bishop_check_table[i] = ZERO;
    slide_check_table[i] = ZERO;
    knight_check_table[i] = ZERO;
  }

  // look for connections and set bits for allowed attacks
  for(i = 0; i < 64; i++) {
   for(j = 0; j < 64; j++) {
     // setup the taxi-cab distance table
     taxi_cab[i][j] = MAX((ABS(FILE(i)-FILE(j))),(ABS(RANK(i)-RANK(j))));
     if((FILE(i)==FILE(j) || RANK(i)==RANK(j)) && i != j) {
       rook_check_table[i] |= (1ULL<<j);
       slide_check_table[i] |= (1ULL<<j);
       check_table[i] |= (1ULL<<j);
     } 
   }
  }

  for(i = 0; i < 64; i++) {
    j = i; while(RANK(j) && FILE(j) && j >= 0)
	     { j -= 9; bishop_check_table[i] |= (1ULL<<j); slide_check_table[i] |= (1ULL<<j); check_table[i] |= (1ULL<<j); }
    j = i; while(RANK(j) && FILE(j) < 7 && j >= 0)
	     { j -= 7; bishop_check_table[i] |= (1ULL<<j); slide_check_table[i] |= (1ULL<<j); check_table[i] |= (1ULL<<j); }
    j = i; while(RANK(j) < 7 && FILE(j) && j <= 63)
	     { j += 7; bishop_check_table[i] |= (1ULL<<j); slide_check_table[i] |= (1ULL<<j); check_table[i] |= (1ULL<<j); }
    j = i; while(RANK(j) < 7 && FILE(j) < 7 && j <= 63)
	     { j += 9; bishop_check_table[i] |= (1ULL<<j); slide_check_table[i] |= (1ULL<<j); check_table[i] |= (1ULL<<j); }
    if(FILE(i) < 6 && RANK(i) < 7) { knight_check_table[i] |= (1ULL<<(i+10)); }
    if(FILE(i) < 6 && RANK(i)) { knight_check_table[i] |= (1ULL<<(i-6)); }
    if(FILE(i) > 1 && RANK(i) < 7) { knight_check_table[i] |= (1ULL<<(i+6)); }
    if(FILE(i) > 1 && RANK(i)) { knight_check_table[i] |= (1ULL<<(i-10)); }
    if(FILE(i) < 7 && RANK(i) < 6) { knight_check_table[i] |= (1ULL<<(i+17)); }
    if(FILE(i) && RANK(i) < 6) { knight_check_table[i] |= (1ULL<<(i+15)); }
    if(FILE(i) < 7 && RANK(i) > 1) { knight_check_table[i] |= (1ULL<<(i-15)); }
    if(FILE(i) && RANK(i) > 1) { knight_check_table[i] |= (1ULL<<(i-17)); }
  }

}

/* function to set a single score or search value */

void set_score_value(char dummy[50], float val) 
{

   if(!strcmp(dummy, "NO_ROOT_LMR_SCORE")) {
     NO_ROOT_LMR_SCORE = int(val);
   } else if(!strcmp(dummy, "DRAW_SCORE")) {
     DRAW_SCORE = val;
   } else if(!strcmp(dummy, "VAR1")) {
     VAR1 = val;
   } else if(!strcmp(dummy, "VAR2")) {
     VAR2 = val;
   } else if(!strcmp(dummy, "VAR3")) {
     VAR3 = val;
   } else if(!strcmp(dummy, "VAR4")) {
     VAR4 = val;
   } else if(!strcmp(dummy, "VERIFY_MARGIN")) {
     VERIFY_MARGIN = int(val);
   } else if(!strcmp(dummy, "PAWN_VALUE")) {
     value[PAWN] = int(val);
   } else if(!strcmp(dummy, "KING_VALUE")) {
     value[KING] = int(val);
   } else if(!strcmp(dummy, "KNIGHT_VALUE")) {
     value[KNIGHT] = int(val);
   } else if(!strcmp(dummy, "BISHOP_VALUE")) {
     value[BISHOP] = int(val);
   } else if(!strcmp(dummy, "ROOK_VALUE")) {
     value[ROOK] = int(val);
   } else if(!strcmp(dummy, "QUEEN_VALUE")) {
     value[QUEEN] = int(val);
   } else if(!strcmp(dummy, "DOUBLED_PAWN_EARLY")) {
     DOUBLED_PAWN_EARLY = int(val);
   } else if(!strcmp(dummy, "DOUBLED_PAWN_LATE")) {
     DOUBLED_PAWN_LATE = int(val);
   } else if(!strcmp(dummy, "WEAK_PAWN_EARLY")) {
     WEAK_PAWN_EARLY = int(val);
   } else if(!strcmp(dummy, "WEAK_PAWN_LATE")) {
     WEAK_PAWN_LATE = int(val);
   } else if(!strcmp(dummy, "BACKWARD_PAWN_EARLY")) {
     BACKWARD_PAWN_EARLY = int(val);
   } else if(!strcmp(dummy, "BACKWARD_PAWN_LATE")) {
     BACKWARD_PAWN_LATE = int(val);
   } else if(!strcmp(dummy, "PAWN_ISLAND_EARLY")) {
     PAWN_ISLAND_EARLY = int(val);
   } else if(!strcmp(dummy, "PAWN_ISLAND_LATE")) {
     PAWN_ISLAND_LATE = int(val);
   } else if(!strcmp(dummy, "PASSED_PAWN")) {
     PASSED_PAWN = int(val);
   } else if(!strcmp(dummy, "BISHOP_PAIR")) {
     BISHOP_PAIR = int(val);
   } else if(!strcmp(dummy, "CON_PASSED_PAWNS")) {
     CON_PASSED_PAWNS = int(val);
   } else if(!strcmp(dummy, "OUTSIDE_PASSED_PAWN")) {
     OUTSIDE_PASSED_PAWN = int(val);
   } else if(!strcmp(dummy, "FREE_PASSED_PAWN")) {
     FREE_PASSED_PAWN = int(val);
   } else if(!strcmp(dummy, "CASTLED")) {
     CASTLED = int(val);
   } else if(!strcmp(dummy, "TRADES_EARLY")) {
     TRADES_EARLY = int(val);
   } else if(!strcmp(dummy, "TRADES_LATE")) {
     TRADES_LATE = int(val);
   } else if(!strcmp(dummy, "NO_POSSIBLE_CASTLE")) {
     NO_POSSIBLE_CASTLE = int(val);
   } else if(!strcmp(dummy, "ROOK_MOBILITY")) {
     ROOK_MOBILITY = int(val);
   } else if(!strcmp(dummy, "QUEEN_MOBILITY")) {
     QUEEN_MOBILITY = int(val);
   } else if(!strcmp(dummy, "KNIGHT_MOBILITY")) {
     KNIGHT_MOBILITY = int(val);
   } else if(!strcmp(dummy, "BISHOP_MOBILITY")) {
     BISHOP_MOBILITY = int(val);
   } else if(!strcmp(dummy, "PAWN_DUO")) {
     PAWN_DUO = int(val);
   } else if(!strcmp(dummy, "SPACE")) {
     SPACE = int(val);
   } else if(!strcmp(dummy, "BOXED_IN_ROOK")) {
     BOXED_IN_ROOK = int(val);
   } else if(!strcmp(dummy, "MINOR_OUTPOST")) {
     MINOR_OUTPOST = int(val);
   } else if(!strcmp(dummy, "MINOR_OUTP_UNGUARD")) {
     MINOR_OUTP_UNGUARD = int(val);
   } else if(!strcmp(dummy, "MINOR_BLOCKER")) {
     MINOR_BLOCKER = int(val);
   } else if(!strcmp(dummy, "BMINOR_OUTPOST")) {
     BMINOR_OUTPOST = int(val);
   } else if(!strcmp(dummy, "BMINOR_OUTP_UNGUARD")) {
     BMINOR_OUTP_UNGUARD = int(val);
   } else if(!strcmp(dummy, "BMINOR_BLOCKER")) {
     BMINOR_BLOCKER = int(val);
   } else if(!strcmp(dummy, "PAWN_THREAT_MINOR")) {
     PAWN_THREAT_MINOR = int(val);
   } else if(!strcmp(dummy, "PAWN_THREAT_MAJOR")) {
     PAWN_THREAT_MAJOR = int(val);
   } else if(!strcmp(dummy, "ROOK_OPEN_FILE")) {
     ROOK_OPEN_FILE = int(val);
//   } else if(!strcmp(dummy, "ROOK_ON_7TH")) {
//     ROOK_ON_7TH = int(val);
   } else if(!strcmp(dummy, "ROOK_HALF_OPEN_FILE")) {
     ROOK_HALF_OPEN_FILE = int(val);
//   } else if(!strcmp(dummy, "CONNECTED_ROOKS")) {
//     CONNECTED_ROOKS = int(val);
   } else if(!strcmp(dummy, "SIDE_ON_MOVE_EARLY")) {
     SIDE_ON_MOVE_EARLY = int(val);
   } else if(!strcmp(dummy, "SIDE_ON_MOVE_LATE")) {
     SIDE_ON_MOVE_LATE = int(val);
   } else if(!strcmp(dummy, "WRONG_BISHOP")) {
     WRONG_BISHOP = int(val);
   } else { 
     cout << "Score label " << dummy << " is not found!\n"; 
     cout.flush();
   }

}

/* function to set a single score or search value */

void modify_score_value(char dummy[50], float val) 
{

   if(!strcmp(dummy, "NO_ROOT_LMR_SCORE")) {
     NO_ROOT_LMR_SCORE += int(val);
   } else if(!strcmp(dummy, "DRAW_SCORE")) {
     DRAW_SCORE += val;
   } else if(!strcmp(dummy, "VAR1")) {
     VAR1 += val;
   } else if(!strcmp(dummy, "VAR2")) {
     VAR2 += val;
   } else if(!strcmp(dummy, "VAR3")) {
     VAR3 += val;
   } else if(!strcmp(dummy, "VAR4")) {
     VAR4 += val;
   } else if(!strcmp(dummy, "VERIFY_MARGIN")) {
     VERIFY_MARGIN += int(val);
   } else if(!strcmp(dummy, "PAWN_VALUE")) {
     value[PAWN] += int(val);
   } else if(!strcmp(dummy, "KING_VALUE")) {
     value[KING] += int(val);
   } else if(!strcmp(dummy, "KNIGHT_VALUE")) {
     value[KNIGHT] += int(val);
   } else if(!strcmp(dummy, "BISHOP_VALUE")) {
     value[BISHOP] += int(val);
   } else if(!strcmp(dummy, "ROOK_VALUE")) {
     value[ROOK] += int(val);
   } else if(!strcmp(dummy, "QUEEN_VALUE")) {
     value[QUEEN] += int(val);
   } else if(!strcmp(dummy, "DOUBLED_PAWN_EARLY")) {
     DOUBLED_PAWN_EARLY += int(val);
   } else if(!strcmp(dummy, "DOUBLED_PAWN_LATE")) {
     DOUBLED_PAWN_LATE += int(val);
   } else if(!strcmp(dummy, "WEAK_PAWN_EARLY")) {
     WEAK_PAWN_EARLY += int(val);
   } else if(!strcmp(dummy, "WEAK_PAWN_LATE")) {
     WEAK_PAWN_LATE += int(val);
   } else if(!strcmp(dummy, "BACKWARD_PAWN_EARLY")) {
     BACKWARD_PAWN_EARLY += int(val);
   } else if(!strcmp(dummy, "BACKWARD_PAWN_LATE")) {
     BACKWARD_PAWN_LATE += int(val);
   } else if(!strcmp(dummy, "PAWN_ISLAND_EARLY")) {
     PAWN_ISLAND_EARLY += int(val);
   } else if(!strcmp(dummy, "PAWN_ISLAND_LATE")) {
     PAWN_ISLAND_LATE += int(val);
   } else if(!strcmp(dummy, "PASSED_PAWN")) {
     PASSED_PAWN += int(val);
   } else if(!strcmp(dummy, "BISHOP_PAIR")) {
     BISHOP_PAIR += int(val);
   } else if(!strcmp(dummy, "CON_PASSED_PAWNS")) {
     CON_PASSED_PAWNS += int(val);
   } else if(!strcmp(dummy, "OUTSIDE_PASSED_PAWN")) {
     OUTSIDE_PASSED_PAWN += int(val);
   } else if(!strcmp(dummy, "FREE_PASSED_PAWN")) {
     FREE_PASSED_PAWN += int(val);
   } else if(!strcmp(dummy, "CASTLED")) {
     CASTLED += int(val);
   } else if(!strcmp(dummy, "TRADES_EARLY")) {
     TRADES_EARLY += int(val);
   } else if(!strcmp(dummy, "TRADES_LATE")) {
     TRADES_LATE += int(val);
   } else if(!strcmp(dummy, "NO_POSSIBLE_CASTLE")) {
     NO_POSSIBLE_CASTLE += int(val);
   } else if(!strcmp(dummy, "ROOK_MOBILITY")) {
     ROOK_MOBILITY += int(val);
   } else if(!strcmp(dummy, "QUEEN_MOBILITY")) {
     QUEEN_MOBILITY += int(val);
   } else if(!strcmp(dummy, "KNIGHT_MOBILITY")) {
     KNIGHT_MOBILITY += int(val);
   } else if(!strcmp(dummy, "BISHOP_MOBILITY")) {
     BISHOP_MOBILITY += int(val);
   } else if(!strcmp(dummy, "PAWN_DUO")) {
     PAWN_DUO += int(val);
   } else if(!strcmp(dummy, "SPACE")) {
     SPACE += int(val);
   } else if(!strcmp(dummy, "BOXED_IN_ROOK")) {
     BOXED_IN_ROOK += int(val);
   } else if(!strcmp(dummy, "MINOR_OUTPOST")) {
     MINOR_OUTPOST += int(val);
   } else if(!strcmp(dummy, "MINOR_OUTP_UNGUARD")) {
     MINOR_OUTP_UNGUARD += int(val);
   } else if(!strcmp(dummy, "MINOR_BLOCKER")) {
     MINOR_BLOCKER += int(val);
   } else if(!strcmp(dummy, "BMINOR_OUTPOST")) {
     BMINOR_OUTPOST += int(val);
   } else if(!strcmp(dummy, "BMINOR_OUTP_UNGUARD")) {
     BMINOR_OUTP_UNGUARD += int(val);
   } else if(!strcmp(dummy, "BMINOR_BLOCKER")) {
     BMINOR_BLOCKER += int(val);
   } else if(!strcmp(dummy, "PAWN_THREAT_MINOR")) {
     PAWN_THREAT_MINOR += int(val);
   } else if(!strcmp(dummy, "PAWN_THREAT_MAJOR")) {
     PAWN_THREAT_MAJOR += int(val);
   } else if(!strcmp(dummy, "ROOK_OPEN_FILE")) {
     ROOK_OPEN_FILE += int(val);
   } else if(!strcmp(dummy, "ROOK_HALF_OPEN_FILE")) {
     ROOK_HALF_OPEN_FILE += int(val);
//   } else if(!strcmp(dummy, "CONNECTED_ROOKS")) {
//     CONNECTED_ROOKS += int(val);
//   } else if(!strcmp(dummy, "ROOK_ON_7TH")) {
//     ROOK_ON_7TH += int(val);
   } else if(!strcmp(dummy, "SIDE_ON_MOVE_EARLY")) {
     SIDE_ON_MOVE_EARLY += int(val);
   } else if(!strcmp(dummy, "SIDE_ON_MOVE_LATE")) {
     SIDE_ON_MOVE_LATE += int(val);
   } else if(!strcmp(dummy, "WRONG_BISHOP")) {
     WRONG_BISHOP += int(val);
/*
   } else if(!strcmp(dummy, "PAWN_OFFSET_PSQ_EARLY")) {
     OFFSET_PSQ[0][PAWN] += val;
     fill_psq(PAWN);
   } else if(!strcmp(dummy, "PAWN_OFFSET_PSQ_LATE")) {
     OFFSET_PSQ[1][PAWN] += val;
     fill_psq(PAWN);
   } else if(!strcmp(dummy, "PAWN_SCALE_PSQ_EARLY")) {
     SCALE_PSQ[0][PAWN] += val;
     fill_psq(PAWN);
   } else if(!strcmp(dummy, "PAWN_SCALE_PSQ_LATE")) {
     SCALE_PSQ[1][PAWN] += val;
     fill_psq(PAWN);
   } else if(!strcmp(dummy, "KNIGHT_OFFSET_PSQ_EARLY")) {
     OFFSET_PSQ[0][KNIGHT] += val;
     fill_psq(KNIGHT);
   } else if(!strcmp(dummy, "KNIGHT_OFFSET_PSQ_LATE")) {
     OFFSET_PSQ[1][KNIGHT] += val;
     fill_psq(KNIGHT);
   } else if(!strcmp(dummy, "KNIGHT_SCALE_PSQ_EARLY")) {
     SCALE_PSQ[0][KNIGHT] += val;
     fill_psq(KNIGHT);
   } else if(!strcmp(dummy, "KNIGHT_SCALE_PSQ_LATE")) {
     SCALE_PSQ[1][KNIGHT] += val;
     fill_psq(KNIGHT);
   } else if(!strcmp(dummy, "BISHOP_OFFSET_PSQ_EARLY")) {
     OFFSET_PSQ[0][BISHOP] += val;
     fill_psq(BISHOP);
   } else if(!strcmp(dummy, "BISHOP_OFFSET_PSQ_LATE")) {
     OFFSET_PSQ[1][BISHOP] += val;
     fill_psq(BISHOP);
   } else if(!strcmp(dummy, "BISHOP_SCALE_PSQ_EARLY")) {
     SCALE_PSQ[0][BISHOP] += val;
     fill_psq(BISHOP);
   } else if(!strcmp(dummy, "BISHOP_SCALE_PSQ_LATE")) {
     SCALE_PSQ[1][BISHOP] += val;
     fill_psq(BISHOP);
   } else if(!strcmp(dummy, "ROOK_OFFSET_PSQ_EARLY")) {
     OFFSET_PSQ[0][ROOK] += val;
     fill_psq(ROOK);
   } else if(!strcmp(dummy, "ROOK_OFFSET_PSQ_LATE")) {
     OFFSET_PSQ[1][ROOK] += val;
     fill_psq(ROOK);
   } else if(!strcmp(dummy, "ROOK_SCALE_PSQ_EARLY")) {
     SCALE_PSQ[0][ROOK] += val;
     fill_psq(ROOK);
   } else if(!strcmp(dummy, "ROOK_SCALE_PSQ_LATE")) {
     SCALE_PSQ[1][ROOK] += val;
     fill_psq(ROOK);
   } else if(!strcmp(dummy, "QUEEN_OFFSET_PSQ_EARLY")) {
     OFFSET_PSQ[0][QUEEN] += val;
     fill_psq(QUEEN);
   } else if(!strcmp(dummy, "QUEEN_OFFSET_PSQ_LATE")) {
     OFFSET_PSQ[1][QUEEN] += val;
     fill_psq(QUEEN);
   } else if(!strcmp(dummy, "QUEEN_SCALE_PSQ_EARLY")) {
     SCALE_PSQ[0][QUEEN] += val;
     fill_psq(QUEEN);
   } else if(!strcmp(dummy, "QUEEN_SCALE_PSQ_LATE")) {
     SCALE_PSQ[1][QUEEN] += val;
     fill_psq(QUEEN);
   } else if(!strcmp(dummy, "KING_OFFSET_PSQ_EARLY")) {
     OFFSET_PSQ[0][KING] += val;
     fill_psq(KING);
   } else if(!strcmp(dummy, "KING_OFFSET_PSQ_LATE")) {
     OFFSET_PSQ[1][KING] += val;
     fill_psq(KING);
   } else if(!strcmp(dummy, "KING_SCALE_PSQ_EARLY")) {
     SCALE_PSQ[0][KING] += val;
     fill_psq(KING);
   } else if(!strcmp(dummy, "KING_SCALE_PSQ_LATE")) {
     SCALE_PSQ[1][KING] += val;
     fill_psq(KING);
*/
   } else { 
     cout << "Score label " << dummy << " is not found!\n"; 
     cout.flush();
   }

}

/* function to get a single score or search value */

float get_score_value(char dummy[50]) 
{

   if(!strcmp(dummy, "NO_ROOT_LMR_SCORE")) {
     return(NO_ROOT_LMR_SCORE);
   } else if(!strcmp(dummy, "DRAW_SCORE")) {
     return(DRAW_SCORE);
   } else if(!strcmp(dummy, "VAR1")) {
     return(VAR1);
   } else if(!strcmp(dummy, "VAR2")) {
     return(VAR2);
   } else if(!strcmp(dummy, "VAR3")) {
     return(VAR3);
   } else if(!strcmp(dummy, "VAR4")) {
     return(VAR4);
   } else if(!strcmp(dummy, "VERIFY_MARGIN")) {
     return(VERIFY_MARGIN);
   } else if(!strcmp(dummy, "NO_QCHECKS")) {
     return(NO_QCHECKS);
   } else if(!strcmp(dummy, "PAWN_VALUE")) {
     return(value[PAWN]);
   } else if(!strcmp(dummy, "KING_VALUE")) {
     return(value[KING]);
   } else if(!strcmp(dummy, "KNIGHT_VALUE")) {
     return(value[KNIGHT]);
   } else if(!strcmp(dummy, "BISHOP_VALUE")) {
     return(value[BISHOP]);
   } else if(!strcmp(dummy, "ROOK_VALUE")) {
     return(value[ROOK]);
   } else if(!strcmp(dummy, "QUEEN_VALUE")) {
     return(value[QUEEN]);
   } else if(!strcmp(dummy, "DOUBLED_PAWN_EARLY")) {
     return(DOUBLED_PAWN_EARLY);
   } else if(!strcmp(dummy, "DOUBLED_PAWN_LATE")) {
     return(DOUBLED_PAWN_LATE);
   } else if(!strcmp(dummy, "WEAK_PAWN_EARLY")) {
     return(WEAK_PAWN_EARLY);
   } else if(!strcmp(dummy, "WEAK_PAWN_LATE")) {
     return(WEAK_PAWN_LATE);
   } else if(!strcmp(dummy, "BACKWARD_PAWN_EARLY")) {
     return(BACKWARD_PAWN_EARLY);
   } else if(!strcmp(dummy, "BACKWARD_PAWN_LATE")) {
     return(BACKWARD_PAWN_LATE);
   } else if(!strcmp(dummy, "PAWN_ISLAND_EARLY")) {
     return(PAWN_ISLAND_EARLY);
   } else if(!strcmp(dummy, "PAWN_ISLAND_LATE")) {
     return(PAWN_ISLAND_LATE);
   } else if(!strcmp(dummy, "PASSED_PAWN")) {
     return(PASSED_PAWN);
   } else if(!strcmp(dummy, "BISHOP_PAIR")) {
     return(BISHOP_PAIR);
   } else if(!strcmp(dummy, "CON_PASSED_PAWNS")) {
     return(CON_PASSED_PAWNS);
   } else if(!strcmp(dummy, "OUTSIDE_PASSED_PAWN")) {
     return(OUTSIDE_PASSED_PAWN);
   } else if(!strcmp(dummy, "FREE_PASSED_PAWN")) {
     return(FREE_PASSED_PAWN);
   } else if(!strcmp(dummy, "CASTLED")) {
     return(CASTLED);
   } else if(!strcmp(dummy, "TRADES_EARLY")) {
     return(TRADES_EARLY);
   } else if(!strcmp(dummy, "TRADES_LATE")) {
     return(TRADES_LATE);
   } else if(!strcmp(dummy, "NO_POSSIBLE_CASTLE")) {
     return(NO_POSSIBLE_CASTLE);
   } else if(!strcmp(dummy, "ROOK_MOBILITY")) {
     return(ROOK_MOBILITY);
   } else if(!strcmp(dummy, "QUEEN_MOBILITY")) {
     return(QUEEN_MOBILITY);
   } else if(!strcmp(dummy, "KNIGHT_MOBILITY")) {
     return(KNIGHT_MOBILITY);
   } else if(!strcmp(dummy, "BISHOP_MOBILITY")) {
     return(BISHOP_MOBILITY);
   } else if(!strcmp(dummy, "PAWN_DUO")) {
     return(PAWN_DUO);
   } else if(!strcmp(dummy, "SPACE")) {
     return(SPACE);
   } else if(!strcmp(dummy, "BOXED_IN_ROOK")) {
     return(BOXED_IN_ROOK);
   } else if(!strcmp(dummy, "MINOR_OUTPOST")) {
     return(MINOR_OUTPOST);
   } else if(!strcmp(dummy, "MINOR_OUTP_UNGUARD")) {
     return(MINOR_OUTP_UNGUARD);
   } else if(!strcmp(dummy, "MINOR_BLOCKER")) {
     return(MINOR_BLOCKER);
   } else if(!strcmp(dummy, "BMINOR_OUTPOST")) {
     return(BMINOR_OUTPOST);
   } else if(!strcmp(dummy, "BMINOR_OUTP_UNGUARD")) {
     return(BMINOR_OUTP_UNGUARD);
   } else if(!strcmp(dummy, "BMINOR_BLOCKER")) {
     return(BMINOR_BLOCKER);
   } else if(!strcmp(dummy, "PAWN_THREAT_MINOR")) {
     return(PAWN_THREAT_MINOR);
   } else if(!strcmp(dummy, "PAWN_THREAT_MAJOR")) {
     return(PAWN_THREAT_MAJOR);
   } else if(!strcmp(dummy, "ROOK_OPEN_FILE")) {
     return(ROOK_OPEN_FILE);
   } else if(!strcmp(dummy, "ROOK_HALF_OPEN_FILE")) {
     return(ROOK_HALF_OPEN_FILE);
//   } else if(!strcmp(dummy, "CONNECTED_ROOKS")) {
//     return(CONNECTED_ROOKS);
//   } else if(!strcmp(dummy, "ROOK_ON_7TH")) {
//     return(ROOK_ON_7TH);
   } else if(!strcmp(dummy, "SIDE_ON_MOVE_EARLY")) {
     return(SIDE_ON_MOVE_EARLY);
   } else if(!strcmp(dummy, "SIDE_ON_MOVE_LATE")) {
     return(SIDE_ON_MOVE_LATE);
   } else if(!strcmp(dummy, "WRONG_BISHOP")) {
     return(WRONG_BISHOP);
/*
   } else if(!strcmp(dummy, "PAWN_OFFSET_PSQ_EARLY")) {
     return(OFFSET_PSQ[0][PAWN]);
   } else if(!strcmp(dummy, "PAWN_OFFSET_PSQ_LATE")) {
     return(OFFSET_PSQ[1][PAWN]);
   } else if(!strcmp(dummy, "PAWN_SCALE_PSQ_EARLY")) {
     return(SCALE_PSQ[0][PAWN]);
   } else if(!strcmp(dummy, "PAWN_SCALE_PSQ_LATE")) {
     return(SCALE_PSQ[1][PAWN]);
   } else if(!strcmp(dummy, "KNIGHT_OFFSET_PSQ_EARLY")) {
     return(OFFSET_PSQ[0][KNIGHT]);
   } else if(!strcmp(dummy, "KNIGHT_OFFSET_PSQ_LATE")) {
     return(OFFSET_PSQ[1][KNIGHT]);
   } else if(!strcmp(dummy, "KNIGHT_SCALE_PSQ_EARLY")) {
     return(SCALE_PSQ[0][KNIGHT]);
   } else if(!strcmp(dummy, "KNIGHT_SCALE_PSQ_LATE")) {
     return(SCALE_PSQ[1][KNIGHT]);
   } else if(!strcmp(dummy, "BISHOP_OFFSET_PSQ_EARLY")) {
     return(OFFSET_PSQ[0][BISHOP]);
   } else if(!strcmp(dummy, "BISHOP_OFFSET_PSQ_LATE")) {
     return(OFFSET_PSQ[1][BISHOP]);
   } else if(!strcmp(dummy, "BISHOP_SCALE_PSQ_EARLY")) {
     return(SCALE_PSQ[0][BISHOP]);
   } else if(!strcmp(dummy, "BISHOP_SCALE_PSQ_LATE")) {
     return(SCALE_PSQ[1][BISHOP]);
   } else if(!strcmp(dummy, "ROOK_OFFSET_PSQ_EARLY")) {
     return(OFFSET_PSQ[0][ROOK]);
   } else if(!strcmp(dummy, "ROOK_OFFSET_PSQ_LATE")) {
     return(OFFSET_PSQ[1][ROOK]);
   } else if(!strcmp(dummy, "ROOK_SCALE_PSQ_EARLY")) {
     return(SCALE_PSQ[0][ROOK]);
   } else if(!strcmp(dummy, "ROOK_SCALE_PSQ_LATE")) {
     return(SCALE_PSQ[1][ROOK]);
   } else if(!strcmp(dummy, "QUEEN_OFFSET_PSQ_EARLY")) {
     return(OFFSET_PSQ[0][QUEEN]);
   } else if(!strcmp(dummy, "QUEEN_OFFSET_PSQ_LATE")) {
     return(OFFSET_PSQ[1][QUEEN]);
   } else if(!strcmp(dummy, "QUEEN_SCALE_PSQ_EARLY")) {
     return(SCALE_PSQ[0][QUEEN]);
   } else if(!strcmp(dummy, "QUEEN_SCALE_PSQ_LATE")) {
     return(SCALE_PSQ[1][QUEEN]);
   } else if(!strcmp(dummy, "KING_OFFSET_PSQ_EARLY")) {
     return(OFFSET_PSQ[0][KING]);
   } else if(!strcmp(dummy, "KING_OFFSET_PSQ_LATE")) {
     return(OFFSET_PSQ[1][KING]);
   } else if(!strcmp(dummy, "KING_SCALE_PSQ_EARLY")) {
     return(SCALE_PSQ[0][KING]);
   } else if(!strcmp(dummy, "KING_SCALE_PSQ_LATE")) {
     return(SCALE_PSQ[1][KING]);
*/
   } else { 
     cout << "Score label " << dummy << " is not found!\n"; 
     cout.flush();
     return 0;
   }

}

/*
void fill_psq(int piece) {

    float rank_coef, center_coef, offset,scale;    

    for(int stage = 0; stage < 4; stage++) {
      //for(int piece = 1; piece < 2; piece++) {
        for(int sqr = 0; sqr < 64; sqr++) {  
          if(stage == 0) {
            offset = OFFSET_PSQ[0][piece];
            scale = SCALE_PSQ[0][piece];
          } else if(stage == 1) {
            offset = (OFFSET_PSQ[0][piece]+OFFSET_PSQ[1][piece]*0.5)/1.5;
            scale = (SCALE_PSQ[0][piece]+SCALE_PSQ[1][piece]*0.5)/1.5;
          } else if(stage == 2) {
            offset = (OFFSET_PSQ[0][piece]*0.5+OFFSET_PSQ[1][piece])/1.5;
            scale = (SCALE_PSQ[0][piece]*0.5+SCALE_PSQ[1][piece])/1.5;
          } else {
            offset = OFFSET_PSQ[1][piece];
            scale= SCALE_PSQ[1][piece];
          }        
          int value = scale*piece_sq_save[stage][piece][sqr]+offset;
          if(value > 127) value = 127;
          if(value < -127) value = -127;
          piece_sq[stage][piece][sqr]=value;       

          //cout << setw(5) << value << ", ";
          //if(FILE(sqr) == 7) cout << "\n";

        }
        //cout << "\n";
      //}
    }

}
*/

void print_psq() {

    for(int stage = 0; stage < 4; stage++) {
      cout << "{\n";
      for(int piece = 0; piece < 7; piece++) {
        cout << "*** Table for piece = " << pstring[piece] << " in stage " << stage;
        cout << "\n{\n";
        for(int sqr = 0; sqr < 64; sqr++) {  
          cout << setw(5) << int(piece_sq[stage][piece][sqr]);
          if(sqr < 63) cout << ", ";
          if(FILE(sqr) == 7) cout << "\n";
        }
        if(piece == 6) cout << "} ";
        cout << "},\n";
      }
    }
    cout << "};\n";    
}
