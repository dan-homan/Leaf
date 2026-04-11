/* Global constants used all over the program */

#ifndef CONST_H
#define CONST_H

// Piece names for output routines
const char name[7] = { ' ', 'P', 'N', 'B', 'R', 'Q', 'K' };
const char bname[7] = { ' ', 'p', 'n', 'b', 'r', 'q', 'k' };
const char pstring[7][7] = { " ", "PAWN", "KNIGHT", "BISHOP", 
                                  "ROOK", "QUEEN", "KING" };

// Castle mask as suggested by Tom Kerrigan's Simple Chess Program
// The basic idea is to speed up change of castle rights by AND
// operations.
unsigned char castle_mask[64] = {
               13, 15, 15, 15, 12, 15, 15, 14,
               15, 15, 15, 15, 15, 15, 15, 15,
               15, 15, 15, 15, 15, 15, 15, 15,
               15, 15, 15, 15, 15, 15, 15, 15,
               15, 15, 15, 15, 15, 15, 15, 15,
               15, 15, 15, 15, 15, 15, 15, 15,
               15, 15, 15, 15, 15, 15, 15, 15,
                7, 15, 15, 15,  3, 15, 15, 11,
};

// starting position
const char i_pos[256] = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR";


#endif  /* CONST_H */

