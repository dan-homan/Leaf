// EXchess source code, (c) Daniel C. Homan  1997-2013
// Released under the GNU public license, see file license.txt

/* Execute move function */


#include "define.h"
#include "chess.h"
#include "const.h"
#include "funct.h"
#include "hash.h"
#include "extern.h"

// Function to execute the move.
// If the move is legal, the function returns a 1, otherwise 0.
// Note that the move is made, regardless - so proper precautions
// need to be taken, if the move might need to be undone
int position::exec_move(move emove, int ply)
{
  move_t mv = emove.b;
  int pi;

  // basic sanity check for illegal moves that could
  //  cause problems.  Return 0 in these cases.
  if(mv.from > 63 || mv.to > 63 ||
     PSIDE(sq[mv.from]) != wtm || PTYPE(sq[mv.to]) == KING) { 
    write_out("Illegal move detected!\n");
    return 0;
  }

  // save the original (from) square and
  //  then blank it... doing so now allows
  //  various types of castling in Chess960
  square from_sq = sq[mv.from];
  sq[mv.from] = EMPTY;

  // if this is a castle, check that it is legal then
  // move the rook, the king move is given below
  // --> should be correct for Chess960
  if(mv.type&CASTLE) {
    //if(check) return 0;   // -- should be prevented in move generation
       switch(mv.to) {
	 /* white kingside castle */
           case 6:
	      for(pi = MIN(5,mv.from+1); pi <= MAX(6,Krook[WHITE]-1); pi++) { if(sq[pi] && pi != Krook[WHITE]) return 0; }
	      for(pi = mv.from+1; pi < 6; pi++) { if(attacked(pi,BLACK)) return 0; }
              sq[5] = sq[Krook[WHITE]];
              if(Krook[WHITE] != 5) sq[Krook[WHITE]] = EMPTY;
	      // note -2 on the WROOK value is to access the right part of the array
              Or(hcode, hval[WROOK-2][5]);
              Or(hcode, hval[WROOK-2][Krook[WHITE]]);
              /* update piece list */ 
              for(pi=1;pi<=plist[WHITE][ROOK][0];pi++)
                if(plist[WHITE][ROOK][pi] == Krook[WHITE]) {
                   plist[WHITE][ROOK][pi] = 5;
                   break;
                }
              break;
	 /* white queenside castle */  
           case 2:
	      for(pi = MIN(2,Qrook[WHITE]+1); pi <= MAX(3,mv.from-1); pi++) { if(sq[pi] && pi != Qrook[WHITE]) return 0; }
	      for(pi = mv.from-1; pi > 2; pi--) { if(attacked(pi,BLACK)) return 0; }  // note if King starts on file 1, no check necessary
              sq[3] = sq[Qrook[WHITE]];
              if(Qrook[WHITE] != 3) sq[Qrook[WHITE]] = EMPTY;
	      // note -2 on the WROOK value is to access the right part of the array
              Or(hcode, hval[WROOK-2][3]);
              Or(hcode, hval[WROOK-2][Qrook[WHITE]]);
              /* update piece list */
              for(pi=1;pi<=plist[WHITE][ROOK][0];pi++)
                if(plist[WHITE][ROOK][pi] == Qrook[WHITE]) {
                   plist[WHITE][ROOK][pi] = 3;
                   break;
                }
              break;
	 /* black kingside castle */  
           case 62:
	      for(pi = MIN(61,mv.from+1); pi <= MAX(62,Krook[BLACK]-1); pi++) { if(sq[pi] && pi != Krook[BLACK]) return 0; }
	      for(pi = mv.from+1; pi < 62; pi++) { if(attacked(pi,WHITE)) return 0; }
              sq[61] = sq[Krook[BLACK]];
              if(Krook[BLACK] != 61) sq[Krook[BLACK]] = EMPTY;
              Or(hcode, hval[BROOK][61]);
              Or(hcode, hval[BROOK][Krook[BLACK]]);
              /* update piece list */
              for(pi=1;pi<=plist[BLACK][ROOK][0];pi++)
                if(plist[BLACK][ROOK][pi] == Krook[BLACK]) {
                   plist[BLACK][ROOK][pi] = 61;
                   break;
                }
              break;
	 /* black queenside castle */
           case 58:
  	      for(pi = MIN(58,Qrook[BLACK]+1); pi <= MAX(59,mv.from-1); pi++) { if(sq[pi] && pi != Qrook[BLACK]) return 0; }
	      for(pi = mv.from-1; pi > 58; pi--) { if(attacked(pi,WHITE)) return 0; }  // note if King starts on file 1, no check necessary
              sq[59] = sq[Qrook[BLACK]];
              if(Qrook[BLACK] != 59) sq[Qrook[BLACK]] = EMPTY;
              Or(hcode, hval[BROOK][59]);
              Or(hcode, hval[BROOK][Qrook[BLACK]]);
              /* update piece list */
              for(pi=1;pi<=plist[BLACK][ROOK][0];pi++)
                if(plist[BLACK][ROOK][pi] == Qrook[BLACK]) {
                   plist[BLACK][ROOK][pi] = 59;
                   break;
                }
              break;
           }
  }

  // update piece list for moving piece
  for(pi=1;pi<=plist[wtm][PTYPE(from_sq)][0];pi++)
   if(plist[wtm][PTYPE(from_sq)][pi] == mv.from) {
      plist[wtm][PTYPE(from_sq)][pi] = mv.to;
      break;
   }

  if(PTYPE(sq[mv.to])) {
   // Remove hashcode for the target square 
   Or(hcode, hval[HASH_ID(sq[mv.to])][mv.to]);
   if(PTYPE(sq[mv.to]) == PAWN) { 
     Or(pcode, hval[HASH_ID(sq[mv.to])][mv.to]); 
   } else {
     pieces[wtm^1]--;    // adjust total piece count
     gstage++;           // modify game_stage
     if(gstage > 15) { gstage = 15; }
     Or(pcode, gstage_code[gstage-1]);
     Or(pcode, gstage_code[gstage]);
   }
   // Remove piece from piece list
   for(pi=1;pi<=plist[wtm^1][PTYPE(sq[mv.to])][0];pi++)
    if(plist[wtm^1][PTYPE(sq[mv.to])][pi] == mv.to) {
       plist[wtm^1][PTYPE(sq[mv.to])][pi] =
       plist[wtm^1][PTYPE(sq[mv.to])][plist[wtm^1][PTYPE(sq[mv.to])][0]];
       plist[wtm^1][PTYPE(sq[mv.to])][0]--;
       break;
    }
   // adjust material score   
   material += value[PTYPE(sq[mv.to])];
  }

  // Move the new piece to the target square
  sq[mv.to] = from_sq;
  // Update the hash code to reflect the move
  Or(hcode, hval[HASH_ID(from_sq)][mv.from]);
  Or(hcode, hval[HASH_ID(from_sq)][mv.to]);
  if(PTYPE(from_sq) == PAWN) {
    Or(pcode, hval[HASH_ID(from_sq)][mv.from]);
    Or(pcode, hval[HASH_ID(from_sq)][mv.to]);
  }

  // if move is en-passant, finish it
  if(mv.type&EP) {
    if(wtm) {
      sq[mv.to-8] = EMPTY;
      Or(hcode, hval[BPAWN][mv.to-8]);
      Or(pcode, hval[BPAWN][mv.to-8]);
      // Update piece lists 
      for(pi=1;pi<=plist[wtm^1][PAWN][0];pi++)
        if(plist[BLACK][PAWN][pi] == mv.to-8) {
           plist[BLACK][PAWN][pi] = 
           plist[BLACK][PAWN][plist[wtm^1][PAWN][0]];
	   plist[BLACK][PAWN][0]--;
           break;
        }
    } else {
      sq[mv.to+8] = EMPTY;
      // note -2 on the WPAWN value is to access the right part of the array
      Or(hcode, hval[WPAWN-2][mv.to+8]);
      Or(pcode, hval[WPAWN-2][mv.to+8]);
      // Update piece lists
      for(pi=1;pi<=plist[wtm^1][PAWN][0];pi++)
       if(plist[WHITE][PAWN][pi] == mv.to+8) {
          plist[WHITE][PAWN][pi] = 
          plist[WHITE][PAWN][plist[wtm^1][PAWN][0]];
	  plist[WHITE][PAWN][0]--;
          break;
       }
    }
    material += value[PAWN];
  }

  // if we are in check, move isn't legal
  // return 0
  if(!check) {
   if(check || mv.type&EP || PTYPE(sq[mv.to]) == KING)
   {
     if(attacked(plist[wtm][KING][1], wtm^1)) return 0;
   } else if(slide_check_table[mv.from]&(1ULL<<plist[wtm][KING][1])) {
     if(simple_check(mv.from)) return 0;
   }
  }

  // if the move is a promotion, promote it
  if(mv.type&PROMOTE) {
    // Remove the pawn from the hash code
    Or(hcode, hval[HASH_ID(sq[mv.to])][mv.to]);
    Or(pcode, hval[HASH_ID(sq[mv.to])][mv.to]);
    // Change the piece type to the promoted piece
    sq[mv.to] = (sq[mv.to]&8)+mv.promote;
    // Add the new piece to the piece lists
    plist[wtm][mv.promote][0]++;
    plist[wtm][mv.promote][plist[wtm][mv.promote][0]]=mv.to;
    // Remove the pawn from the piece lists
    for(pi=1;pi<=plist[wtm][PAWN][0];pi++)
      if(plist[wtm][PAWN][pi] == mv.to) {
         plist[wtm][PAWN][pi] = 
         plist[wtm][PAWN][plist[wtm][PAWN][0]];
	 plist[wtm][PAWN][0]--;
         break;
      }
    // adjust material score
    material += value[mv.promote] - value[PAWN];
    // add piece to hash code 
    Or(hcode, hval[HASH_ID(sq[mv.to])][mv.to]);
    // adjust total piece count
    pieces[wtm]++;
    gstage--;           // modify game_stage
    if(gstage < 0) { gstage = 0; }
    Or(pcode, gstage_code[gstage+1]);
    Or(pcode, gstage_code[gstage]);
  }

  // update position characteristics
  wtm = wtm^1;
  material = -material;
  Or(hcode, hstm); 

  // undo hash code for en-passant and castling status
  if(ep) Or(hcode, ep_code[FILE(ep)]);
  Or(hcode, castle_code[castle]);

  // if move is a pawn push 2 spaces, set en passant flag
  if(((mv.type&PAWN_PUSH) && ABS(mv.from-mv.to) == 16) && 
     ((FILE(mv.to) < 7 && PTYPE(sq[mv.to+1]) == PAWN && PSIDE(sq[mv.to+1]) == wtm)
      || (FILE(mv.to) && PTYPE(sq[mv.to-1]) == PAWN && PSIDE(sq[mv.to-1]) == wtm)))
   { ep = (mv.from+mv.to)/2; } else { ep = 0; }
  // if move is not a capture or a pawn move, increase fifty count
  if((mv.type&CAPTURE) || (mv.type&PAWN_PUSH))
   { fifty = 0; } else { fifty++; }
  // update castling status
  castle = castle&castle_mask[mv.from];
  castle = castle&castle_mask[mv.to];
  // put this move in as the last move
  last = emove;

  // update hash code for en-passant and castling status
  if(ep) Or(hcode, ep_code[FILE(ep)]);
  Or(hcode, castle_code[castle]);

  // check whether other side is placed in check
  int ptype = PTYPE(sq[mv.to]); 
  int ksq = plist[wtm][KING][1];
  uint64_t ksq_bit = (1ULL<<ksq);

  check = 0;

  if((ptype == BISHOP || ptype == QUEEN)
     && (bishop_check_table[mv.to]&ksq_bit))
    check = dia_slide_attack(mv.to, ksq);
  else if((ptype == ROOK || ptype == QUEEN)
	  && (rook_check_table[mv.to]&ksq_bit))
    check = hor_slide_attack(mv.to, ksq);
  else if(ptype == PAWN && (bishop_check_table[mv.to]&ksq_bit)) {
    // note the bishop_check_table guarantees that the pawn/king
    //   relationships below are OK... otherwise we would have to
    //   restrict the files for these comparisons
    if(wtm && (mv.to == ksq+9 || mv.to == ksq+7))
      { check = 1; return 1; }
    else if(wtm^1 && (mv.to == ksq-9 || mv.to == ksq-7))
      { check = 1; return 1; }
  } else if(ptype == KNIGHT && (knight_check_table[mv.to]&ksq_bit)) check = 1;

  if((slide_check_table[mv.from]&ksq_bit) && check^1) 
    check = simple_check(mv.from);
  if((mv.type&EP) && (check^1))
    check = attacked(ksq, wtm^1);
  if((mv.type&CASTLE) && (check^1))
    check = attacked(ksq, wtm^1);

  return 1;
}








