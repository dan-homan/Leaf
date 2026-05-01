// EXchess source code, (c) Daniel C. Homan  1997-2013
// Released under the GNU public license, see file license.txt

/* Piece captures and checks */

#include "define.h"
#include "chess.h"
#include "const.h"
#include "funct.h"
#include "extern.h"

/*------------------------- Semi-legal capture/check generator ---------------------*/

void position::captchecks(move_list *list, int delta_score)
{
  int i;
  list->count = 0;

  for(i=1;i<=plist[wtm][PAWN][0];i++) 
    pawn_cc(list, plist[wtm][PAWN][i], delta_score);
  for(i=1;i<=plist[wtm][KNIGHT][0];i++) 
    knight_cc(list, plist[wtm][KNIGHT][i], delta_score);
  for(i=1;i<=plist[wtm][BISHOP][0];i++) 
    bishop_cc(list, plist[wtm][BISHOP][i], delta_score);
  for(i=1;i<=plist[wtm][ROOK][0];i++) 
    rook_cc(list, plist[wtm][ROOK][i], delta_score);
  for(i=1;i<=plist[wtm][QUEEN][0];i++) {
    bishop_cc(list, plist[wtm][QUEEN][i], delta_score);
    rook_cc(list, plist[wtm][QUEEN][i], delta_score);
  }
  for(i=1;i<=plist[wtm][KING][0];i++) 
    king_cc(list, plist[wtm][KING][i], delta_score);

}

/* Add move to move list */
// Function to add move to move list
// Move is scored for alpha-beta algorithm
void position::add_cc(int fsq, int tsq, move_list *list, char type, int delta_score)
{

  int i = list->count;                 // move list index
  int pawn_bonus = 0;

  // add move to list
  list->mv[i].m.b.from = fsq;
  list->mv[i].m.b.to = tsq;
  list->mv[i].m.b.type = type;
  list->mv[i].m.b.promote = 0;

  // put a promotion straight into the list and return
  if((tsq > 55 || tsq < 8) && PTYPE(sq[fsq]) == PAWN)
  {
    list->mv[i].m.b.type |= PROMOTE; 
    list->mv[i].m.b.promote = QUEEN;
    //if(swap(tsq,(*this),wtm,fsq) >= 0) list->mv[i].score += 1000;
    list->mv[i].score = delta_score+1000;
    list->count++;
    return;
  }

  // Is is a pawn push to the 7th rank?  If so, mark it as such
  if(type&PAWN_PUSH) {
    if(RANK(tsq) == 1 || RANK(tsq) == 6) list->mv[i].m.b.type |= PAWN_PUSH7;
  }

  // Give an initial score to the move below delta_score
  // so that by default it will not be put in the list 
  // unless a step below changes the score
  list->mv[i].score = delta_score-123;  
  
  // if it is a capture, see if we capture enough material to stay in list
  if(type&CAPTURE) {
    // If the move is a pawn capture
    //  give a bonus by the rank of the pawn
    if(PTYPE(sq[tsq]) == PAWN) {
      if(gstage > 9) { // bonus for pawns about to queen in end-game
	if(wtm && RANK(tsq) == 1) pawn_bonus += 35*gstage;   
	if(!wtm && RANK(tsq) == 6) pawn_bonus += 35*gstage;
      }
    }
    // if it is an EP capture, include ptype value for the pawn captured
    if(type&EP) pawn_bonus += value[PAWN];
    // adjust pawn bonus for bishop capture
    if(PTYPE(sq[tsq]) == BISHOP && plist[wtm^1][BISHOP][0] == 2) pawn_bonus += BISHOP_PAIR;
    // if this is a promising capture, try a static swap
    if(value[PTYPE(sq[tsq])]+pawn_bonus >= delta_score) { 
      list->mv[i].score = swap(tsq,(*this),wtm,fsq) + pawn_bonus;
      // put capture in list immediately if >= delta_score
      if(list->mv[i].score >= delta_score) {
	list->count++;
	return;
      }
      // see if it is a revealed check
      if(slide_check_table[fsq]&(1ULL<<plist[wtm^1][KING][1]))  {
	position temp_pos = (*this);
	temp_pos.wtm ^= 1;
	temp_pos.sq[tsq] = temp_pos.sq[fsq];
	temp_pos.sq[fsq] = EMPTY;
	// if it is a revealed check, add to list immediately
	if(temp_pos.simple_check(fsq)) { 
	  list->mv[i].score = delta_score + value[PTYPE(sq[tsq])] + pawn_bonus;
	  list->count++;
	  return;
	}
      }
      // otherwise just return without putting this losing capture in list
      return;   
    }
  }

  // Any "promising" captures with possible capture value >= delta_score have
  // been 'swapped' at this point.  Survivors are either "promising" but not
  // up to delta_score OR not promising, in which case they weren't 'swapped'
  // in the code above.  Either of these cases, or non-captures as well, could
  // be checks that are worth looking at... see below

  int ptype = PTYPE(sq[fsq]); 
  int ksq = plist[wtm^1][KING][1];
  uint64_t ksq_bit = (1ULL<<ksq);
  int ocheck = 0;

  if((ptype == BISHOP || ptype == QUEEN)
     && (bishop_check_table[tsq]&ksq_bit)) {
    ocheck = dia_slide_attack(tsq,ksq);
  } else if((ptype == ROOK || ptype == QUEEN)
	    && (rook_check_table[tsq]&ksq_bit))
    ocheck = hor_slide_attack(tsq,ksq);
  else if(ptype == PAWN && (bishop_check_table[tsq]&ksq_bit)) {
    // note the bishop_check_table guarantees that the pawn/king
    //   relationships below are OK... otherwise we would have to
    //   restrict the files for these comparisons
    if(!wtm && (tsq == ksq+9 || tsq == ksq+7))
      { ocheck = 1; }
    else if(wtm && (tsq == ksq-9 || tsq == ksq-7))
      { ocheck = 1; }
  } 
  else if(ptype == KNIGHT && (knight_check_table[tsq]&ksq_bit)) ocheck = 1;    
  
  if(ocheck) { 
    // use swap to determine if our attacker is captured.  If so,
    // this is not a useful check.  Note that we subtract off the value of the
    // piece on the square in case this is a 'non-promising' capture (see above)...
    // in which case the capture of the piece on this square cannot bring us up 
    // to delta_score and the check is only useful if our piece cannot be captured
    if(swap(tsq,(*this),wtm,fsq) - value[PTYPE(sq[tsq])] >= 0) 
       list->mv[i].score = delta_score;
  }

  if(list->mv[i].score < delta_score) {
    // looking for a revealed check
    if(slide_check_table[fsq]&ksq_bit)  {
      position temp_pos = (*this);
      temp_pos.wtm ^= 1;
      temp_pos.sq[tsq] = temp_pos.sq[fsq];
      temp_pos.sq[fsq] = EMPTY;
      ocheck = temp_pos.simple_check(fsq);
      if(ocheck) list->mv[i].score = delta_score + value[PTYPE(sq[tsq])] + pawn_bonus;
    } 
  } 
 
  if(list->mv[i].score >= delta_score) {
    list->count++;           // increase list count
  }

}

/*------------------------------- Bishop Moves --------------------------*/
void position::bishop_cc(move_list *list, int sqr, int ds)
{
  int mm, nn, ii, tsq;

  mm = FILE(sqr); nn = RANK(sqr);

  ii = 1;
  while (mm + ii <= 7 && nn + ii <= 7)
  {
   tsq = SQR((mm+ii),(nn+ii));
   if (!sq[tsq])
     { add_cc(sqr, tsq, list, 0, ds); }
   else if (PSIDE(sq[tsq]) != wtm)
   { add_cc(sqr, tsq, list, 1, ds); break; }
   else break;
   ii++;
  }

  ii = 1;
  while (mm - ii >= 0 && nn + ii <= 7)
  {
   tsq = SQR((mm-ii),(nn+ii));
   if (!sq[tsq])
   { add_cc(sqr, tsq, list, 0, ds); }
   else if (PSIDE(sq[tsq]) != wtm)
   { add_cc(sqr, tsq, list, 1, ds); break; }
   else break;
   ii++;
  }

  ii = 1;
  while(mm + ii <= 7 && nn - ii >= 0)
  {
   tsq = SQR((mm+ii),(nn-ii));
   if (!sq[tsq])
   { add_cc(sqr, tsq, list, 0, ds); }
   else if (PSIDE(sq[tsq]) != wtm)
   { add_cc(sqr, tsq, list, 1, ds); break; }
   else break;
   ii++;
  }

  ii = 1;
  while (mm - ii >= 0 && nn - ii >= 0)
  {
   tsq = SQR((mm-ii),(nn-ii));
   if (!sq[tsq])
   { add_cc(sqr, tsq, list, 0, ds); }
   else if (PSIDE(sq[tsq]) != wtm)
   { add_cc(sqr, tsq, list, 1, ds); break; }
   else break;
   ii++;
  }

}

/*--------------------------- Rook Moves ---------------------------*/

void position::rook_cc(move_list *list, int sqr, int ds)
{
  int mm, nn, ii, tsq;

  mm = FILE(sqr); nn = RANK(sqr);

  ii = 1;
  while (mm + ii <= 7)
  {
   tsq = SQR((mm+ii),nn);
   if (!sq[tsq])
   { add_cc(sqr, tsq, list, 0, ds); }
   else if (PSIDE(sq[tsq]) != wtm)
   { add_cc(sqr, tsq, list, 1, ds); break; }
   else break;
   ii++;
  }

  ii = 1;
  while (mm - ii >= 0)
  {
   tsq = SQR((mm-ii),nn);
   if (!sq[tsq])
   { add_cc(sqr, tsq, list, 0, ds); }
   else if (PSIDE(sq[tsq]) != wtm)
   { add_cc(sqr, tsq, list, 1, ds); break; }
   else break;
   ii++;
  }

  ii = 1;
  while(nn - ii >= 0)
  {
   tsq = SQR(mm,(nn-ii));
   if (!sq[tsq])
   { add_cc(sqr, tsq, list, 0, ds); }
   else if (PSIDE(sq[tsq]) != wtm)
   { add_cc(sqr, tsq, list, 1, ds); break; }
   else break;
   ii++;
  }

  ii = 1;
  while (nn + ii <= 7)
  {
   tsq = SQR(mm,(nn+ii));
   if (!sq[tsq])
   { add_cc(sqr, tsq, list, 0, ds); }
   else if (PSIDE(sq[tsq]) != wtm)
   { add_cc(sqr, tsq, list, 1, ds); break; }
   else break;
   ii++;
  }

}

/*--------------------------- Knight Moves ----------------------------*/
void position::knight_cc(move_list *list, int sqr, int ds)
{
  int tsq;

  if(FILE(sqr) < 6 && RANK(sqr) < 7) {
   tsq = sqr + 10;
   if(sq[tsq]) {
     if(PSIDE(sq[tsq]) != wtm) add_cc(sqr, tsq, list, 1, ds);
   } else add_cc(sqr, tsq, list, 0, ds);
  }
  if(FILE(sqr) < 6 && RANK(sqr)) {
   tsq = sqr - 6;
   if(sq[tsq]) {
     if(PSIDE(sq[tsq]) != wtm) add_cc(sqr, tsq, list, 1, ds);
   } else add_cc(sqr, tsq, list, 0, ds);
  }
  if(FILE(sqr) > 1 && RANK(sqr) < 7) {
   tsq = sqr + 6;
   if(sq[tsq]) {
     if(PSIDE(sq[tsq]) != wtm) add_cc(sqr, tsq, list, 1, ds);
   } else add_cc(sqr, tsq, list, 0, ds);
  }
  if(FILE(sqr) > 1 && RANK(sqr)) {
   tsq = sqr - 10;
   if(sq[tsq]) {
     if(PSIDE(sq[tsq]) != wtm) add_cc(sqr, tsq, list, 1, ds);
   } else add_cc(sqr, tsq, list, 0, ds);
  }
  if(FILE(sqr) < 7 && RANK(sqr) < 6) {
   tsq = sqr + 17;
   if(sq[tsq]) {
     if(PSIDE(sq[tsq]) != wtm) add_cc(sqr, tsq, list, 1, ds);
   } else add_cc(sqr, tsq, list, 0, ds);
  }
  if(FILE(sqr) && RANK(sqr) < 6) {
   tsq = sqr + 15;
   if(sq[tsq]) {
     if(PSIDE(sq[tsq]) != wtm) add_cc(sqr, tsq, list, 1, ds);
   } else add_cc(sqr, tsq, list, 0, ds);
  }
  if(FILE(sqr) < 7 && RANK(sqr) > 1) {
   tsq = sqr - 15;
   if(sq[tsq]) {
     if(PSIDE(sq[tsq]) != wtm) add_cc(sqr, tsq, list, 1, ds);
   } else add_cc(sqr, tsq, list, 0, ds);
  }
  if(FILE(sqr) && RANK(sqr) > 1) {
   tsq = sqr - 17;
   if(sq[tsq]) {
     if(PSIDE(sq[tsq]) != wtm) add_cc(sqr, tsq, list, 1, ds);
   } else add_cc(sqr, tsq, list, 0, ds);
  }

}

/*--------------------------- King Moves ----------------------------*/
void position::king_cc(move_list *list, int sqr, int ds)
{
  int tsq;

  if(FILE(sqr) && RANK(sqr) < 7) {
   tsq = sqr + 7;
   if(sq[tsq]) {
     if(PSIDE(sq[tsq]) != wtm) add_cc(sqr, tsq, list, 1, ds);
   } else add_cc(sqr, tsq, list, 0, ds);
  }
  if(RANK(sqr) < 7) {
   tsq = sqr + 8;
   if(sq[tsq]) {
     if(PSIDE(sq[tsq]) != wtm) add_cc(sqr, tsq, list, 1, ds);
   } else add_cc(sqr, tsq, list, 0, ds);
  }
  if(FILE(sqr) < 7 && RANK(sqr) < 7) {
   tsq = sqr + 9;
   if(sq[tsq]) {
     if(PSIDE(sq[tsq]) != wtm) add_cc(sqr, tsq, list, 1, ds);
   } else add_cc(sqr, tsq, list, 0, ds);
  }
  if(FILE(sqr)) {
   tsq = sqr - 1;
   if(sq[tsq]) {
     if(PSIDE(sq[tsq]) != wtm) add_cc(sqr, tsq, list, 1, ds);
   } else add_cc(sqr, tsq, list, 0, ds);
  }
  if(FILE(sqr) < 7) {
   tsq = sqr + 1;
   if(sq[tsq]) {
     if(PSIDE(sq[tsq]) != wtm) add_cc(sqr, tsq, list, 1, ds);
   } else add_cc(sqr, tsq, list, 0, ds);
  }
  if(FILE(sqr) < 7 && RANK(sqr)) {
   tsq = sqr - 7;
   if(sq[tsq]) {
     if(PSIDE(sq[tsq]) != wtm) add_cc(sqr, tsq, list, 1, ds);
   } else add_cc(sqr, tsq, list, 0, ds);
  }
  if(RANK(sqr)) {
   tsq = sqr - 8;
   if(sq[tsq]) {
     if(PSIDE(sq[tsq]) != wtm) add_cc(sqr, tsq, list, 1, ds);
   } else add_cc(sqr, tsq, list, 0, ds);
  }
  if(FILE(sqr) && RANK(sqr)) {
   tsq = sqr - 9;
   if(sq[tsq]) {
     if(PSIDE(sq[tsq]) != wtm) add_cc(sqr, tsq, list, 1, ds);
   } else add_cc(sqr, tsq, list, 0, ds);
  }

}

/*------------------------ Pawn Moves ------------------------------*/
void position::pawn_cc(move_list *list, int sqr, int ds)
{

  if(wtm) {                        // if it is white's pawn
    if(!PTYPE(sq[sqr+8])) {
     add_cc(sqr, sqr+8, list, PAWN_PUSH, ds);
     if(RANK(sqr) == 1) {
       if(!PTYPE(sq[sqr+16])) { add_cc(sqr, sqr+16, list, PAWN_PUSH, ds); }
      }
   }
   if(FILE(sqr)) {
     if (PTYPE(sq[sqr+7]) && PSIDE(sq[sqr+7]) != wtm)
      { add_cc(sqr, sqr+7, list, (CAPTURE|PAWN_PUSH), ds); }
    else if((sqr+7) == ep && ep)
      { add_cc(sqr, sqr+7, list, (EP|CAPTURE|PAWN_PUSH), ds); }
   }
   if(FILE(sqr) < 7) {
     if (PTYPE(sq[sqr+9]) && PSIDE(sq[sqr+9]) != wtm)
      { add_cc(sqr, sqr+9, list, (CAPTURE|PAWN_PUSH), ds); }
    else if((sqr+9) == ep && ep)
      { add_cc(sqr, sqr+9, list, (EP|CAPTURE|PAWN_PUSH), ds); }
   }
  } else {                           // or if it is black's pawn
    if(!PTYPE(sq[sqr-8])) {
     add_cc(sqr, sqr-8, list, PAWN_PUSH, ds);
     if(RANK(sqr) == 6) {
       if(!PTYPE(sq[sqr-16])) { add_cc(sqr, sqr-16, list, PAWN_PUSH, ds); }
      }
   }
   if(FILE(sqr)) {
     if (PTYPE(sq[sqr-9]) && PSIDE(sq[sqr-9]) != wtm)
      { add_cc(sqr, sqr-9, list, (CAPTURE|PAWN_PUSH), ds); }
    else if((sqr-9) == ep && ep)
      { add_cc(sqr, sqr-9, list, (EP|CAPTURE|PAWN_PUSH), ds); }
   }
   if(FILE(sqr) < 7) {
     if (PTYPE(sq[sqr-7]) && PSIDE(sq[sqr-7]) != wtm)
      { add_cc(sqr, sqr-7, list, (CAPTURE|PAWN_PUSH), ds); }
    else if((sqr-7) == ep && ep)
      { add_cc(sqr, sqr-7, list, (EP|CAPTURE|PAWN_PUSH), ds); }
   }
  }
}
























