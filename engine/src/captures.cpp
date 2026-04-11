// EXchess source code, (c) Daniel C. Homan  1997-2016
// Released under the GNU public license, see file license.txt

/* Piece captures only */

#include "define.h"
#include "chess.h"
#include "const.h"
#include "funct.h"

/*------------------------- Semi-legal capture generator ---------------------*/

int position::captures(move_list *list, int delta_score)
{
  int i;
  list->count = 0;

  for(i=1;i<=plist[wtm][PAWN][0];i++) 
    pawn_capts(list, plist[wtm][PAWN][i], delta_score);
  for(i=1;i<=plist[wtm][KNIGHT][0];i++) 
    knight_capts(list, plist[wtm][KNIGHT][i], delta_score);
  for(i=1;i<=plist[wtm][BISHOP][0];i++) 
    bishop_capts(list, plist[wtm][BISHOP][i], delta_score);
  for(i=1;i<=plist[wtm][ROOK][0];i++) 
    rook_capts(list, plist[wtm][ROOK][i], delta_score);
  for(i=1;i<=plist[wtm][QUEEN][0];i++) {
    bishop_capts(list, plist[wtm][QUEEN][i], delta_score);
    rook_capts(list, plist[wtm][QUEEN][i], delta_score);
  }
  for(i=1;i<=plist[wtm][KING][0];i++) 
    king_capts(list, plist[wtm][KING][i], delta_score);

  return 0;
}

/* Add capture function */
// Function to add a capture to the capture list
// The move is also scored for the alpha-beta search
// fsq = from square,  tsq = to square
int position::add_capt(int fsq, int tsq, move_list *list, char type, int delta_score)
{
  int i = list->count;      // index of move list

  // add move to list...
  list->mv[i].m.b.from = fsq;
  list->mv[i].m.b.to = tsq;
  list->mv[i].m.b.type = type;
  list->mv[i].m.b.promote = 0;

  // Give an initial score to the move below delta_score
  // so that by default it will not be put in the list 
  // unless a step below changes the score
  list->mv[i].score = delta_score-100;  

  // give a score for a promotion + put in move list
  if((tsq > 55 || tsq < 8) && PTYPE(sq[fsq]) == PAWN) {

    list->mv[i].m.b.type |= PROMOTE;
    list->mv[i].m.b.promote = QUEEN;
    if(swap(tsq,(*this),wtm,fsq) >= 0) list->mv[i].score += 1000;
    //list->mv[i].score += 1000;

  } else {   // score regular captures in the normal way

    // If the move is a pawn capture 
    //  give a bonus by the rank of the pawn
    int pawn_bonus = 0;
    if(PTYPE(sq[tsq]) == PAWN) {
      if(gstage > 9) { // bonus for pawns about to queen in end-game
	if(wtm && RANK(tsq) == 1) pawn_bonus += 35*gstage;   
	if(!wtm && RANK(tsq) == 6) pawn_bonus += 35*gstage;
      }
    }

    if(PTYPE(sq[tsq]) == BISHOP && plist[wtm^1][BISHOP][0] == 2) pawn_bonus += BISHOP_PAIR;
    
    if(value[PTYPE(sq[tsq])]+pawn_bonus >= delta_score) {
      list->mv[i].score = swap(tsq,(*this),wtm,fsq)+pawn_bonus;

      // looking for a revealed check if score isn't high enough
      if(list->mv[i].score < delta_score && (slide_check_table[fsq]&(1ULL<<plist[wtm^1][KING][1])))  {
	position temp_pos = (*this);
	temp_pos.wtm ^= 1;
	temp_pos.sq[tsq] = temp_pos.sq[fsq];
	temp_pos.sq[fsq] = 0;
	if(temp_pos.simple_check(fsq)) list->mv[i].score = value[PTYPE(sq[tsq])]+pawn_bonus;
      }
      
    } 
    
  }

    
  if(list->mv[i].score >= delta_score) {
    list->count++;           // increase list count
  }

  return 0;

}

/*---------------------------- Bishop Captures --------------------------*/
int position::bishop_capts(move_list *list, int sqr, int ds)
{
  int mm, nn, ii;
  int tsq;                            // to square

  mm = FILE(sqr); nn = RANK(sqr);     // set rank and file of bishop

  ii = 1;
  while (mm + ii <= 7 && nn + ii <= 7)
  {
   tsq = SQR((mm+ii),(nn+ii));                   // set to square
   if (!sq[tsq]) { ii++; }                       // if empty, move on
   else if (PSIDE(sq[tsq]) != wtm)               // else if other side,
     { add_capt(sqr, tsq, list, 1, ds); break; } // add to capture list
   else break;                                   // break if our piece
  }

  ii = 1;
  while (mm - ii >= 0 && nn + ii <= 7)
  {
   tsq = SQR((mm-ii),(nn+ii));
   if (!sq[tsq]) { ii++; }
   else if (PSIDE(sq[tsq]) != wtm)
     { add_capt(sqr, tsq, list, 1, ds); break; }
   else break;
  }

  ii = 1;
  while(mm + ii <= 7 && nn - ii >= 0)
  {
   tsq = SQR((mm+ii),(nn-ii));
   if (!sq[tsq]) { ii++; }
   else if (PSIDE(sq[tsq]) != wtm)
     { add_capt(sqr, tsq, list, 1, ds); break; }
   else break;
  }

  ii = 1;
  while (mm - ii >= 0 && nn - ii >= 0)
  {
   tsq = SQR((mm-ii),(nn-ii));
   if (!sq[tsq]) { ii++; }
   else if (PSIDE(sq[tsq]) != wtm)
     { add_capt(sqr, tsq, list, 1, ds); break; }
   else break;
  }

  return 0;

}

/*--------------------------- Rook Captures ---------------------------*/
int position::rook_capts(move_list *list, int sqr, int ds)
{
  int mm, nn, ii;
  int tsq;                              // to square

  mm = FILE(sqr); nn = RANK(sqr);       // set file and rank of rook

  ii = 1;
  while (mm + ii <= 7)
  {
   tsq = SQR((mm+ii),nn);                         // set to square
   if (!sq[tsq]) { ii++; }                        // if empty, move on
   else if (PSIDE(sq[tsq]) != wtm)                // else if other side,
     { add_capt(sqr, tsq, list, 1, ds); break; }  // add to capture list
   else break;                                    // else if our side, break
  }

  ii = 1;
  while (mm - ii >= 0)
  {
   tsq = SQR((mm-ii),nn);
   if (!sq[tsq]) { ii++; }
   else if (PSIDE(sq[tsq]) != wtm)
     { add_capt(sqr, tsq, list, 1, ds); break; }
   else break;
  }

  ii = 1;
  while(nn - ii >= 0)
  {
   tsq = SQR(mm,(nn-ii));
   if (!sq[tsq]) { ii++; }
   else if (PSIDE(sq[tsq]) != wtm)
     { add_capt(sqr, tsq, list, 1, ds); break; }
   else break;
  }

  ii = 1;
  while (nn + ii <= 7)
  {
   tsq = SQR(mm,(nn+ii));
   if (!sq[tsq]) { ii++; }
   else if (PSIDE(sq[tsq]) != wtm)
     { add_capt(sqr, tsq, list, 1, ds); break; }
   else break;
  }

  return 0;

}

/*--------------------------- Knight Captures ----------------------------*/
int position::knight_capts(move_list *list, int sqr, int ds)
{
  int tsq;                                // to square

  if(FILE(sqr) < 6 && RANK(sqr) < 7) {
   tsq = sqr + 10;                                   // set to square
   if(PSIDE(sq[tsq]) != wtm) {                        // if occupied by
     if(sq[tsq]) add_capt(sqr, tsq, list, 1, ds);   // other side, add
    }                                                // to capture list
  }
  if(FILE(sqr) < 6 && RANK(sqr)) {
   tsq = sqr - 6;
   if(PSIDE(sq[tsq]) != wtm) {
     if(sq[tsq]) add_capt(sqr, tsq, list, 1, ds);
    }
  }
  if(FILE(sqr) > 1 && RANK(sqr) < 7) {
   tsq = sqr + 6;
   if(PSIDE(sq[tsq]) != wtm) {
     if(sq[tsq]) add_capt(sqr, tsq, list, 1, ds);
    }
  }
  if(FILE(sqr) > 1 && RANK(sqr)) {
   tsq = sqr - 10;
   if(PSIDE(sq[tsq]) != wtm) {
     if(sq[tsq]) add_capt(sqr, tsq, list, 1, ds);
    }
  }
  if(FILE(sqr) < 7 && RANK(sqr) < 6) {
   tsq = sqr + 17;
   if(PSIDE(sq[tsq]) != wtm) {
     if(sq[tsq]) add_capt(sqr, tsq, list, 1, ds);
    }
  }
  if(FILE(sqr) && RANK(sqr) < 6) {
   tsq = sqr + 15;
   if(PSIDE(sq[tsq]) != wtm) {
     if(sq[tsq]) add_capt(sqr, tsq, list, 1, ds);
    }
  }
  if(FILE(sqr) < 7 && RANK(sqr) > 1) {
   tsq = sqr - 15;
   if(PSIDE(sq[tsq]) != wtm) {
     if(sq[tsq]) add_capt(sqr, tsq, list, 1, ds);
    }
  }
  if(FILE(sqr) && RANK(sqr) > 1) {
   tsq = sqr - 17;
   if(PSIDE(sq[tsq]) != wtm) {
     if(sq[tsq]) add_capt(sqr, tsq, list, 1, ds);
    }
  }

  return 0;

}

/*--------------------------- King Captures ----------------------------*/
int position::king_capts(move_list *list, int sqr, int ds)
{
  int tsq;                            // to square

  if(FILE(sqr) && RANK(sqr) < 7) {
   tsq = sqr + 7;                                   // set to square
   if(PSIDE(sq[tsq]) != wtm) {                       // if occupied by
     if(sq[tsq]) add_capt(sqr, tsq, list, 1, ds);  // other side, add
    }                                               // to capture list
  }
  if(RANK(sqr) < 7) {
   tsq = sqr + 8;
   if(PSIDE(sq[tsq]) != wtm) {
     if(sq[tsq]) add_capt(sqr, tsq, list, 1, ds);
    }
  }
  if(FILE(sqr) < 7 && RANK(sqr) < 7) {
   tsq = sqr + 9;
   if(PSIDE(sq[tsq]) != wtm) {
     if(sq[tsq]) add_capt(sqr, tsq, list, 1, ds);
    }
  }
  if(FILE(sqr)) {
   tsq = sqr - 1;
   if(PSIDE(sq[tsq]) != wtm) {
     if(sq[tsq]) add_capt(sqr, tsq, list, 1, ds);
    }
  }
  if(FILE(sqr) < 7) {
   tsq = sqr + 1;
   if(PSIDE(sq[tsq]) != wtm) {
     if(sq[tsq]) add_capt(sqr, tsq, list, 1, ds);
    }
  }
  if(FILE(sqr) < 7 && RANK(sqr)) {
   tsq = sqr - 7;
   if(PSIDE(sq[tsq]) != wtm) {
     if(sq[tsq]) add_capt(sqr, tsq, list, 1, ds);
    }
  }
  if(RANK(sqr)) {
   tsq = sqr - 8;
   if(PSIDE(sq[tsq]) != wtm) {
     if(sq[tsq]) add_capt(sqr, tsq, list, 1, ds);
    }
  }
  if(FILE(sqr) && RANK(sqr)) {
   tsq = sqr - 9;
   if(PSIDE(sq[tsq]) != wtm) {
     if(sq[tsq]) add_capt(sqr, tsq, list, 1, ds);
    }
  }

  return 0;
}

/*------------------------ Pawn Captures ------------------------------*/
// Not including en passant!
// Do include promotions!
int position::pawn_capts(move_list *list, int sqr, int ds)
{

  if(wtm) {                        // if it is white's pawn
   if(FILE(sqr)) {
     if (PTYPE(sq[sqr+7]) && PSIDE(sq[sqr+7]) != wtm)
      { add_capt(sqr, sqr+7, list, (PAWN_PUSH|CAPTURE), ds); }  
   }
   if(FILE(sqr) < 7) {
     if (PTYPE(sq[sqr+9]) && PSIDE(sq[sqr+9]) != wtm)
      { add_capt(sqr, sqr+9, list, (PAWN_PUSH|CAPTURE), ds); }
   }
   if(RANK(sqr) == 6) {
     if (!PTYPE(sq[sqr+8]))
      { add_capt(sqr, sqr+8, list, (PAWN_PUSH|PROMOTE), ds); }
   }
  } else {                           // or if it is black's pawn
   if(FILE(sqr)) {
     if (PTYPE(sq[sqr-9]) && PSIDE(sq[sqr-9]) != wtm)
      { add_capt(sqr, sqr-9, list, (PAWN_PUSH|CAPTURE), ds); }
   }
   if(FILE(sqr) < 7) {
     if (PTYPE(sq[sqr-7]) && PSIDE(sq[sqr-7]) != wtm)
      { add_capt(sqr, sqr-7, list, (PAWN_PUSH|CAPTURE), ds); }
   }
   if(RANK(sqr) == 1) {
     if (!PTYPE(sq[sqr-8]))
      { add_capt(sqr, sqr-8, list, (PAWN_PUSH|PROMOTE), ds); }
   }
  }

  return 0;
}
























