// EXchess source code, (c) Daniel C. Homan  1997-2016
// Released under the GNU public license, see file license.txt

/* Piece moves */

#include "define.h"
#include "chess.h"
#include "const.h"
#include "funct.h"
#include "extern.h"

/*------------------------- Semi-legal moves generator ---------------------*/

void position::allmoves(move_list *list, ts_thread_data *tdata)
{
  int i;
  list->count = 0;

  for(i=1;i<=plist[wtm][PAWN][0];i++) 
    pawn_moves(list, plist[wtm][PAWN][i], tdata);
  for(i=1;i<=plist[wtm][KNIGHT][0];i++) 
    knight_moves(list, plist[wtm][KNIGHT][i], tdata);
  for(i=1;i<=plist[wtm][BISHOP][0];i++) 
    bishop_moves(list, plist[wtm][BISHOP][i], tdata);
  for(i=1;i<=plist[wtm][ROOK][0];i++) 
    rook_moves(list, plist[wtm][ROOK][i], tdata);
  for(i=1;i<=plist[wtm][QUEEN][0];i++) {
    bishop_moves(list, plist[wtm][QUEEN][i], tdata);
    rook_moves(list, plist[wtm][QUEEN][i], tdata);
  }
  for(i=1;i<=plist[wtm][KING][0];i++) 
    king_moves(list, plist[wtm][KING][i], tdata);

}

int position::verify_move(move_list *list, ts_thread_data *tdata, move tmove)
{
  list->count = 0;

  if(PSIDE(sq[tmove.b.from]) != wtm) return 0;


  if(PTYPE(sq[tmove.b.from]) == PAWN)  
    pawn_moves(list, tmove.b.from, tdata);
  else if(PTYPE(sq[tmove.b.from]) == KNIGHT)  
    knight_moves(list, tmove.b.from, tdata);
  else if(PTYPE(sq[tmove.b.from]) == BISHOP)  
    bishop_moves(list, tmove.b.from, tdata);
  else if(PTYPE(sq[tmove.b.from]) == ROOK)  
    rook_moves(list, tmove.b.from, tdata);
  else if(PTYPE(sq[tmove.b.from]) == QUEEN) { 
    bishop_moves(list, tmove.b.from, tdata);
    rook_moves(list, tmove.b.from, tdata);
  }
  else if(PTYPE(sq[tmove.b.from]) == KING)  
    king_moves(list, tmove.b.from, tdata);

  for(int i=0;i<list->count;i++) 
    if(list->mv[i].m.b.to == tmove.b.to) return 1;


  return 0;

}

/* Add move to move list */
// Function to add move to move list
// Move is scored for alpha-beta algorithm
void position::add_move(int fsq, int tsq, move_list *list, char type, ts_thread_data *tdata)
{
  int i = list->count;                 // move list index

  // add move to list
  list->mv[i].m.b.from = fsq;
  list->mv[i].m.b.to = tsq;
  list->mv[i].m.b.type = type;
  list->mv[i].m.b.promote = 0;

  if(check) {
   if(!(check_table[tsq]&(1ULL<<plist[wtm][KING][1]))
      && !((knight_check_table[tsq]&(1ULL<<plist[wtm][KING][1]))
           && PTYPE(sq[tsq]) == KNIGHT)
      && !(type&EP)
      && PTYPE(sq[fsq]) != KING)
    return;
   position temp_pos = (*this);
   temp_pos.sq[tsq] = temp_pos.sq[fsq];
   temp_pos.sq[fsq] = EMPTY;
   // if this is a capture, remove piece from temporary plist
   //  -- necessary to make the 'attacks' function work properly
   if(type&(CAPTURE|EP)) {
     int ptype = PTYPE(sq[tsq]); 
     int capt_square = tsq;
     if(type&EP) { ptype = PAWN; capt_square = (wtm ? (tsq-8) : (tsq+8)); }
     for(int pi=1; pi<=plist[wtm^1][ptype][0];pi++)
       if(plist[wtm^1][ptype][pi] == capt_square) {
	 temp_pos.plist[wtm^1][ptype][pi] =
	   plist[wtm^1][ptype][plist[wtm^1][ptype][0]];
	 temp_pos.plist[wtm^1][ptype][0]--;
	 break;
       }       
   }
   if(type&EP) {
     if(wtm) temp_pos.sq[tsq-8] = EMPTY;
     else temp_pos.sq[tsq+8] = EMPTY;
   }
   if(PTYPE(temp_pos.sq[tsq]) == KING) {
     if(temp_pos.attacked(tsq, (wtm^1))) return;
   } else {
     if(temp_pos.attacked(plist[wtm][KING][1], (wtm^1))) return;
   }
  }   

  // is it a promotion move? if so, generate all types of promotions
  if((tsq > 55 || tsq < 8) && PTYPE(sq[fsq]) == PAWN)
   {
     list->mv[i].m.b.type |= PROMOTE; list->mv[i].m.b.promote = QUEEN;
     list->mv[i].score = 20000000; list->count++;
     list->mv[list->count] = list->mv[i];
     list->mv[list->count].m.b.promote = ROOK;
     list->mv[list->count].score -= 9000050; list->count++;
     list->mv[list->count] = list->mv[i];
     list->mv[list->count].m.b.promote = BISHOP;
     list->mv[list->count].score -= 9000060; list->count++;
     list->mv[list->count] = list->mv[i];
     list->mv[list->count].m.b.promote = KNIGHT;
     list->mv[list->count].score -= 9000070; list->count++;
     // if one of these is a hash_move, score it first
     for(int j = i; j < list->count; j++) {
       if(list->mv[j].m.t == hmove.t) { list->mv[j].score = 50000000; }
     }
     return;
   }

  // Is is a pawn push to the 7th rank?  If so, mark it as such
  if(type&PAWN_PUSH) {
    if(RANK(tsq) == 1 || RANK(tsq) == 6) list->mv[i].m.b.type |= PAWN_PUSH7;
  }

  // if it is a hash_move, score it first
  if(list->mv[i].m.t == hmove.t) { 
    list->mv[i].score = 50000000;
  }
  // else if it is a capture...
  else if(type&CAPTURE) {
    //  give a bonus for captures of pawns about to queen
    int pawn_bonus = 0;
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
    // now score the move
    if(value[PTYPE(sq[tsq])]+pawn_bonus+50 >= value[PTYPE(sq[fsq])]) {  // +50 allows minor exchanges
      list->mv[i].score = 10000000 + 1000*PTYPE(sq[tsq])+pawn_bonus - PTYPE(sq[fsq]);
    } else if(swap(tsq,(*this),wtm,fsq)+pawn_bonus+50 >= 0) {           // +50 allows minor exchanges
      list->mv[i].score = 10000000 + 1000*PTYPE(sq[tsq])+pawn_bonus - PTYPE(sq[fsq]);
    // looking for a revealed check if score isn't high enough
    } else if(slide_check_table[fsq]&(1ULL<<plist[wtm^1][KING][1]))  {
	position temp_pos = (*this);
	temp_pos.wtm ^= 1;
	temp_pos.sq[tsq] = temp_pos.sq[fsq];
	temp_pos.sq[fsq] = EMPTY;
	if(temp_pos.simple_check(fsq))
	  list->mv[i].score = 10000000 + 1000*PTYPE(sq[tsq])+pawn_bonus - PTYPE(sq[fsq]);
	else list->mv[i].score = 0;	
    } else list->mv[i].score = 0;
  }
  // if move has no score give a 'killer score' or a history score         
  else { 
    if(rmove.t == list->mv[i].m.t) {                   // reply/counter move
      list->mv[i].score = 8000000;
    } else if(tdata->killer1[wtm] == list->mv[i].m.t) { // killer 1
      list->mv[i].score = 6000000;
    } else if(tdata->killer2[wtm] == list->mv[i].m.t) { // killer 2
      list->mv[i].score = 4000000;
    } else if(tdata->killer3[wtm] == list->mv[i].m.t) {  // killer 3
      list->mv[i].score = 2000000;
    } else {
      // give it a history score
      list->mv[i].score = tdata->history[sq[fsq]][tsq];
    }
  }

  // bonus for moves that complete a combination
  if(list->mv[i].m.t == cmove.t && !(list->mv[i].score == 0 && (type&CAPTURE))) {
    list->mv[i].score += 9000000;
  }
  
  // if move is scored less than a killer, see if it might
  // be a Pawn Push to 7th and thus worth more....
  if(!(type&CAPTURE) && list->mv[i].score < 2000000 && pieces[wtm^1] < 6
     && (type&PAWN_PUSH7)) {
    if(swap(tsq,(*this),wtm,fsq) >= 0) list->mv[i].score = 2000000;
  }

 
  // if move is scored less than a killer, see if it might
  // be a safe check and thus worth more....
  if(!(type&CAPTURE) && list->mv[i].score < 2000000) {
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
      // this is not a useful check.  
      if(swap(tsq, (*this),wtm,fsq) < 0) ocheck = 0;
    }

    if(!ocheck) {
      // looking for a revealed check
      if(slide_check_table[fsq]&ksq_bit)  {
	position temp_pos = (*this);
	temp_pos.wtm ^= 1;
	temp_pos.sq[tsq] = temp_pos.sq[fsq];
	temp_pos.sq[fsq] = EMPTY;
	ocheck = temp_pos.simple_check(fsq);
      } 
    }

    // bonus for safe check...
    if(ocheck) { 
      list->mv[i].score = 2000000;
      // more if this will be dangerous for other side
      if(qchecks[wtm] > 2) { 
	list->mv[i].score += 10000000; 
      }
    }
 
  }
 
  list->count++;                       // increment list count

}

/*------------------------------- Bishop Moves --------------------------*/
void position::bishop_moves(move_list *list, int sqr, ts_thread_data *tdata)
{
  int mm, nn, ii, tsq;

  mm = FILE(sqr); nn = RANK(sqr);

  ii = 1;
  while (mm + ii <= 7 && nn + ii <= 7)
  {
   tsq = SQR((mm+ii),(nn+ii));
   if (!sq[tsq])
   { add_move(sqr, tsq, list, 0, tdata); }
   else if (PSIDE(sq[tsq]) != wtm)
   { add_move(sqr, tsq, list, 1, tdata); break; }
   else break;
   ii++;
  }

  ii = 1;
  while (mm - ii >= 0 && nn + ii <= 7)
  {
   tsq = SQR((mm-ii),(nn+ii));
   if (!sq[tsq])
   { add_move(sqr, tsq, list, 0, tdata); }
   else if (PSIDE(sq[tsq]) != wtm)
   { add_move(sqr, tsq, list, 1, tdata); break; }
   else break;
   ii++;
  }

  ii = 1;
  while(mm + ii <= 7 && nn - ii >= 0)
  {
   tsq = SQR((mm+ii),(nn-ii));
   if (!sq[tsq])
   { add_move(sqr, tsq, list, 0, tdata); }
   else if (PSIDE(sq[tsq]) != wtm)
   { add_move(sqr, tsq, list, 1, tdata); break; }
   else break;
   ii++;
  }

  ii = 1;
  while (mm - ii >= 0 && nn - ii >= 0)
  {
   tsq = SQR((mm-ii),(nn-ii));
   if (!sq[tsq])
   { add_move(sqr, tsq, list, 0, tdata); }
   else if (PSIDE(sq[tsq]) != wtm)
   { add_move(sqr, tsq, list, 1, tdata); break; }
   else break;
   ii++;
  }

}

/*--------------------------- Rook Moves ---------------------------*/

void position::rook_moves(move_list *list, int sqr, ts_thread_data *tdata)
{
  int mm, nn, ii, tsq;

  mm = FILE(sqr); nn = RANK(sqr);

  ii = 1;
  while (mm + ii <= 7)
  {
   tsq = SQR((mm+ii),nn);
   if (!sq[tsq])
   { add_move(sqr, tsq, list, 0, tdata); }
   else if (PSIDE(sq[tsq]) != wtm)
   { add_move(sqr, tsq, list, 1, tdata); break; }
   else break;
   ii++;
  }

  ii = 1;
  while (mm - ii >= 0)
  {
   tsq = SQR((mm-ii),nn);
   if (!sq[tsq])
   { add_move(sqr, tsq, list, 0, tdata); }
   else if (PSIDE(sq[tsq]) != wtm)
   { add_move(sqr, tsq, list, 1, tdata); break; }
   else break;
   ii++;
  }

  ii = 1;
  while(nn - ii >= 0)
  {
   tsq = SQR(mm,(nn-ii));
   if (!sq[tsq])
   { add_move(sqr, tsq, list, 0, tdata); }
   else if (PSIDE(sq[tsq]) != wtm)
   { add_move(sqr, tsq, list, 1, tdata); break; }
   else break;
   ii++;
  }

  ii = 1;
  while (nn + ii <= 7)
  {
   tsq = SQR(mm,(nn+ii));
   if (!sq[tsq])
   { add_move(sqr, tsq, list, 0, tdata); }
   else if (PSIDE(sq[tsq]) != wtm)
   { add_move(sqr, tsq, list, 1, tdata); break; }
   else break;
   ii++;
  }

}

/*--------------------------- Knight Moves ----------------------------*/
void position::knight_moves(move_list *list, int sqr, ts_thread_data *tdata)
{
  int tsq;

  if(FILE(sqr) < 6 && RANK(sqr) < 7) {
   tsq = sqr + 10;
   if(sq[tsq]) {
     if(PSIDE(sq[tsq]) != wtm) add_move(sqr, tsq, list, 1, tdata);
   } else add_move(sqr, tsq, list, 0, tdata);
  }
  if(FILE(sqr) < 6 && RANK(sqr)) {
   tsq = sqr - 6;
   if(sq[tsq]) {
     if(PSIDE(sq[tsq]) != wtm) add_move(sqr, tsq, list, 1, tdata);
   } else add_move(sqr, tsq, list, 0, tdata);
  }
  if(FILE(sqr) > 1 && RANK(sqr) < 7) {
   tsq = sqr + 6;
   if(sq[tsq]) {
     if(PSIDE(sq[tsq]) != wtm) add_move(sqr, tsq, list, 1, tdata);
   } else add_move(sqr, tsq, list, 0, tdata);
  }
  if(FILE(sqr) > 1 && RANK(sqr)) {
   tsq = sqr - 10;
   if(sq[tsq]) {
     if(PSIDE(sq[tsq]) != wtm) add_move(sqr, tsq, list, 1, tdata);
   } else add_move(sqr, tsq, list, 0, tdata);
  }
  if(FILE(sqr) < 7 && RANK(sqr) < 6) {
   tsq = sqr + 17;
   if(sq[tsq]) {
     if(PSIDE(sq[tsq]) != wtm) add_move(sqr, tsq, list, 1, tdata);
   } else add_move(sqr, tsq, list, 0, tdata);
  }
  if(FILE(sqr) && RANK(sqr) < 6) {
   tsq = sqr + 15;
   if(sq[tsq]) {
     if(PSIDE(sq[tsq]) != wtm) add_move(sqr, tsq, list, 1, tdata);
   } else add_move(sqr, tsq, list, 0, tdata);
  }
  if(FILE(sqr) < 7 && RANK(sqr) > 1) {
   tsq = sqr - 15;
   if(sq[tsq]) {
     if(PSIDE(sq[tsq]) != wtm) add_move(sqr, tsq, list, 1, tdata);
   } else add_move(sqr, tsq, list, 0, tdata);
  }
  if(FILE(sqr) && RANK(sqr) > 1) {
   tsq = sqr - 17;
   if(sq[tsq]) {
     if(PSIDE(sq[tsq]) != wtm) add_move(sqr, tsq, list, 1, tdata);
   } else add_move(sqr, tsq, list, 0, tdata);
  }

}

/*--------------------------- King Moves ----------------------------*/
void position::king_moves(move_list *list, int sqr, ts_thread_data *tdata)
{
  int tsq;

  if(FILE(sqr) && RANK(sqr) < 7) {
   tsq = sqr + 7;
   if(sq[tsq]) {
     if(PSIDE(sq[tsq]) != wtm) add_move(sqr, tsq, list, 1, tdata);
   } else add_move(sqr, tsq, list, 0, tdata);
  }
  if(RANK(sqr) < 7) {
   tsq = sqr + 8;
   if(sq[tsq]) {
     if(PSIDE(sq[tsq]) != wtm) add_move(sqr, tsq, list, 1, tdata);
   } else add_move(sqr, tsq, list, 0, tdata);
  }
  if(FILE(sqr) < 7 && RANK(sqr) < 7) {
   tsq = sqr + 9;
   if(sq[tsq]) {
     if(PSIDE(sq[tsq]) != wtm) add_move(sqr, tsq, list, 1, tdata);
   } else add_move(sqr, tsq, list, 0, tdata);
  }
  if(FILE(sqr)) {
   tsq = sqr - 1;
   if(sq[tsq]) {
     if(PSIDE(sq[tsq]) != wtm) add_move(sqr, tsq, list, 1, tdata);
   } else add_move(sqr, tsq, list, 0, tdata);
  }
  if(FILE(sqr) < 7) {
   tsq = sqr + 1;
   if(sq[tsq]) {
     if(PSIDE(sq[tsq]) != wtm) add_move(sqr, tsq, list, 1, tdata);
   } else add_move(sqr, tsq, list, 0, tdata);
  }
  if(FILE(sqr) < 7 && RANK(sqr)) {
   tsq = sqr - 7;
   if(sq[tsq]) {
     if(PSIDE(sq[tsq]) != wtm) add_move(sqr, tsq, list, 1, tdata);
   } else add_move(sqr, tsq, list, 0, tdata);
  }
  if(RANK(sqr)) {
   tsq = sqr - 8;
   if(sq[tsq]) {
     if(PSIDE(sq[tsq]) != wtm) add_move(sqr, tsq, list, 1, tdata);
   } else add_move(sqr, tsq, list, 0, tdata);
  }
  if(FILE(sqr) && RANK(sqr)) {
   tsq = sqr - 9;
   if(sq[tsq]) {
     if(PSIDE(sq[tsq]) != wtm) add_move(sqr, tsq, list, 1, tdata);
   } else add_move(sqr, tsq, list, 0, tdata);
  }

  // genrating castling moves ...
  if(!check) {
    if(wtm) {
      if (castle&1) add_move(sqr, 6, list, 2, tdata);
      if (castle&2) add_move(sqr, 2, list, 2, tdata);
    } else {
      if (castle&4) add_move(sqr, 62, list, 2, tdata);
      if (castle&8) add_move(sqr, 58, list, 2, tdata);
    }
  }
}

/*------------------------ Pawn Moves ------------------------------*/
void position::pawn_moves(move_list *list, int sqr, ts_thread_data *tdata)
{

  if(wtm) {                        // if it is white's pawn
    if(!PTYPE(sq[sqr+8])) {
     add_move(sqr, sqr+8, list, PAWN_PUSH, tdata);
     if(RANK(sqr) == 1) {
       if(!PTYPE(sq[sqr+16])) { add_move(sqr, sqr+16, list, PAWN_PUSH, tdata); }
      }
   }
   if(FILE(sqr)) {
     if (PTYPE(sq[sqr+7]) && PSIDE(sq[sqr+7]) != wtm)
      { add_move(sqr, sqr+7, list, (CAPTURE|PAWN_PUSH), tdata); }
    else if((sqr+7) == ep && ep)
      { add_move(sqr, sqr+7, list, (EP|CAPTURE|PAWN_PUSH), tdata); }
   }
   if(FILE(sqr) < 7) {
     if (PTYPE(sq[sqr+9]) && PSIDE(sq[sqr+9]) != wtm)
      { add_move(sqr, sqr+9, list, (CAPTURE|PAWN_PUSH), tdata); }
    else if((sqr+9) == ep && ep)
      { add_move(sqr, sqr+9, list, (EP|CAPTURE|PAWN_PUSH), tdata); }
   }
  } else {                           // or if it is black's pawn
    if(!PTYPE(sq[sqr-8])) {
     add_move(sqr, sqr-8, list, PAWN_PUSH, tdata);
     if(RANK(sqr) == 6) {
       if(!PTYPE(sq[sqr-16])) { add_move(sqr, sqr-16, list, PAWN_PUSH, tdata); }
      }
   }
   if(FILE(sqr)) {
     if (PTYPE(sq[sqr-9]) && PSIDE(sq[sqr-9]) != wtm)
      { add_move(sqr, sqr-9, list, (CAPTURE|PAWN_PUSH), tdata); }
    else if((sqr-9) == ep && ep)
      { add_move(sqr, sqr-9, list, (EP|CAPTURE|PAWN_PUSH), tdata); }
   }
   if(FILE(sqr) < 7) {
     if (PTYPE(sq[sqr-7]) && PSIDE(sq[sqr-7]) != wtm)
      { add_move(sqr, sqr-7, list, (CAPTURE|PAWN_PUSH), tdata); }
    else if((sqr-7) == ep && ep)
      { add_move(sqr, sqr-7, list, (EP|CAPTURE|PAWN_PUSH), tdata); }
   }
  }
}
























