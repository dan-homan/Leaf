// EXchess source code, (c) Daniel C. Homan  1997-2013
// Released under the GNU public license, see file license.txt

/* Functions to swap off attacks/material on a given square */

#include "define.h"
#include "chess.h"
#include "const.h"

struct swap_rec {
  short v;               // piece value  
  short s;               // piece square   
};

// functions for finding the attacks
int knt_attcks(int sq, position *p, int attcks[2], swap_rec attackers[2][17]);
int dia_attcks(int sq, position *p, int attcks[2], swap_rec attackers[2][17]);
int hor_attcks(int sq, position *p, int attcks[2], swap_rec attackers[2][17]);
int dia_attcks_directed(int sq, int lsq, position *p, int attcks[2], swap_rec attackers[2][17]);
int hor_attcks_directed(int sq, int lsq, position *p, int attcks[2], swap_rec attackers[2][17]);

// Sort the lowest value attacking piece to the
// top of the list
inline void SSort(swap_rec *Lb, swap_rec *Ub)
{
  swap_rec V, *I, *J;

   V = *Lb; J = Lb;
   for(I = Lb+1; I <= Ub; I++) {
       if (I->v < J->v) { J = I; }
   }
   *Lb = *J;
   *J = V;
}

int swap(int sq, position p, int side, int from)
{
  int val = value[PTYPE(p.sq[sq])], lsq;
  int attcks[2];  // counts of attacks for each side
  swap_rec attackers[2][17];  

  // initalize attackers arrays
  attackers[0][0].v = 0; attcks[0] = 0;
  attackers[1][0].v = 0; attcks[1] = 0;

  // put in the value of the possible capture      
  // of the initial attacker (only happens if the
  // other side has an attack on this square) 
  attackers[side][0].v = value[PTYPE(p.sq[from])];
  attcks[side]++; 

  // When from == sq, this routine just gives
  // the swap value of staying put on that square
  if(from == sq) {
    val = 0;
  } else { 
    p.sq[from] = EMPTY;
  }
 
  // find all the attcks on sq
  dia_attcks(sq, &p, attcks, attackers);
  knt_attcks(sq, &p, attcks, attackers);
  hor_attcks(sq, &p, attcks, attackers);
  
  attackers[0][attcks[0]].v = 0;
  attackers[1][attcks[1]].v = 0;

  // swap off the attacks, starting with other side
  // and with the least value
  int swapside = side^1; 
  int count[2] = { 0, 0 }, best_side = -1000000, best_otherside = val;

  count[side]++; // for the first capture

  while(count[swapside] < attcks[swapside]) {
    SSort(&attackers[swapside][count[swapside]], &attackers[swapside][attcks[swapside]-1]);      
    if(swapside == side) {
     val += attackers[swapside^1][count[swapside^1]-1].v;
     if(val < best_otherside) best_otherside = MAX(val,best_side);
     if(val < best_side) return(best_side);
    } else {
     val -= attackers[swapside^1][count[swapside^1]-1].v;
     if(val > best_side) best_side = MIN(val,best_otherside);
     if(val > best_otherside) return(best_otherside);
    }
    // square from which the capture came
    lsq = attackers[swapside][count[swapside]].s;
    // add in any revealed attacks due to this capture
    if((FILE(lsq) == FILE(sq) || RANK(lsq) == RANK(sq)) && lsq)
      hor_attcks_directed(sq, lsq, &p, attcks, attackers);
    else if(PTYPE(p.sq[lsq]) != KNIGHT && lsq) dia_attcks_directed(sq, lsq, &p, attcks, attackers);
    count[swapside]++;
    swapside ^= 1;
  }      

  if(swapside == side) { // other side went last and could have passed
	if(val < best_otherside) best_otherside = MAX(val,best_side);
	return(best_otherside);
  } else {               // our side went last and could have passed 
	if(val > best_side) best_side = MIN(val,best_otherside);
	return(best_side);
  }

}

/*----------------- Calculate diagonal attcks ------------------*/

int dia_attcks(int sq, position *p, int attcks[2], swap_rec attackers[2][17])
{
  int mm = FILE(sq), nn = RANK(sq);
  int fsq, fside;
    
  int ii = 1;
  while (mm + ii <= 7 && nn + ii <= 7)
  {
   fsq = SQR((mm+ii),(nn+ii));
   fside = PSIDE(p->sq[fsq]);
   if (p->sq[fsq])
   {
    if (ii == 1 && ID(p->sq[fsq]) == BPAWN)
    { attackers[fside][attcks[fside]].v = value[PAWN]; 
      attackers[fside][attcks[fside]].s = fsq; attcks[fside]++; }
    else if (ii == 1 && PTYPE(p->sq[fsq]) == KING)
    { attackers[fside][attcks[fside]].v = value[KING]; 
      attackers[fside][attcks[fside]].s = fsq; attcks[fside]++; }
    else if (PTYPE(p->sq[fsq]) == QUEEN || PTYPE(p->sq[fsq]) == BISHOP)
    { attackers[fside][attcks[fside]].v = value[PTYPE(p->sq[fsq])]; 
      attackers[fside][attcks[fside]].s = fsq; attcks[fside]++; }
    break;
   }
   ii++;
  }

  ii = 1;
  while ((mm - ii) >= 0 && nn + ii <= 7)
  {
   fsq = SQR((mm-ii),(nn+ii));
   fside = PSIDE(p->sq[fsq]);
   if (p->sq[fsq])
   {
    if (ii == 1 && ID(p->sq[fsq]) == BPAWN)
    { attackers[fside][attcks[fside]].v = value[PAWN]; 
      attackers[fside][attcks[fside]].s = fsq; attcks[fside]++; }
    else if (ii == 1 && PTYPE(p->sq[fsq]) == KING)
    { attackers[fside][attcks[fside]].v = value[KING]; 
      attackers[fside][attcks[fside]].s = fsq; attcks[fside]++; }
    else if (PTYPE(p->sq[fsq]) == QUEEN || PTYPE(p->sq[fsq]) == BISHOP)
    { attackers[fside][attcks[fside]].v = value[PTYPE(p->sq[fsq])]; 
      attackers[fside][attcks[fside]].s = fsq; attcks[fside]++; }
    break;
   }
   ii++;
  }

  ii = 1;
  while ((mm - ii) >= 0 && (nn - ii) >= 0)
  {
   fsq = SQR((mm-ii),(nn-ii));
   fside = PSIDE(p->sq[fsq]);
   if (p->sq[fsq])
   {
    if (ii == 1 && ID(p->sq[fsq]) == WPAWN)
    { attackers[fside][attcks[fside]].v = value[PAWN]; 
      attackers[fside][attcks[fside]].s = fsq; attcks[fside]++; }
    else if (ii == 1 && PTYPE(p->sq[fsq]) == KING)
    { attackers[fside][attcks[fside]].v = value[KING]; 
      attackers[fside][attcks[fside]].s = fsq; attcks[fside]++; }
    else if (PTYPE(p->sq[fsq]) == QUEEN || PTYPE(p->sq[fsq]) == BISHOP)
    { attackers[fside][attcks[fside]].v = value[PTYPE(p->sq[fsq])]; 
      attackers[fside][attcks[fside]].s = fsq; attcks[fside]++; }
    break;
   }
   ii++;
  }

  ii = 1;
  while (mm + ii <= 7 && nn - ii >= 0)
  {
   fsq = SQR((mm+ii),(nn-ii));
   fside = PSIDE(p->sq[fsq]);
   if (p->sq[fsq])
   {
    if (ii == 1 && ID(p->sq[fsq]) == WPAWN)
    { attackers[fside][attcks[fside]].v = value[PAWN]; 
      attackers[fside][attcks[fside]].s = fsq; attcks[fside]++; }
    else if (ii == 1 && PTYPE(p->sq[fsq]) == KING)
    { attackers[fside][attcks[fside]].v = value[KING]; 
      attackers[fside][attcks[fside]].s = fsq; attcks[fside]++; }
    else if (PTYPE(p->sq[fsq]) == QUEEN || PTYPE(p->sq[fsq]) == BISHOP)
    { attackers[fside][attcks[fside]].v = value[PTYPE(p->sq[fsq])]; 
      attackers[fside][attcks[fside]].s = fsq; attcks[fside]++; }
    break;
   }
   ii++;
  }


  return 0;
}

// directed version of the above routine to only add attacks
//  from a particular direction beyond the last square 
int dia_attcks_directed(int sq, int lsq, position *p, int attcks[2], swap_rec attackers[2][17])
{
  int mm = FILE(sq), nn = RANK(sq), lmm = FILE(lsq), lnn = RANK(lsq);
  int fsq, fside, dir;

  if(lmm - mm > 0 && lnn - nn > 0) dir = 1;
  else if(lmm - mm < 0 && lnn - nn > 0) dir = 2;   
  else if(lmm - mm < 0 && lnn - nn < 0) dir = 3;
  else dir = 4;
  // treat attacks as coming to the last square.  This 
  //   is a trick to speed things up and is OK because we
  //   are looking for an attacker behind that square
  mm = lmm; nn = lnn;
    
  int ii = 1;  
  while (dir == 1 && mm + ii <= 7 && nn + ii <= 7)
  {
   fsq = SQR((mm+ii),(nn+ii));
   fside = PSIDE(p->sq[fsq]);
   if (p->sq[fsq])
   {
    if (PTYPE(p->sq[fsq]) == QUEEN || PTYPE(p->sq[fsq]) == BISHOP)
    { attackers[fside][attcks[fside]].v = value[PTYPE(p->sq[fsq])]; 
      attackers[fside][attcks[fside]].s = fsq; attcks[fside]++; }
    break;
   }
   ii++;
  }

  ii = 1;
  while (dir == 2 && (mm - ii) >= 0 && nn + ii <= 7)
  {
   fsq = SQR((mm-ii),(nn+ii));
   fside = PSIDE(p->sq[fsq]);
   if (p->sq[fsq])
   {
    if (PTYPE(p->sq[fsq]) == QUEEN || PTYPE(p->sq[fsq]) == BISHOP)
    { attackers[fside][attcks[fside]].v = value[PTYPE(p->sq[fsq])]; 
      attackers[fside][attcks[fside]].s = fsq; attcks[fside]++; }
    break;
   }
   ii++;
  }

  ii = 1;
  while (dir == 3 && (mm - ii) >= 0 && (nn - ii) >= 0)
  {
   fsq = SQR((mm-ii),(nn-ii));
   fside = PSIDE(p->sq[fsq]);
   if (p->sq[fsq])
   {
    if (PTYPE(p->sq[fsq]) == QUEEN || PTYPE(p->sq[fsq]) == BISHOP)
    { attackers[fside][attcks[fside]].v = value[PTYPE(p->sq[fsq])]; 
      attackers[fside][attcks[fside]].s = fsq; attcks[fside]++; }
    break;
   }
   ii++;
  }

  ii = 1;
  while (dir == 4 && mm + ii <= 7 && nn - ii >= 0)
  {
   fsq = SQR((mm+ii),(nn-ii));
   fside = PSIDE(p->sq[fsq]);
   if (p->sq[fsq])
   {
    if (PTYPE(p->sq[fsq]) == QUEEN || PTYPE(p->sq[fsq]) == BISHOP)
    { attackers[fside][attcks[fside]].v = value[PTYPE(p->sq[fsq])]; 
      attackers[fside][attcks[fside]].s = fsq; attcks[fside]++; }
    break;
   }
   ii++;
  }


  return 0;
}

/*----------------- Calculate Horizontal attcks ------------------*/

int hor_attcks(int sq, position *p, int attcks[2], swap_rec attackers[2][17])
{
  int mm = FILE(sq), nn = RANK(sq);
  int fsq, fside;
 
  int ii = 1;
  while (mm + ii <= 7)
  {
   fsq = SQR((mm+ii), nn);
   fside = PSIDE(p->sq[fsq]);
   if (p->sq[fsq])
   {
    if (ii == 1 && PTYPE(p->sq[fsq]) == KING)
    { attackers[fside][attcks[fside]].v = value[KING]; 
      attackers[fside][attcks[fside]].s = fsq; attcks[fside]++; }
    else if (PTYPE(p->sq[fsq]) == QUEEN || PTYPE(p->sq[fsq]) == ROOK)
    { attackers[fside][attcks[fside]].v = value[PTYPE(p->sq[fsq])]; 
      attackers[fside][attcks[fside]].s = fsq; attcks[fside]++; }
    break;
   }
   ii++;
  }

  ii = 1;
  while ((mm - ii) >= 0)
  {
   fsq = SQR((mm-ii), nn);
   fside = PSIDE(p->sq[fsq]);
   if (p->sq[fsq])
   {
    if (ii == 1 && PTYPE(p->sq[fsq]) == KING)
    { attackers[fside][attcks[fside]].v = value[KING]; 
      attackers[fside][attcks[fside]].s = fsq; attcks[fside]++; }
    else if (PTYPE(p->sq[fsq]) == QUEEN || PTYPE(p->sq[fsq]) == ROOK)
    { attackers[fside][attcks[fside]].v = value[PTYPE(p->sq[fsq])]; 
      attackers[fside][attcks[fside]].s = fsq; attcks[fside]++; }
    break;
   }
   ii++;
  }

  ii = 1;
  while (nn - ii >= 0)
  {
   fsq = SQR((mm),(nn-ii));
   fside = PSIDE(p->sq[fsq]);
   if (p->sq[fsq])
   {
    if (ii == 1 && PTYPE(p->sq[fsq]) == KING)
    { attackers[fside][attcks[fside]].v = value[KING]; 
      attackers[fside][attcks[fside]].s = fsq; attcks[fside]++; }
    else if (PTYPE(p->sq[fsq]) == QUEEN || PTYPE(p->sq[fsq]) == ROOK)
    { attackers[fside][attcks[fside]].v = value[PTYPE(p->sq[fsq])]; 
      attackers[fside][attcks[fside]].s = fsq; attcks[fside]++; }
    break;
   }
   ii++;
  }

  ii = 1;
  while (nn + ii <= 7)
  {
   fsq = SQR((mm),(nn+ii));
   fside = PSIDE(p->sq[fsq]);
   if (p->sq[fsq])
   {
    if (ii == 1 && PTYPE(p->sq[fsq]) == KING)
    { attackers[fside][attcks[fside]].v = value[KING]; 
      attackers[fside][attcks[fside]].s = fsq; attcks[fside]++; }
    else if (PTYPE(p->sq[fsq]) == QUEEN || PTYPE(p->sq[fsq]) == ROOK)
    { attackers[fside][attcks[fside]].v = value[PTYPE(p->sq[fsq])]; 
      attackers[fside][attcks[fside]].s = fsq; attcks[fside]++; }
    break;
   }
   ii++;
  }


  return 0;
}

// directed version of the above routine to only add attacks
//  from a particular direction beyond the last square 
int hor_attcks_directed(int sq, int lsq, position *p, int attcks[2], swap_rec attackers[2][17])
{
  int mm = FILE(sq), nn = RANK(sq), lmm = FILE(lsq), lnn = RANK(lsq);
  int fsq, fside, dir;
 
  if(lmm - mm > 0) dir = 1;
  else if(lmm - mm < 0) dir = 2;   
  else if(lnn - nn < 0) dir = 3;
  else dir = 4;
  // treat attacks as coming to the last square.  This 
  //   is a trick to speed things up and is OK because we
  //   are looking for an attacker behind that square
  mm = lmm; nn = lnn;

  int ii = 1;
  while (dir == 1 && mm + ii <= 7)
  {
   fsq = SQR((mm+ii), nn);
   fside = PSIDE(p->sq[fsq]);
   if (p->sq[fsq])
   {
    if (PTYPE(p->sq[fsq]) == QUEEN || PTYPE(p->sq[fsq]) == ROOK)
    { attackers[fside][attcks[fside]].v = value[PTYPE(p->sq[fsq])]; 
      attackers[fside][attcks[fside]].s = fsq; attcks[fside]++; }
    break;
   }
   ii++;
  }

  ii = 1;
  while (dir == 2 && (mm - ii) >= 0)
  {
   fsq = SQR((mm-ii), nn);
   fside = PSIDE(p->sq[fsq]);
   if (p->sq[fsq])
   {
    if (PTYPE(p->sq[fsq]) == QUEEN || PTYPE(p->sq[fsq]) == ROOK)
    { attackers[fside][attcks[fside]].v = value[PTYPE(p->sq[fsq])]; 
      attackers[fside][attcks[fside]].s = fsq; attcks[fside]++; }
    break;
   }
   ii++;
  }

  ii = 1;
  while (dir == 3 && nn - ii >= 0)
  {
   fsq = SQR((mm),(nn-ii));
   fside = PSIDE(p->sq[fsq]);
   if (p->sq[fsq])
   {
    if (PTYPE(p->sq[fsq]) == QUEEN || PTYPE(p->sq[fsq]) == ROOK)
    { attackers[fside][attcks[fside]].v = value[PTYPE(p->sq[fsq])]; 
      attackers[fside][attcks[fside]].s = fsq; attcks[fside]++; }
    break;
   }
   ii++;
  }

  ii = 1;
  while (dir == 4 && nn + ii <= 7)
  {
   fsq = SQR((mm),(nn+ii));
   fside = PSIDE(p->sq[fsq]);
   if (p->sq[fsq])
   {
    if (PTYPE(p->sq[fsq]) == QUEEN || PTYPE(p->sq[fsq]) == ROOK)
    { attackers[fside][attcks[fside]].v = value[PTYPE(p->sq[fsq])]; 
      attackers[fside][attcks[fside]].s = fsq; attcks[fside]++; }
    break;
   }
   ii++;
  }


  return 0;
}

/*------------------------ Knight Attcks Counted -------------------*/
int knt_attcks(int sq, position *p, int attcks[2], swap_rec attackers[2][17])
{
  int fsq, fside;

  if(FILE(sq) < 6 && RANK(sq) < 7) {
   fsq = sq + 10;
   fside = PSIDE(p->sq[fsq]);
   if(PTYPE(p->sq[fsq]) == KNIGHT) 
    { attackers[fside][attcks[fside]].v = value[KNIGHT]; 
      attackers[fside][attcks[fside]].s = fsq; attcks[fside]++; }    
  }
  if(FILE(sq) < 6 && RANK(sq)) {
   fsq = sq - 6;
   fside = PSIDE(p->sq[fsq]);
   if(PTYPE(p->sq[fsq]) == KNIGHT) 
    { attackers[fside][attcks[fside]].v = value[KNIGHT]; 
      attackers[fside][attcks[fside]].s = fsq; attcks[fside]++; }    
  }
  if(FILE(sq) > 1 && RANK(sq) < 7) {
   fsq = sq + 6;
   fside = PSIDE(p->sq[fsq]);
   if(PTYPE(p->sq[fsq]) == KNIGHT) 
    { attackers[fside][attcks[fside]].v = value[KNIGHT]; 
      attackers[fside][attcks[fside]].s = fsq; attcks[fside]++; }    
  }
  if(FILE(sq) > 1 && RANK(sq)) {
   fsq = sq - 10;
   fside = PSIDE(p->sq[fsq]);
   if(PTYPE(p->sq[fsq]) == KNIGHT) 
    { attackers[fside][attcks[fside]].v = value[KNIGHT]; 
      attackers[fside][attcks[fside]].s = fsq; attcks[fside]++; }    
  }
  if(FILE(sq) < 7 && RANK(sq) < 6) {
   fsq = sq + 17;
   fside = PSIDE(p->sq[fsq]);
   if(PTYPE(p->sq[fsq]) == KNIGHT) 
    { attackers[fside][attcks[fside]].v = value[KNIGHT]; 
      attackers[fside][attcks[fside]].s = fsq; attcks[fside]++; }    
  }
  if(FILE(sq) && RANK(sq) < 6) {
   fsq = sq + 15;
   fside = PSIDE(p->sq[fsq]);
   if(PTYPE(p->sq[fsq]) == KNIGHT) 
    { attackers[fside][attcks[fside]].v = value[KNIGHT]; 
      attackers[fside][attcks[fside]].s = fsq; attcks[fside]++; }    
  }
  if(FILE(sq) < 7 && RANK(sq) > 1) {
   fsq = sq - 15;
   fside = PSIDE(p->sq[fsq]);
   if(PTYPE(p->sq[fsq]) == KNIGHT) 
    { attackers[fside][attcks[fside]].v = value[KNIGHT]; 
      attackers[fside][attcks[fside]].s = fsq; attcks[fside]++; }    
  }
  if(FILE(sq) && RANK(sq) > 1) {
   fsq = sq - 17;
   fside = PSIDE(p->sq[fsq]);
   if(PTYPE(p->sq[fsq]) == KNIGHT) 
    { attackers[fside][attcks[fside]].v = value[KNIGHT]; 
      attackers[fside][attcks[fside]].s = fsq; attcks[fside]++; }    
  }
 
 return 0;
}






