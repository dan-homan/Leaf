// EXchess source code, (c) Daniel C. Homan  1997-2013
// Released under the GNU public license, see file license.txt

/* Attack functions */


#include "define.h"
#include "chess.h"
#include "funct.h"
#include "const.h"

/* Function to do a simple look to see if we are in check */
// Works by checking for a revealed attack from beyond
// square "move_sq" which was vacated on the previous move
int position::simple_check(int move_sq)
{
  int ksq = plist[wtm][KING][1];  // king-square

  if(rook_check_table[move_sq]&(1ULL<<ksq)) return hor_slide_attack_xray(ksq, move_sq);
  else if(bishop_check_table[move_sq]&(1ULL<<ksq)) return dia_slide_attack_xray(ksq, move_sq);
  else return 0;

}

// function to determine if squares A and B can
// have a sliding attack between them
int position::dia_slide_attack(int A, int B) {
  int Ax = FILE(A), Ay = RANK(A);
  int Bx = FILE(B), By = RANK(B);
  int test_sq;

  // assume basic connection and legality of diagonal moves
  // between A and B... if this is wrong, use line below
  //if(A == B || (ABS(Ax-Bx) != ABS(Ay-By))) return 0;

  if(Ax > Bx)
    if(Ay > By && A > B+9) 
      for(test_sq = A-9; test_sq > B; test_sq -= 9) 
	if(sq[test_sq]) return 0;
  
  if(Ax > Bx)
    if(Ay < By && A < B-7) 
      for(test_sq = A+7; test_sq < B; test_sq += 7) 
	if(sq[test_sq]) return 0;

  if(Ax < Bx)
    if(Ay > By && A > B+7) 
      for(test_sq = A-7; test_sq > B; test_sq -= 7) 
	if(sq[test_sq]) return 0;
  
  if(Ax < Bx)
    if(Ay < By && A < B-9) 
      for(test_sq = A+9; test_sq < B; test_sq += 9) 
	if(sq[test_sq]) return 0;
 
  return 1;   // if there are no blockers the two squares
              // can be connected by a sliding attack
  
}

// function to determine if squares A and B can
// have a sliding attack between them with the 
// attacker (attacking square A) lies 
// somewhere beyond square B
int position::dia_slide_attack_xray(int A, int B) {
  int Ax = FILE(A), Ay = RANK(A);
  int Bx = FILE(B), By = RANK(B);
  int test_sq;

  int Aside = -1;
  if(PTYPE(sq[A])) Aside = PSIDE(sq[A]); 

  if(Ax > Bx)
    if(Ay > By) {
      for(test_sq = A-9; test_sq > B; test_sq -= 9) 
	if(sq[test_sq]) return 0;
      for(test_sq = B-9; test_sq >= 0 && FILE(test_sq) < 7; test_sq -= 9) {
	if(!sq[test_sq]) continue; 
	if(PSIDE(sq[test_sq]) != Aside 
	   && (PTYPE(sq[test_sq]) == QUEEN || PTYPE(sq[test_sq]) == BISHOP)) return 1;
	else return 0;
      }
      return 0;
    }
  
  if(Ax > Bx)
    if(Ay < By) {
      for(test_sq = A+7; test_sq < B; test_sq += 7) 
	if(sq[test_sq]) return 0;
      for(test_sq = B+7; test_sq <= 63 && FILE(test_sq) < 7; test_sq += 7) {
	if(!sq[test_sq]) continue; 
	if(PSIDE(sq[test_sq]) != Aside 
	   && (PTYPE(sq[test_sq]) == QUEEN || PTYPE(sq[test_sq]) == BISHOP)) return 1;
	else return 0;
      }
      return 0;
    }

  if(Ax < Bx)
    if(Ay > By) {
      for(test_sq = A-7; test_sq > B; test_sq -= 7) 
	if(sq[test_sq]) return 0;
      for(test_sq = B-7; test_sq >= 0 && FILE(test_sq) > 0; test_sq -= 7) {
	if(!sq[test_sq]) continue; 
	if(PSIDE(sq[test_sq]) != Aside 
	   && (PTYPE(sq[test_sq]) == QUEEN || PTYPE(sq[test_sq]) == BISHOP)) return 1;
	else return 0;
      }
      return 0;
    }
  
  if(Ax < Bx)
    if(Ay < By) {
      for(test_sq = A+9; test_sq < B; test_sq += 9) 
	if(sq[test_sq]) return 0;
      for(test_sq = B+9; test_sq <= 63 && FILE(test_sq) > 0; test_sq += 9) {
	if(!sq[test_sq]) continue; 
	if(PSIDE(sq[test_sq]) != Aside 
	   && (PTYPE(sq[test_sq]) == QUEEN || PTYPE(sq[test_sq]) == BISHOP)) return 1;
	else return 0;
      }
      return 0;
    }
 
  return 0;  
  
}

// function to determine if squares A and B can
// have a sliding attack between them
int position::hor_slide_attack(int A, int B) {
  int Ax = FILE(A), Ay = RANK(A);
  int Bx = FILE(B), By = RANK(B);
  int test_sq;

  // assume basic connection and legality of horizontal moves
  // between A and B... if this is wrong, use line below
  //if(A == B || (Ax != Bx && Ay != By)) return 0;

  if(Ax > Bx+1)
      for(test_sq = A-1; test_sq > B; test_sq -= 1) 
	if(sq[test_sq]) return 0;

  if(Ax < Bx-1)
      for(test_sq = A+1; test_sq < B; test_sq += 1) 
	if(sq[test_sq]) return 0;
 
  if(Ay > By+1)
      for(test_sq = A-8; test_sq > B; test_sq -= 8) 
	if(sq[test_sq]) return 0;

  if(Ay < By-1)
      for(test_sq = A+8; test_sq < B; test_sq += 8) 
	if(sq[test_sq]) return 0;
 
  return 1;   // if there are no blockers the two squares
              // can be connected by a sliding attack
  
}


// function to determine if squares A and B can
// have a sliding attack between them with the 
// attacker (attacking square A) lies 
// somewhere beyond square B
int position::hor_slide_attack_xray(int A, int B) {
  int Ax = FILE(A), Ay = RANK(A);
  int Bx = FILE(B), By = RANK(B);
  int test_sq;

  int Aside = -1;
  if(PTYPE(sq[A])) Aside = PSIDE(sq[A]); 

  if(Ax > Bx) {
      for(test_sq = A-1; test_sq > B; test_sq -= 1) 
	if(sq[test_sq]) return 0;
      for(test_sq = B-1; test_sq >= 0 && FILE(test_sq) < 7; test_sq -=1) {
	if(!sq[test_sq]) continue; 
	if(PSIDE(sq[test_sq]) != Aside 
	   && (PTYPE(sq[test_sq]) == QUEEN || PTYPE(sq[test_sq]) == ROOK)) return 1;
	else return 0;
      }
      return 0;
    }

  if(Ax < Bx) {
      for(test_sq = A+1; test_sq < B; test_sq += 1) 
	if(sq[test_sq]) return 0;
      for(test_sq = B+1; test_sq <= 63 && FILE(test_sq) > 0; test_sq +=1) {
	if(!sq[test_sq]) continue; 
	if(PSIDE(sq[test_sq]) != Aside 
	   && (PTYPE(sq[test_sq]) == QUEEN || PTYPE(sq[test_sq]) == ROOK)) return 1;
	else return 0;
      }
      return 0;
    }
 
  if(Ay > By) {
      for(test_sq = A-8; test_sq > B; test_sq -= 8) 
	if(sq[test_sq]) return 0;
      for(test_sq = B-8; test_sq >=0; test_sq -= 8) {
	if(!sq[test_sq]) continue; 
	if(PSIDE(sq[test_sq]) != Aside 
	   && (PTYPE(sq[test_sq]) == QUEEN || PTYPE(sq[test_sq]) == ROOK)) return 1;
	else return 0;
      }
      return 0;
    }
 
  if(Ay < By) {
      for(test_sq = A+8; test_sq < B; test_sq += 8) 
	if(sq[test_sq]) return 0;
      for(test_sq = B+8; test_sq <=63; test_sq += 8) {
	if(!sq[test_sq]) continue; 
	if(PSIDE(sq[test_sq]) != Aside 
	   && (PTYPE(sq[test_sq]) == QUEEN || PTYPE(sq[test_sq]) == ROOK)) return 1;
	else return 0;
      }
      return 0;
    }

  return 0;
  
}

//-------------------------------------------------------------
// Function to determine if a square is attacked at least once
//   by the given side
//-------------------------------------------------------------
int position::attacked(int sqr, int side)
{
  int i;
  uint64_t sqr_bit = (1ULL<<sqr);

  // look for pawn attack first
  if(side == BLACK && sqr < 48) { 
    if(FILE(sqr) && ID(sq[sqr+7]) == BPAWN) return 1;
    if(FILE(sqr) < 7 && ID(sq[sqr+9]) == BPAWN) return 1;
  }
  if(side == WHITE && sqr > 15) { 
    if(FILE(sqr) && ID(sq[sqr-9]) == WPAWN) return 1;
    if(FILE(sqr) < 7 && ID(sq[sqr-7]) == WPAWN) return 1;
  }

  // loop over pieces and see if they attack this square
  for(i = 1; i <= plist[side][KNIGHT][0]; i++) {
    if(knight_check_table[plist[side][KNIGHT][i]]&sqr_bit) return 1;
  }
  for(i = 1; i <= plist[side][BISHOP][0]; i++) {
    if(bishop_check_table[plist[side][BISHOP][i]]&sqr_bit)
      if(dia_slide_attack(plist[side][BISHOP][i], sqr)) return 1;
  }
  for(i = 1; i <= plist[side][ROOK][0]; i++) {
    if(rook_check_table[plist[side][ROOK][i]]&sqr_bit)
      if(hor_slide_attack(plist[side][ROOK][i], sqr)) return 1;
  }
  for(i = 1; i <= plist[side][QUEEN][0]; i++) {
    if(bishop_check_table[plist[side][QUEEN][i]]&sqr_bit)
      if(dia_slide_attack(plist[side][QUEEN][i], sqr)) return 1;
    if(rook_check_table[plist[side][QUEEN][i]]&sqr_bit)
      if(hor_slide_attack(plist[side][QUEEN][i], sqr)) return 1;
  }

  // see if the king attacks that square
  if(taxi_cab[plist[side][KING][1]][sqr] == 1) return 1;

  return 0;
}


