/*-------------------------- Sorting Routines ------------------------*/  
// simple inline sorts are used in the main search, but here we
// have some quicksort wrappers for the opening book functions.
//


#include "define.h"
#include "chess.h"
#include "const.h"
#include "funct.h"
#include <stdlib.h>

// For opening book records
int comp_book_rec(const void *AA, const void *BB) {
  book_rec *A = (book_rec *) AA;
  book_rec *B = (book_rec *) BB;
  if(A->pos_code > B->pos_code) return 1;
  if(A->pos_code < B->pos_code) return -1;
  return 0;
}

void QuickSortBook(book_rec *Lb, book_rec *Ub) {

  qsort(Lb, (Ub-Lb), sizeof(book_rec), &comp_book_rec);

}


// For move lists
int comp_move_score(const void *AA, const void *BB) {
  move_rec *A = (move_rec *) AA;
  move_rec *B = (move_rec *) BB;
  if(A->score > B->score) return -1;
  if(A->score < B->score) return 1;
  return 0;
}

void QuickSortMove(move_rec *Lb, move_rec *Ub) {

  qsort(Lb, (Ub-Lb), sizeof(move_rec), &comp_move_score);

}






