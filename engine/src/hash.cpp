// EXchess source code, (c) Daniel C. Homan  1997-2017
// Released under the GNU public license, see file license.txt

//--------------------------------------------------------------
// void set_hash_size(unsigned int Mbytes)
// void open_hash()
// void close_hash()
// void put_hash(h_code (*h_key), int score, int alpha, int beta,
// 	      int depth, int hmove_t, int mate_ext, int h_id)
// int get_hash(h_code (*h_key), int *hflag, int *hdepth, int *mate_ext, move *gmove)
// int get_move(h_code (*h_key))
// int put_move(h_code (h_key), int putmove, int h_id)
// void position::gen_code()
// void start_code()
//--------------------------------------------------------------
/*
  Functions dealing with the standard hash table and the pawn hash tables.
  The standard hash table is divided into "buckets" with four entries each. 
  This allows a "smart" depth priority replacement strategy.
  The pawn hash is very much like the standard hash - but simpler.  Only
  the hash key and score is stored.  The pawn table is always replace.
*/


#include "define.h"
#include "chess.h"
#include "funct.h"
#include "const.h"
#include "hash.h"
#include "extern.h"

#include <assert.h>
#include <cstdlib>

// Round n up to the next multiple of 64 for aligned_alloc()
static inline size_t align64(size_t n) { return (n + 63ULL) & ~63ULL; }

//--------------------------------------------
// Function to set the hash table size 
//--------------------------------------------
void set_hash_size(unsigned int Mbytes)
{
  unsigned int Tab_entry_size = sizeof(hash_bucket);
  unsigned int Pawn_entry_size = sizeof(pawn_rec);
  unsigned int Score_entry_size = sizeof(score_rec);
  unsigned int Cmove_entry_size = sizeof(cmove_rec);

  unsigned int Max_tab_size, Max_pawn_size, Max_score_size, Max_cmove_size;

  /*   -- subtracting off cache_size creates a difference in 
          node counts between TB and non-TB versions even in
          non-TB positions, so commenting out.  
#if TABLEBASES
  Mbytes -= int(CACHE_SIZE);  // subtract off EGTB CACHE 
#endif
  */
  if(Mbytes < 16) {
    cout << "Minimum total hash size is 16 MB, using 16 MB\n";
    cout.flush();
    Mbytes = 16;
  }

  uint64_t size_available = uint64_t(Mbytes)*1048576;

  TAB_SIZE = MIN(MAX_UINT, 0.75*size_available/(Tab_entry_size));

  size_available -= uint64_t(TAB_SIZE)*Tab_entry_size;

  CMOVE_SIZE = MIN(MAX_UINT, 0.25*size_available/(Cmove_entry_size));

  size_available -= uint64_t(CMOVE_SIZE)*Cmove_entry_size;

  SCORE_SIZE = MIN(MAX_UINT, 0.60*size_available/(Score_entry_size));

  size_available -= uint64_t(SCORE_SIZE)*Score_entry_size;

  PAWN_SIZE =  MIN(MAX_UINT, size_available/(Pawn_entry_size));

  close_hash();
  open_hash();
}


//--------------------------------------------
// Function with allocates space for the hash 
//   tables and generates the hash codes
//--------------------------------------------
void open_hash()
{
 hash_table = (hash_bucket*)aligned_alloc(64, align64(TAB_SIZE * sizeof(hash_bucket)));
 // initialize key values in table
 hash_bucket *h;
 for(uintptr_t i = 0; i < TAB_SIZE; i++) {
   h = hash_table + i;
   for(int j = 0; j < 4; j++) {
     h->rec[j].hr_key = 0ULL;
     h->rec[j].hr_hmove.t = 0;
     h->rec[j].hr_score = HASH_MISS;
     h->rec[j].hr_depth = -2;
     h->rec[j].hr_data = 0;
   }
 }
 pawn_table = (pawn_rec*)aligned_alloc(64, align64(PAWN_SIZE * sizeof(pawn_rec)));
 pawn_rec *p;
 for(uintptr_t i = 0; i < PAWN_SIZE; i++) {
   p = pawn_table+i;
   p->key = 0ULL;
   p->data.score = 0;
   p->data.pawn_attacks[BLACK] = 0ULL;
   p->data.pawn_attacks[WHITE] = 0ULL;
   p->data.open_files = 255;
   p->data.half_open_files_w = 255;
   p->data.half_open_files_b = 255;
   p->data.passed_w = 0;
   p->data.passed_b = 0;
   p->data.padding1 = 0;
 }
 score_table = (score_rec*)aligned_alloc(64, align64(SCORE_SIZE * sizeof(score_rec)));
 score_rec *s;
 for(uintptr_t i = 0; i < SCORE_SIZE; i++) {
   s = score_table+i;
   s->key = 0ULL;
   s->score = 0;
   s->qchecks[0] = 0;
   s->qchecks[1] = 0;
   s->padding1 = 0;
 }
 cmove_table = (cmove_rec*)aligned_alloc(64, align64(CMOVE_SIZE * sizeof(cmove_rec)));
 cmove_rec *c;
 for(uintptr_t i = 0; i < CMOVE_SIZE; i++) {
   c = cmove_table+i;
   c->key1 = 0ULL;
   c->move1 = 0;
   c->depth1 = -2;
   c->padding1 = 0;
   c->id = 0;
   c->key2 = 0ULL;
   c->move2 = 0;
   c->depth2 = -2;
   c->padding2 = 0;
   c->padding3 = 0;
 }
 lmr_init_tables();
}


//--------------------------------------------
// function to close the hash table 
//--------------------------------------------
void close_hash()
{
 free(hash_table);
 free(pawn_table);
 free(score_table);
 free(cmove_table);
}


//--------------------------------------------
// function to stuff a hash entry into 
//    the hash table 
//--------------------------------------------
void put_hash(h_code (*h_key), int score, int alpha, int beta,
	      int depth, int hmove_t, int h_id, int ply)
{
 hash_bucket *h; int flag, low_depth_rec = 0; 

 // is this an upper bound, lower bound, or exact score?
 if (score >= beta) 
   { flag = FLAG_B; if(score < 30000) score = beta;  }
 else if (score <= alpha) 
   { flag = FLAG_A; if(score > -30000) score = alpha;  }
 else { flag = FLAG_P; }

 // adjust any mate score to be in range of a short
 //  and make distance to MATE recoverable in get_hash
 if(score >= 30000) { 
   if(score >= MATE-1000) {
     score = 31000+(MATE-score)-ply;
     assert(score >= 31000);
   } else if(flag == FLAG_B || flag == FLAG_P) {
     score = 30000; flag = FLAG_B; 
   } else return;
 } else if(score <= -30000) { 
   if(score <= -MATE+1000) {
     score = -31000-(MATE+score)+ply;
     assert(score <= -31000);
   } else if(flag == FLAG_A || flag == FLAG_P) {
     score = -30000; flag = FLAG_A; 
   } else return;
 }

 // find location in table
 h = hash_table + (((TAB_SIZE-1)*((*h_key)&MAX_UINT))/MAX_UINT);
 assert(h < hash_table+TAB_SIZE && h >= hash_table);
 //h = hash_table + ((*h_key)&(TAB_SIZE-1));

 // find the lowest depth entry OR first entry that doesn't match current
 //  hash id for this search OR first entry that *is* an exact key match 
 //   -- note that this last condition makes sense because if an exact
 //      match with a good score had been found, it would have been
 //      used at the top of the search node when the hash table was probed
 for(int i=0; i < 4; i++) {
   if(GET_ID(h->rec[i].hr_data) != h_id || h->rec[i].get_key() == (*h_key)) {
     low_depth_rec = i; 
     break;
   }
   if(h->rec[i].hr_depth < h->rec[low_depth_rec].hr_depth) {
     low_depth_rec = i;
   }
 }

 // enter data into the lowest depth entry as determined above
 h->rec[low_depth_rec].set_key((*h_key),int16_t(score),SET_DATA(h_id,flag,0),depth,hmove_t);
 h->rec[low_depth_rec].hr_hmove.t = hmove_t;
 h->rec[low_depth_rec].hr_score = int16_t(score);
 h->rec[low_depth_rec].hr_depth = depth;
 h->rec[low_depth_rec].hr_data = SET_DATA(h_id,flag,0);

}


//--------------------------------------------
// function to find and return a hash entry
//--------------------------------------------
int get_hash(h_code (*h_key), int *hflag, int *hdepth, move *gmove, int ply, int *singular)
{
  hash_bucket *h; int best_depth_rec = -1; 

 // find the right location in the table
 h = hash_table + (((TAB_SIZE-1)*((*h_key)&MAX_UINT))/MAX_UINT);
 //h = hash_table + ((*h_key)&(TAB_SIZE-1));

 // find the highest depth entry that matches the hash key 
 for(int i=0; i < 4; i++) {
   if(h->rec[i].get_key() == (*h_key)) {
     if(best_depth_rec < 0) best_depth_rec = i;  
     else {
       if(h->rec[i].hr_depth > h->rec[best_depth_rec].hr_depth)
	 best_depth_rec = i;
     }
   }
 }

 // if this is a hash miss, return a HASH_MISS
 if(best_depth_rec < 0) { 
   gmove->t = NOMOVE;
   (*hflag) = 0;
   return HASH_MISS; 
 }

 // otherwise record values and return score
 (*gmove) = h->rec[best_depth_rec].hr_hmove;
 (*singular) = (gmove->b.type&SINGULAR);
 // unset singular status, if set 
 gmove->b.type &= NOT_SINGULAR;
 int score = h->rec[best_depth_rec].hr_score;
 (*hdepth) = h->rec[best_depth_rec].hr_depth;
 (*hflag) = GET_FLAG(h->rec[best_depth_rec].hr_data);

 // verify that data has not changed and hash move values
 //  do not exceed range...
 if(h->rec[best_depth_rec].get_key() != (*h_key)
    || gmove->b.from > 63 || gmove->b.to > 63) { 
   gmove->t = NOMOVE;
   (*hflag) = 0;
   (*singular) = 0;
   return HASH_MISS;
 }

 // resurrect a MATE limit
 if(score >= 31000) return (MATE+(31000-score)-ply);
 if(score <= -31000) return (-MATE-(31000+score)+ply);

 return score;

}


//-----------------------------------------------------
// function to retrieve the hash move from the table 
//-----------------------------------------------------
int get_move(h_code (*h_key))
{
  hash_bucket *h; int best_depth_rec = -1; 

 // find the right location in the table
 h = hash_table + (((TAB_SIZE-1)*((*h_key)&MAX_UINT))/MAX_UINT);
 //h = hash_table + ((*h_key)&(TAB_SIZE-1));

 // find the highest depth entry that matches the hash key 
 for(int i=0; i < 4; i++) {
   if(h->rec[i].get_key() == (*h_key)) {
     if(best_depth_rec < 0) best_depth_rec = i;
     else {
       if(h->rec[i].hr_depth > h->rec[best_depth_rec].hr_depth)
	 best_depth_rec = i;
     }
   }
 }

 // if this is a hash miss, return NOMOVE
 if (best_depth_rec < 0) return NOMOVE;  

 // copy relevant data
 move hmove;
 hmove.t = h->rec[best_depth_rec].hr_hmove.t;

 // unset any singular bit that might be set
 hmove.b.type &= NOT_SINGULAR;

 // verify that data has not changed and that hash move
 //  values are within acceptable range
 if(h->rec[best_depth_rec].get_key() != (*h_key)
    || hmove.b.from > 63 || hmove.b.to > 63) return NOMOVE; 

 return hmove.t;

}


//-----------------------------------------------
// function to stuff a hash move into the table 
//-----------------------------------------------
int put_move(h_code (h_key), int putmove, int h_id)
{
  hash_bucket *h; int low_depth_rec = 0;
  h = hash_table + (((TAB_SIZE-1)*((h_key)&MAX_UINT))/MAX_UINT);
  //h = hash_table + ((h_key)&(TAB_SIZE-1));

  // find the lowest depth entry OR first entry that doesn't match current
  //  hash id for this search
  for(int i=0; i < 4; i++) {
    if(GET_ID(h->rec[i].hr_data) != h_id) {
      low_depth_rec = i; 
      break;
    }
    if(h->rec[i].hr_depth < h->rec[low_depth_rec].hr_depth) {
      low_depth_rec = i;
    }
  }
  
  h->rec[low_depth_rec].set_key(h_key, HASH_MISS, 0, -2, putmove);
  h->rec[low_depth_rec].hr_hmove.t = putmove;
  h->rec[low_depth_rec].hr_score = HASH_MISS;      // will cause score to be ignored by get_hash()
  h->rec[low_depth_rec].hr_depth = -2;
  h->rec[low_depth_rec].hr_data = SET_DATA(h_id,0,0);
  
  return 0;
}


//------------------------------------------------------
// generate the hash code for a given position 
//  ALSO calculate material score and fill piecelists
//------------------------------------------------------
void position::gen_code()
{
  int i;
  hcode = ZERO;
  pcode = ZERO;
  material = 0;
  pieces[0] = 0; 
  pieces[1] = 0;

  for(i=0; i<=KING; i++) {
    plist[0][i][0] = 0;
    plist[1][i][0] = 0;
  }

  for(i = 0; i < 64; i++) {
   Or(hcode, hval[HASH_ID(sq[i])][i]);
   Or(pcode, hval[HASH_ID(sq[i])][i]);
   if(sq[i]) {   // if square is occupied 
     if(PSIDE(sq[i]) == wtm) material += value[PTYPE(sq[i])];
     else material -= value[PTYPE(sq[i])];
     if(PTYPE(sq[i]) > PAWN) pieces[PSIDE(sq[i])]++;
     plist[PSIDE(sq[i])][PTYPE(sq[i])][0]++;
     plist[PSIDE(sq[i])][PTYPE(sq[i])][plist[PSIDE(sq[i])][PTYPE(sq[i])][0]] = i;
   }
  }

  if(wtm) Or(hcode, hstm); 
  Or(hcode, castle_code[castle]);
  if(ep) Or(hcode, ep_code[FILE(ep)]);

  gstage = 18 - pieces[0] - pieces[1];
  if(gstage < 0) { gstage = 0; }
  if(gstage > 15) { gstage = 15; }
  Or(pcode, gstage_code[gstage]);

}



