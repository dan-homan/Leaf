/* header file for hash functions */

#ifndef HASH_H
#define HASH_H

#define FLAG_A 1
#define FLAG_B 2
#define FLAG_P 3
#define HASH_MISS -21000
#define HASH_MOVE -21001

#define GET_ID(x) (x&31)
#define GET_FLAG(x) ((x&96)>>5)
#define GET_MATE_EXT(x) ((x&128)>>7)         
#define SET_DATA(id,flag,mate) ((id&31)+((flag&3)<<5)+((mate&1)<<7))

// Note: Mate extension variable currently is unused

//--------------------------------------------------------
// Hash tables of various kinds are defined here...
// -- each now uses a lock-less strategy for eventual 
//    conversion to multi-threaded search, as developed 
//    by Robert Hyatt and Tim Mann, see
//    http://www.cis.uab.edu/hyatt/hashing.html
//--------------------------------------------------------

/* standard hash record - 16 bytes long */
struct hash_rec
{
  h_code hr_key;
  move hr_hmove;
  int16_t hr_score;
  char hr_depth;   // note that stored depths cannot be larger than 127
  unsigned char hr_data;  // first 5 bits = id, next 2 = flag, next 1 = mate_ext
  
  // lock-less hashing of hash key
  inline void set_key(h_code uncoded_key,int16_t sc, unsigned char dat, char dep, int32_t hmove_t) {
    hr_key = uncoded_key^(h_code(sc)|(h_code(dat)<<16)|(h_code(dep)<<24)|(h_code(hmove_t)<<32));
  }
  inline h_code get_key() {
    return (hr_key^(h_code(hr_score)|(h_code(hr_data)<<16)|(h_code(hr_depth)<<24)|(h_code(hr_hmove.t)<<32)));
  }

};


/* Bucket for 4 hash recs */
struct hash_bucket
{
  hash_rec rec[4];
};

// pawn data used in pawn hash record 
// -- 24 bytes long
struct pawn_data {
  int16_t score;
  uint64_t pawn_attacks[2];
  unsigned char open_files;
  unsigned char half_open_files_w;
  unsigned char half_open_files_b;
  unsigned char passed_w;
  unsigned char passed_b;
  int8_t padding1;
};

/* pawn hash record - 32 bytes long */
struct pawn_rec
{
  h_code key;

  pawn_data data;

  // lock-less hashing of hash key
  inline void set_key(h_code uncoded_key,int16_t sc, uint64_t bpa, uint64_t wpa) {
    key = uncoded_key^(h_code(sc)|h_code(bpa)|h_code(wpa));
  }
  inline h_code get_key() {
    return (key^(h_code(data.score)|(h_code(data.pawn_attacks[BLACK]))|(h_code(data.pawn_attacks[WHITE]))));
  }
};

/* score hash record - 16 bytes long */
struct score_rec
{
  h_code key;
  int16_t score;
  char qchecks[2];

  int32_t padding1;

  // lock-less hashing of hash key
  inline void set_key(h_code uncoded_key,int16_t sc, char qc0, char qc1) {
    key = uncoded_key^(h_code(sc)|(h_code(qc0)<<16)|(h_code(qc1)<<24));
  }
  inline h_code get_key() {
    return (key^(h_code(score)|(h_code(qchecks[0])<<16)|(h_code(qchecks[1])<<24)));
  }
};

/* combination move hash record - 32 bytes long */ 
struct cmove_rec
{
  h_code key1;
  int32_t move1;
  char depth1;
  unsigned char id;
  int16_t padding1;
  h_code key2;
  int32_t move2;
  char depth2;
  char padding2;
  int16_t padding3;

  // lock-less hashing of hash key
  inline void set_key1(h_code uncoded_key,int32_t mv, char depth) {
    key1 = uncoded_key^(h_code(mv)|(h_code(depth)<<32));
  }
  inline h_code get_key1() {
    return (key1^(h_code(move1)|(h_code(depth1)<<32)));
  }
  inline void set_key2(h_code uncoded_key,int32_t mv, char depth) {
    key2 = uncoded_key^(h_code(mv)|(h_code(depth)<<32));
  }
  inline h_code get_key2() {
    return (key2^(h_code(move2)|(h_code(depth2)<<32)));
  }
};

/* Number of hash related functions */
void open_hash();
void close_hash();
void set_hash_size(unsigned int Mbytes);
void put_hash(h_code *h_key, int score, int alpha, int beta, int depth, int hmove, int h_id, int ply);
int get_hash(h_code *h_key, int *hflag, int *hdepth, move *gmove, int ply, int *singular);
int get_move(h_code *h_key);
int put_move(h_code h_key, int putmove, int h_id);

/* Macro for or'ing two hash codes */
#define Or(A, B)   A ^= B;

// total size of hash tables in MB
int HASH_SIZE = 32;

/* hash table variables -- total as given is about 16 MB of hash */
unsigned int TAB_SIZE  =  131072;   // hash table size (bucket entries) - override in search.par
unsigned int PAWN_SIZE =   65536;   // pawn hash sizes (entries) - override in search.par
unsigned int SCORE_SIZE =  65536;   // score hash sizes (entries) - override in search.par
unsigned int CMOVE_SIZE =  16384;   // combination move hash size (entries)

hash_bucket *hash_table;            // pointer to start of hash table
pawn_rec *pawn_table;               // pointer to start of pawn table
score_rec *score_table;             // pointer to start of score table
cmove_rec *cmove_table;             // pointer to cmove table

//
// hash codes for sides to-move, pieces, castling, en-passant, and game_stage
//

#include "hash_values.h"


#endif /* HASH_H */
