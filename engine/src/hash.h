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
  
  // Lock-less hashing (Hyatt/Mann): store key ^ packed(data), so a torn read is
  // caught when get_key() fails to reproduce the probe key.
  //
  // Each field is cast to its UNSIGNED width before widening to h_code.  A
  // signed field would sign-extend to 64 bits and set every higher bit, so the
  // fields OR-ed in above it would land on bits that are already 1 and
  // contribute nothing -- leaving torn writes in them undetectable.  That is
  // the common case, not an edge case: roughly half of all stored scores are
  // negative, and depth -2 is both this table's initial value and what
  // put_move() writes.  (At score == -1 the mask degenerated to all ones, i.e.
  // the stored key stopped depending on the data at all.)
  //
  // Field layout, disjoint by construction: score 0..15, data 16..23,
  // depth 24..31, move 32..63.
  inline void set_key(h_code uncoded_key,int16_t sc, unsigned char dat, char dep, int32_t hmove_t) {
    hr_key = uncoded_key^((h_code)(uint16_t)sc
                         |((h_code)(uint8_t)dat<<16)
                         |((h_code)(uint8_t)dep<<24)
                         |((h_code)(uint32_t)hmove_t<<32));
  }
  inline h_code get_key() {
    return (hr_key^((h_code)(uint16_t)hr_score
                   |((h_code)(uint8_t)hr_data<<16)
                   |((h_code)(uint8_t)hr_depth<<24)
                   |((h_code)(uint32_t)hr_hmove.t<<32)));
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

  // Lock-less hashing -- see hash_rec above on the unsigned casts.
  //
  // The two 64-bit bitboards are XOR-mixed rather than OR-ed: OR-ing two full
  // 64-bit values makes them overlap completely and throws away most of their
  // information (any pair with the same OR is indistinguishable).  WHITE's board
  // is rotated by 32 first so that swapping the two boards changes the result.
  // Coverage is unchanged from before: score + both attack bitboards.  The byte
  // fields (open / half-open files, passed masks) are still not covered.
  inline void set_key(h_code uncoded_key,int16_t sc, uint64_t bpa, uint64_t wpa) {
    key = uncoded_key^(h_code)(uint16_t)sc^bpa^((wpa<<32)|(wpa>>32));
  }
  inline h_code get_key() {
    return (key^(h_code)(uint16_t)data.score^data.pawn_attacks[BLACK]
               ^((data.pawn_attacks[WHITE]<<32)|(data.pawn_attacks[WHITE]>>32)));
  }
};

/* score hash record - 16 bytes long */
struct score_rec
{
  h_code key;
  int16_t score;
  char qchecks[2];

  int32_t padding1;

  // Lock-less hashing -- see hash_rec above on the unsigned casts.
  // Field layout: score 0..15, qchecks[0] 16..23, qchecks[1] 24..31.
  inline void set_key(h_code uncoded_key,int16_t sc, char qc0, char qc1) {
    key = uncoded_key^((h_code)(uint16_t)sc
                      |((h_code)(uint8_t)qc0<<16)
                      |((h_code)(uint8_t)qc1<<24));
  }
  inline h_code get_key() {
    return (key^((h_code)(uint16_t)score
                |((h_code)(uint8_t)qchecks[0]<<16)
                |((h_code)(uint8_t)qchecks[1]<<24)));
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

  // Lock-less hashing -- see hash_rec above on the unsigned casts.  depth1/depth2
  // initialise to -2, which under the old signed widening flooded bits 32..63 and
  // erased the move's contribution entirely.
  // Field layout: move 0..31, depth 32..39.
  inline void set_key1(h_code uncoded_key,int32_t mv, char depth) {
    key1 = uncoded_key^((h_code)(uint32_t)mv|((h_code)(uint8_t)depth<<32));
  }
  inline h_code get_key1() {
    return (key1^((h_code)(uint32_t)move1|((h_code)(uint8_t)depth1<<32)));
  }
  inline void set_key2(h_code uncoded_key,int32_t mv, char depth) {
    key2 = uncoded_key^((h_code)(uint32_t)mv|((h_code)(uint8_t)depth<<32));
  }
  inline h_code get_key2() {
    return (key2^((h_code)(uint32_t)move2|((h_code)(uint8_t)depth2<<32)));
  }
};

/* Number of hash related functions */
void open_hash();
void close_hash();
void clear_hash();
void set_hash_size(unsigned int Mbytes);
void put_hash(h_code *h_key, int score, int alpha, int beta, int depth, int hmove, int h_id, int ply);
int get_hash(h_code *h_key, int *hflag, int *hdepth, move *gmove, int ply, int *singular);
int get_move(h_code *h_key);
int put_move(h_code h_key, int putmove, int h_id);

/* Macro for or'ing two hash codes */
#define Or(A, B)   A ^= B;

// total size of hash tables in MB — now engine_cfg.hash_size in engine_globals.h

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
