/* EGTB Probe functions */
// This file interfaces EXchess with the egtb probing code of Eugene Nalimov
// and Andrew Kadatch.  The tablebase initialization and probing 
// functions included below are modified from the examples given in
// Eugene Namilov's 'tbgen' distribution, and Dr. Namilov reserves
// all rights to these functions.  They are not part of the GNU open 
// source release of EXchess.  
//
// This file is included for the users who wish to make EXchess work 
// with the Namilov endgame tablebases.  To do so, simply download the 
// 'tbgen' distribution and put the files in a sub-directory 'tb'. 
// Then change the 'TABLEBASES' define in the file 'define.h' to turn
// on tablebase probing.  Then just recompile...
//

#include "define.h"

#if TABLEBASES
#define INDEX unsigned long
#define square int
#define piece int
#define NEW
#define XX 64
#define T41_INCLUDE


inline square SqFindKing (square *sq)
{
 return sq[4];
}

inline square SqFindFirst (square *sq, piece pi)
{
  if(pi==sq[1]) return sq[5];
  if(pi==sq[2]) return sq[6];
  return sq[7];
}

inline square SqFindSecond (square *sq, piece pi)
{
  if(pi==sq[1] && pi==sq[2]) return sq[6];
  return sq[7];
}

inline square SqFindThird (square *sq, piece pi)
{
 return sq[7];
}

inline square SqFindOne (square *sq, piece pi)
{
 return sq[5];
}

#undef piece
#include "../tb_src/tbindex.c"
#undef square

#include "chess.h"
#include "funct.h"

// Thanks to Dann Corbit for adding the ability to do a environment
// variable fetch for the endgame tablebase path

float CACHE_SIZE = 4;
char EGTB_PATH[FILENAME_MAX] = "./tb";
void      *EGTB_cache =                    (void*)0;
extern int EGTB;

int init_tb() {
 char *where;
 if ((where = getenv("NALIMOV_PATH")) != NULL)
 {
         strncpy(EGTB_PATH, where, sizeof EGTB_PATH);
 }
 EGTB = IInitializeTb(EGTB_PATH);
 EGTB_cache = malloc(int(1048576*CACHE_SIZE));
 FTbSetCacheSize(EGTB_cache,int(1048576*CACHE_SIZE));
 return 0;
}

int probe_tb(position *p, int ply)
{

 int *psqW, i, j;
 int *psqB;
 int rgiCounters[10] = {0,0,0,0,0,0,0,0,0,0};
 int iTb;
 int side;
 int fInvert;
 int sqEnP;
 int wi = 1, W[8] = {6,0,0,0,0,0,0,0}; 
 int bi = 1, B[8] = {6,0,0,0,0,0,0,0}; 
 int tbScore;
 INDEX ind;

 W[4] = p->plist[1][KING][1];
 B[4] = p->plist[0][KING][1];

 for(i = 1; i < 6; i++) {
  for(j = 1; j <= p->plist[WHITE][i][0]; j++) {
    rgiCounters[i-1]++;
    W[wi] = i; W[wi+4]=p->plist[WHITE][i][j]; wi++;
  } 
  for(j = 1; j <= p->plist[BLACK][i][0]; j++) {
    rgiCounters[i+4]++;
    B[bi] = i; B[bi+4]=p->plist[BLACK][i][j]; bi++;
  } 
 }   

 iTb = IDescFindFromCounters(rgiCounters);

 if(0 == iTb) return -1;

 if (iTb > 0) {
  side = int(p->wtm^1); // white and black are reversed
  fInvert = 0;
  psqW = W;
  psqB = B;
 } else {
  side = p->wtm;
  fInvert = 1;
  psqW = B;
  psqB = W;
  iTb = -iTb;
 }

 if (!FRegistered(iTb, side)) return -1;
 if(p->ep != 0) sqEnP = p->ep; else sqEnP = XX;
 ind = PfnIndCalc(iTb, side) (psqW, psqB, sqEnP, fInvert);
 //FReadTableToMemory (iTb, side, NULL);
 tbScore = L_TbtProbeTable (iTb, side, ind);

 if(tbScore == L_bev_broken) return -1;

 if(tbScore > 0) return ((tbScore-L_bev_mi1-1)*2+MATE-ply);
 if(tbScore < 0) return ((tbScore+L_bev_mi1)*2 - MATE+ply);
 return 0;

}

#endif

