// EXchess source code, (c) Daniel C. Homan  1997-2013
// Released under the GNU public license, see file license.txt


/* Check functions */

#include "define.h"
#include "chess.h"
#include "funct.h"
#include "const.h"

/* Check to see if we are in check */
// Returns a one if side to move is in check
// Returns a zero if not.
int position::in_check()
{
  check = attacked(plist[wtm][KING][1], wtm^1);
  return check;
}


/* Check to see if this is check-mate */
// Returns 1 if check_mate
// Returns 2 if stale_mate
// Returns 0 if ok
int position::in_check_mate()
{
  int ok = 0;                 // ok flag
  move_list tlist;            // temporary list
  position tpos;              // temporary position

  allmoves(&tlist, &game.ts.tdata[0]);      // find the semi-legal moves

  for(int i = 0; i < tlist.count; i++)
   {  
     tpos = (*this);
     if(tpos.exec_move(tlist.mv[i].m, 0)) { ok = 1; break; }
   }

  if(ok) return 0;                      // ok!
  else {
    if(this->in_check()) return 1;      // check_mate!
    else return 2;                      // stale_mate!
  }

}















