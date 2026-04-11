// EXchess source code, (c) Daniel C. Homan  1997-2013
// Released under the GNU public license, see file license.txt


// Change Log
//-------------
// 2011_12_12 (and 2011_12_11) -- Changed long GetTime() to int GetTime() and
//    changed internal definitions from long to int...  also changed all time
//    variables in search.cpp, main.cpp to be "int" rather than "long".  Seems
//    to be required in 64-bit compile under MacOS X.  Hopefully will be portable
//    to other systems.

#include <cstdlib>

#include "define.h"
#include "chess.h"
#include "const.h"
#include "funct.h"

#if MSVC
 #include <sys/timeb.h>
#else
 #include <time.h>
 #include <sys/time.h>
#endif

// Function return the time since an arbitrary reference in
// 100ths of a second
int GetTime()
{
#if MSVC
  struct timeb tval;

  ftime(&tval);
  return (tval.time*100 + int(float(tval.millitm)/10));
#else
  struct timeval tval;
  struct timezone tzone;

  if(!gettimeofday(&tval,&tzone))
   return (tval.tv_sec*100 + int(float(tval.tv_usec)/10000));
  else return (time(NULL)*100);
#endif
}




