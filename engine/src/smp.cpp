// EXchess source code, (c) Daniel C. Homan  1997-2016
// Released under the GNU public license, see file license.txt

/* SMP Functions */
//-----------------------------------------------------------------------------
// void *daughter_thread_search(void *id) 
// int tree_search::search_threads(int alpha, int beta, int depth, int threads) 
//-----------------------------------------------------------------------------

#include <pthread.h>
#include <unistd.h>

#include "define.h"
#include "chess.h"
#include "search.h"

//-----------------------------------------------
// Global variables to control SMP search
//-----------------------------------------------

pthread_mutex_t egtb_lock;
pthread_mutex_t log_lock;
unsigned int THREADS = 1;

//----------------------------------------------
// Function to run daughter threads and call 
// the search when they recieve the go signal
// from the main thread
//  NOTE: using a double lock system to 
//        ensure that threads are kept
//        synchronized.  I originally thought
//        a single lock (just 'runlock') with
//        a 'go_cond' signal would be enough,
//        but some threads were so slow that 
//        the later call to lock 'runlock' 
//        would happen *before* the cond_wait
//        was activated!!  I needed a second
//        'waitlock' to ensure the cond_wait
//        was activated before proceeding
// NOTE2: These ONLY apply to searches in the 
//        main game.ts tree search structure.
//        Other tree search structures will
//        not be able to call searches in 
//        multiple threads with the current
//        design and will be limited to
//        a single thread (the original one).
// NOTE3: The quit_thread variable is currently
//        unused by the rest of the program, so
//        threads will sit in their idle state
//        waiting for a go_cond from the main
//        thread.  Threads must be cancelled to
//        be closed, see the close_extra_threads
//        function below.
//----------------------------------------------
void *daughter_thread_search(void *id) {

  int tid = (intptr_t) id;

  pthread_setcanceltype(PTHREAD_CANCEL_ASYNCHRONOUS,NULL);

  pthread_mutex_lock(&game.ts.runlock[tid]);
  pthread_mutex_lock(&game.ts.waitlock[tid]);

  game.ts.tdata[tid].running = 1;
  game.ts.tdata[tid].quit_thread = 0;

  // Write out a logfile string to indicate that
  //  this thread was created
  char wstring[50];
  snprintf(wstring, sizeof(wstring), "Creating Extra Thread tid=%d\n", int(tid));
  write_out(wstring);

  // Prepare another string for use when starting
  //  a search in this thread
  snprintf(wstring, sizeof(wstring), "Starting Search in Thread tid=%d\n", int(tid));

  while(!game.ts.tdata[tid].quit_thread) {
    pthread_cond_wait(&game.ts.go_cond[tid], &game.ts.runlock[tid]);
    pthread_mutex_unlock(&game.ts.waitlock[tid]);
    //write_out(wstring);                 // log search start, for debugging
    game.ts.tdata[tid].n[0].root_pvs();   // start search
    //-----------------------------------------
    // If this thread returns a real score,
    // then it finished first, so stop others
    //-----------------------------------------
    if(game.ts.tdata[tid].g != -TIME_FLAG) {
      // Stop the main thread, the rest will be stopped
      //   after the main returns
      game.ts.tdata[0].done=tid;
    }
    pthread_mutex_lock(&game.ts.waitlock2[tid]);
    pthread_mutex_unlock(&game.ts.waitlock2[tid]);
    pthread_mutex_lock(&game.ts.waitlock[tid]);
  }
  pthread_mutex_unlock(&game.ts.runlock[tid]);
  pthread_mutex_unlock(&game.ts.waitlock[tid]);

  game.ts.tdata[tid].running = 0;

  return NULL;
}

//----------------------------------------------------
// function to search a bunch of threads
//----------------------------------------------------
int tree_search::search_threads(int alpha, int beta, int depth, int threads) {

  int ti, pi; 

  if(threads > initialized_threads+1) {
    threads = initialized_threads+1;
    write_out("Not enough initialized threads in tree_search structure, limiting to initialized threads only!\n");
  }

  //----------------------------------------------------------------
  // setup thread data and start/continue the parallel threads
  //  -- at the moment, all parallel threads are searching the
  //     same alpha-beta window, but perhaps with a greater 
  //     depth than the main thread with (ti = 0, ID = 0).
  //  -- experimenting with different schemes for this
  //----------------------------------------------------------------
  for(ti=0; ti<threads; ti++) {
    // setup data  
    tdata[ti].alpha = alpha;
    tdata[ti].beta = beta;
    tdata[ti].depth = depth;
    tdata[ti].quit_thread = 0;
    tdata[ti].fail = 0;
    tdata[ti].tb_hit = 0;
    /*
    if(depth > 2 && threads > 2) { 
      if(ti > threads/3) tdata[ti].depth += 1; 
      if(ti > (3*threads)/4) tdata[ti].depth += 1;
      if(ti > (7*threads)/8) tdata[ti].depth += 1;
    }
    tdata[ti].depth = MIN(tdata[ti].depth,MAX_MAIN_TREE-1);
    */
    tdata[ti].done = 0;
    if(ti) {     
      tdata[ti].pc[0][0].t = tdata[0].pc[0][0].t;
    } 
  }
  //-------------------------------------
  // send go signal to parallel threads 
  //-------------------------------------
  for(ti=1; ti<threads; ti++) {
    pthread_mutex_lock(&waitlock2[ti]);
    pthread_mutex_lock(&runlock[ti]);
    pthread_cond_signal(&go_cond[ti]);
    pthread_mutex_unlock(&runlock[ti]);
    pthread_mutex_lock(&waitlock[ti]);
    pthread_mutex_unlock(&waitlock2[ti]);
  }

  //-------------------------------------
  // run the search in the main thread
  //-------------------------------------
  tdata[0].n[0].root_pvs();

  //-------------------------------------------------------------------
  // wait for the parallel threads to finish current work using locks
  //-------------------------------------------------------------------
  for(ti=1; ti<threads; ti++) {
    tdata[ti].done=1;
    pthread_mutex_unlock(&waitlock[ti]);
    pthread_mutex_lock(&runlock[ti]);
    pthread_mutex_unlock(&runlock[ti]);
  }
  //-------------------------------------------------------------------
  // Check whether a parallel thread returned a real score because it 
  // finished before the main thread.  If so, replace score and
  // PV information 
  //-------------------------------------------------------------------
  for(ti=1; ti<threads; ti++) {
    if(tdata[ti].g != -TIME_FLAG && !game.terminate_search
       && (tdata[0].g == -TIME_FLAG || tdata[0].depth < tdata[ti].depth)
       && tdata[ti].pc[0][0].t) {
      // Replace g in the 0 thread  
      tdata[0].g = tdata[ti].g;
      tdata[0].depth = tdata[ti].depth;
      // update the iteration depth to ensure consistent
      // researches if this thread had a greater depth
      if(tdata[0].depth > max_ply-1) {
	max_ply += tdata[0].depth-(max_ply-1);
      }
      // update the PV
      for(pi=0; tdata[ti].pc[0][pi].t; pi++) {
	tdata[0].pc[0][pi].t = tdata[ti].pc[0][pi].t;
      }
      tdata[0].pc[0][pi].t = NOMOVE;
      // display the results of switch
      if(tdata[0].g <= alpha) tdata[0].fail = -1;
      if(tdata[0].g >= beta) tdata[0].fail = 1;
      if(post && !xboard) search_display(tdata[0].g);
      write_out("Replacing main thread with alternate thread\n");
      // only log fail highs or lows because exact scores will
      //  be displayed after the iteration (and this is the end
      //  of the iteration if it is not a fail high/low).
      if(tdata[0].fail) log_search(tdata[0].g);
      tdata[0].fail = 0;
    }
  }
 
  /*
  // Try to debug a difficult problem... (see NOTE above for resolution)
  if(tdata[0].done && tdata[0].g == -TIME_FLAG) {
    write_out("Premature Termination of Search\n");
    char wstring[100];
    for(ti=0; ti<threads; ti++) {
      snprintf(wstring, sizeof(wstring), "move number: %d, g: %d, depth: %d, done: %d\n", tdata[ti].pc[0][0].t, tdata[ti].g, tdata[ti].depth, tdata[ti].done);
      write_out(wstring);
    }
  }
  */

  //-------------------------------------------------------
  // If we ran out of time, check to see if another thread
  //  found a move with a higher score that is > alpha
  //   -- only if we are not pondering or not analysis
  //-------------------------------------------------------
  if(tdata[0].g == -TIME_FLAG && !ponder 
     && !analysis_mode && !game.terminate_search) {
    for(ti=1; ti<threads; ti++) {
      if(tdata[ti].n[0].best > alpha 
	 && tdata[ti].n[0].best > tdata[0].n[0].best
	 && tdata[ti].pc[0][0].t) {
	tdata[0].n[0].best = tdata[ti].n[0].best;
	// update the PV
	for(pi=0; tdata[ti].pc[0][pi].t; pi++) {
	  tdata[0].pc[0][pi].t = tdata[ti].pc[0][pi].t;
	}
	tdata[0].pc[0][pi].t = NOMOVE;
	// display the results of switch
	if(post && !xboard) search_display(tdata[0].n[0].best);
	write_out("Replacing main thread with alternate thread at timeout\n");
	log_search(tdata[0].n[0].best);
      }
    }
  }

  //-------------------------------------
  // return score from main thread
  //-------------------------------------
  return tdata[0].g;

}


