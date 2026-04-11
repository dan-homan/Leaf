// EXchess source code, (c) Daniel C. Homan  1997-2017
// Released under the GNU public license, see file license.txt
 
// Support functions for tree_search class
//----------------------------------------------------------------
//  tree_search();           // initializer function 
//  ~tree_search();          // destructor function 
//  void initialize_extra_threads();
//  void close_extra_threads();
//  void delete_thread_data();
//  void create_thread_data(game_rec *gr, int thread_count);
//  void history_stats();
//----------------------------------------------------------------

#include <pthread.h>
#include <unistd.h>
#include <cstdlib>

#include "define.h"
#include "chess.h"
#include "const.h"
#include "hash.h"

//--------------------------------------------------------
// Initialization Function
//--------------------------------------------------------
tree_search::tree_search() {

  max_ply = 0;
  start_time = 0;
  limit = 0;
  max_limit = 0;
  time_double = 0;
  ponder = 0;
  last_ponder = 0;
  ponder_time = 0;
  ponder_time_double = 0;
  root_alpha = -MATE;
  root_beta = +MATE;
  root_tb_score = -1;
  root_wtm = 0;
  start_depth = 1;
  last_depth = 1;
  g_last = 0;
  h_id = 1;
  analysis_mode = 0;
  tsuite = 0;
  no_book = 0;
  max_search_depth = MAXD;
  max_search_time = MAXT;
  fail_low = 0;
  fail_high = 0;
  singular_response.t = NOMOVE;
  initialized_threads = 0;
  
  // NOTE: thread and board data need to be initialized after 
  //       the game object is created... see start of main() in main.cpp

}

//--------------------------------------------------------
// Destructor Function
//--------------------------------------------------------
tree_search:: ~tree_search() {          // destructor function
  close_extra_threads();
  delete_thread_data();
}

//--------------------------------------------------------
//  Function to create new search threads as needed
//--------------------------------------------------------
void tree_search::initialize_extra_threads() {

  //----------------------------------------------------
  // initialize theads and mutexes on first call of this
  // function 
  //  -- the daughter threads are numbered with 
  //     thread IDs = 1, 2, ... up to THREADS-1.
  //  -- the original (main) thread has ID = 0 (thread_index = 0)
  //     and doesn't need initialization
  //----------------------------------------------------
  for(thread_index=initialized_threads+1; thread_index<THREADS; thread_index++) {
    pthread_mutex_init(&runlock[thread_index],NULL);
    pthread_mutex_init(&waitlock[thread_index],NULL);
    pthread_mutex_init(&waitlock2[thread_index],NULL);
    pthread_cond_init(&go_cond[thread_index],NULL);
    if(!tdata[thread_index].running) {
      pthread_create(&search_thread[thread_index], NULL, daughter_thread_search, (void *) (intptr_t)thread_index);
      while(!tdata[thread_index].running) { 
	//timespec wait_time[1] = {0, 10000000};
	write_out("waiting...\n"); 
	// sleep for 0.01 seconds
	//nanosleep(wait_time, NULL);
	usleep(10000);
      } 
    }
    initialized_threads++;
  }

}
//--------------------------------------------------------
// Function to close and wait for extra threads to avoid memory leaks
//  --> called from the tree_search destructor
//  --> NOTE: Under normal operation, these extra threads will always
//      be running in their while loop (see daughter_thread_search function)
//      and waiting on the next "go_cond" from the main thread
//      until cancelled explicitly
//--------------------------------------------------------
void tree_search::close_extra_threads() {
  for(thread_index=1;thread_index <= initialized_threads;thread_index++) {
    pthread_cancel(search_thread[thread_index]);
    pthread_join(search_thread[thread_index],NULL);
  }
}

//--------------------------------------------------------
//  Function to free memory held by thread data
//--------------------------------------------------------
void tree_search::delete_thread_data() {
  delete [] tdata;
}
//-------------------------------------------------------------------
//  Function to create and link nodes in tree search data structure
//-------------------------------------------------------------------
void tree_search::create_thread_data(game_rec *gr, int thread_count) {

  tdata = new ts_thread_data[thread_count];

  // initialize nodes and thread ID for each thread data structure
  // in the search tress
  for(int t=0; t < thread_count; t++) {
    tdata[t].ID = t;
    tdata[t].fail = 0;
    tdata[t].tb_hit = 0;
    if(!t) { tdata[0].running = 1; }  // main thread is always running
    else { tdata[t].running = 0; }
    tdata[t].quit_thread = 0;
    // initialize principle continuation
    for(int i = 0; i < MAXD+1; i++) {
      for(int j = 0; j < MAXD+1; j++) {
	tdata[t].pc[i][j].t = 0;
      }
    }
    // initialize history and reply table
    for(int i = 0; i < 15; i++)
      for(int j = 0; j < 64; j++) { 
	tdata[t].history[i][j] = 0; 
	tdata[t].reply[i][j] = 0; 
      }
    // initialize thread data 
    tdata[t].init_thread_data(0);

    /* setup linked list of nodes for search */
    for(int iply = 0; iply <= MAXD; iply++) {
      tdata[t].n[iply].ply=iply;
      tdata[t].n[iply].tdata = &(gr->ts.tdata[t]);
      tdata[t].n[iply].ts = &(gr->ts);
      tdata[t].n[iply].gr = gr; 
      if(!iply) { tdata[t].n[iply].prev = &tdata[t].n[iply]; }
      else { tdata[t].n[iply].prev = &tdata[t].n[iply-1]; }
      if(iply == MAXD) { tdata[t].n[iply].next = &tdata[t].n[iply]; }
      else { tdata[t].n[iply].next = &tdata[t].n[iply+1]; } 
    }
  }

}

/* Function to show statistics of the history table */
void tree_search::history_stats() {
  
  int bins[16] = { -1000001,-1000000,-100000,-10000,-1000,-100,-10,0,10,100,1000,10000,100000,999999,1000000,2000000000 };
  int count[16] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  
  for(int t=0; t < THREADS; t++) {

    cout << "History data for thread: " << t << "\n";

    for(int i=0; i<15; i++) {
      for(int j=0; j<64; j++) {
	for(int k=0; k<16; k++) {
	  if(tdata[t].history[i][j] <= bins[k]) { count[k]++; k = 16; }
	}
      }
    }

    for(int k=0; k<16; k++) {
      cout << "Bin " << bins[k] << " count = " << count[k] << "\n";
    }

  }

}


