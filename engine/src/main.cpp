// EXchess source code, (c) Daniel C. Homan  1997-2017
// Released under the GNU public license, see file license.txt

// Thanks to Jim Ablett for fixes to the polling code which 
// allow EXchess to work with the mingw compiler on windows!

/*  Main functions controlling program */
//-------------------------------------------
// int main(int argc, char *argv[])
// void takeback(int tm)
// void make_move()
// void drawboard()
// void help()
// void type_moves()
// void type_capts()
// void parse_command()
// void performance()
// void save_game()
// int inter()
// void write_out(const char *outline) {
//-------------------------------------------

#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <csignal>
#include <fstream>
#include <stdint.h>
#include <pthread.h>

#include "define.h"
#include "stdtypes.h"
#include "chess.h"
#include "search.h"

// MUST come after define.h to disable asserts
#include <assert.h>

#if MSVC || MINGW
 #include <windows.h>
 #include <time.h>
 #include <conio.h>
 #if MSVC
  #undef BLACK
  #undef WHITE
  #define BLACK 0
  #define WHITE 1
 #endif
#else
 #include <time.h>
 #include <unistd.h>
 #include <sys/types.h>
 #include <sys/time.h>
#endif

#if NNUE && NNUE_EMBED
extern "C" const unsigned char gNnueNetData[];
extern "C" const unsigned int  gNnueNetSize;
#endif

// Custom headers, defining external functions and struct types for
// board, piece and moves.  And defining global variables.

#include "const.h"
#include "funct.h"
#include "hash.h"
#include "extern.h"
#include "engine_globals.h"

// Main Game Structure
game_rec game;

// Grouped global state
ProtocolState proto;
EngineConfig  engine_cfg;
SearchConfig  search_cfg;
ThreadConfig  thread_cfg;

// Flags for input and display
int display_board = 0;
char response[60];        // first word of input string from command prompt

// set of values for tracking interface lag
#define LAG_COUNT  10
int interface_lags[LAG_COUNT];
int interface_lag_count = 0;
int next_lag_mod = 0;
int average_lag = 0;

// Basic control flags not in structs
int ics = 0, ALLEG = 0, hintflag = 0;
int ponder_flag = 1, shout_book = 0;

#if UNIX
 fd_set read_fds;
 struct timeval timeout = { 0, 0 };
#endif

#if FLTK_GUI
 int FLTKmain(int argc, char** argv);
#endif

/*------------------------- Main Function ---------------------------*/
//  Main control function that interacts with the User

int main(int argc, char *argv[])
{
  char mstring[10];
  move hint;

#ifdef TEST_ASSERT
  assert(0);  // testing that asserts are off
#endif

  //-------------------------------
  // Parse and record exec path
  //-------------------------------
  strcpy(engine_cfg.exec_path, argv[0]);
  int last_slash = -1;
  for(int j = 0; j < FILENAME_MAX; j++) {
    if(engine_cfg.exec_path[j] == '\0') break;
    if(engine_cfg.exec_path[j] == '\\') last_slash = j;
    if(engine_cfg.exec_path[j] == '/') last_slash = j;
  }  
  if(last_slash == 0) strcpy(engine_cfg.exec_path, "./");
  else engine_cfg.exec_path[last_slash+1] = '\0';

  // initialize lock for proto.logfile
  // -- must come before any function that might
  //    try to access these locks, even "write_out"
  pthread_mutex_init(&thread_cfg.log_lock,NULL);

  //---------------------------------
  // Initialize variables for 
  //  -- search params 
  //  -- hash table
  //  -- check tables
  //  -- random number seeds
  //  -- threads and locks
  //  -- board and game record
  //----------------------------------
  set_search_param();
  set_hash_size(engine_cfg.hash_size);
  gen_check_table();
  srand(time(NULL));
#if NNUE
  {
    // --init-nnue: create a fresh random-initialised .nnue without reading an existing one.
    //   PSQT = classical material (own=+V, enemy=-V; P=100 N=377 B=399 R=596 Q=1197 cp,
    //   same across all 8 buckets).  Pure-PSQT: the bucketed PSQT is the sole material
    //   channel (no dense piece_val).
    // --init-nnue-noprior: PSQT = uniform 100 cp (own=+V, enemy=-V; P=N=B=R=Q=100 cp,
    //   same across all 8 buckets).  Materially blind from move 1 but value[PAWN] stays
    //   anchored at 100 cp; N/B/R/Q learn from a 100 cp baseline.
    // --init-nnue-classical: PSQT = classical material + 4-stage piece-square tables
    //   (gstage-interpolated across the 8 NNUE buckets).  Use to bootstrap training from
    //   the classical hand-tuned positional prior; lets FC/FT learn finer patterns
    //   instead of re-deriving piece values and positional shape.
    // All require --write-nnue <filename>; write the .nnue AND a fresh companion
    // .tdleaf.bin (the FP32 training shadow), then exit.  Refuses to run if the
    // companion .tdleaf.bin already exists (the save path is a MERGE-save, so writing
    // over an existing file would blend the fresh init with stale trained weights).
    bool init_nnue_mode = false;
    int  init_nnue_prior = NNUE_PRIOR_MATERIAL;
    for (int ai = 1; ai < argc; ai++) {
      if (strcmp(argv[ai], "--init-nnue") == 0) { init_nnue_mode = true; }
      if (strcmp(argv[ai], "--init-nnue-noprior") == 0)   { init_nnue_mode = true; init_nnue_prior = NNUE_PRIOR_NOPRIOR; }
      if (strcmp(argv[ai], "--init-nnue-classical") == 0) { init_nnue_mode = true; init_nnue_prior = NNUE_PRIOR_CLASSICAL; }
    }

    if (init_nnue_mode) {
#if !TDLEAF
      fprintf(stderr, "--init-nnue requires a TDLEAF build (compile with -D TDLEAF=1)\n");
      return 1;
#else
      // Find the --write-nnue <filename> argument.
      const char *write_nnue_path = nullptr;
      for (int ai = 1; ai < argc - 1; ai++)
        if (strcmp(argv[ai], "--write-nnue") == 0) { write_nnue_path = argv[ai + 1]; break; }
      if (!write_nnue_path) {
        fprintf(stderr, "--init-nnue requires --write-nnue <filename>\n");
        return 1;
      }
      // Refuse if the companion .tdleaf.bin already exists.  nnue_save_fc_weights()
      // is a MERGE-save: writing a fresh init over an existing file would blend the
      // new init with stale trained weights rather than resetting.  Fail fast (before
      // writing the .nnue) so the caller removes the old file first.
      {
        char tdbin[FILENAME_MAX];
        snprintf(tdbin, sizeof(tdbin), "%s", write_nnue_path);
        char *dot = strrchr(tdbin, '.');
        if (dot && strcmp(dot, ".nnue") == 0) strcpy(dot, ".tdleaf.bin");
        else strncat(tdbin, ".tdleaf.bin", sizeof(tdbin) - strlen(tdbin) - 1);
        FILE *ex = fopen(tdbin, "rb");
        if (ex) {
          fclose(ex);
          fprintf(stderr, "--init-nnue: companion '%s' already exists; refusing to "
                          "merge-save a fresh init over it.  Remove it first "
                          "(rm '%s') then re-run.\n", tdbin, tdbin);
          return 1;
        }
      }
      // Allocate arrays, init float shadows, then fill with random distributions.
      nnue_alloc_arrays();
      nnue_init_fp32_weights();
      nnue_init_zero_weights(init_nnue_prior);
      // Write .nnue file.
      if (!nnue_write_nnue(write_nnue_path)) {
        fprintf(stderr, "--write-nnue: failed to write %s\n", write_nnue_path);
        return 1;
      }
      // Write a fresh companion .tdleaf.bin (the FP32 training shadow + Adam state)
      // so the first training session resumes exactly from this init.
      {
        char tdbin[FILENAME_MAX];
        snprintf(tdbin, sizeof(tdbin), "%s", write_nnue_path);
        char *dot = strrchr(tdbin, '.');
        if (dot && strcmp(dot, ".nnue") == 0)
          strcpy(dot, ".tdleaf.bin");
        else
          strncat(tdbin, ".tdleaf.bin", sizeof(tdbin) - strlen(tdbin) - 1);
        if (!nnue_save_fc_weights(tdbin))
          fprintf(stderr, "--init-nnue: failed to write %s\n", tdbin);
        else
          printf("TDLeaf: initial weights saved to %s\n", tdbin);
      }
      return 0;
#endif
    } else {
#if NNUE_EMBED
      nnue_load_from_memory(gNnueNetData, gNnueNetSize);
#else
      char nnue_path[FILENAME_MAX];
      snprintf(nnue_path, sizeof(nnue_path), "%s%s", engine_cfg.exec_path, NNUE_NET);
      if (!nnue_load(nnue_path)) {
        nnue_load(NNUE_NET);
      }
#endif
      if (nnue_available) write_out("NNUE evaluation loaded.\n");
      else                write_out("NNUE file not found, using classical evaluation.\n");
#if TDLEAF
      if (nnue_available) {
        // Attempt to load previously learned weights (companion .tdleaf.bin file).
        char tdleaf_path[512];
        snprintf(tdleaf_path, sizeof(tdleaf_path), "%s%s",
                 engine_cfg.exec_path, NNUE_TDLEAF_BIN);
        bool loaded = nnue_load_fc_weights(tdleaf_path);
        if (!loaded)
            loaded = nnue_load_fc_weights(NNUE_TDLEAF_BIN);
        if (!loaded)
          fprintf(stderr, "TDLeaf: no weights file found — using pretrained .nnue weights.\n"
                          "TDLeaf: run with --init-nnue --write-nnue <file> to create a fresh net.\n");
      }
#endif
    }
  }
#if NNUE
  // Extract PSQT-derived piece values and update the search's value[] array.
  // Must be called after both nnue_load() and nnue_load_fc_weights() complete
  // so that any piece_val correction from .tdleaf.bin is included.
  if (nnue_available) nnue_extract_piece_values();
#endif
#if NNUE && TDLEAF
  // --batch-train <tsv[,tsv...]>: offline supervised training on quiet-position
  // sets (see nnue_batch_train.cpp for options).  Runs after the normal
  // .nnue + .tdleaf.bin load above so training starts from the current
  // checkpoint state, then exits.
  {
    bool bt_mode = false;
    for (int ai = 1; ai < argc - 1; ai++)
      if (strcmp(argv[ai], "--batch-train") == 0) { bt_mode = true; break; }
    if (bt_mode) {
      if (!nnue_available) {
        fprintf(stderr, "--batch-train requires a loaded NNUE network\n");
        return 1;
      }
      return nnue_batch_train(argc, argv);
    }
  }
#endif
#endif

#if NNUE
  // Handle --write-nnue <filename>: export current NNUE weights (FC trained via TDLeaf,
  // FT copied verbatim from the loaded .nnue) to a complete .nnue file, then exit.
  for (int ai = 1; ai < argc; ai++) {
    if (strcmp(argv[ai], "--write-nnue") == 0 && ai + 1 < argc) {
      if (!nnue_available) {
        fprintf(stderr, "--write-nnue: no NNUE loaded\n");
        return 1;
      }
      if (!nnue_write_nnue(argv[ai + 1])) {
        fprintf(stderr, "--write-nnue: failed to write %s\n", argv[ai + 1]);
        return 1;
      }
      return 0;
    }
  }
#endif


  // initialize all thread data
  game.ts.create_thread_data(&game, MAX_THREADS);
  game.ts.initialize_extra_threads();

  // initialize board and movelist
  game.setboard(i_pos, 'w', "KQkq", "-");

  //-------------------------------
  // opening proto.logging file 
  //-------------------------------
  if(proto.logging) {
    char lfile[FILENAME_MAX];
    strcpy(lfile, engine_cfg.exec_path);
    strcpy(lfile, "run.log");
    for(int li = 1; li <= ABS(engine_cfg.max_logs); li++) {
      // if there is a single log file (engine_cfg.max_logs = 1), 
      // just append to that or create one if none is there
      if(li==1 && engine_cfg.max_logs == 1) {
	proto.logfile.open(lfile, ios::in); 
	if(proto.logfile.is_open()) { 
	  proto.logfile.close(); 
	  proto.logfile.open(lfile, ios::app); 
	  break;
	} else { 
	  proto.logfile.open(lfile, ios::out);
	  break;
	}
      }
      // if there is a single log file with engine_cfg.max_logs = -1, 
      // overwrite the exisiting log that is there
      if(li==1 && engine_cfg.max_logs == -1) {
	proto.logfile.open(lfile, ios::out);
	break;	
      }
      // otherwise find a new file name for log up to engine_cfg.max_logs
      if(li < 10) snprintf(lfile, sizeof(lfile), "%srun_00%i.log", engine_cfg.exec_path, li);
      else if(li < 100) snprintf(lfile, sizeof(lfile), "%srun_0%i.log", engine_cfg.exec_path, li);
      else snprintf(lfile, sizeof(lfile), "%srun_%i.log", engine_cfg.exec_path, li);
      proto.logfile.open(lfile, ios::in);
      if(!proto.logfile.is_open()) {
	proto.logfile.close();
	proto.logfile.open(lfile, ios::out);       
	break;
      } else {
	proto.logfile.close();
      }
    }
    if(!proto.logfile.is_open()) {
      cout << "Error(Can't open proto.logging file!)\n";
      proto.logging = 0;
    } else {
      proto.logfile.clear();
      proto.logfile << "===============================\n"; 
      proto.logfile << "=\n";
      proto.logfile << "= Log file for Leaf v" << VERS << VERS2 << "\n";
      proto.logfile << "=\n";
      proto.logfile << "===============================\n"; 
    }
  }

  //-----------------------------------
  // parsing command line args
  //  -- virtually no error checking!
  //-----------------------------------
  for(int argi = 1; argi < argc; argi++) {
    // turn on UCI mode via command-line flag
    if(!strcmp(argv[argi], "--uci")) {
      proto.uci_mode = 1;
      proto.interface_mode = 1;
      continue;
    }
    // turn on proto.xboard mode via command-line flag
    if(!strcmp(argv[argi], "--xboard")) {
      proto.xboard = 1;
      proto.interface_mode = 1;
      signal(SIGINT, SIG_IGN);
      continue;
    }
    // turn on proto.xboard mode
    if(!strcmp(argv[argi], "xb")) {
      proto.xboard = 1;
      proto.interface_mode = 1;
      continue;
    }
    // enable proto.logging
    if(!strcmp(argv[argi], "--log")) {
      proto.logging = 1;
      continue;
    }
    // set the number of cores to use
    if(!strcmp(argv[argi], "cores")) { 
      int thread_val = atoi(argv[argi+1]);
      if(thread_val > MAX_THREADS) {
	cout << "Error(MAX_THREADS set to " << MAX_THREADS << " so cores limited to this value)\n";
	proto.logfile << "Error(MAX_THREADS set to " << MAX_THREADS << " so cores limited to this value)\n";
        thread_cfg.threads = MAX_THREADS;
	game.ts.initialize_extra_threads();
      } else if(thread_val < 1) {
	cout << "Error(thread_cfg.threads must be at least 1, no change made)\n";
	proto.logfile << "Error(thread_cfg.threads must be at least 1, no change made)\n";
      } else {
	thread_cfg.threads = thread_val;
	game.ts.initialize_extra_threads();
      }
      argi += 1;
      continue;
    }
    // set hash size in MB
    // FORMAT:  hash <size_in_MB>   
    if(!strcmp(argv[argi], "hash")) { 
      engine_cfg.hash_size = ABS(atoi(argv[argi+1]));
      set_hash_size(engine_cfg.hash_size);
      argi += 1;
      continue;
    }
    // command line score value change, use centipawn as smallest unit
    // FORMAT:  setvalue <parameter_name> <parameter_value>
    if(!strcmp(argv[argi], "setvalue")) { 
      set_score_value(argv[argi+1], atof(argv[argi+2]));
      argi += 2; 
      continue;
    }
    // test command has to be last on command line, will exit after
    // FORMAT:  test <epd_file> <results_file> <time_in_seconds> <depth_limit>
    if(!strcmp(argv[argi], "test")) {
      game.test_suite(argv[argi+1], argv[argi+2], atof(argv[argi+3]), atoi(argv[argi+4]));
      close_hash();
      return 0;
    }
  }

#if FLTK_GUI
#if UNIX
  /* throw away cout output */
  std::streambuf *nullbuf;
  ofstream nullstr;
  nullstr.open("/dev/null");
  nullbuf = nullstr.rdbuf();
  cout.rdbuf(nullbuf);
#endif
  /* Start main loop for GUI interface */
  return FLTKmain(argc,argv);
#endif

  // Suppress proto.xboard startup signal setup here if already set via flag
  if (proto.xboard && !proto.uci_mode) {
    signal(SIGINT, SIG_IGN);
  }

  // Protocol auto-detection (only if protocol not already set via flags)
  if (!proto.uci_mode && !proto.xboard) {
    std::string first_line;
    if (std::getline(std::cin, first_line)) {
      std::istringstream iss(first_line);
      std::string first_token;
      iss >> first_token;
      if (first_token == "uci") {
        proto.uci_mode = 1;
        proto.interface_mode = 1;
      } else if (first_token == "xboard") {
        proto.xboard = 1;
        proto.interface_mode = 1;
        signal(SIGINT, SIG_IGN);
      } else {
        // CLI mode: print banner then process this line as the first command
        cout << "\nLeaf Chess Engine version " << VERS << VERS2 << " (beta),"
             << "\nCopyright (C) 1997-2026 Daniel C. Homan, Granville OH, USA"
             << "\nLeaf comes with ABSOLUTELY NO WARRANTY. This is free"
             << "\nsoftware, and you are welcome to redistribute it under"
             << "\ncertain conditions. This program is distributed under the"
             << "\nGNU public license.  See the file license.txt"
             << "\nfor more information.\n\n";
        cout << "Hash size = " << TAB_SIZE << " buckets of 4 entries, "
             << TAB_SIZE*sizeof(hash_bucket)/1048576 << " Mbytes\n";
        cout << "Pawn size = " << PAWN_SIZE << " individual entries, "
             << PAWN_SIZE*sizeof(pawn_rec)/1048576 << " Mbytes\n";
        cout << "Score size = " << SCORE_SIZE << " individual entries, "
             << SCORE_SIZE*sizeof(score_rec)/1048576 << " Mbytes\n";
        cout << "Cmove size = " << CMOVE_SIZE << " individual entries, "
             << CMOVE_SIZE*sizeof(cmove_rec)/1048576 << " Mbytes\n\n";
        cout << "Type 'help' for a list of commands.\n";
        if(MATERIAL_ONLY) {
            cout << "\nWARNING -- Material-only eval mode active.\n";
        }
        strncpy(response, first_token.c_str(), sizeof(response)-1);
        response[sizeof(response)-1] = '\0';
        parse_command();
      }
    }
  }

  // If UCI mode, run the UCI loop and exit cleanly
  if (proto.uci_mode) {
    uci_loop(&game);
    if(proto.logging) proto.logfile.close();
#if TDLEAF && NNUE && !TDLEAF_READONLY
    if (nnue_available) {
      char tdleaf_save[FILENAME_MAX];
      snprintf(tdleaf_save, sizeof(tdleaf_save), "%s%s",
               engine_cfg.exec_path, NNUE_TDLEAF_BIN);
      tdleaf_flush_batch(tdleaf_save);
    }
#endif
    close_hash();
    return 0;
  }

  /* main loop for text interface */

  while (game.program_run)
   {

    // find a hint move, check book first then look in pv
    if(hintflag) {
      hint.t = 0;
      if(game.ts.last_ponder) hint = game.ts.ponder_move;
      else if(game.book) hint = opening_book(game.pos.hcode, game.pos, &game);
      if(!hint.t) hint = game.ts.tdata[0].pc[0][1];
      if(hint.t) {
	game.pos.print_move(hint, mstring, &game.ts.tdata[0]);
       cout << "Hint: " << mstring << "\n";
      }
      hintflag = 0;
    }

    // pondering if possible
    if(game.T > 2 && game.p_side == game.pos.wtm && !game.over
       && !game.both && !game.ts.last_ponder && !game.force_mode && ponder_flag)
    {
      if(!proto.xboard) cout << "pondering... (press any key to interrupt)\n";
      cout.flush();
      game.ts.ponder = 1;
      game.ts.search(game.pos, 1, game.T+1, &game);
      game.ts.ponder = 0;
      game.ts.last_ponder = 1;
    }

    // if analysis_mode, do some analysis
    if(game.ts.analysis_mode && !game.over && !game.force_mode) {
      game.p_side = !game.pos.wtm;
      game.ts.search(game.pos, 36000000, game.T, &game);
      game.p_side = game.pos.wtm;
    }

    // if we received a quit command while pondering or analyzing
    if(!game.program_run) break;

    if(!game.pos.wtm)                        // if it is black's turn
    {
     if(game.both) game.p_side = 0;
     if(!proto.xboard) cout << "Black-To-Move[" << floor((double)game.T/2) << "]: ";
     if(proto.logging) proto.logfile << "Black-To-Move[" << floor((double)game.T/2) << "]: ";
    }
    else                                         // or if it is white's
    {
     if(game.both) game.p_side = 1;
     if(!proto.xboard) cout << "White-To-Move[" << (floor((double)game.T/2) + 1) << "]: ";
     if(proto.logging) proto.logfile << "White-To-Move[" << (floor((double)game.T/2) + 1) << "]: ";
    }

    cout.flush();

    // process any move that happened while during pondering or analysis
    if(game.process_move) { make_move(); game.T++; game.process_move = 0; continue; }

    if(game.p_side == game.pos.wtm || game.over || game.force_mode) {
      if (!(cin >> response)) break;  // EOF (pipe closed by GUI) — exit cleanly
      if((game.ts.last_ponder || game.ts.analysis_mode) && UNIX) cout << "\n";
      parse_command();      // parse it
    } else {
      if(!proto.xboard) cout << "Thinking ...\n";
      if(proto.logging) proto.logfile << "Thinking ...\n";
      cout.flush();
      make_move();
      game.ts.last_ponder = 0;
      game.T++;
    }

    cout.flush();
    if(proto.logging) proto.logfile.flush();
   }

  if(proto.logging) proto.logfile.close();

#if TDLEAF && NNUE && !TDLEAF_READONLY
  if (nnue_available) {
    char tdleaf_save[FILENAME_MAX];
    snprintf(tdleaf_save, sizeof(tdleaf_save), "%s%s", engine_cfg.exec_path, NNUE_TDLEAF_BIN);
    tdleaf_flush_batch(tdleaf_save);
  }
#endif

  close_hash();

  return 0;
}

// Function to takeback moves
// tm is the number of moves to take back.
// 1 or 2 with current setup
void takeback(int tm)
{
 int temp_turn = game.T;
 // no book learning yet
 game.learn_count = 0; game.learned = 0;
 // game is not over
 game.over = 0;
 game.pos = game.reset;
 game.T = temp_turn; if(!(game.T % 2)) game.p_side = 0;
 if(game.p_side == 0 && tm == 1) game.p_side = 1;
 for (int ip = 0; ip <= game.T-2-tm; ip++)
 {
  game.pos.exec_move(game.game_history[ip], 0);
 }
 if(!proto.xboard && !FLTK_GUI) drawboard();
 game.T = game.T - tm;
 // find quasi-legal moves
 game.pos.allmoves(&game.movelist, &game.ts.tdata[0]);     
}


// Function to make the next move... If it is the computer's turn, this
// function calls the search algorithm, takes the best move given by that
// search, and makes the move - unless it is a check move: then it flagges
// stale-mate.....
// The function also looks to see if this move places the opponent in check
// or check-mate.

void make_move()
{
   int mtime = GetTime(); int time_limit, legal, ri;
   char mstring[10]; int time_div = 73; int rep_count = 0;
   char outstring[400];

   if (game.p_side != game.pos.wtm)
   {
     //------------------------------
     // Estimate search time to use
     //------------------------------
     if(proto.interface_mode || game.mttc) {
       // Estimate the amount of time remaining to time control
       int projected_time = game.timeleft[game.pos.wtm];
       int moves_remaining = game.mttc+1;
       if(!game.mttc || moves_remaining > 40) {
	 moves_remaining = 40;
       }
       projected_time += moves_remaining*game.inc*100;
       if(interface_lag_count >= LAG_COUNT) {
         projected_time -= moves_remaining*average_lag;
       }
       // Don't let lags reduce projected time by more than half
       projected_time = MAX(projected_time, game.timeleft[game.pos.wtm]/2);
       // Base limit used on this move on the projected time
       time_limit = 75*projected_time/(100*moves_remaining);
       if(ponder_flag) { time_limit = (115*time_limit)/100; }
       // if opponent made the expected move, reduce the time...
       /*
       if(game.ts.tdata[0].pc[0][1].t == game.pos.last.t && game.ts.tdata[0].pc[0][1].t) {
	 time_limit = 100*time_limit/100;
       }
       */
       // Use no more than half of the time left
       time_limit = MIN(time_limit, game.timeleft[game.pos.wtm]/2);
       // Reserve a fixed overhead buffer for small-increment time controls
       // to account for interface latency and OS scheduling jitter per move
       if(game.inc > 0.0f && game.inc < 0.10f) {
         time_limit = MAX(1, time_limit - search_cfg.move_overhead_cs);
       }
     } else {
       time_limit = game.timeleft[game.pos.wtm];
     }
     //---------------------------
     // Now complete the search
     //---------------------------
     game.best = game.ts.search(game.pos, time_limit, game.T, &game);
     assert(!game.searching);
#if TDLEAF && NNUE && !TDLEAF_READONLY
     if (nnue_available) {
       tdleaf_record_ply(game.td_game,
                         game.pos,
                         game.ts.tdata[0].n[0].acc,
                         game.ts.tdata[0].pc[0],
                         game.ts.g_last,
                         game.ts.id_scores,
                         game.ts.id_score_count,
                         game.ts.last_depth);
     }
#endif
     //---------------------------
     // Adjust remaining time
     //---------------------------
     game.timeleft[game.pos.wtm] -= float(GetTime() - mtime); 
     game.timeleft[game.pos.wtm] += float(game.inc*100);
   }

   // reduce time control by one move after every pair of moves in the game, should avoid the
   // winboard 'bug' introduced when forced moves are made by at the start of the game for the engine
   if(!(game.T&1)) {
    if(game.mttc) { 
      game.mttc--; 
      if(!game.mttc) { 
	game.timeleft[0] += game.base*100; 
	game.timeleft[1] += game.base*100; 
	game.mttc = game.omttc; 
      } 
    }
    if(game.mttc <= 0 && !proto.interface_mode) {
      game.timeleft[0] = game.base*100;
      game.timeleft[1] = game.base*100;
      game.mttc = game.omttc; 
    }
   }

   // execute the move....
   game.temp = game.pos;
   legal = game.temp.exec_move(game.best, 0);

   // if game isn't over, make the move
   if(!game.over) {
    
    // Is the move legal? if not Error ....
    if (legal) {
      game.pos.print_move(game.best, mstring, &game.ts.tdata[0]);
      strcpy(game.lmove,mstring);
      // if it is the computer's turn - echo the move
      if(game.p_side != game.pos.wtm || FLTK_GUI) {
	if(!proto.xboard) {
	  if(game.pos.wtm) {
	    cout << (ceil((double)game.T/2) + 1) << ". ";
	    snprintf(outstring, sizeof(outstring), "%i. ", (int(((double)game.T)/2) + 1));
	    write_out(outstring);
	  } else {
	    cout << ceil((double)game.T/2) << ". ... ";
	    snprintf(outstring, sizeof(outstring), "%i. ... ", int(((double)game.T)/2));
	    write_out(outstring);
	  }
	} else {
	  cout << "move ";
	}
        cout << mstring << "\n"; cout.flush();
        write_out(mstring); write_out("\n");
      }

      game.last = game.pos;        // Save last position
      game.pos = game.temp;        // actually execute move

      // Check if we have, check_mate, stale_mate, or a continuing game...
      switch (game.pos.in_check_mate()) {
       case 0:
         if(game.pos.fifty >= 100) { 
           game.over = 1;
	   cout << "1/2-1/2 {50 moves}\n";
           snprintf(game.overstring, sizeof(game.overstring), "1/2-1/2 {50 moves}");
	   write_out(game.overstring);
           if(ics) cout << "tellics draw\n"; 
	 } else if(game.pos.in_check() && !proto.xboard) {
	   cout << "Check!\n"; 
	 }
         // check for a 3-rep
         for(ri = game.T-2; ri >= game.T-game.pos.fifty && rep_count < 2; ri -= 2)
          if(game.ts.tdata[0].plist[ri] == game.pos.hcode) {
           rep_count++;
           if(rep_count > 1) {
	     game.over = 1;
	     cout << "1/2-1/2 {3-rep}\n";
	     snprintf(game.overstring, sizeof(game.overstring), "1/2-1/2 {3-rep}");
	     write_out(game.overstring);
	     if(ics) cout << "tellics draw\n";
           }
          }    
         break;
       case 1:
         game.over = 1;
         if(!game.pos.wtm) { 
	   cout << "1-0 {White Mates}\n";
           snprintf(game.overstring, sizeof(game.overstring), "1-0 {White Mates}");
	   write_out(game.overstring);
         } else {
	   cout << "0-1 {Black Mates}\n";
           snprintf(game.overstring, sizeof(game.overstring), "0-1 {Black Mates}");
	   write_out(game.overstring);
	 }
         break;
       case 2:
         game.over = 1;
         cout << "1/2-1/2 {Stalemate}\n";
	 snprintf(game.overstring, sizeof(game.overstring), "1/2-1/2 {Stalemate}");
	 write_out(game.overstring);
      }
      // Flush stdout before TDLeaf writes to stderr; without this the result
      // line stays in the stdout pipe buffer while the stderr TDLeaf message
      // is already visible to the driving script, causing it to miss the message.
      if (game.over) cout.flush();

      game.game_history[game.T-1] = game.best; // record the move in the history list
      // update position list for all threads
      for(int ti=0; ti<MAX_THREADS;ti++) {
	game.ts.tdata[ti].plist[game.T] = game.pos.hcode;
      }
      if(!proto.xboard && display_board) drawboard();  // draw the resulting board
    } else { 
      game.over = 1; 
      cout << "Error - please reset"; 
      snprintf(game.overstring, sizeof(game.overstring), "Error - please reset\n");
      write_out(game.overstring);
    }
   } 

   // update the quasi-legal moves in this situation
   game.pos.allmoves(&game.movelist, &game.ts.tdata[0]);

#if TDLEAF && NNUE && !TDLEAF_READONLY
   // If the game just ended via checkmate/stalemate/draw, trigger TDLeaf update.
   if (game.over && nnue_available && game.td_game.n_plies > 0) {
     // Skip early 3-rep draws — degenerate repetition cycles provide noisy gradients
     if (strstr(game.overstring, "3-rep") && game.td_game.n_plies < TDLEAF_MIN_PLIES_REP) {
       fprintf(stderr, "[TDLeaf] Skipping early 3-rep draw (%d plies < %d)\n",
               game.td_game.n_plies, TDLEAF_MIN_PLIES_REP);
       game.td_game.n_plies      = 0;
       game.td_game.engine_color = -1;
     } else {
       float td_result = 0.5f;
       if (strstr(game.overstring, "1-0"))  td_result = 1.0f;
       else if (strstr(game.overstring, "0-1")) td_result = 0.0f;
       char tdleaf_save[FILENAME_MAX];
       snprintf(tdleaf_save, sizeof(tdleaf_save), "%s%s", engine_cfg.exec_path, NNUE_TDLEAF_BIN);
       tdleaf_update_after_game(game.td_game, td_result, tdleaf_save);
       tdleaf_replay(game.td_game, td_result, tdleaf_save);
       game.td_game.n_plies      = 0;  // prevent double-trigger
       game.td_game.engine_color = -1;
     }
   }
#endif

}

// This function draws the graphical board in a very simple way
void drawboard()
{
  char mstring[10];     // character string to hold move

 // the following for loop steps through the board and paints each square
  for (int j = 7; j >= 0; j--)
  {
   cout << "\n  +---+---+---+---+---+---+---+---+\n" << (j+1) << " | ";
   for (int i = 0; i <= 7; i++)
   {
     if(PTYPE(game.pos.sq[SQR(i,j)]) && !PSIDE(game.pos.sq[SQR(i,j)])) 
       cout << "\b<" << name[PTYPE(game.pos.sq[SQR(i,j)])] << ">| ";
     else if(PSIDE(game.pos.sq[SQR(i,j)]) == 1) cout << name[PTYPE(game.pos.sq[SQR(i,j)])] << " | ";
    else if(!((i+j)&1)) cout << "\b:::| ";
    else cout << "  | ";
   }
   if(j==7) { if(game.pos.wtm) cout << "   White to move";
                          else cout << "   Black to move"; }
   if(j==6) {
     cout << "   castle: ";
     if(game.pos.castle&1) cout << "K";
     if(game.pos.castle&2) cout << "Q";
     if(game.pos.castle&4) cout << "k";
     if(game.pos.castle&8) cout << "q";
     if(!game.pos.castle)  cout << "-";
   }
   if(j==5 && game.pos.ep)
     cout << "   ep: " << char(FILE(game.pos.ep) + 97) << (RANK(game.pos.ep) + 1);
   if(j==4 && game.pos.last.t) {
     cout << "   last: ";
     game.last.print_move(game.pos.last, mstring, &game.ts.tdata[0]);
     cout << mstring;
    }
   if(j==3) cout << "   fifty: " << ceil((double)game.pos.fifty/2);
   if(j==2) cout << "   Computer time: " << int(game.timeleft[game.p_side^1]/100) << " seconds";
  }
   cout << "\n  +---+---+---+---+---+---+---+---+";
   cout << "\n    a   b   c   d   e   f   g   h  \n\n";
}


// Help function
void help()
{

 cout <<   "\n Commands ........ ";
 cout << "\n\n   Enter a move in standard algebraic notation,";
 cout <<   "\n      Nf3, e4, O-O, d8=Q, Bxf7, Ned7, etc....";
 cout <<   "\n      Other notation's like: g1f3, e2e4, etc... are also ok.";
 cout << "\n\n   new            -> start a new game";
 cout <<   "\n   quit           -> end Leaf";
 cout <<   "\n   save           -> save the game to a text file";
 cout <<   "\n   go             -> computer takes side on move";
 cout <<   "\n   white          -> white to move, Leaf takes black";
 cout <<   "\n   black          -> black to move, Leaf takes white";
 cout <<   "\n   book           -> toggle opening book";
 cout <<   "\n   proto.post           -> turn on display of computer thinking";
 cout <<   "\n   nopost         -> turn off display of computer thinking";
 cout <<   "\n   setboard rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -";
 cout <<   "\n                  -> setup board using EPD/FEN notation";
 cout <<   "\n                     The last two fields are castling rights and";
 cout <<   "\n                      the en-passant square, if possible. If";
 cout <<   "\n                      either is not, use a '-' instead.";
 cout <<   "\n   level 40 5 0   -> set level of play:";
 cout <<   "\n                       1st number is the number of move till time control";
 cout <<   "\n                       2nd number is the base time control in minutes";
 cout <<   "\n                       3rd number is the increment in seconds";
 cout <<   "\n   takeback       -> takeback last move";
 cout <<   "\n------------ Hit \"Enter\" for remaining commands: ";
 cin.get(); cin.get();
 cout <<   "\n   hint           -> get a hint from the program";
 cout <<   "\n   testsuite      -> run a testsuite";
 cout <<   "\n   display        -> display the board";
 cout <<   "\n   nodisplay      -> turn off board display";
 cout <<   "\n   list           -> list the legal moves";
 cout <<   "\n   clist          -> list the legal captures";
 cout <<   "\n   score          -> score the current position";
 cout <<   "\n   analyze        -> enter analysis mode";
 cout <<   "\n   exit           -> exit analysis mode";
 cout <<   "\n   ponder         -> toggle pondering";
 cout <<   "\n   hash n         -> set total hash to n megabytes";
 cout <<   "\n   build          -> build a new opening book from a pgn file";
 cout <<   "\n   edit_book      -> directly edit the current opening book";

 cout << "\n\n";
}

/* Function to print the possible moves to the screen */
// Useful for debugging
void type_moves()
{
  int j = 0;   // dummy count variable to determine when to
               // send a newline character
  char mstring[10]; // character string for the move

  for(int i = 0; i < game.movelist.count; i++) {
      game.temp = game.pos;
      // if it is legal, print it!
      if(game.temp.exec_move(game.movelist.mv[i].m, 0)) {
         if(!(j%6) && j) cout << "\n";    // newline if we have printed
                                          // 6 moves on a line
         else if(j) cout << ", ";         // comma to separate moves
         game.pos.print_move(game.movelist.mv[i].m, mstring, &game.ts.tdata[0]);   // print the move
	                                                                  // to the string
         cout << mstring;
         j++;                              // increment printed moves variable
      }
  }
  cout << "\n";
}

/* Function to print out the possible captures to the screen */
// Useful for debugging
void type_capts()
{
  int j = 0;              // dummy variable for counting printed moves
  char mstring[10];       // character string to hold move
  move_list clist;        // capture list

  game.pos.captures(&clist, -10000);
  for(int i = 0; i < clist.count; i++) {
      game.temp = game.pos;
      // if it is legal, print it!
      if(game.temp.exec_move(clist.mv[i].m, 0)) {
         if(!(j%6) && j) cout << "\n";    // newline if we have printed
                                          // 6 moves on a line
         else if(j) cout << ", ";         // comma to separate moves
         game.pos.print_move(clist.mv[i].m, mstring, &game.ts.tdata[0]);         // print the move
                                                                        // to the string
         cout << mstring;
         j++;                              // increment printed moves variable
      }
  }
  cout << "\n";
}


// ------------ function to write stuff to proto.logfile or to search posting buffer 
//
void write_out(const char *outline) {
  pthread_mutex_lock(&thread_cfg.log_lock);
  if(proto.logging) proto.logfile << outline;
#if FLTK_GUI
  if(FLTK_post) { 
   searchout_buffer->append(outline);
   searchout->move_down();
   searchout->show_insert_position();
  }
#endif
  pthread_mutex_unlock(&thread_cfg.log_lock);
}











