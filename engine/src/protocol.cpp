// protocol.cpp — proto.xboard/CECP and interactive CLI command dispatch + interrupt
// polling.  Extracted from main.cpp for readability.
//
// Unity-build position: after main.cpp (needs game, response[], protocol
// globals), before uci.cpp.
// ===========================================================================

/* Function to parse the command from the user */
// Some of these commands are proto.xboard/winboard specific

void parse_command()
{
  char inboard[256], ms, castle[5], ep[3], basestring[12];
  char outstring[400];
  char testfile[100] = "";
  char resfile[100] = "";
  char * options;
  char line[256];
  int rating1, rating2, Mbytes, fsq, tsq, protN;
  int interface_time;
  int computed_lag;
  float min = 0.0, sec = 0.0;

  // proto.logging initial command
  if(proto.logging) proto.logfile << "Parsing command: " << response << "\n";

  // default is to assume command will terminate an active search
  game.terminate_search = 1;

  // if the command is an instruction
  if(!strcmp(response, "level"))
   { cin >> game.mttc >> basestring >> game.inc;
     sscanf(basestring, "%f:%f", &min, &sec);
     game.base = min*60.0+sec;
     game.omttc = game.mttc; 
     game.timeleft[0] = game.base*100; 
     game.timeleft[1] = game.base*100; }
  else if(!strcmp(response, "time")) { 
    cin >> interface_time; 
    computed_lag = game.timeleft[game.p_side^1] - interface_time;
    game.timeleft[game.p_side^1] = interface_time;
    //-------------------------------------------------
    // Keep a running list of up to LAG_COUNT lags,
    //  --> next_lag_mod variable tells which part of
    //      this array the next lag should replace,
    //      tis allows a continuously updated list
    //  --> interface_lag_count, count of lags
    //      recorded so far, up to LAG_COUNT
    //-------------------------------------------------
    interface_lags[next_lag_mod] = computed_lag;
    next_lag_mod++;
    if(next_lag_mod == LAG_COUNT) {
      next_lag_mod = 0;
    }
    if(interface_lag_count < LAG_COUNT) {
      interface_lag_count++;
    }
    //-------------------------------------------------
    // Compute the average lag to use in time alottment
    //   -- computed only over most recent LAG_COUNT
    //   -- max and min lags are discarded in average
    //      to give a more robust result 
    //-------------------------------------------------
    average_lag = 0;
    int max_lag = -1000; int min_lag = 1000;
    for(int li = 0; li < interface_lag_count; li++) {
      average_lag += interface_lags[li];
      if(interface_lags[li] > max_lag) {
	max_lag = interface_lags[li];
      }
      if(interface_lags[li] < min_lag) {
	min_lag = interface_lags[li];
      }
    }
    if(interface_lag_count > 2) {
      average_lag -= (max_lag+min_lag);
      average_lag /= (interface_lag_count-2);
    } else {
      average_lag /= interface_lag_count;
    }
    average_lag = MAX(average_lag, 0);  // don't have a lag < 0
    //-------------------------------------------------
    // Report lag statistics to log file
    //-------------------------------------------------
    snprintf(outstring, sizeof(outstring), "Lag measured to be %3i centi-sec, running <lag> = %3i\n", computed_lag, average_lag);
    write_out(outstring);
  }
  else if(!strcmp(response, "otim")) { cin >> game.timeleft[game.p_side];}
  else if(!strcmp(response, "display") && !proto.xboard)
   { display_board = 1; drawboard(); }
  else if(!strcmp(response, "nodisplay") && !proto.xboard)
   { display_board = 0; }
  else if(!strcmp(response, "force")) { game.both = 1; game.force_mode = 1; }
  else if(!strcmp(response, "black"))
   { game.pos.wtm = 0; game.p_side = 0; game.both = 0;
     game.pos.gen_code();
     game.pos.in_check();  }
  else if(!strcmp(response, "white"))
   { game.pos.wtm = 1; game.p_side = 1; game.both = 0;
     game.pos.gen_code();
     game.pos.in_check();  }
  else if(!strcmp(response, "go"))
    { game.p_side = game.pos.wtm^1; game.both = 0; game.force_mode = 0; game.ts.analysis_mode = 0;  }
  else if(!strcmp(response, "playother"))
    { game.p_side = game.pos.wtm; game.both = 0; game.force_mode = 0; game.ts.analysis_mode = 0;  }
  else if(!strcmp(response, "edit")) { game.board_edit(); }
  else if(!strcmp(response, "testsuite") && !proto.xboard) { game.test_suite(testfile,resfile,0, 0); }
  else if(!strcmp(response, "analyze"))
   { proto.post = 1; game.learn_bk = 0; game.ts.analysis_mode = 1; game.book = 0; game.both = 1;
     if(!proto.xboard) cout << "Analysis mode: Enter commands/moves as ready.\n\n";      
   }
  else if(!strcmp(response, "exit"))
    { cout << "Analysis mode: off\n"; game.ts.analysis_mode = 0;  }
  else if(!strcmp(response, "new"))
   { if(!game.ts.analysis_mode) { game.both = 0; game.book = 1; game.learn_bk = engine_cfg.book_learning; }
     game.ts.no_book = 0;
     game.ts.max_search_depth = MAXD;
     game.ts.max_search_time = MAXT;
     game.setboard(i_pos, 'w', "KQkq", "-");
     game.force_mode = 0;
  }
  else if(!strcmp(response, "?")) { 
    game.terminate_search = 1; 
  }
  else if(!strcmp(response, ".")) { 
      game.terminate_search = 0; 
      cout << "Error (unknown command): " << response << "\n"; 
      proto.logfile << "Error (unknown command): " << response << "\n"; 
  }
  else if(!strcmp(response, "draw")) { 
    // a draw request can/will terminate a ponder but not
    // a move search... note that if we don't terminate
    // a ponder, an immediate followup command from proto.xboard might be
    // missed leading to a search that doesn't terminate
    if(!game.ts.ponder) { game.terminate_search = 0; } 
  }
  else if(!strcmp(response, "remove"))
    { takeback(2); }
  else if(!strcmp(response, "takeback") && !proto.xboard)
    { takeback(2); }
  else if(!strcmp(response, "undo")) { takeback(1); }
  else if(!strcmp(response, "sd")) { cin >> game.ts.max_search_depth; }
  else if(!strcmp(response, "st")) { cin >> game.ts.max_search_time; }
  else if(!strcmp(response, "bk") || !strcmp(response, "book"))
   { if(game.book) { game.book = 0; cout << " Book off\n\n"; }
     else { game.book = 1; cout << " Book on\n\n"; } }
  else if(!strcmp(response, "hint")) { hintflag = 1; }
  else if(!strcmp(response, "edit_book") && !proto.xboard) {
   edit_book(game.pos.hcode, &game.pos);
  }
  else if(!strcmp(response, "shout")) { shout_book = 1; }
  else if(!strcmp(response, "post")) { proto.post = 1; }
  else if(!strcmp(response, "cores")) { 
    int thread_val;
    cin >> thread_val;
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
  }
  else if(!strcmp(response, "swap") && !proto.xboard) { 
    cin >> fsq >> tsq;  
    cout << swap(tsq,game.pos,game.pos.wtm,fsq) << "\n"; 
  }
  else if(!strcmp(response, "xboard")) {
    /* only do something if proto.xboard mode is not already set */
    if(!proto.xboard) {
      proto.xboard = 1;
      proto.interface_mode = 1;
      // catch signals for proto.xboard interface
      signal(SIGINT, SIG_IGN);
    }
  }
  else if(!strcmp(response, "variant")) { cin.getline(line,256); }
  else if(!strcmp(response, "protover")) {
    cin >> protN;
    if(protN > 1) {
      cout << "\n";
      cout << "feature setboard=1\n";
      cout << "feature playother=1\n";
      cout << "feature usermove=1\n"; 
      cout << "feature ics=1\n";
      cout << "feature smp=1\n";
      cout << "feature memory=1\n";
      cout << "feature variants=\"normal,nocastle,fischerandom\"\n";
      cout << "feature myname=\"Leaf v" << VERS << VERS2 << "\"\n";
      snprintf(outstring, sizeof(outstring), "feature option=\"Playing Strength -slider %i 1 100\n", game.knowledge_scale);
      cout << outstring;
      cout << "feature done=1\n";
      cout.flush();
    }
  }
  else if(!strcmp(response, "option")) { 
    cin.getline(line, 256);
    if(strstr(line, "=") != NULL) {    // Thanks to Alex Guerrero for this fix!
      options = strtok(line,"=");
      if(!strcmp(options, " Playing Strength")) {
	options = strtok(NULL,"\n");
	game.knowledge_scale = atoi(options);
      }
    }
  }
  else if(!strcmp(response, "build") && !proto.xboard) { 
    game.setboard(i_pos, 'w', "KQkq", "-");
    build_book(game.pos); 
  }
  else if(!strcmp(response, "memory") || !strcmp(response, "hash"))
    { cin >> Mbytes; set_hash_size(ABS(Mbytes));
      if(!proto.xboard) {
	cout << " Hash size = " << TAB_SIZE << " buckets of 4 entries, "
	     << TAB_SIZE*sizeof(hash_bucket)/1048576 << " Mbytes\n";
	cout << " Pawn size = " << PAWN_SIZE << " individual entries, "
	     << PAWN_SIZE*sizeof(pawn_rec)/1048576 << " Mbytes\n"; 
	cout << "Score size = " << SCORE_SIZE << " individual entries, "
	     << SCORE_SIZE*sizeof(score_rec)/1048576 << " Mbytes\n";
	cout << "Cmove size = " << CMOVE_SIZE << " individual entries, "
	     << CMOVE_SIZE*sizeof(cmove_rec)/1048576 << " Mbytes\n\n"; 
      }
    }
  //else if(!strcmp(response, "sum") || !strcmp(response, "ics"))
  // { ics = 1; if(!proto.xboard) cout << " Search summary is on\n\n"; }
  else if(!strcmp(response, "ponder") && !proto.xboard)
   { if(ponder_flag) { ponder_flag = 0; cout << " Pondering off\n\n"; }
     else { ponder_flag = 1; cout << " Pondering on\n\n"; } }
  else if(!strcmp(response, "easy")) // pondering off in proto.xboard/winboard
   { ponder_flag = 0; }
  else if(!strcmp(response, "hard")) // pondering on in proto.xboard/winboard
   { ponder_flag = 1; }
  else if(!strcmp(response, "list") && !proto.xboard) {  type_moves(); }
  else if(!strcmp(response, "clist") && !proto.xboard) { type_capts(); }
  else if(!strcmp(response, "print_psq") && !proto.xboard) { print_psq(); }
  else if(!strcmp(response, "score") && !proto.xboard)
    { game.p_side = game.pos.wtm^1;
#if NNUE
      if (nnue_available) {
        NNUEAccumulator tmp_acc;
        nnue_init_accumulator(tmp_acc, game.pos);
        cout << "score = " << game.pos.score_pos(&game, &game.ts.tdata[0], &tmp_acc) << "\n";
      } else
#endif
      cout << "score = " << game.pos.score_pos(&game,&game.ts.tdata[0]) << "\n";
      cout << "material = " << game.pos.material << "\n";
      game.p_side = game.pos.wtm; }
  else if(!proto.xboard && !strcmp(response, "help")) { help(); }
  else if(!strcmp(response, "nopost")) { proto.post = 0; }
  else if((!strcmp(response, "save") || !strcmp(response, "SR")) && !proto.xboard) { save_game(); }
  else if(!strcmp(response, "quit")) { game.over = 1; game.program_run = 0;  }
  else if(!strcmp(response, "result")) {
    game.over = 1;
#if TDLEAF && NNUE && !TDLEAF_READONLY
    if (nnue_available && game.td_game.n_plies > 0) {
      // proto.xboard sends: "result 1-0 {description}" or "0-1" or "1/2-1/2"
      char result_str[20]   = "";
      char result_desc[256] = "";
      cin >> result_str;
      cin.getline(result_desc, sizeof(result_desc));
      // Skip learning if the game ended on time — the result reflects clock
      // management, not chess quality.  Both cutechess-cli and XBoard include
      // the word "time" in the description for time-forfeit results
      // (e.g. "{Black forfeits on time}", "{White wins on time}").
      if (strstr(result_desc, "time") == nullptr) {
        float td_result = 0.5f;
        if (!strcmp(result_str, "1-0"))      td_result = 1.0f;
        else if (!strcmp(result_str, "0-1")) td_result = 0.0f;
        char tdleaf_save[FILENAME_MAX];
        snprintf(tdleaf_save, sizeof(tdleaf_save), "%s%s", engine_cfg.exec_path, NNUE_TDLEAF_BIN);
        tdleaf_update_after_game(game.td_game, td_result, tdleaf_save);
      }
      game.td_game.n_plies      = 0;  // always reset — prevent carry-over to next game
      game.td_game.engine_color = -1;
    }
#endif
  }
  else if(!strcmp(response, "setvalue")) { 
    char par_string[50]; float par_val;
    cin >> par_string >> par_val;
    set_score_value(par_string, par_val);
  }
  else if(!strcmp(response, "getvalue")) {
    char par_string[50];
    cin >> par_string;
    cout << par_string << ": "<< get_score_value(par_string) << "\n";;
  }
  else if(!strcmp(response, "performance") && !proto.xboard) { performance(); }
  else if(!strcmp(response, "history_stats") && !proto.xboard) 
    { game.ts.analysis_mode = 0; game.ts.history_stats(); }
  else if(!strcmp(response, "name")) 
    { cin.getline(line, 256); if(proto.logging) proto.logfile << "--> Opponent: " << line << "\n"; }
  else if(!strcmp(response, "ics")) 
    { ics = 1; cin.getline(line, 256); if(proto.logging) proto.logfile << "--> ICS Host: " << line << "\n"; }
  else if(!strcmp(response, "rating")) 
    { cin >> rating1 >> rating2; 
      if(proto.logging) proto.logfile << "--> My rating  : " << rating1 << "\n"
                          << "--> Opp. rating: " << rating2 << "\n"; }
  else if(!strcmp(response, "setboard"))
    { cin >> inboard >> ms >> castle >> ep; game.setboard(inboard, ms, castle, ep);  }
  else if(!strcmp(response, "accepted")) { cin.getline(line, 256); }  // ignore accepted command
  else if(!strcmp(response, "rejected")) { cin.getline(line, 256); }  // ignore rejected command
  // no-op: the interface already defaults to interactive when neither "uci"
  // nor "xboard" is the first command received
  else if(!strcmp(response, "interactive")) { }
  else if(!strcmp(response, "netinfo")) {
#if NNUE
    nnue_print_diag_info();
#else
    cout << "Classical evaluation build (no NNUE)\n";
#endif
  }
  // if command is a move
  else if(!strcmp(response, "usermove")) { 
    cin >> response; 
    cout << "Got Usermove!\n";
    game.best = game.pos.parse_move(response, &game.ts.tdata[0]);
    if(game.best.t) { 
      if(!game.searching) { make_move(); game.T++; }
      else { 
	game.process_move = 1; 	  
      }
    } else { 
      cout << "Illegal move: " << response << "\n"; 
      proto.logfile << "Illegal move: " << response << "\n"; 
      cin.getline(line, 256);
    }
    cout.flush();
  } 
  else { 
    game.best = game.pos.parse_move(response, &game.ts.tdata[0]);
    if(game.best.t) {    
      if(!game.searching) { make_move(); game.T++; }
      else { 
	game.process_move = 1;   
      }
    } else { 
      cout << "Error (unknown command): " << response << "\n"; 
      proto.logfile << "Error (unknown command): " << response << "\n"; 
      cin.getline(line, 256);
    }
    cout.flush();
  }
}

// Function to run a perfomance test on generating and making move
void performance()
{
 position perf_pos = game.pos;
 move_list perf_list;
 int gen_count = 0;
 int start_time = GetTime();
 int loop = 0, perfi;

 while(1) {
  perf_pos.allmoves(&perf_list, &game.ts.tdata[0]);   
  gen_count += perf_list.count;
  loop++; if(loop > 1000) { loop = 0; if(GetTime()-start_time > 500) break; }
 }

 cout << "Generated " << gen_count << " moves in " << float(GetTime()-start_time)/100 << " seconds\n";

 loop = 0; start_time = GetTime(); gen_count = 0;

 while(1) {
  perf_pos.allmoves(&perf_list, &game.ts.tdata[0]);   
  gen_count += perf_list.count;
  for(perfi = 0; perfi < perf_list.count; perfi++) {
   game.temp = perf_pos;
   game.temp.exec_move(perf_list.mv[perfi].m, 0);
  }   
  loop++; if(loop > 1000) { loop = 0; if(GetTime()-start_time > 500) break; }
 }

 cout << "Generated/Made/Unmade " << gen_count << " moves in " << float(GetTime()-start_time)/100 << " seconds\n";
}

// Save game function to save the game to a text file
void save_game()
{
  int TURN; TURN = game.T;
  char gname[] = "lastgame.gam";
  char resp, mstring[10];
  char Event[30], White[30], Black[30], Date[30], result[30];

  cout << "\nFile Name : ";
  cin >> gname;
  cout << "Custom Header? (y/n): ";
  cin >> resp;

  if(resp == 'y' || resp == 'Y')
  {
    cout << "Event: ";  cin >> Event;
    cout << "Date: "; cin >> Date;
    cout << "White: ";  cin >> White;
    cout << "Black: ";  cin >> Black;
  } else {
    strcpy(Event, "Chess Match");
    strcpy(Date, "??.??.????");
    if (game.p_side)
     { strcpy(White, "Human"); strcpy(Black, "Leaf"); }
    else
     { strcpy(White, "Leaf"); strcpy(Black, "Human"); }
  }

  ofstream outfile(gname);

  outfile <<   "[Event: " << Event << " ]";
  outfile << "\n[Date: " << Date << " ]";
  outfile << "\n[White: " << White << " ]";
  outfile << "\n[Black: " << Black << " ]";

  // set the result string
  switch (game.pos.in_check_mate())
   {
    case 0:
     if(game.pos.fifty >= 100)
      { strcpy(result, " 1/2-1/2 {50 moves}"); }
     else strcpy(result, " adjourned");
      break;
    case 1:
     if(!game.pos.wtm) strcpy(result, " 1-0 {White Mates}");
     else strcpy(result, " 0-1 {Black Mates}");
     break;
    case 2:
     strcpy(result, " 1/2-1/2 {Stalemate}");
   }

  outfile << "\n[Result: " << result << " ]\n\n";

  // set the board up from the starting position
  game.setboard(i_pos, 'w', "KQkq", "-");

  // play through the game and record the moves in a file
  for(int i = 1; i < TURN; i++)
   {
     game.pos.print_move(game.game_history[i-1], mstring, &game.ts.tdata[0]);
     if (game.pos.wtm) outfile << (ceil((double)i/2) + 1) << ". " << mstring;
     else outfile << mstring;
     outfile << " ";
     if(!(game.T%8)) outfile << "\n";
     game.pos.exec_move(game.game_history[i-1], 0);
     game.T++;
   }

   outfile << result;

   // update quasi-legal moves
   game.pos.allmoves(&game.movelist, &game.ts.tdata[0]);     

}

#if FLTK_GUI

#include <FL/Fl.H>
#include <FL/Fl_Double_Window.H>
//#include <FL/Fl_Bitmap.H>
//#include <FL/fl_draw.H>
//#include <FL/Fl_Menu_Item.H>
//#include <FL/fl_ask.H>

#endif

//-----------------------------------------------------------------------------------
// Function returns a 1 if a pondering session should be interrupted
int inter()
{
#if FLTK_GUI
  Fl::check();
  if(abortflag) { abortflag = 0; return 1; } else { return 0; }
#endif

  if (proto.uci_mode) {
    return uci_check_interrupt();
  }

  int interrupt = 0;
 
  if(!game.ts.ponder && !game.ts.analysis_mode && !proto.interface_mode) return 0;

  if(!proto.xboard && cin.rdbuf() -> in_avail() > 1) interrupt = 1;

  if(!interrupt) {
#if MSVC || MINGW  
    static int init = 0, pipe;
    static HANDLE inh;
    DWORD dw;
    if(proto.xboard) {     // winboard interrupt code taken from crafty
      if (!init) {
	init = 1;
	inh = GetStdHandle(STD_INPUT_HANDLE);
	pipe = !GetConsoleMode(inh, &dw);
	if (!pipe) {
	  SetConsoleMode(inh, dw & ~(ENABLE_MOUSE_INPUT|ENABLE_WINDOW_INPUT));
	  FlushConsoleInputBuffer(inh);
	  FlushConsoleInputBuffer(inh);
	}
      }
      if(pipe) {
	if(!PeekNamedPipe(inh, NULL, 0, NULL, &dw, NULL)) interrupt = 1;
	interrupt = dw;
      } else {
	GetNumberOfConsoleInputEvents(inh, &dw);
	dw <= 1 ? 0 : dw;
	interrupt = dw;
      }
    }
    if(kbhit()) interrupt = 1;

#else // unix

    FD_ZERO(&read_fds);
    FD_SET(0,&read_fds);
    timeout.tv_sec = timeout.tv_usec = 0;
    select(1,&read_fds,NULL,NULL,&timeout);
    if((game.ts.ponder || game.ts.analysis_mode || proto.interface_mode) && FD_ISSET(0,&read_fds)) interrupt = 1;

#endif
  }

  if(interrupt && proto.xboard) {
    if (!(cin >> response)) {
      // EOF: pipe closed by GUI — stop search so main loop can exit cleanly
      game.terminate_search = 1;
      proto.logfile << "Search Interrupted: stdin EOF\n";
    } else {
      parse_command();
      if(!game.terminate_search) {
        interrupt = 0;
      } else {
        proto.logfile << "Search Interrupted by GUI command\n";
      }
    }
  }

  return interrupt;
}
