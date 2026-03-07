// EXchess source code, (c) Daniel C. Homan  1997-2017
// Released under the GNU public license, see file license.txt
 
// Functions for game_rec class
//----------------------------------------------------------------
//  game_rec();
//  void setboard(const char inboard[256], const char ms, const char castle[5], const char ep[3]);
//  void test_suite(const char *, const char *, float, int); 
//  void board_edit();
//  void build_fen_list();
//  void train_eval();
//----------------------------------------------------------------

#include <cstdlib>

#include "define.h"
#include "chess.h"
#include "const.h"
#include "hash.h"

game_rec::game_rec() {

	book = 1;
	both = 0;
	mttc = 0;
	omttc = 0;
	inc = 0;
	base = 5.0;
	timeleft[0] = 500.0;
	timeleft[1] = 500.0;
        learn_bk = 0;
	knowledge_scale = 100;
	searching = 0;
	process_move = 0;
	terminate_search = 0;
	force_mode = 0;
	program_run = 1;

#if TDLEAF
	td_game.n_plies = 0;
#endif

	/* tree search parameters */

        // declared in initializer of tree_search

        // NOTE: thread and board data need to be initialized after 
	//       the game object is created... see start of main() in main.cpp
}

//-----------------------------------------------------------
// This function sets up the board from EPD format
//-----------------------------------------------------------
void game_rec::setboard(const char inboard[256], const char ms, const char castle[5], const char ep[3])
{
  int rx = 0, ry = 7, i;  // control variables

  // no book learning yet
  learn_count = 0; learned = 0;

  // game is not over
  over = 0;

#if TDLEAF
  td_game.n_plies = 0;
#endif

  // reset time control variables
  mttc = omttc;
  timeleft[0] = base*100.0;
  timeleft[1] = base*100.0;

  // Side to move
  if (ms == 'b') { T = 2; p_side = 0; pos.wtm = 0; }
  else { T = 1; p_side = 1; pos.wtm = 1; }

  // Other game parameters
  pos.fifty = 0; pos.last.t = NOMOVE;
  pos.hmove.t = 0;
  pos.rmove.t = 0;
  pos.cmove.t = 0;

  // en-passant
  if(ep[0] != '-') {
    int ex, ey;
    ex = int(ep[0])-97;
    ey = atoi(&ep[1])-1;
    pos.ep = SQR(ex,ey);
    //cout << "ep square is " << ex << " " << ey << " " << pos.ep << "\n";
  }
  else pos.ep = 0;

  // clear the board
  for(int ci = 0; ci < 64; ci++) { pos.sq[ci] = EMPTY; }

  // initialize game stage -- will be modified by gen_code() below
  pos.gstage = 16;  

  int wking_file = -1;
  int bking_file = -1;

  // Setting up the board
  for (int ri = 0; ri < 256; ri++)
  {
    switch (inboard[ri])
    {
      case '/': ry--; rx = 0; break;
      case '1': pos.sq[SQR(rx,ry)] = EMPTY; rx++; break;
      case '2': for(i=1;i<=2;i++) { pos.sq[SQR(rx,ry)] = EMPTY; rx++; } break;
      case '3': for(i=1;i<=3;i++) { pos.sq[SQR(rx,ry)] = EMPTY; rx++; } break;
      case '4': for(i=1;i<=4;i++) { pos.sq[SQR(rx,ry)] = EMPTY; rx++; } break;
      case '5': for(i=1;i<=5;i++) { pos.sq[SQR(rx,ry)] = EMPTY; rx++; } break;
      case '6': for(i=1;i<=6;i++) { pos.sq[SQR(rx,ry)] = EMPTY; rx++; } break;
      case '7': for(i=1;i<=7;i++) { pos.sq[SQR(rx,ry)] = EMPTY; rx++; } break;
      case '8': for(i=1;i<=8;i++) { pos.sq[SQR(rx,ry)] = EMPTY; rx++; } break;
      case 'p': pos.sq[SQR(rx,ry)] = 0;
	pos.sq[SQR(rx,ry)] += PAWN; rx++; break;
      case 'n': pos.sq[SQR(rx,ry)] = 0;
           pos.sq[SQR(rx,ry)] += KNIGHT; rx++; break;
      case 'b': pos.sq[SQR(rx,ry)] = 0;
           pos.sq[SQR(rx,ry)] += BISHOP; rx++; break;
      case 'r': pos.sq[SQR(rx,ry)] = 0;
           pos.sq[SQR(rx,ry)] += ROOK; rx++; break;
      case 'q': pos.sq[SQR(rx,ry)] = 0;
           pos.sq[SQR(rx,ry)] += QUEEN; rx++; break;
      case 'k': pos.sq[SQR(rx,ry)] = 0;
	   pos.sq[SQR(rx,ry)] += KING; bking_file=rx; rx++; break;
      case 'P': pos.sq[SQR(rx,ry)] = 8;
	pos.sq[SQR(rx,ry)] += PAWN; rx++; break;
      case 'N': pos.sq[SQR(rx,ry)] = 8;
           pos.sq[SQR(rx,ry)] += KNIGHT; rx++; break;
      case 'B': pos.sq[SQR(rx,ry)] = 8;
           pos.sq[SQR(rx,ry)] += BISHOP; rx++; break;
      case 'R': pos.sq[SQR(rx,ry)] = 8;
           pos.sq[SQR(rx,ry)] += ROOK; rx++; break;
      case 'Q': pos.sq[SQR(rx,ry)] = 8;
           pos.sq[SQR(rx,ry)] += QUEEN; rx++; break;
      case 'K': pos.sq[SQR(rx,ry)] = 8;
	   pos.sq[SQR(rx,ry)] += KING; wking_file = rx; rx++; break;
    }
   if(ry <= 0 && rx >= 8) break;
   if(inboard[ri] == '\0') break;
  }

  // initializing castling status
  // -- generalized for chess960, both X-FEN and SHREDDER-FEN
  pos.castle = 0;
  pos.Krook[WHITE] = -100;
  pos.Qrook[WHITE] = -100; 
  pos.Krook[BLACK] = -100; 
  pos.Qrook[BLACK] = -100;
  for(i = 0; i < 4; i++)  {
    if(castle[i] == '\0' || castle[i] == '-') {
      break;
    }
    switch (castle[i]) {
     case 'K':
       pos.Krook[WHITE] = -1;  // set after piece lists are made
       pos.castle = pos.castle^1; break;
     case 'Q':
       pos.Qrook[WHITE] = 64;  // set after piece lists are made
       pos.castle = pos.castle^2; break;
     case 'k':
       pos.Krook[BLACK] = -1;  // set after piece lists are made
       pos.castle = pos.castle^4; break;
     case 'q':
       pos.Qrook[BLACK] = 64;  // set after piece lists are made
       pos.castle = pos.castle^8; break;
     case 'A':
       if(wking_file < 0) { 
	 assert(1); break; // should be illegal
       } else {
	 pos.Qrook[WHITE] = 0;  
	 pos.castle = pos.castle^2; break;
       }
     case 'B':
       if(wking_file < 1) {
	 assert(1); break; // should be illegal
       } else {
	 pos.Qrook[WHITE] = 1;  
	 pos.castle = pos.castle^2; break;
       }
     case 'C':
       if(wking_file < 2) {
	 pos.Krook[WHITE] = 2;  
	 pos.castle = pos.castle^1; break;
       } else {
	 pos.Qrook[WHITE] = 2;  
	 pos.castle = pos.castle^2; break;
       }
     case 'D':
       if(wking_file < 3) {
	 pos.Krook[WHITE] = 3;  
	 pos.castle = pos.castle^1; break;
       } else {
	 pos.Qrook[WHITE] = 3;  
	 pos.castle = pos.castle^2; break;
       }
     case 'E':
       if(wking_file < 4) {
	 pos.Krook[WHITE] = 4;  
	 pos.castle = pos.castle^1; break;
       } else {
	 pos.Qrook[WHITE] = 4;  
	 pos.castle = pos.castle^2; break;
       }
     case 'F':
       if(wking_file < 5) {
	 pos.Krook[WHITE] = 5;  
	 pos.castle = pos.castle^1; break;
       } else {
	 pos.Qrook[WHITE] = 5;  
	 pos.castle = pos.castle^2; break;
       }
     case 'G':
       if(wking_file < 6) {
	 pos.Krook[WHITE] = 6;  
	 pos.castle = pos.castle^1; break;
       } else {
         assert(1); break; // should be illegal
       }
     case 'H':
       if(wking_file < 7) {
	 pos.Krook[WHITE] = 7;  
	 pos.castle = pos.castle^1; break;
       } else {
         assert(1); break; // should be illegal
       }
     case 'a':
       if(bking_file < 0) {
	 assert(1); break; // should be illegal
       } else {
	 pos.Qrook[BLACK] = 56;  
	 pos.castle = pos.castle^8; break;
       }
     case 'b':
       if(bking_file < 1) {
	 assert(1); break; // should be illegal
       } else {
	 pos.Qrook[BLACK] = 57;  
	 pos.castle = pos.castle^8; break;
       }
     case 'c':
       if(bking_file < 2) {
	 pos.Krook[BLACK] = 58;  
	 pos.castle = pos.castle^4; break;
       } else {
	 pos.Qrook[BLACK] = 58;  
	 pos.castle = pos.castle^8; break;
       }
     case 'd':
       if(bking_file < 3) {
	 pos.Krook[BLACK] = 59;  
	 pos.castle = pos.castle^4; break;
       } else {
	 pos.Qrook[BLACK] = 59;  
	 pos.castle = pos.castle^8; break;
       }
     case 'e':
       if(bking_file < 4) {
	 pos.Krook[BLACK] = 60;  
	 pos.castle = pos.castle^4; break;
       } else {
	 pos.Qrook[BLACK] = 60;  
	 pos.castle = pos.castle^8; break;
       }
     case 'f':
       if(bking_file < 5) {
	 pos.Krook[BLACK] = 61;  
	 pos.castle = pos.castle^4; break;
       } else {
	 pos.Qrook[BLACK] = 61;  
	 pos.castle = pos.castle^8; break;
       }
     case 'g':
       if(bking_file < 6) {
	 pos.Krook[BLACK] = 62;  
	 pos.castle = pos.castle^4; break;
       } else {
         assert(1); break; // should be illegal
       }
     case 'h':
       if(bking_file < 7) {
	 pos.Krook[BLACK] = 63;  
	 pos.castle = pos.castle^4; break;
       } else {
         assert(1); break; // should be illegal
       }
    } 
  } 

  // generate the hash_code for this position
  //  -- this also updates the piece lists and game_stage
  pos.gen_code();

  // Setup castling mask (generalized for chess960), see const.h
  for(i = 0; i < 64; i++) { castle_mask[i] = 15; }
  castle_mask[pos.plist[WHITE][KING][1]] = 12;
  castle_mask[pos.plist[BLACK][KING][1]] = 3;
  // set the minimum rook as castling queens rook if not specified by
  //  SHREDDER-FEN above
  if(pos.Qrook[WHITE] == 64) {
    for(i = 1; i <= pos.plist[WHITE][ROOK][0]; i++) {
      if(pos.plist[WHITE][ROOK][i] < pos.Qrook[WHITE] && RANK(pos.plist[WHITE][ROOK][i]) == 0) { 
	pos.Qrook[WHITE] = pos.plist[WHITE][ROOK][i];
      }
    }
  }
  if(pos.Qrook[BLACK] == 64) {
    for(i = 1; i <= pos.plist[BLACK][ROOK][0]; i++) {
      if(pos.plist[BLACK][ROOK][i] < pos.Qrook[BLACK] && RANK(pos.plist[BLACK][ROOK][i]) == 7) {
	pos.Qrook[BLACK] = pos.plist[BLACK][ROOK][i];
      }
    }
  }
  // set the maximum rook as castling kings rook if not specified by
  //  SHREDDER-FEN above
  if(pos.Krook[WHITE] == -1) {
    for(i = 1; i <= pos.plist[WHITE][ROOK][0]; i++) {
      if(pos.plist[WHITE][ROOK][i] > pos.Krook[WHITE] && RANK(pos.plist[WHITE][ROOK][i]) == 0) { 
	pos.Krook[WHITE] = pos.plist[WHITE][ROOK][i];
      }
    }
  }
  if(pos.Krook[BLACK] == -1) {
    for(i = 1; i <= pos.plist[BLACK][ROOK][0]; i++) {
      if(pos.plist[BLACK][ROOK][i] > pos.Krook[BLACK] && RANK(pos.plist[BLACK][ROOK][i]) == 7) {
	pos.Krook[BLACK] = pos.plist[BLACK][ROOK][i];
      }
    }
  }
  // Finish setting castling mask for rook squares
  if(pos.castle&1) { castle_mask[pos.Krook[WHITE]] = 14; }
  if(pos.castle&2) { castle_mask[pos.Qrook[WHITE]] = 13; }
  if(pos.castle&4) { castle_mask[pos.Krook[BLACK]] = 11; }
  if(pos.castle&8) { castle_mask[pos.Qrook[BLACK]] =  7; }

  //for(int i = 0; i < 64; i++) { cout << " " << castle_mask[i]; if(!((i+1)%8)) cout << "\n"; }

  // set some final parameters
  pos.qchecks[0] = 0;
  pos.qchecks[1] = 0;
  pos.in_check();

  // add position code to plist for all threads
  //  and clear all history tables
  for(int ti=0; ti<MAX_THREADS; ti++) { 
    ts.tdata[ti].plist[T-1] = pos.hcode; 
    if(T==2) ts.tdata[ti].plist[0] = 0ULL;
    // initialize history and reply table
    for(int i = 0; i < 15; i++)
      for(int j = 0; j < 64; j++) { 
	ts.tdata[ti].history[i][j] = 0; 
	ts.tdata[ti].reply[i][j] = 0; 
      }
  }
  reset = pos;

  // for search display in text mode
  ts.last_displayed_move.t = NOMOVE;

  // find the quasi-legal moves in this situation
  pos.allmoves(&movelist, &ts.tdata[0]);    


}

//-----------------------------------------------------------
// This function is a special edit mode for xboard/winboard
//  -- NOT compatitable with Chess960
//-----------------------------------------------------------
void game_rec::board_edit()
{
  char edcom[4];    // edit command
  int edside = 1;   // side being edited
  int ex, ey;       // edit coordinates

  // no book learning yet
  learn_count = 0; learned = 0;

  // game is not over
  over = 0; 

  // reset time control variables
  mttc = omttc;
  timeleft[0] = base*100.0;
  timeleft[1] = base*100.0;

  // Other game parameters
  pos.fifty = 0; pos.last.t = NOMOVE;
  pos.hmove.t = 0;
  pos.rmove.t = 0;
  pos.cmove.t = 0;

  // initialize game stage -- will be modified by gen_code() below
  pos.gstage = 16;  

  while(edside > -1) {
    cin >> edcom;
    if(edcom[0] == '#') {
      // clear the board
      for(int ci = 0; ci < 64; ci++) { pos.sq[ci] = EMPTY; }
    } else if(edcom[0] == 'c') {
      edside ^= 1;           // change side to edit
      continue;
    } else if(edcom[0] == '.') {
      edside = -1;  // exit edit mode
    } else {
      ex = CHAR_FILE(edcom[1]);
      ey = CHAR_ROW(edcom[2]);
      if(edside) pos.sq[SQR(ex,ey)] = 8;
      else pos.sq[SQR(ex,ey)] = 0;
      switch(edcom[0]) {
      case 'P':
	pos.sq[SQR(ex,ey)] += PAWN; break;
      case 'N':
	pos.sq[SQR(ex,ey)] += KNIGHT; break;
      case 'B':
	pos.sq[SQR(ex,ey)] += BISHOP; break;
      case 'R':
	pos.sq[SQR(ex,ey)] += ROOK; break;
      case 'Q':
	pos.sq[SQR(ex,ey)] += QUEEN; break;
      case 'K':
	pos.sq[SQR(ex,ey)] += KING;
	break;
      case 'X':
	pos.sq[SQR(ex,ey)] = EMPTY; break;
      }
    }
  }
  
  // setup castling rights... edit assumes castling is OK if
  // the king and rook are on their starting squares 
  if(ID(pos.sq[SQR(4,0)]) == WKING) {
    if(ID(pos.sq[SQR(7,0)]) == WROOK) pos.castle |= 1;  // kingside
    if(ID(pos.sq[SQR(0,0)]) == WROOK) pos.castle |= 2;  // queenside
  }  
  if(ID(pos.sq[SQR(4,7)]) == BKING) {
    if(ID(pos.sq[SQR(7,7)]) == BROOK) pos.castle |= 4;  // kingside
    if(ID(pos.sq[SQR(0,7)]) == BROOK) pos.castle |= 8;  // queenside
  }  
  
  // generate the hash_code for this position
  //  -- function also creates the piece lists and counts and game stage
  pos.gen_code();
  
  // set some final parameters
  pos.qchecks[0] = 0;
  pos.qchecks[1] = 0;
  pos.in_check();

  // add position code to plist for all threads
  //  -- also initialize history table
  for(int ti=0; ti<MAX_THREADS; ti++) { 
    ts.tdata[ti].plist[T-1] = pos.hcode; 
    // initialize history and reply table
    for(int i = 0; i < 15; i++)
      for(int j = 0; j < 64; j++) { 
	ts.tdata[ti].history[i][j] = 0; 
	ts.tdata[ti].reply[i][j] = 0; 
      }
  }
  reset = pos;

  // for search display in text mode
  ts.last_displayed_move.t = NOMOVE;

  // find the quasi-legal moves in this situation
  pos.allmoves(&movelist, &ts.tdata[0]);    

}

//------------------------------------------------------------------
// Function to run a test suite.  The function is 
// designed to work with the tree_search::search_display() function 
// to determine when the best move was first found and held on to.
//------------------------------------------------------------------
void game_rec::test_suite(char *testfile, char *resfile, float testtime, int fixed_depth)
{
  char testpos[256], ms, bookm[10], h1[5] = "KQkq", h2[3], h4[256], id[256];
  char reply = 'n';
  char mstring[10];
  char filein[100], fileout[100];
  int inter = 0, correct = 0, total = 0, dtotal = 0;
  unsigned int e; int wac = 0, stime[300], bmexit;
  float total_time_sq = 0, total_depth = 0;
  unsigned __int64 nodes = 0, test_time = 0, depth_time = 0;

  ts.bmcount = 0; ts.tsuite = 1;
  learn_bk = 0;
  // turn off opening book and turn on search posting
  book = 0; post = 1; 


  if(!testtime) {
    cout << "\nEnter file name for test suite in EPD format: ";
    cin >> filein; testfile = &filein[0];
    
    cout << "\nEnter file name for test results: ";
    cin >> fileout; resfile = &fileout[0];
    
    cout << "\nInteractive run? (y/n): "; cin >> reply;
    if (reply == 'y') inter = 1;
    else { cout << "\nEnter search time per move: "; cin >> testtime; }
    
    fixed_depth = MAXD;
  }

  if(!fixed_depth || fixed_depth > MAXD) fixed_depth = MAXD;

  // set a maximum search depth to be completed
  game.ts.max_search_depth = fixed_depth;
  
  cout << "\n--------------------*** " << testfile << " ***--------------------\n";
  if(!strcmp(testfile, "wac.epd")) wac = 1;

  ifstream infile(testfile, IOS_IN_TEXT);
  ofstream outfile(resfile);
  if (!(infile.is_open())) { cout << "\nUnable to open file. "; return; }
  if (!(outfile.is_open())) { cout << "\nUnable to open results file. "; return; }

  do
   {
    ts.soltime = -1;
    if (reply != 's') {
      //for(int j = 0; j < 4; j++) { h1[j] = '*'; }
      infile >> testpos;
      if(infile.eof() || testpos[0] == '*') {
       cout << "\nNo more test positions.\n";
       break;
      }
      infile >> ms >> h1 >> h2 >> ts.bmtype;
      ts.bmcount = 0;
      setboard(testpos, ms, h1, h2);

      do {
       infile >> bookm;
       bmexit = 0;
       for(unsigned int d = 0; d < sizeof(bookm); d++) {
        if(bookm[d] == '\0') break;
        if(bookm[d] == ';')
         { bookm[d] = '\0'; bmexit = 1; break; }
       }
       ts.bmoves[ts.bmcount] = pos.parse_move(bookm, &ts.tdata[0]);
       ts.bmcount++;
      } while(!bmexit && ts.bmcount < sizeof(ts.bmoves)/sizeof(ts.bmoves[0]));

      infile >> h4;
      infile.getline(id,sizeof(id));

      p_side = pos.wtm;

      if(inter) drawboard(); else cout << "\n";
      if (ms == 'w') { cout << "\nWhite to Move"; } else { cout << "\nBlack to Move"; }

      cout << "  Book Move(s):";
      for(e = 0; e < ts.bmcount; e++) {
	pos.print_move(ts.bmoves[e], mstring, &ts.tdata[0]);
       cout  << " " << mstring;
       if(e < (ts.bmcount-1)) cout << ",";
      }

      cout << "\n  Test Position: " << id << "\b ";

      if (inter) {
        cout << "\n\nPress 's' to search, 'n' for the next position, 'q' to exit: ";
        cin >> reply;
        if(reply == 'n') continue;
        if(reply == 'q') break;
        cout << "Please enter a search time (in seconds): ";
        cin >> testtime;
      }
    }

    if(!inter) cout << "\n";

    ts.best_depth = 0;
    p_side = pos.wtm^1;
    // reset hash tables of all types
    close_hash();
    open_hash();
    best = ts.search(pos, int(testtime*100), T, &game);
    p_side = pos.wtm;

    // update total node count
    for(int ti=0; ti<THREADS; ti++) { nodes += game.ts.tdata[ti].node_count; }
    // record time for this problem and total time used so far
    int used_time = GetTime() - ts.start_time;
    test_time += used_time;
    
    for(e = 0; e < ts.bmcount; e++) {
     if(best.t == ts.bmoves[e].t && ts.bmtype[0] == 'b')
     { correct++; if(ts.soltime < 0) ts.soltime = 0; break; }
	 if(best.t == ts.bmoves[e].t && ts.bmtype[0] == 'a') { break; }
	 if(e == ts.bmcount-1 && ts.bmtype[0] == 'a') 
     { correct++; if(ts.soltime < 0) ts.soltime = 0; break; } 
    }
    total++; 

    if(ts.best_score < (MATE>>1)) { 
      dtotal++;
      total_depth += ts.best_depth;
      depth_time += used_time;
    }

    pos.print_move(best, mstring, &ts.tdata[0]);
    pos.print_move(ts.bmoves[0], bookm, &ts.tdata[0]);

    cout << "\nSearched Move: " << mstring << "\n";
    cout << "Right = " << correct << "/" << total;
    cout << " Stime = " << setprecision(3) << ts.soltime;
    cout << " Total NPS = " << int((nodes)/(float(test_time)/100));

    cout.flush();

    if(ts.soltime > -1) total_time_sq += ts.soltime;
    if(wac) stime[total-1] = int(ts.soltime);

    if(correct)
     { cout << " <sol.time> = "
            << setprecision(3) << float(total_time_sq)/float(correct); }

    if(total_depth) {
     cout << " <depth> = " << float(total_depth)/float(dtotal);
     cout << " <time to depth> = " << float(depth_time)/(100.0*float(dtotal));
    }

    outfile << "\n" << id << " Smove: " << mstring;
    outfile << " Stime = " << ts.soltime;
    outfile << " Right = " << correct << "/" << total;
    outfile << " Total NPS = " << int((nodes)/(float(test_time)/100));
    if(correct)
     { outfile << " <sol.time> = "
              << setprecision(3) << float(total_time_sq)/float(correct); }
    if(total_depth) {
      outfile << " <depth> = " << float(total_depth)/float(dtotal);
      outfile << " <time to depth> = " << float(depth_time)/(100.0*float(dtotal));
    }

    if (inter) {
      cout << "\n\nPress 's' to search again, 'n' for the next position, 'q' to exit: ";
      cin >> reply;
    }

   } while (reply != 'q');

  if(wac && total >= 300) {
    cout << "           0    20    40    60    80   100   120   140   160   180   200   220   240   260   280\n";
    cout << "      -------------------------------------------------------------------------------------------\n";
    for(e = 1; e <= 20 ; e++) {
      cout << setw(4) << e << " |" 
           << setw(6) << stime[e-1] 
           << setw(6) << stime[20+e-1]
           << setw(6) << stime[40+e-1] 
           << setw(6) << stime[60+e-1] 
           << setw(6) << stime[80+e-1] 
           << setw(6) << stime[100+e-1] 
           << setw(6) << stime[120+e-1] 
           << setw(6) << stime[140+e-1] 
           << setw(6) << stime[160+e-1] 
           << setw(6) << stime[180+e-1]
           << setw(6) << stime[200+e-1] 
           << setw(6) << stime[220+e-1] 
           << setw(6) << stime[240+e-1] 
           << setw(6) << stime[260+e-1] 
           << setw(6) << stime[280+e-1] << "\n";      
    }
    cout << "\n";
    outfile << "\n           0    20    40    60    80   100   120   140   160   180   200   220   240   260   280\n";
    outfile << "      -------------------------------------------------------------------------------------------\n";
    for(e = 1; e <= 20 ; e++) {
      outfile << setw(4) << e << " |" 
           << setw(6) << stime[e-1] 
           << setw(6) << stime[20+e-1]
           << setw(6) << stime[40+e-1] 
           << setw(6) << stime[60+e-1] 
           << setw(6) << stime[80+e-1] 
           << setw(6) << stime[100+e-1] 
           << setw(6) << stime[120+e-1] 
           << setw(6) << stime[140+e-1] 
           << setw(6) << stime[160+e-1] 
           << setw(6) << stime[180+e-1]
           << setw(6) << stime[200+e-1] 
           << setw(6) << stime[220+e-1] 
           << setw(6) << stime[240+e-1] 
           << setw(6) << stime[260+e-1] 
           << setw(6) << stime[280+e-1] << "\n";      
    }
    outfile << "\n";
  }

  outfile.close();
  infile.close();
  ts.tsuite = 0;
  return;
}

/* Function to build a list of fen positions from a text file (pgn) of games */
void game_rec::build_fen_list()
{

  /* variables */
  char file[100];                    // file names
  char instring[100], line[200];     // strings from input files
  move bmove;                        // book move under consideration
  unsigned __int64 pcode;            // hash code for position
  int i = -1, j = 0, p,q;            // loop variables
  int r, s;                    
  int count = 0, thresh, LINE_DEPTH; // control variables
  int start_bk = 0, good_move = 1;
  int white_elo = 0, black_elo = 0;
  int plycount = 0, movecount = 0;
  int last_move_written = -5;
  int not_interesting_move = 0; 
  int search_score = 0, static_score = 0;     
  float game_score = 0.0;
  int game_id = 0;

  /* find out what the user wants */
  cout << " Enter name of PGN text file: ";
  cin >> file;
  // Consider no more than 150 moves
  LINE_DEPTH = 300;

  cout << " Building FEN file.... please wait.\n";

  /* open the pgn file and start work */
  ifstream infile(file);
  if(!infile) { cout << "File not found!\n"; return; }

  infile.seekg(0,ios::end);
  unsigned int file_size = infile.tellg();
  infile.seekg(0,ios::beg);

  ts.tsuite = 1;

  while(!infile.eof()) {   /* start !infile.eof() loop */

    infile >> instring;
    for(r=1;r<5;r++) {
     if(instring[r] == '\0') break;
	 if(instring[r] == '.' && instring[r+1] != '\0') {
	   for(s=0;instring[r]!='\0';r++,s++) { 
        instring[s] = instring[r+1];
       } 
     }
    } 

    switch(instring[0]) {
     case '[':
       infile.getline(line,199);
       if(!strncmp(line, " \"1-0", 4) && !strcmp(instring, "[Result")) {
	 game_score = 1.0; break;
       }
       if(!strncmp(line, " \"0-1", 4) && !strcmp(instring, "[Result")) {
	 game_score = 0.0; break;
       }
       if(!strncmp(line, " \"1/2-1/2", 4) && !strcmp(instring, "[Result")) {
	 game_score = 0.5; break;
       }
       if(!strcmp(instring, "[PlyCount")) {
	 sscanf(line," \"%d\"]", &plycount); break;
       }
       if(!strcmp(instring, "[WhiteElo")) {
	 sscanf(line," \"%d\"]", &white_elo); break;
       }
       if(!strcmp(instring, "[BlackElo")) {
	 sscanf(line," \"%d\"]", &black_elo); break;
       }
       if(!strcmp(instring, "[Event")) {
	 white_elo = 0;
	 black_elo = 0;
	 plycount = 0;
	 game_score = -1.0;
	 not_interesting_move = 0;
	 search_score = 0;
       }
       i++; count=0; movecount = 0;
       ts.no_book = 1;
       ts.max_search_depth = 5;
       ts.max_search_time = 100;
       setboard(i_pos, 'w', "KQkq", "-");
       force_mode = 1;
       over = 0;
       last_move_written = -5;
       break;
     case '#':
       i++; count=0; infile.getline(line,99);
       ts.no_book = 1;
       ts.max_search_depth = 5;
       ts.max_search_time = 100;
       setboard(i_pos, 'w', "KQkq", "-");
       force_mode = 1;
       over = 0;
       last_move_written = -5;
       break;
     case '1': break;
     case '2': break;
     case '3': break;
     case '4': break;
     case '5': break;
     case '6': break;
     case '7': break;
     case '8': break;
     case '9': break;
     case '{': break;
     default :
       count++; 
       // Conditions that cause us to skip
       //  the rest of the moves
       if(count > LINE_DEPTH) break;
       if(abs(search_score) > 1000) break;
       if(white_elo < 2800) break;
       if(black_elo < 2800) break;
       if(count == 1) {
	 game_id++;
	 if(!(game_id%100)) {
	   float percent_done = 100.0*infile.tellg()/file_size;
	   cout << "Working on game number " << game_id << ", percent done = " << percent_done << "\n";
	   cout.flush();
         }
       }
       // Now parse and make the move
       bmove = pos.parse_move(instring, &ts.tdata[0]);
       if(!bmove.t) { count = LINE_DEPTH; break; }
       pos.exec_move(bmove, 1);
       // Check if we have, check_mate, stale_mate, or a continuing game...
       if(pos.in_check_mate()) { over = 1; }
       else {
	 if(pos.fifty >= 100) { 
	   over = 1;
	 }
	 // check for a 3-rep
	 int rep_count = 0;
	 for(int ri = T-2; ri >= T-pos.fifty && rep_count < 2; ri -= 2) {
	   if(ts.tdata[0].plist[ri] == pos.hcode) {
	     rep_count++;
	     if(rep_count > 1) {
	       over = 1;
	       count = LINE_DEPTH;
	     }
	   }
	 }
       }
       if(over) { count = LINE_DEPTH; break; }
       game_history[T-1] = bmove; // record the move in the history list
       // update position list for all threads
       for(int ti=0; ti<MAX_THREADS;ti++) {
	 ts.tdata[ti].plist[T] = pos.hcode;
       }
       T++;
       movecount++;
       // keep track of how long since an interesting move
       if(!bmove.b.type) {
          not_interesting_move++;
       } else {
          not_interesting_move = 0;
       }
       // Only add a new position to the list if more than 2 half-moves have gone by
       if(movecount-last_move_written > 2 && not_interesting_move < 10) {
	 // Check to see if this position meets our requirements for 'quiet'
	 //int static_score = pos.score_pos(this,&ts.tdata[0]);
         ts.tdata[0].n[1].pos = pos;
         static_score = ts.tdata[0].n[1].qsearch(-MATE,MATE,0);
	 logging = 0;
	 ts.search(pos, 10, T, this);
	 logging = 1;
	 search_score = ts.g_last;
	 if(abs(search_score) < 1000 && abs(static_score-search_score) < 100) {
	   last_move_written = movecount;
	   // write fen string out to log file
	   pos.write_fen(0);
	   char outstring[50];
	   snprintf(outstring, sizeof(outstring), " %d %d %d %d %3.2f %d %d %d\n", white_elo, black_elo, movecount, plycount, game_score, static_score, search_score, game_id);
	   write_out(outstring);
	   logfile.flush();
	 }
       }
       break;
     }
  }            /* end !infile.eof() loop */


}

//------------------------------------------------------------------
// Function to train the evaluation function against a list of
//  'quiet' positions
//------------------------------------------------------------------
struct train_pos {
  position pos;
  float game_score;
  int game_ply;
  int max_game_ply;
};

struct train_val {
  char label[50];
  float jump_value;
};       


void game_rec::train_eval()
{
  char testpos[256], ms, h1[5] = "KQkq", h2[3], h4[256], id[256];
  char filein[60], *testfile;
  char line[200];
  train_pos *train;
  train = new train_pos[7000000];
  int train_count = 0;
  int white_elo, black_elo, game_ply, max_game_ply, static_score, search_score;
  float game_score;

  cout << "\nEnter file name for training suite in pseudo FEN format: ";
  cin >> filein; testfile = &filein[0];

  ifstream infile(testfile, IOS_IN_TEXT);
  if (!(infile.is_open())) { cout << "\nUnable to open file. "; return; }

  while(!infile.eof()) {
     infile >> testpos;
     if(infile.eof() || testpos[0] == '*') {
       cout << "\nNo more training positions.\n";
       break;
     }
     infile >> ms >> h1 >> h2 >> white_elo >> black_elo >> game_ply >> max_game_ply >> game_score >> static_score >> search_score;
     infile.getline(line,199);
 
     // Skip draws
     //if(game_score == 0.5) { continue; }
     
     // Skip positions too early in opening
     if(game_ply <= 10) { continue; }
     
     // Eliminate very bad matches
     //float scaled_game_score = 0.5+(game_score-0.5)*tanh(2.0*pow((1.0*game_ply)/max_game_ply,1.0));
     //if(ms == 'b') { static_score *= -1; }
     //if((1.0-scaled_game_score)*static_score > 200) { continue; }
     //if((scaled_game_score)*static_score < -200) { continue; }

     setboard(testpos, ms, h1, h2);
     // Add position to training list
     train[train_count].pos = pos;
     train[train_count].game_score = game_score;
     train[train_count].game_ply = game_ply;
     train[train_count].max_game_ply = max_game_ply;
     train_count++;
  }

  cout << "Done reading " << train_count << " positions from file, now training...\n";

 
 #define TRAIN_QUANT_COUNT 39
  int train_quant_count = TRAIN_QUANT_COUNT;

  train_val train_quantities[TRAIN_QUANT_COUNT+1] = {  
    //{ "VAR2", 0.01 },
    //{ "VAR1", 0.01 },
    { "KNIGHT_VALUE", 1 },
    { "BISHOP_VALUE", 1 },
    { "ROOK_VALUE", 1 },
    { "QUEEN_VALUE", 1 },
    { "SIDE_ON_MOVE_EARLY", 1 },
    { "SIDE_ON_MOVE_LATE", 1 },
    { "DOUBLED_PAWN_EARLY", 1 },
    { "DOUBLED_PAWN_LATE", 1 },
    { "WEAK_PAWN_EARLY", 1 },
    { "WEAK_PAWN_LATE", 1 },
    { "BACKWARD_PAWN_EARLY", 1 },
    { "BACKWARD_PAWN_LATE", 1 },
    { "PAWN_ISLAND_EARLY", 1 },
    { "PAWN_ISLAND_LATE", 1 },
    { "PAWN_DUO", 1 },
    { "SPACE", 1 },
    { "PASSED_PAWN", 1 },
    { "WRONG_BISHOP", 1 },
    { "CON_PASSED_PAWNS", 1 },
    { "OUTSIDE_PASSED_PAWN", 1 },
    { "FREE_PASSED_PAWN", 1 },
    { "BISHOP_MOBILITY", 1 },
    { "ROOK_MOBILITY", 1 },
    { "QUEEN_MOBILITY", 1 },
    { "KNIGHT_MOBILITY", 1 }, 
    { "BISHOP_PAIR", 1 },
    { "MINOR_OUTPOST", 1 },
    { "MINOR_OUTP_UNGUARD", 1 },
    { "MINOR_BLOCKER", 1 },
    { "BMINOR_OUTPOST", 1 },
    { "BMINOR_OUTP_UNGUARD", 1 },
    { "BMINOR_BLOCKER", 1 },
    { "PAWN_THREAT_MINOR", 1 },
    { "PAWN_THREAT_MAJOR", 1 },
    { "ROOK_OPEN_FILE", 1 },
    { "ROOK_HALF_OPEN_FILE", 1 },
    { "BOXED_IN_ROOK", 1 }, 
    //"CASTLED",
    //"NO_POSSIBLE_CASTLE",
    { "TRADES_EARLY", 1 },
    { "TRADES_LATE", 1 },
    /*
    { "PAWN_SCALE_PSQ_EARLY", 0.05 },
    { "PAWN_SCALE_PSQ_LATE", 0.05 },
    { "PAWN_OFFSET_PSQ_EARLY", 1 },
    { "PAWN_OFFSET_PSQ_LATE", 1 },
    { "KNIGHT_OFFSET_PSQ_EARLY", 1 },
    { "KNIGHT_OFFSET_PSQ_LATE", 1 },
    { "KNIGHT_SCALE_PSQ_EARLY", 0.1 },
    { "KNIGHT_SCALE_PSQ_LATE", 0.1 },
    { "BISHOP_OFFSET_PSQ_EARLY", 1 },
    { "BISHOP_OFFSET_PSQ_LATE", 1 },
    { "BISHOP_SCALE_PSQ_EARLY", 0.1 },
    { "BISHOP_SCALE_PSQ_LATE", 0.1 },
    { "ROOK_SCALE_PSQ_EARLY", 0.1 },
    { "ROOK_SCALE_PSQ_LATE", 0.1 },
    { "ROOK_OFFSET_PSQ_EARLY", 1 },
    { "ROOK_OFFSET_PSQ_LATE", 1 },
    { "QUEEN_SCALE_PSQ_EARLY", 0.1 },
    { "QUEEN_SCALE_PSQ_LATE", 0.1 },
    { "QUEEN_OFFSET_PSQ_EARLY", 1 },
    { "QUEEN_OFFSET_PSQ_LATE", 1 },
    { "KING_SCALE_PSQ_EARLY", 0.1 },
    { "KING_SCALE_PSQ_LATE", 0.1 },
    { "KING_OFFSET_PSQ_EARLY", 1 },
    { "KING_OFFSET_PSQ_LATE", 1 },
    */
    //{ "ROOK_ON_7TH", 1 },
    { "NO_REAL_PARAMETER", 1 }
};     


  double sum_square_diff;
  double sum_square_diff_prime;

  // setup search limits
  ts.max_search_depth = 1;
  logging = 0;
  // setup qsearch parameters
  ts.tdata[0].n[0].pos = pos;

  float cvalue = 0.0046;  // value for scaling conversion to logistic scores

  //for(cvalue = 0.0045; cvalue < 0.0055; cvalue += 0.0001) {
  // Set a baseline score	
  sum_square_diff = 0.0;	
  for(int i = 0; i < train_count; i++) {
    ts.tdata[0].n[1].pos = train[i].pos;
    int train_score = ts.tdata[0].n[1].qsearch(-MATE,MATE,0);
    //int train_score = train[i].pos.score_pos(this, &ts.tdata[0]);
    int game_length = MIN(110,train[i].max_game_ply);
    // Make a score for black from the point of view of white
    if(!train[i].pos.wtm) { train_score = -train_score; }
    // Turn the training score into a logistic function
    double logistic_train_score = 1.0/(1.0+exp(-cvalue*train_score));
    double scaled_game_score = 0.5+(train[i].game_score-0.5)*tanh(2.0*pow((1.0*train[i].game_ply)/game_length,1.0));
    sum_square_diff += (scaled_game_score-logistic_train_score)*(scaled_game_score-logistic_train_score); 
    //cout << setprecision(10) << logistic_train_score << " " << scaled_game_score << " " << sum_square_diff << "\n";
    //if(i > 100) return;
  }
  // adjust for number of positions
  sum_square_diff = sqrt(sum_square_diff/train_count);

  //cout << cvalue << "   " << sum_square_diff << "\n";
  //}
  //return; 

  cout << "Current best score = " << sum_square_diff << "\n";

  /*  -- The following is a very slow, stepwise add/subtract approach to testing changes -- 
  for(int k = 0; k < 3; k++) {
    cout << "---------------------------------------- Trial " << k << "\n";
    for(int quantity = 0; quantity < train_quant_count; quantity++) {
      
      // Now try new values and check against baseline, adjusting as the baseline improves
      int reverse = 0;
      while(reverse < 2) {
	modify_score_value(train_quantities[quantity].label, train_quantities[quantity].jump_value);
	sum_square_diff_prime = 0.0;
	for(int i = 0; i < train_count; i++) {
	  ts.tdata[0].n[1].pos = train[i].pos;
	  int train_score_prime = ts.tdata[0].n[1].qsearch(-MATE,MATE,0);
	  //int train_score_prime = train[i].pos.score_pos(this, &ts.tdata[0]);
          int game_length = MIN(110,train[i].max_game_ply);
	  if(!train[i].pos.wtm) { train_score_prime = -train_score_prime; }
	  double logistic_train_score_prime = 1.0/(1.0+exp(-cvalue*train_score_prime));
	  double scaled_game_score = 0.5+(train[i].game_score-0.5)*tanh(2.0*pow((1.0*train[i].game_ply)/game_length,1.0));
	  sum_square_diff_prime += (scaled_game_score-logistic_train_score)*(scaled_game_score-logistic_train_score); 
	}
	// adjust for number of positions
	sum_square_diff_prime /= train_count;
	if(sum_square_diff_prime < sum_square_diff) {
	  cout << train_quantities[quantity].label << " " << get_score_value(train_quantities[quantity].label) << "   " << sum_square_diff << " " << sum_square_diff_prime << "\n";
	  sum_square_diff = sum_square_diff_prime;
	  reverse = 1;
	} else {
	  modify_score_value(train_quantities[quantity].label, -train_quantities[quantity].jump_value);
	  if(!reverse) {
	    train_quantities[quantity].jump_value *= -1;
	  }
	  reverse++;
	}
      }
      cout << train_quantities[quantity].label << " " << get_score_value(train_quantities[quantity].label) << "   " << sum_square_diff << " " << sum_square_diff_prime << "\n";
      
    }
  }    
  */


  /* -- The following is a generation/genetic type change approach -- */
#define GENERATIONS 100
#define CHILDREN 10
#define CHANGE_PROBABILITY 0.1
#define NEGATIVE_CHANGE_PROBABILITY 0.5
#define LARGE_CHANGE_PROBABILITY 0.5

  float best_train_values[TRAIN_QUANT_COUNT];
  float new_train_values[CHILDREN][TRAIN_QUANT_COUNT];
  double best_score = sum_square_diff;
  double best_child_score;
  int best_child; 

  for(int quantity = 0; quantity < train_quant_count; quantity++) {
    best_train_values[quantity] = get_score_value(train_quantities[quantity].label);
  }

  for(int generation = 0; generation < GENERATIONS; generation++) {
    best_child = 0;
    best_child_score = best_score;
    for(int child = 0; child < CHILDREN; child++) {

      // Review, modify, store, and set all training quantities
      for(int quantity = 0; quantity < train_quant_count; quantity++) {
	new_train_values[child][quantity] = best_train_values[quantity];
	if(float(rand())/RAND_MAX < CHANGE_PROBABILITY) {
	  float change = train_quantities[quantity].jump_value;
	  if(float(rand())/RAND_MAX < LARGE_CHANGE_PROBABILITY) {
	    if(0.1*best_train_values[quantity] > change) {
	      change = 0.1*best_train_values[quantity];
	    }
	  }
	  if(float(rand())/RAND_MAX < NEGATIVE_CHANGE_PROBABILITY) {
	    change = -change;
	  }
	  new_train_values[child][quantity] += change;
	}
	set_score_value(train_quantities[quantity].label, new_train_values[child][quantity]);
      }

      // Now obtain a score for this child
      double sum_square_diff_prime = 0.0;
      for(int i = 0; i < train_count; i++) {
	ts.tdata[0].n[1].pos = train[i].pos;
	int train_score_prime = ts.tdata[0].n[1].qsearch(-MATE,MATE,0);
	//int train_score_prime = train[i].pos.score_pos(this, &ts.tdata[0]);
	int game_length = MIN(110,train[i].max_game_ply);
	if(!train[i].pos.wtm) { train_score_prime = -train_score_prime; }
	double logistic_train_score_prime = 1.0/(1.0+exp(-cvalue*train_score_prime));
	double scaled_game_score = 0.5+(train[i].game_score-0.5)*tanh(2.0*pow((1.0*train[i].game_ply)/game_length,1.0));
	sum_square_diff_prime += (scaled_game_score-logistic_train_score_prime)*(scaled_game_score-logistic_train_score_prime); 
      }
      
      // adjust for number of positions
      sum_square_diff_prime = sqrt(sum_square_diff_prime/train_count);
      

      cout << "Child " << child << " of generation " << generation << " has score = " << sum_square_diff_prime << "\n";

      if(sum_square_diff_prime < best_child_score) {
	best_child = child;
	best_child_score = sum_square_diff_prime;
      } 

      if(best_score - best_child_score > 0.000010) {
	child = CHILDREN;
	cout << "Large enough score change to terminate generation early!\n";
      }
     
    }

    // If the generation was successful, update best quantities
    if(best_child_score < best_score) {
      best_score = best_child_score;
      cout << "Successful generation!  New best score = " << best_score << "\n";
      for(int quantity = 0; quantity < train_quant_count; quantity++) {
	best_train_values[quantity] = new_train_values[best_child][quantity];
	cout << train_quantities[quantity].label << " " << int(best_train_values[quantity]) << "\n"; 
      }
    } else {
      cout << "Generation failed to produce a better child!\n";
    }

    // Reset score values to current best
    for(int quantity = 0; quantity < train_quant_count; quantity++) {
      set_score_value(train_quantities[quantity].label, best_train_values[quantity]);
    }

  }


}

