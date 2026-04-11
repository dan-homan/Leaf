// EXchess source code, (c) Daniel C. Homan  1997-2017
// Released under the GNU public license, see file license.txt
 
// Functions for game_rec class
//----------------------------------------------------------------
//  game_rec();
//  void setboard(const char inboard[256], const char ms, const char castle[5], const char ep[3]);
//  void test_suite(const char *, const char *, float, int); 
//  void board_edit();
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

