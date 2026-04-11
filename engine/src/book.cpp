// EXchess source code, (c) Daniel C. Homan  1997-2012
// Released under the GNU public license, see file license.txt

/* Book.Cpp functions to construct a book file from
   a pgn-like text.  Also includes functions to probe
   the book during search, edit the book, and incorporate
   learned information into the book */

#include "define.h"
#include "chess.h"
#include "funct.h"
#include "const.h"
#include "extern.h"
#include <cstdio>
#include <string.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <ctime>

/* variables for book learning */
book_rec learn_book[100];          // book learning array
int learn_filepos[100];            // file positions of played moves 
int learn_choice[100];             // number of move choices

/* variables for returning book moves */
int GAMBIT_SCORE = 80;                     // Gambit threshold 

char BOOK_FILE[FILENAME_MAX] = "main_bk.dat";
char START_BOOK[FILENAME_MAX] = "start_bk.dat";

/* Function to build an opening book from a text file (pgn) of games */
void build_book(position ipos)
{

  /* variables for book building */
  fstream chunk_file[TEMP_FILES];    // temporary files
  ofstream out;                      // final file
  int chunk_count = 1;               // number of temp files
  book_rec chunk_record[TEMP_FILES]; // current record in each temp file
  book_rec *record, *record_place;   // pointers to working records in memory
  char file[100], chunk[20];         // file names
  char instring[100], line[100];     // strings from input files
  char outbook[100], resp;
  position temp_pos;                 // temporary position
  move bmove;                        // book move under consideration
  unsigned __int64 pcode;            // hash code for position
  int i = -1, j = 0, p,q;            // loop variables
  int r, s;                    
  int count = 0, thresh, LINE_DEPTH; // control variables
  int start_bk = 0, good_move = 1;
  int Leaf_white = 0;
  int Leaf_black = 0;
  int win_white = 0;
  int win_black = 0;

  /* initialize working record space in memory */
  record = new book_rec[BOOK_POS];
  for(p = 0; p < BOOK_POS; p++) {
   record[p].pos_code = ZERO;
   record[p].score = 0;
   record[p].gambit = 0;
   // default to 10 wins and losses to prevent a very
   //  small number of learned games from eliminating a
   //  move from consideration
   record[p].wins = 10;
   record[p].losses = 10;
  }

  /* find out what the user wants */
  cout << " Enter name of book text file: ";
  cin >> file;
  cout << " Will this be a starting book\n  (special format for pgn file, see start.pgn)? (y/n): ";
  cin >> resp;
  if(resp != 'y') {
   cout << " Enter line depth of book: ";
   cin >> LINE_DEPTH;
   cout << " Enter the minimum number of times a move must be played: ";
   cin >> thresh;
  } else { LINE_DEPTH = 60; thresh = 1; start_bk = 1;}

  cout << " Enter the name for the output book: ";
  cin >> outbook;
  cout << " Building book.... please wait.\n";

  /* open the pgn file and start work */
  ifstream infile(file);
  if(!infile) { cout << "File not found!\n"; return; }

  infile.seekg(0,ios::end);
  unsigned int file_size = infile.tellg();
  infile.seekg(0,ios::beg);

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
       infile.getline(line,99);
       if(!strncmp(line, " \"Leaf", 6) && !strcmp(instring, "[White")) {
	 Leaf_white = 1; break;
       }
       if(!strncmp(line, " \"Leaf", 6) && !strcmp(instring, "[Black")) {
	 Leaf_black = 1; break;
       }
       if(!strncmp(line, " \"1-0", 4) && !strcmp(instring, "[Result")) {
	 win_white = 1; break;
       }
       if(!strncmp(line, " \"0-1", 4) && !strcmp(instring, "[Result")) {
	 win_black = 1; break;
       }
       if(!strcmp(instring, "[Event")) {
	 Leaf_white = 0;
	 Leaf_black = 0;
	 win_white = 0;
	 win_black = 0;
       }
       i++; count=0; temp_pos = ipos;
       break;
     case '#':
       i++; count=0; infile.getline(line,99); temp_pos = ipos; 
       Leaf_white = 0;
       Leaf_black = 0;
       win_white = 0;
       win_black = 0;
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
       count++; if(count > LINE_DEPTH) break;
       if(start_bk) {
         q = MIN(strlen(instring),99);
	 if(instring[q-1] == '!') { 
	  good_move = 1;
          instring[q-1] = '\0'; 
         } else { good_move = 0; }
       }
       bmove = temp_pos.parse_move(instring, &game.ts.tdata[0]);
       if(!bmove.t) { count = LINE_DEPTH; break; }
       temp_pos.exec_move(bmove, 1);
       pcode = temp_pos.hcode;
       record_place = record;
       for(j = 0; j < BOOK_POS-1; j++) {
          if(record_place->pos_code == pcode ||
              !record_place->pos_code) break;
          record_place++;
       }
       if(!(j%1000) && j) {
         cout << "Adding " << j << "th record to chunk "
              << chunk_count << ", " << setprecision(3)
              << (float(infile.tellg())/file_size)*100 <<  "% done\n";
         cout.flush();
       }
       record_place->pos_code = pcode;
       if(good_move) {
	 if(record_place->score < 65535) record_place->score++;
	 if(record_place->wins < 65535) {
	   if((Leaf_white && win_white && (temp_pos.wtm^1)) || 
	      (Leaf_black && win_black && (temp_pos.wtm))) {
	     record_place->wins++;
	   }
	 }
	 if(record_place->losses < 65535) {
	   if((Leaf_white && win_black && (temp_pos.wtm^1)) || 
	      (Leaf_black && win_white && (temp_pos.wtm))) {
	     record_place->losses++;
	   }
	 }	 
       }
       break;
     }

     if (j>=BOOK_POS-1) {
      if(chunk_count >= TEMP_FILES) break;
      cout << "Sorting records for chunk " << chunk_count << "\n";
      for(j = 0; j <= BOOK_POS-1; j++)
       { record_place = record+j;
         if(!record_place->pos_code) break; }
      QuickSortBook(record, record_place-1);
      snprintf(chunk, sizeof(chunk), "temp_bk.%i", chunk_count);
      chunk_file[chunk_count-1].open(chunk, IOS_OUT);
      for(i = 0; i < j; i++) {
       record_place = record+i;
       chunk_file[chunk_count-1].write((char *) record_place, sizeof(book_rec));
      }
      chunk_file[chunk_count-1].close();
      chunk_file[chunk_count-1].open(chunk, IOS_IN);
      chunk_count++;

      /* initialize record structure for next chunk
         of the file to be digested */
      for(p = 0; p < BOOK_POS; p++) {
       record[p].pos_code = ZERO;
       record[p].score = 0;
       record[p].gambit = 0;
       // default to 10 wins and losses to prevent a very
       //  small number of learned games from eliminating a
       //  move from consideration
       record[p].wins = 10;
       record[p].losses = 10;
      }
      j = 0;
     }
  }            /* end !infile.eof() loop */



  for(j = 0; j < BOOK_POS-1; j++)
   { record_place = record+j;
     if(!record_place->pos_code) break; }

  /* Sort and Write the last chunk to disk */
  if(j) {
    cout << "Sorting records for last chunk\n ";
    cout.flush();
    QuickSortBook(record, record_place-1);
    snprintf(chunk, sizeof(chunk), "temp_bk.%i", chunk_count);
    chunk_file[chunk_count-1].open(chunk, IOS_OUT);
    for(i = 0; i < j; i++) {
     record_place = record+i;
     chunk_file[chunk_count-1].write((char *) record_place, sizeof(book_rec));
    }
    chunk_file[chunk_count-1].close();
    chunk_file[chunk_count-1].open(chunk, IOS_IN);
  }


  /* Write out big book file by sorting through the individual
     chunks */

  cout << "Writing book file...\n";
  out.open(outbook, IOS_OUT);
  int min = 0;

   // first reading in a record from each chunk
   for(i = 0; i < chunk_count; i++) {
      if(chunk_file[i].eof()) { chunk_file[i].close(); break; }
      chunk_file[i].seekg(0,ios::beg);
      chunk_file[i].read((char *) &chunk_record[i], sizeof(book_rec));
   }

 while(1) {

   // Now probe for minimum (meaning smallest hash-code) entry
   min = 0;
   for(i = 0; i < chunk_count; i++) {
     if(min == i && !chunk_record[i].pos_code) { min++; continue; }
     if(chunk_record[min].pos_code > chunk_record[i].pos_code &&
        chunk_record[i].pos_code) min = i;
   }

   if(min == chunk_count) break;

   // merge all minimum records together - reading in more when necessary
   for(i = 0; i < chunk_count; i++) {
     if(i == min) continue;
     if(chunk_record[min].pos_code == chunk_record[i].pos_code) {
       if(chunk_record[min].score+chunk_record[i].score < 65535) 
	 chunk_record[min].score += chunk_record[i].score;
       else chunk_record[min].score = 65535;
       if(chunk_record[min].wins+chunk_record[i].wins < 65535) 
	 chunk_record[min].wins += chunk_record[i].wins;
       else chunk_record[min].wins = 65535;
       if(chunk_record[min].losses+chunk_record[i].losses < 65535) 
	 chunk_record[min].losses += chunk_record[i].losses;
       else chunk_record[min].losses = 65535;
       if(!chunk_file[i].eof())
	 chunk_file[i].read((char *) &chunk_record[i], sizeof(book_rec)); \
       else chunk_record[i].pos_code = ZERO;
     }
   }

   if (chunk_record[min].score >= thresh) {
     out.write((char *) &chunk_record[min], sizeof(book_rec));
   }

   if(!chunk_file[min].eof())
       chunk_file[min].read((char *) &chunk_record[min], sizeof(book_rec));
   else chunk_record[min].pos_code = ZERO;

 }

  out.close();
  delete [] record;

  // remove temporary files
  for(j = 1; j <= chunk_count; j++) {
     chunk_file[j-1].close();
     snprintf(chunk, sizeof(chunk), "temp_bk.%i", j);
     if(remove(chunk) == -1) cout << "Error deleting " << chunk << "\n";
  }

}


//--------------------------------------------------------
// Function to find a position in the book
//--------------------------------------------------------
int find_record(position p, move m, int file_size, book_rec *book_record, fstream *book_f, int file_pos)
{
  int jump = int(file_size/2);
  unsigned __int64 pcode = ZERO;
  position temporary_pos;  // working position
  
  // initialize file position
  file_pos = 0;

  // make move to create new position to search for
  temporary_pos = p;
  if(!temporary_pos.exec_move(m, 1)) return 0;
  pcode = temporary_pos.hcode;

  // jump to beginning of file
  book_f->seekg(0,ios::beg);
 
  //----------------------------------
  // seek within file for position
  //  -- ordered by hash code (pcode),
  //     so jumping and narrowing down
  //     position by 1/2 each time is
  //     a good strategy
  //-----------------------------------
  while(jump) {
   book_f->seekg(int(jump*sizeof(book_rec)), ios::cur);
   file_pos += jump;
   book_f->read((char *) book_record, sizeof(book_rec));
   book_f->seekg(-int(sizeof(book_rec)), ios::cur);
   if(book_record->pos_code == pcode) return 1;
   if(jump == 1) {
     if(file_pos > 10) {
       book_f->seekg(-int(10*sizeof(book_rec)), ios::cur);
       file_pos -= 10;
     } else {
       book_f->seekg(0,ios::beg);
       file_pos = 0;
     }
     for(int i = 0; (i <= 20 && file_pos < file_size); i++) {
       book_f->read((char *) book_record, sizeof(book_rec));
       if(book_record->pos_code == pcode) {
        book_f->seekg(-int(sizeof(book_rec)), ios::cur);
        return 1;
       }
       file_pos++;
       if(book_record->pos_code > pcode) { jump = 0; i = 21; }
       if(file_pos > file_size) { jump = 0; i = 21; }
     }
     jump = 0;
   } else {
     jump = int(ABS(jump)/2); if(!jump) jump = 1;
     if(book_record->pos_code > pcode) jump = -jump;
     if(jump == -1) jump = 1;
   }
 }

 book_record->pos_code = pcode;   // for ease in adding records to book
 return 0;

}

//--------------------------------------------------------
/* Opening Book Function */
// This function retrieves a move from the opening book.
// It does this by making each of the individual moves
// in a given position and checking if the subsequent
// position is in the book.  If so, the move in question
// becomes a candidate move.  Information is stored to
// facilitate easy learning during the game.
//--------------------------------------------------------
move opening_book(h_code hash_code, position p, game_rec *gr)
{
  int file_size, mflag = 0, j;
  int candidates = 0, total_score = 0;
  move_list list;
  move nomove; nomove.t = 0;
  char book_file[100];          // file name for the book
  book_rec book_record;         // record of move considered
  int file_pos = 0;                 // file position of record
  fstream book_f;               // actual file handle for book
  float total_win_losses, win_loss_exponent;   // floats for win/loss learning calculations

  /************* First try starting book *************/
  /* no check on gambit scores and no learning, all moves
     are played with equal probability (score irrelevant) */    

  // generate legal moves
  p.allmoves(&list, &(gr->ts.tdata[0]));

  // look for book
  strcpy(book_file, exec_path);    // try executable directory
  strcat(book_file, START_BOOK);
  book_f.open(book_file, IOS_IN);

  if(!book_f.is_open()) {
   // try working directory
   book_f.open(START_BOOK, IOS_IN);
  }

  if(book_f.is_open()) {   // if no book is found jump ahead to main book

   book_f.seekg(0,ios::end);
   file_size = book_f.tellg()/sizeof(book_rec);

   for(j = 0; j < list.count; j++) {

    // find a record for the current move, if no record, return 0 else 1.
    mflag = find_record(p,list.mv[j].m, file_size, &book_record, &book_f, file_pos);

    // If there is a record for this position in the book
    // then do some testing to set the move score
    if(mflag) {
     if(book_record.score > 0) book_record.score = 1;      // define all non-zero start books moves as equal
     list.mv[j].score = book_record.score;
     if(book_record.score > 0) {
        candidates++;
        total_score += book_record.score;
     }
    } else list.mv[j].score = 0; 
 
   }

   // Sort moves based on their score
   QuickSortMove(&list.mv[0], &list.mv[list.count-1]);

   // Now select a move from the list to play
   if(candidates) {
    int running_score = 0;
    float random = float(rand())/RAND_MAX;
    for(j = 0; j < candidates; j++) {
     running_score += list.mv[j].score;
     if(random <= (float(running_score)/total_score)) {
       book_f.close();
       return list.mv[j].m;      
      }
     }
    }
 
    book_f.close();
  }

  /************* Now go to main book *************/
  book_f.clear();

  // generate legal moves
  p.allmoves(&list, &(gr->ts.tdata[0]));

  // look for book
  strcpy(book_file, exec_path);  // try executable directory
  strcat(book_file, BOOK_FILE);
  book_f.open(book_file, IOS_IN);

  if(!book_f.is_open()) {
   // try working directory
   book_f.open(BOOK_FILE, IOS_IN);
  }

  if(!book_f.is_open()) { return nomove; }   // if no book is found

  book_f.seekg(0,ios::end);
  file_size = book_f.tellg()/sizeof(book_rec);

  total_score = 0;

  //------------------------------------------------------------
  // First eliminate any moves played less than 5% of the time
  //------------------------------------------------------------
  for(j = 0; j < list.count; j++) {

   // find a record for the current move, if no record, return 0 else 1.
   mflag = find_record(p,list.mv[j].m,file_size, &book_record, &book_f, file_pos);

   if(mflag) {
     list.mv[j].score = book_record.score;
     total_score += book_record.score;
   } else list.mv[j].score = 0;

  }
  
  if(total_score) {
    for(j = 0; j < list.count; j++) {
      if(20*list.mv[j].score/total_score < 1) {
	list.mv[j].score = 0;
      }
    }
  }

  total_score = 0;

  //-----------------------------------------------------------------
  // Then use win/loss records to assign the score for rest of moves
  //-----------------------------------------------------------------
  for(j = 0; j < list.count; j++) {

   // find a record for the current move, if no record, return 0 else 1.
   mflag = find_record(p,list.mv[j].m,file_size, &book_record, &book_f, file_pos);

   // If there is a record for this position in the book
   //  -- and the move wasn't eliminated in the 5% filter above
   // then do some testing to set the move score
   if(mflag && list.mv[j].score > 0) {
     total_win_losses = MAX(float(book_record.wins+book_record.losses), 1.0);    
     win_loss_exponent = float(book_record.wins-book_record.losses)/total_win_losses;
     // squaring total_win_losses to increase importance of these statistics
     //   -- cap this value at 4000 to prevent overflow of total move scoring in extreme cases
     total_win_losses=MIN(total_win_losses*total_win_losses,4000.0);
     list.mv[j].score = int(float(book_record.score)*(pow(total_win_losses,win_loss_exponent)));
     // only consider moves that pass a certain win threshold
     if(book_record.score > 0 && 15*book_record.wins >= 10*book_record.losses) {
       candidates++;
       total_score += int(float(book_record.score)*(pow(total_win_losses,win_loss_exponent)));
     } else list.mv[j].score = 0;
   } else list.mv[j].score = 0;

  }

 // Sort moves based on their score
 QuickSortMove(&list.mv[0], &list.mv[list.count-1]);

 // Now select a move from the list to play
 if(candidates) {
  int running_score = 0;
  float random = float(rand())/RAND_MAX;

  for(j = 0; j < candidates; j++) {
   running_score += list.mv[j].score;
   if(random <= (float(running_score)/total_score)) {
     /* now find this record in the book again.
       this is done to get the file position. */
      mflag = find_record(p,list.mv[j].m,file_size, &book_record, &book_f, file_pos);

      // Now that we found it, store learning
      // information for the move
      if(mflag) {
       if(candidates) learn_choice[gr->learn_count] = 1;
       else learn_choice[gr->learn_count] = 0;
       learn_book[gr->learn_count] = book_record;
       learn_filepos[gr->learn_count] = file_pos;
       gr->learn_count++;
       book_f.close();
       return list.mv[j].m;
      }
    }
   }
  }

  book_f.close();

  return nomove;
}


//-------------------------------------------
// Book editing function 
//  -- very rough right now
//-------------------------------------------
int edit_book(h_code hash_code, position *p)
{

  char mstring[10];
  int file_size, mflag = 0, j, outflag = 0;
  int search_time = 0, total_score = 0;
  move_list list;
  char resp[2];
  char book_file[100];   // file name for the book
  book_rec book_record;         // record of move considered
  int file_pos;                 // file position of record
  fstream book_f;               // actual file handle for book
  float total_win_losses, win_loss_exponent;

  // generate legal moves
  p->allmoves(&list, &(game.ts.tdata[0]));

  // Get file from user
  cout << " Enter book file to modify: ";
  cin >> book_file;

  book_f.open(book_file, IOS_IN | IOS_OUT);

  if(!book_f) 
   { cout << " File not found!\n"; return 0; }   // if no book is found

  book_f.seekg(0,ios::end);
  file_size = book_f.tellg()/sizeof(book_rec);

  post = 1;   // turn on search posting
  game.book = 0;   // turn off book in search

  total_score = 0;

  //------------------------------------------------------------
  // First eliminate any moves played less than 5% of the time
  //------------------------------------------------------------
  for(j = 0; j < list.count; j++) {

   // find a record for the current move, if no record, return 0 else 1.
    mflag = find_record((*p),list.mv[j].m,file_size, &book_record, &book_f, file_pos);

   if(mflag) {
     list.mv[j].score = book_record.score;
     total_score += book_record.score;
   } else list.mv[j].score = 0;

  }

  for(j = 0; j < list.count; j++) {
    if(20*list.mv[j].score/total_score < 1) {
      list.mv[j].score = 0;
    }
  }

  total_score = 0;

  //-----------------------------------------------------------------
  // Then use win/loss records to assign the score for rest of moves
  //-----------------------------------------------------------------
  for(j = 0; j < list.count; j++) {

   // find a record for the current move, if no record, return 0 else 1.
    mflag = find_record((*p),list.mv[j].m,file_size, &book_record, &book_f, file_pos);

   // If there is a record for this position in the book
   //  -- and the move wasn't eliminated in the 5% filter above
   // then do some testing to set the move score
   if(mflag && list.mv[j].score > 0) {
     total_win_losses = MAX(float(book_record.wins+book_record.losses), 1.0);    
     win_loss_exponent = float(book_record.wins-book_record.losses)/total_win_losses;
     // squaring total_win_losses to increase importance of these statistics
     //   -- cap this value at 4000 to prevent overflow of total move scoring in extreme cases
     total_win_losses=MIN(total_win_losses*total_win_losses,4000.0);
     list.mv[j].score = int(float(book_record.score)*(pow(total_win_losses,win_loss_exponent)));
     // only consider moves that pass a certain win threshold
     if(book_record.score > 0 && 15*book_record.wins >= 10*book_record.losses) {
       total_score += int(float(book_record.score)*(pow(total_win_losses,win_loss_exponent)));
     } else list.mv[j].score = 0;
   }
  }

  cout << "******************* Book Editing Mode ***********************\n";
  cout << "The frequency with which moves are played is given by its\n"
       << "percentage (which is computed from the scores of all the moves).\n"
       << "A move with a percentage of 0, has a less than a 0.1% chance\n"
       << "of being played.  To prevent such a move entirely, its\n"
       << "raw score must be set to zero (via editing).\n"
       << "To add a move to the book, simply try to edit that move.\n";

  for(j = 0; j < list.count; j++) {
   // see if move is in the book
   mflag = find_record((*p),list.mv[j].m,file_size,&book_record, &book_f, file_pos);

   // if so, edit the record if it would normally be played
   if(mflag && book_record.score > 0 && list.mv[j].score > 0) {
     outflag++;
     cout << "  Move: ";
     p->print_move(list.mv[j].m, mstring, &game.ts.tdata[0]);
     cout << setw(5) << mstring << " play %: " << setprecision(3) << setw(4)
          << float(int(1000*(float(list.mv[j].score)/total_score)))/10
          << ", gambit: " << book_record.gambit;
     if(outflag > 1) {
       outflag = 0; cout << "\n";
     }
   }
  }
  if(outflag == 1) cout << "\n";

  /* Edit mode for individual moves */
  move edit_move;
  while(1) {
    cout << "\nEnter a move to be edited/investigated (0 = quit): ";
    cin >> mstring; if(mstring[0] == '0') break;
    edit_move = p->parse_move(mstring, &game.ts.tdata[0]);
    if(find_record((*p), edit_move, file_size, &book_record, &book_f, file_pos)) {
      cout << " raw score: " << book_record.score
           << " gambit flag: " << book_record.gambit 
           << " wins: " << book_record.wins 
           << " losses: " << book_record.losses << "\n";
      cout << " Search time in seconds (0 = no search): ";
      cin >> search_time;
      if(search_time) {
       cout << " Search of position *after* book move, so positive scores are bad! \n";
       // make move to create new position to search for
       position temporary_pos = (*p);
       if(temporary_pos.exec_move(edit_move, 1)) {
	 game.ts.search(temporary_pos, search_time*100, 0, &game);
	 cout << "\n";
       } else { cout << "Move illegal!\n"; }
      }
      cout << " Edit the record for this move (y/n) "; cin >> resp;
      if(!strcmp(resp, "y")) {
        cout << " Enter new percentage (0 = don't play): ";
        total_score -= book_record.score;
        cin >> book_record.score;
        book_record.score = MAX(0,(total_score*book_record.score)/(100-book_record.score));
        total_score += book_record.score;
        if(book_record.score) {
         cout << " Enter gambit flag: ";
         cin >> book_record.gambit;
         cout << " Enter book wins: ";
         cin >> book_record.wins;
         cout << " Enter book losses: ";
         cin >> book_record.losses;
        }
        book_f.seekp((book_f.tellg()),ios::beg);
        book_f.write((char *) &book_record, sizeof(book_rec));
      }
    } else {
      cout << " No such move in book!\n"
           << " Add a new record for this move (y/n)? ";
      cin >> resp;
      if(!strcmp(resp, "y")) {
	for(j = 0; j < list.count; j++) {
         if(list.mv[j].m.t == edit_move.t) break;
        }
        if(j == list.count) {
	 cout << " Move illegal!\n"; continue;
        }
        cout << " Enter raw score (current total = "
             << total_score << "): ";
        cin >> book_record.score;
        cout << " Enter gambit flag: ";
        cin >> book_record.gambit;
        book_record.wins = 0;
        book_rec temp_rec1, temp_rec2;
        book_f.seekg(0,ios::beg);
        book_f.seekp(0,ios::beg);
        temp_rec2 = book_record;
        for(j = 0; j < file_size; j++) {
         book_f.read((char *) &temp_rec1, sizeof(book_rec));
         if(temp_rec1.pos_code > book_record.pos_code) {
           book_f.seekp((int(book_f.tellg())-(sizeof(book_rec))), ios::beg);
           book_f.write((char *) &temp_rec2, sizeof(book_rec));
	   temp_rec2 = temp_rec1;  
         } 
        }
        book_f.write((char *) &temp_rec2, sizeof(book_rec));
        file_size++;
      }
    }
  }

  book_f.close();
  return 0;
}



//----------------------------------------------------
// Function to execute a simple form of book learning 
//   -- This function simply writes a new score for each
//      book position reached into the opening book.
//----------------------------------------------------
void book_learn(int flag, game_rec *gr)
{

  int bi = 0;

  // if we are winning, increase the win number for this
  // opening.  
  if(flag == 1) {
    for(bi = 0; bi < gr->learn_count; bi++) {
      if(learn_book[bi].wins < 65535)  learn_book[bi].wins++;
    }
  }

  // If we are losing, increase the losses number
  if(flag < 1) {
    for(bi = gr->learn_count-1; bi >= 0; bi--) {
      if(learn_book[bi].losses < 65535)  learn_book[bi].losses++;
     }
   }

  // Now write the changes to the file...
  fstream out(BOOK_FILE, IOS_IN|IOS_OUT);

  if(!out) { cout << "\nError(NoBookUpdate)"; out.close(); return; }

  if(flag == 1 || flag == -1) {
    for(bi = 0; bi < gr->learn_count; bi++) {
       out.seekp(learn_filepos[bi]*sizeof(book_rec), ios::beg);
       out.write((char *) &learn_book[bi], sizeof(book_rec));
    }
   } else if(!flag) {
       out.seekp(learn_filepos[bi]*sizeof(book_rec), ios::beg);
       out.write((char *) &learn_book[bi], sizeof(book_rec));
   }

   out.seekp(0,ios::end);
   out.close();

}









