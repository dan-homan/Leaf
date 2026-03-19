/* Functions */

#ifndef FUNCT_H
#define FUNCT_H

#include "define.h"

/* swap.cpp */
int swap(int sq, position p, int side, int from);

/* main.cpp */
void drawboard();          // draw game board and info
void takeback(int tm);
void type_moves();             // print possible moves
void type_capts();             // print possible captures
void parse_command();          // parse user's command
void make_move();              // find and make move
void save_game();
void help();
int inter();                   // interrupt function
void write_out(const char *);   // write information to logfile
void performance();             // performance test function

/* search.cpp */
void pc_update(move pcmove, int ply);
void lmr_init_tables();

/* setup.cpp */
void set_search_param();
void set_score_value(char dummy[50], float val);
void modify_score_value(char dummy[50], float val);
float get_score_value(char dummy[50]);
void gen_check_table();
//void fill_psq(int piece);
void print_psq();

/* sort.cpp */
void QuickSortBook(book_rec *Lb, book_rec *Ub);
void QuickSortMove(move_rec *Lb, move_rec *Ub);

/* book.cpp */
void build_book(position ipos);
void book_learn(int flag, game_rec *gr);
int edit_book(h_code hash_code, position *p);
move opening_book(h_code hash_code, position p, game_rec *gr);

/* util.cpp */
int GetTime();

/* probe.cpp */
#if TABLEBASES
int init_tb();
int probe_tb(position *p, int ply);
#endif

/* uci.cpp */
void uci_loop(game_rec *gr);
void uci_send_info(int score, int depth, int elapsed_cs, unsigned long long nodes, tree_search *ts);
int  uci_check_interrupt();

#endif


