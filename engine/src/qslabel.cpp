// Leaf batch quietness labels (--qs-label)
//
// --qs-label <in.tsv|-> <out.tsv> [--gid-mod M R]
//
// Batch quietness labels for a corpus dump (tdleaf.cpp format: fen, cp,
// result, ply, depth, gid, endply, gate).  For every row it evaluates the FEN
// four ways, all side-to-move POV of the position being evaluated:
//
//   static     the net's static eval of the position
//   qs         full-window qsearch of the position (captures, plus check
//              evasions if in check) -- "what the side to move can grab now"
//   static_n   static eval after a NULL move (opponent to move, ep cleared)
//   qs_n       full-window qsearch after the null move -- "what the opponent
//              threatens to grab".  Empty when the side to move is in check
//              (a null move is illegal there).
//
// qs - static >= 0 measures capture gain available to the side to move;
// qs_n - static_n >= 0 measures the threat against it.  A position is quiet
// in the two-sided sense when both are small.  Neither depends on the row's
// search label, which is the point: the corpus quiet gate (|cp - gate|)
// conditions on the label's residual, this conditions on the position.
//
// The FEN carries no castling / ep / fifty-move state (the dump writes "- -"),
// none of which matters to a capture search.  TT cutoffs are suppressed
// (every node is searched as a PV node under PV_NO_TT_CUTOFF), so a row's
// labels do not depend on which rows preceded it, up to move ordering.
//
// Output columns: gid ply cp gate static qs static_n qs_n incheck pieces
// (cp and gate are passed through, white POV as in the input; the four new
// scores are side-to-move POV as described above).  --gid-mod M R keeps only
// rows whose gid % M == R, a whole-game sample.
// ---------------------------------------------------------------------------

// Board field + side to move -> position.  Castling and ep are off; returns
// false on a malformed board or a missing king.
static bool qsl_set_position(const char *fen, position &pos)
{
  for (int i = 0; i < 64; i++) pos.sq[i] = EMPTY;
  int rx = 0, ry = 7, kings[2] = {0, 0};
  const char *c = fen;
  for (; *c && *c != ' '; c++) {
    if (*c == '/') { ry--; rx = 0; continue; }
    if (*c >= '1' && *c <= '8') { rx += *c - '0'; continue; }
    int pt;
    switch (*c | 32) {
      case 'p': pt = PAWN;   break;
      case 'n': pt = KNIGHT; break;
      case 'b': pt = BISHOP; break;
      case 'r': pt = ROOK;   break;
      case 'q': pt = QUEEN;  break;
      case 'k': pt = KING;   break;
      default: return false;
    }
    if (rx > 7 || ry < 0) return false;
    int white = (*c >= 'A' && *c <= 'Z');
    if (pt == KING) kings[white]++;
    pos.sq[SQR(rx, ry)] = (white ? 8 : 0) + pt;
    rx++;
  }
  if (kings[0] != 1 || kings[1] != 1 || *c != ' ') return false;
  c++;
  if (*c == 'w') pos.wtm = 1; else if (*c == 'b') pos.wtm = 0; else return false;

  pos.castle = 0; pos.ep = 0; pos.fifty = 0;
  pos.Krook[WHITE] = pos.Qrook[WHITE] = pos.Krook[BLACK] = pos.Qrook[BLACK] = -100;
  pos.last.t = NOMOVE; pos.hmove.t = NOMOVE; pos.rmove.t = NOMOVE; pos.cmove.t = NOMOVE;
  pos.qchecks[0] = pos.qchecks[1] = 0;
  pos.gen_code();
  pos.in_check();
  return true;
}

// Full-window qsearch of `pos` on node n[1], with n[0] a not-in-check parent
// (qsearch reads prev->pos.check / prev->moves.count).  premove_score is the
// stand-pat the qsearch uses because pos.last is NOMOVE.  Returns the qsearch
// score and sets *static_out to the static eval.
static int qsl_qsearch(ts_thread_data *td, const position &pos, int *static_out)
{
  search_node &root = td->n[0], &node = td->n[1];
  root.pos = pos; root.pos.check = 0; root.moves.count = 0;
  node.pos = pos;
#if NNUE
  if (nnue_available) {
    node.acc.dirty[0] = node.acc.dirty[1] = true;
    nnue_init_accumulator(node.acc, node.pos);
    root.acc = node.acc;
  }
#endif
#if NNUE
  int st = node.pos.score_pos(node.gr, td, &node.acc);
#else
  int st = node.pos.score_pos(node.gr, td);
#endif
  *static_out = st;
  node.premove_score = st;
  td->pc[1][1].t = NOMOVE;
  return node.qsearch(-MATE, MATE, 0, 1);
}

int qslabel_main(int argc, char *argv[])
{
  const char *in_path = nullptr, *out_path = nullptr;
  long gid_mod = 0, gid_rem = 0;
  for (int ai = 1; ai < argc; ai++) {
    if (!strcmp(argv[ai], "--qs-label") && ai + 2 < argc) {
      in_path = argv[ai + 1]; out_path = argv[ai + 2]; ai += 2;
    } else if (!strcmp(argv[ai], "--gid-mod") && ai + 2 < argc) {
      gid_mod = atol(argv[ai + 1]); gid_rem = atol(argv[ai + 2]); ai += 2;
    }
  }
  if (!in_path || !out_path) {
    fprintf(stderr, "usage: --qs-label <in.tsv|-> <out.tsv> [--gid-mod M R]\n");
    return 1;
  }
  FILE *in = strcmp(in_path, "-") ? fopen(in_path, "r") : stdin;
  FILE *out = fopen(out_path, "w");
  if (!in || !out) { fprintf(stderr, "--qs-label: cannot open input/output\n"); return 1; }

  // Single-threaded, no clock: max_ply 0 keeps SEARCH_INTERRUPT_CHECK from
  // polling time or stdin; turn 1 keeps plist[turn+ply-1] in range.
  tree_search &ts = game.ts;
  ts.max_ply = 0; ts.ponder = 0; ts.max_nodes = 0; ts.turn = 1;
  ts_thread_data *td = &ts.tdata[0];
  td->init_thread_data(0);
  td->done = 0;
  pv_learning_mode = 1;   // with in_pv=1: no TT cutoffs (PV_NO_TT_CUTOFF)

  fprintf(out, "gid\tply\tcp\tgate\tstatic\tqs\tstatic_n\tqs_n\tincheck\tpieces\n");
  static char line[1024];
  uint64_t rows = 0, bad = 0;
  while (fgets(line, sizeof line, in)) {
    if (line[0] == '#' || !strncmp(line, "fen\t", 4)) continue;
    // fen cp result ply depth gid endply gate
    char *f[8]; int nf = 0;
    for (char *p = line; nf < 8; ) {
      f[nf++] = p;
      char *t = strchr(p, '\t');
      if (!t) { char *nl = strchr(p, '\n'); if (nl) *nl = 0; break; }
      *t = 0; p = t + 1;
    }
    if (nf < 8) { bad++; continue; }
    if (gid_mod > 0 && (long)(strtoul(f[5], NULL, 10) % gid_mod) != gid_rem) continue;

    position pos;
    if (!qsl_set_position(f[0], pos)) { bad++; continue; }
    int st, qs = qsl_qsearch(td, pos, &st);

    int incheck = pos.check;
    char nbuf[48] = "\t";
    if (!incheck) {
      // Null move, exactly as pvs() makes it.
      position np = pos; np.wtm ^= 1;
      Or(np.hcode, hstm);
      np.last.t = NOMOVE; np.ep = 0; np.fifty = 0;
      np.material = -np.material;
      np.in_check();
      int st_n, qs_n = qsl_qsearch(td, np, &st_n);
      snprintf(nbuf, sizeof nbuf, "%d\t%d", st_n, qs_n);
    }
    int pieces = 0;
    for (int s = 0; s < 64; s++) if (pos.sq[s] != EMPTY) pieces++;
    fprintf(out, "%s\t%s\t%s\t%s\t%d\t%d\t%s\t%d\t%d\n",
            f[5], f[3], f[1], f[7], st, qs, nbuf, incheck, pieces);
    if (++rows % 1000000 == 0)
      fprintf(stderr, "--qs-label: %llu rows\n", (unsigned long long)rows);
  }
  fprintf(stderr, "--qs-label: done, %llu rows labelled, %llu malformed skipped\n",
          (unsigned long long)rows, (unsigned long long)bad);
  if (in != stdin) fclose(in);
  fclose(out);
  return 0;
}
