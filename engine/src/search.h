// search.h — search constants and the SEARCH_INTERRUPT_CHECK macro.
// Search tuning parameters (NULL_MOVE, VERIFY_MARGIN, etc.) have moved to
// SearchConfig search_cfg in engine_globals.h / main.cpp.

#ifndef SEARCH_H
#define SEARCH_H

#define TIME_FLAG 123456

// Score below alpha at which not to do qchecks
#define NO_QCHECKS 650

// Pruning score margins
#define MARGIN(x)  (50+100*(x)*(x))

#if DEBUG_SEARCH
 ofstream search_outfile;
#endif

#define SEARCH_INTERRUPT_CHECK() \
  if(tdata->ID == 0 && !(tdata->node_count&search_cfg.check_inter)) { \
   if (ts->max_ply > 3 || ts->ponder) {		   \
     int elapsed = GetTime() - ts->start_time;	   \
     if ((elapsed >= MIN(2*ts->limit,ts->max_limit) && !ts->ponder && !proto.uci_in_ponder)	\
         || (ts->tsuite && elapsed >= ts->limit) \
	 || (inter())) {			 \
       return -TIME_FLAG;			 \
     }						 \
     /* Node budget: same abort, same units as the soft/hard time pair.  The  */ \
     /* hard ceiling is MIN(2*node_limit, node_max), exactly mirroring time.  */ \
     if (ts->max_nodes && !ts->ponder && !proto.uci_in_ponder		\
         && tdata->node_count >= MIN(2*ts->node_limit, ts->node_max)) {	\
       return -TIME_FLAG;			 \
     }						 \
     if(FLTK_GUI && ts->ponder			\
	&& ts->root_wtm == gr->pos.wtm) {       \
       return -TIME_FLAG;			\
     }                                          \
   }                                            \
 }                                              \
 if(tdata->done) {                              \
   return -TIME_FLAG;                           \
 }						\


#endif  /* SEARCH_H */
