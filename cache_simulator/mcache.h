#ifndef MCACHE_H
#define MCACHE_H

#include "global_types.h"


typedef enum MCache_ReplPolicy_Enum {
    REPL_LRU=0,
    REPL_RND=1,
    REPL_SRRIP=2, 
    REPL_DRRIP=3, 
    REPL_FIFO=4, 
    REPL_DIP=5, 
    NUM_REPL_POLICY=6
} MCache_ReplPolicy;


typedef struct MCache_Entry MCache_Entry;
typedef struct MCache MCache;

///////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////

struct MCache_Entry {
    Flag    valid;
    Flag    dirty;
    Addr    tag;
    uns     ripctr;
    uns64   last_access;
    Addr    orig_lineaddr;
};


struct MCache{
  uns sets;
  uns assocs;
  MCache_ReplPolicy repl_policy; //0:LRU  1:RND 2:SRRIP
  uns index_policy; // how to index cache

  Flag *is_leader_p0; // leader SET for D(RR)IP
  Flag *is_leader_p1; // leader SET for D(RR)IP
  uns psel;

  MCache_Entry *entries;
  uns *fifo_ptr; // for fifo replacement (per set)
  int touched_wayid;
  int touched_setid;
  int touched_lineid;

  uns64 s_count; // number of accesses
  uns64 s_miss; // number of misses
  uns64 s_evict; // number of evictions
};

///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////

MCache *mcache_new(uns sets, uns assocs, uns repl );
Flag    mcache_access        (MCache *c, Addr addr);
void    mcache_install       (MCache *c, Addr addr);
Flag    mcache_probe         (MCache *c, Addr addr);
Flag    mcache_invalidate    (MCache *c, Addr addr);
Flag    mcache_mark_dirty    (MCache *c, Addr addr);
uns     mcache_get_index     (MCache *c, Addr addr);

uns     mcache_find_victim   (MCache *c, uns set);
uns     mcache_find_victim_lru   (MCache *c, uns set);
uns     mcache_find_victim_rnd   (MCache *c, uns set);
uns     mcache_find_victim_srrip   (MCache *c, uns set);

void    mcache_select_leader_sets(MCache *c,uns sets);
uns     mcache_drrip_get_ripctrval(MCache *c, uns set);
Flag    mcache_dip_check_lru_update(MCache *c, uns set);

void    mcache_print_stats(MCache *c, char *header);
///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////

#endif // MCACHE_H
