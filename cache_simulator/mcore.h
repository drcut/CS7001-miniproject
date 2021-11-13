#ifndef MCORE_H
#define MCORE_H

#include "global_types.h"
#include "mcache.h"
#include "memsys.h"
#include "os.h"
#include <zlib.h>

typedef struct MCore MCore;

////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////

struct MCore {
  uns tid;

  MemSys *memsys;
  MCache *l3cache;
  OS *os;

  char addr_trace_fname[1024];
  FILE *addr_trace;

  uns done;

  uns64 trace_inst_num;
  uns trace_iaddr; // four bytes only IA
  Addr trace_pa;
  Flag trace_wb;
  uns trace_dhits;

  uns64 cycle;
  uns64 inst_num;
  uns64 access_count;
  uns64 miss_count;
  uns64 delay_count; // due to L3/mem access
  uns64 sleep_cycle_count;

  uns64 lifetime_inst_count;

  uns64 done_inst_count;
  uns64 done_cycle_count;
  uns64 done_access_count;
  uns64 done_miss_count;
  uns64 done_delay_count;
};

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

MCore *mcore_new(MemSys *memsys, OS *os, MCache *l3cache,
                 char *addr_trace_fname, uns tid);
void mcore_cycle(MCore *core);
void mcore_print_stats(MCore *c);
void mcore_read_trace(MCore *c);
void mcore_fread_trace(MCore *c);
void mcore_init_trace(MCore *c);

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#endif // MCORE_H
