#include <assert.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <zlib.h>

#include "externs.h"
#include "mcore.h"

#define MCORE_STOP_ON_EOF 0

////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////

MCore *mcore_new(MemSys *memsys, OS *os, MCache *l3cache,
                 char *addr_trace_fname, uns tid) {
  MCore *c = (MCore *)calloc(1, sizeof(MCore));
  c->tid = tid;
  c->memsys = memsys;
  c->os = os;
  c->l3cache = l3cache;

  strcpy(c->addr_trace_fname, addr_trace_fname);
  mcore_init_trace(c);
  mcore_read_trace(c);

  return c;
}

////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////
void mcore_init_trace(MCore *c) {

  // char  command_string[256];
  //  sprintf(command_string,"gunzip -c %s", c->addr_trace_fname);

  if ((c->addr_trace = fopen(c->addr_trace_fname, "r")) == NULL) {

    //---- maybe put random sleep and try again?
    printf("Problems initializing Core: %u ... Trying again\n", (uns)c->tid);
    sleep(rand() % 10);
    if ((c->addr_trace = fopen(c->addr_trace_fname, "r")) == NULL) {
      die_message("Unable to open the input trace file. Dying ...\n");
    }
  }
}

////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////

void mcore_cycle(MCore *c) {
  if (MCORE_STOP_ON_EOF && c->done) {
    return;
  }

  c->cycle += CLOCK_INC_FACTOR;

  // if core is sleeping, return.
  if (c->cycle <= c->sleep_cycle_count) {
    return;
  }

  if (c->inst_num <= c->trace_inst_num) {
    c->inst_num += (CORE_WIDTH * CLOCK_INC_FACTOR); // for faster sims
  }

  if (c->inst_num >= c->trace_inst_num) {

    if (c->trace_wb == FALSE) {
      Flag l3outcome;
      // Addr orig_lineaddr = os_v2p_lineaddr(c->os, c->trace_va, c->tid);
      Addr orig_lineaddr = c->trace_pa;

      uns64 delay = 0;

      c->access_count++;
      l3outcome = mcache_access(c->l3cache, orig_lineaddr);
      delay += L3_LATENCY; // incurred on both hit and miss

      if ((L3_PERFECT == FALSE) && (l3outcome == MISS)) {
        uns64 memsysdelay;
        memsysdelay =
            memsys_access(c->memsys, orig_lineaddr, c->tid, c->cycle + delay);
        delay += memsysdelay;
        mcache_install(c->l3cache, orig_lineaddr);
        c->miss_count++;
      }

      c->delay_count += delay;
      c->sleep_cycle_count = c->cycle + delay;
    }

    mcore_read_trace(c);
  }
}

////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////

void mcore_read_trace(MCore *c) {
  mcore_fread_trace(c);

  if (feof(c->addr_trace)) {

    if (!c->done) {
      c->done_inst_count = c->inst_num;
      c->done_cycle_count = c->cycle;
      c->done_access_count = c->access_count;
      c->done_miss_count = c->miss_count;
      c->done_delay_count = c->delay_count;
      c->done = 1;
    }

    if (!MCORE_STOP_ON_EOF) {
      fclose(c->addr_trace);
      mcore_init_trace(c);
      mcore_fread_trace(c);
      c->lifetime_inst_count += c->inst_num;
      c->inst_num = 0;
    }
  }
}

////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////

void mcore_print_stats(MCore *c) {
  double ipc = (double)(c->done_inst_count) / (double)(c->done_cycle_count);
  double apki =
      1000 * (double)c->done_access_count / (double)c->done_inst_count;
  double mpki = 1000 * (double)c->done_miss_count / (double)c->done_inst_count;
  double missrate = 100 * mpki / apki;
  double avgdelay = (double)c->done_delay_count / (double)c->done_access_count;
  char header[256];
  sprintf(header, "CORE_%02d", c->tid);

  printf("\n%s_ID           \t : %2u", header, c->tid);
  printf("\n%s_TRACE        \t : %s", header, c->addr_trace_fname);
  printf("\n%s_INST         \t : %llu", header, c->done_inst_count);
  printf("\n%s_CYCLES       \t : %llu", header, c->done_cycle_count);
  printf("\n%s_APKI         \t : %4.2f", header, apki);
  printf("\n%s_MPKI         \t : %4.2f", header, mpki);
  printf("\n%s_MISSRATE     \t : %4.2f", header, missrate);
  printf("\n%s_AVGDELAY     \t : %4.2f", header, avgdelay);
  printf("\n%s_IPC          \t : %4.3f", header, ipc);

  printf("\n");

  fclose(c->addr_trace);
}

////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////

char space[128];
void mcore_fread_trace(MCore *c) {
  c->trace_inst_num = 0;
  c->trace_pa = 0;
  c->trace_wb = 0;
  c->trace_dhits = 0;

  int tmp;
  do {
    tmp = fgetc(c->addr_trace);
    if (tmp == '\n')
      break;
    c->trace_pa = c->trace_pa * 10 + (tmp - '0');
  } while (tmp != EOF);
}