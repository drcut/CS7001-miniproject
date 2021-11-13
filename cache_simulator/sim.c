#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "global_types.h"
#include "mcore.h"
#include "memsys.h"
#include "params.h"

MemSys *memsys;
OS *os;
MCache *LLC;
MCore *mcore[MAX_THREADS];

int main(int argc, char **argv) {
  int ii;
  Flag all_cores_done = 0;

  if (argc < 2) {
    die_usage();
  }

  read_params(argc, argv);

  //--------------------------------------------------------------------
  // -- Allocate the nest and cores
  //--------------------------------------------------------------------
  uns num_os_pages = (1024 * MEM_SIZE_MB) / (OS_PAGESIZE / 1024);
  os = os_new(num_os_pages, num_threads);

  uns l3sets = (L3_SIZE_KB * 1024) / (L3_ASSOC * LINESIZE);
  memsys = memsys_new(NUM_THREADS);
  LLC = mcache_new(l3sets, L3_ASSOC, L3_REPL);

  for (ii = 0; ii < num_threads; ii++) {
    mcore[ii] = mcore_new(memsys, os, LLC, addr_trace_filename[ii], ii);
  }

  srand(RAND_SEED);
  //--------------------------------------------------------------------
  // -- Iterate through the traces by cycling all cores till done
  //--------------------------------------------------------------------

  while (!(all_cores_done)) {
    all_cores_done = 1;

    for (ii = 0; ii < num_threads; ii++) {
      mcore_cycle(mcore[ii]);
      all_cores_done &= mcore[ii]->done;
    }

    cycle += CLOCK_INC_FACTOR;
  }

  mcache_print_stats(LLC, (char *)"L3CACHE");

  return 0;
}
