#include <assert.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "global_types.h"
#include "mcore.h"
#include "params.h"

#define ACCESS 5001186
#define MISS 4109930

MCache *initilize_LLC(int set, int assoc, int replace_policy) {
  assert(replace_policy == 0 && "only support LRU policy\n");
  return mcache_new(set, assoc, replace_policy);
}

void generate_trace(MCache *LLC, int ACCESS_CNT, int ACCESS_MISS) {
  // generate an eviction set with size == L3_ASSOC+1
  uns set_id = 0;
  // first insert to set, cold miss
  uns first_addr = set_id + 0 * LLC->sets;
  // mcache_install(LLC, first_addr);
  printf("%lld ", first_addr);

  ACCESS_CNT -= 1;
  ACCESS_MISS -= 1;

  // first generate trace for HIT
  for (int i = 0; i < ACCESS_CNT - ACCESS_MISS; i++) {
    // mcache_install(LLC, first_addr);
    printf("%lld ", first_addr);
  }
  // generate trace for MISS
  // all MISS should be compulsory miss
  uns tag_id = 1;
  for (int i = 0; i < ACCESS_MISS; i++) {
    Addr line_addr =
        (set_id + tag_id * LLC->sets) % (MEM_SIZE_MB * 1024 * 1024 / LINESIZE);
    tag_id++;
    printf("%lld ", line_addr);
  }
}

int main() {
  srand(42);
  // initilization cache simulator
  uns l3sets = (L3_SIZE_KB * 1024) / (L3_ASSOC * LINESIZE);
  MCache *LLC = initilize_LLC(l3sets, L3_ASSOC, L3_REPL);
  generate_trace(LLC, ACCESS, MISS);
}