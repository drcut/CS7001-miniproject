#include <algorithm>
#include <assert.h>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <inttypes.h>
#include <iostream>
#include <list>
#include <math.h>
#include <sstream>
#include <stdlib.h>
#include <string>
#include <sys/time.h>
#include <time.h>

#include <array>
#include <map>
#include <vector>
#define ADDR int64_t
typedef unsigned long long uns64;

using namespace std;

const uns64 MEM_SIZE_MB = 16384;
const int LINESIZE = 64;
const int L3_SIZE_KB = 8192;
const int L3_ASSOC = 16;
const int L3_SETS = (L3_SIZE_KB * 1024) / (L3_ASSOC * LINESIZE);

struct cache {
  struct set {
    struct cache_line {
      bool valid;
      int tag;
      int last_access; // used for LRU
      cache_line() { valid = false; };
    };
    set(){};
    set(int w) {
      ways_cnt = w;
      lines = new cache_line[w];
      access_cnt = 0;
      memset(hit_cnt, 0, sizeof(hit_cnt));
    }
    void print_reuse_stack() {

      int hit_num = 0;
      for (int i = 1; i < L3_ASSOC + 1; i++) {
        printf("dis[%d]=%d\n", i, hit_cnt[i]);
        hit_num += hit_cnt[i];
      }
      printf("miss: %d\n", access_cnt - hit_num);
    }
    int ways_cnt;
    cache_line *lines;
    int access_cnt;
    int hit_cnt[L3_ASSOC + 1];
    bool access(int tag) {
      access_cnt++;
      int last_access_slot = 0;
      for (int i = 0; i < ways_cnt; i++) {
        if (lines[i].valid && lines[i].tag == tag) {
          // cache hit
          // update reuse histogram
          int reuse_dis = 1;
          for (int j = 0; j < ways_cnt; j++) {
            if (lines[j].valid && lines[j].last_access > lines[i].last_access) {
              reuse_dis++;
            }
          }
          hit_cnt[reuse_dis]++;
          lines[i].last_access = access_cnt;
          return true;
        } else if (lines[i].valid == false)
          last_access_slot = i;
        else if (lines[i].last_access < lines[last_access_slot].last_access) {
          last_access_slot = i;
        }
      }
      lines[last_access_slot].valid = true;
      lines[last_access_slot].tag = tag;
      lines[last_access_slot].last_access = access_cnt;
      return false;
    }
  };
  cache(int w, int s) {
    sets = new set[s];
    for (int i = 0; i < s; i++)
      sets[i] = set(w);
    sets_cnt = s;
  }
  void cache_access(int line_addr) {
    // get set
    int set_id = line_addr % sets_cnt;
    int tag = line_addr / sets_cnt;
    if (sets[set_id].access(tag)) {
      hit_cnt++;
    } else {
      miss_cnt++;
    }
  }
  void print_reuse_stack(int set_id) {
    printf("set: %d\n", set_id);
    sets[set_id].print_reuse_stack();
  }
  void print_reuse_stack() {
    for (int i = 0; i < sets_cnt; i++)
      print_reuse_stack(i);
  }
  set *sets;
  int sets_cnt;
  int hit_cnt;
  int miss_cnt;
};
cache LLC(L3_ASSOC, L3_SETS);

int global_hit_cnt[L3_ASSOC + 1];
int global_access_cnt;
void generate_global_hit_cnt() {
  memset(global_hit_cnt, 0, sizeof(global_hit_cnt));
  global_access_cnt = 0;
  for (int i = 0; i < LLC.sets_cnt; i++) {
    global_access_cnt += LLC.sets[i].access_cnt;
  }
  for (int reuse_dis = 1; reuse_dis < L3_ASSOC + 1; reuse_dis++) {
    int acc = 0;
    for (int s = 0; s < LLC.sets_cnt; s++)
      acc += LLC.sets[s].hit_cnt[reuse_dis];
    global_hit_cnt[reuse_dis] = acc;
  }
}
int sampling_reuse_dis() {
  int reuse_dis;
  int acc = 0;
  int r = rand() % global_access_cnt;
  for (reuse_dis = 1; reuse_dis < L3_ASSOC + 1; reuse_dis++) {
    acc += global_hit_cnt[reuse_dis];
    if (acc > r) {
      return reuse_dis;
    }
  }
  return -1;
}
// set sampling: sampling a part of sets' reuse histogram
// use this reuse histogram to represent all sets
void generate_proxy(int inst_cnt, ofstream &fout) {
  int hit_n = 0;
  int miss_n = 0;
  int cold_hit_n = 0;
  cache dummy_cache(L3_ASSOC, L3_SETS);
  int tag_cnt[L3_SETS]; // used to generate cold cache miss
  memset(tag_cnt, 0, sizeof(tag_cnt));
  // do some initilization
  generate_global_hit_cnt();
  // calculate the total access number for whole sets
  int access_cnt = 0;
  for (int i = 0; i < LLC.sets_cnt; i++) {
    access_cnt += LLC.sets[i].access_cnt;
  }
  for (int i = 0; i < inst_cnt; i++) {
    // sampling sets
    int r = rand() % access_cnt;
    int acc = 0;
    int s;
    for (s = 0; s < LLC.sets_cnt; s++) {
      acc += LLC.sets[s].access_cnt;
      if (acc > r)
        break;
    }
    // sampling reuse distance
    // r = rand() % LLC.sets[s].access_cnt;
    // acc = 0;
    // bool cache_hit = false;
    int reuse_dis = sampling_reuse_dis();
    // for (reuse_dis = 1; reuse_dis < L3_ASSOC + 1; reuse_dis++) {
    //   acc += LLC.sets[s].hit_cnt[reuse_dis];
    //   if (acc > r) {
    //     cache_hit = true;
    //     break;
    //   }
    // }
    bool find_reuse = false;
    if (reuse_dis != -1) {
      // get the reuse-dis th recently blocks
      for (int pos = 0; pos < L3_ASSOC; pos++) {
        if (dummy_cache.sets[s].lines[pos].valid) {
          // chech how many ways are younger than it
          int younger_cnt = 1;
          for (int tmp = 0; tmp < L3_ASSOC; tmp++) {
            if (dummy_cache.sets[s].lines[tmp].valid &&
                dummy_cache.sets[s].lines[tmp].last_access >
                    dummy_cache.sets[s].lines[pos].last_access) {
              younger_cnt++;
            }
          }
          if (younger_cnt == reuse_dis) {
            int generated_mem_access =
                s + dummy_cache.sets[s].lines[pos].tag * dummy_cache.sets_cnt;
            fout << generated_mem_access << std::endl;
            dummy_cache.cache_access(generated_mem_access);
            find_reuse = true;
            break;
          }
        }
      }
      if (!find_reuse) {
        // need reuse distance larger than valid blocks
        // use any valid block instead
        for (int pos = 0; pos < L3_ASSOC; pos++) {
          if (dummy_cache.sets[s].lines[pos].valid) {
            int generated_mem_access =
                s + dummy_cache.sets[s].lines[pos].tag * dummy_cache.sets_cnt;
            fout << generated_mem_access << std::endl;
            dummy_cache.cache_access(generated_mem_access);
            find_reuse = true;
            break;
          }
        }
        if (!find_reuse) {
          // try to find any valid block in the cache
          for (int set_id = 0; set_id < L3_SETS; set_id++) {
            for (int way_id = 0; way_id < L3_ASSOC; way_id++) {
              if (find_reuse)
                break;
              if (dummy_cache.sets[set_id].lines[way_id].valid) {
                int generated_mem_access =
                    s + dummy_cache.sets[set_id].lines[way_id].tag *
                            dummy_cache.sets_cnt;
                fout << generated_mem_access << std::endl;
                dummy_cache.cache_access(generated_mem_access);
                find_reuse = true;
                break;
              }
            }
          }
        }
      }
    }
    if (reuse_dis == -1 || !find_reuse) {
      // generate a miss
      int generated_mem_access = s + (tag_cnt[s]++) * dummy_cache.sets_cnt;
      fout << generated_mem_access << std::endl;
      dummy_cache.cache_access(generated_mem_access);
      miss_n++;
    } else {
      hit_n++;
    }
  }
  printf("miss: %d  hit: %d cold hit: %d\n", miss_n, hit_n, cold_hit_n);
}
/*
input: file1: memory trace for physical address
           file2: generated proxy memory trace
*/
int main(int argc, char *argv[]) {
  printf("LLC: ways: %d sets: %d\n", L3_ASSOC, L3_SETS);
  srand(42);
  // input: real memory trace
  ifstream tfile;
  tfile.open(argv[1]);
  // output: proxy memory trace
  ofstream proxy_file;
  proxy_file.open(argv[2]);

  // profiling and get set reuse distance
  ADDR addr;
  timeval begin, end;
  int addr_cnt = 0;
  gettimeofday(&begin, NULL);
  while (!tfile.eof()) {
    string buffer;
    getline(tfile, buffer);
    try {
      addr = std::stoi(buffer);
      LLC.cache_access(addr);
    } catch (...) {
      continue;
    }
  }
  tfile.close();
  gettimeofday(&end, NULL);
  double elapsed =
      (end.tv_sec - begin.tv_sec) + ((end.tv_usec - begin.tv_usec) / 1000000.0);
  printf("Time to profiling: %lf ms\n", elapsed * 1000.0);
  // LLC.print_reuse_stack();
  gettimeofday(&begin, NULL);
  generate_proxy(1000000, proxy_file);
  gettimeofday(&end, NULL);
  elapsed =
      (end.tv_sec - begin.tv_sec) + ((end.tv_usec - begin.tv_usec) / 1000000.0);

  printf("Time to generate %lf ms\n", elapsed * 1000.0);

  return 0;
}
