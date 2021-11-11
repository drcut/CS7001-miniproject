#include <algorithm>
#include <assert.h>
#include <fstream>
#include <functional>
#include <inttypes.h>
#include <iostream>
#include <math.h>
#include <sstream>
#include <stdlib.h>
#include <string>

#include <array>
#include <map>
#include <vector>

const int MAX_REGION_REUSE_DISTANCE = 128;
const int MAX_CASSCADING_DEGREE = 3;

const int MACROBLOCK_SIZE_BITS = 10;

using namespace std;

#define ADDR uint64_t

ADDR get_region_id(ADDR addr) { return (addr >> MACROBLOCK_SIZE_BITS); }

struct CascadedStrideTable {
  std::map<string, std::map<ADDR, int> *> entries;
  CascadedStrideTable(){};
  void print() {
    for (auto entry : entries) {
      printf("history: %s\n", entry.first.c_str());
      for (auto e : *entry.second) {
        printf("next stride: %d cnt: %d\n", e.first, e.second);
      }
    }
  }
  void insert(string previous_strides, ADDR curr_strides) {
    // this previous_pattern is first shown
    if (entries.find(previous_strides) == entries.end()) {
      auto new_map = new std::map<ADDR, int>;
      entries.insert(
          std::pair<string, std::map<ADDR, int> *>(previous_strides, new_map));
    }
    // update the corresponding entry
    auto curr_map = entries.find(previous_strides)->second;
    if (curr_map->find(curr_strides) == curr_map->end()) {
      curr_map->insert(std::pair<ADDR, int>(curr_strides, 0));
    }
    curr_map->find(curr_strides)->second += 1;
  }
  void clear() { entries.clear(); }
} CSTs[MAX_CASSCADING_DEGREE + 1];

void print_CSTs() {
  for (int i = 0; i < MAX_CASSCADING_DEGREE + 1; i++) {
    CSTs[i].print();
  }
}
struct RegionHistoryTable {
  // entry inside the table
  struct entry {
    ADDR last_access;
    std::vector<ADDR> past_strides;
    void print_entry() {
      printf("last access: %d\n", last_access);
      printf("past strides: ");
      for (int i = 0; i < past_strides.size(); i++) {
        printf("%d ", past_strides[i]);
      }
      printf("\n");
    }
    void update(ADDR addr) {
      if (last_access != -1) {
        ADDR new_stride = addr - last_access;
        // TODO: update the cascaded stride tables
        for (int dis = 1; dis < MAX_CASSCADING_DEGREE + 1; dis++) {
          if (dis > past_strides.size())
            break;
          // generate string for previous access pattern
          std::string key = "";
          for (int j = 0; j < dis; j++) {
            key += std::to_string(past_strides[past_strides.size() - dis + j]);
            key += ',';
          }
          CSTs[dis].insert(key, addr);
        }
      }
      last_access = addr;
      past_strides.push_back(addr);
      // print_entry();
      if (past_strides.size() > MAX_CASSCADING_DEGREE + 1) {
        past_strides.erase(past_strides.begin());
      }
    }
    entry() {
      last_access = -1;
      past_strides.clear();
    }
  };
  void print_RegionHistoryTable() {
    for (auto entry : table_entry) {
      printf("Region: %d\n", entry.first);
      entry.second->print_entry();
    }
  }
  // as not all regions will be used,
  // we use set instead of vector
  std::map<ADDR, entry *> table_entry;
  RegionHistoryTable() { table_entry.clear(); }
  void clear() { table_entry.clear(); }
  void access(ADDR addr) {
    ADDR region_id = get_region_id(addr);
    entry *e = NULL;
    // first access to this region
    if (table_entry.find(region_id) == table_entry.end()) {
      e = new entry;
      table_entry.insert(pair<ADDR, entry *>(region_id, e));
    } else {
      e = table_entry.find(region_id)->second;
    }
    e->update(addr);
  }
} RHT;

struct RegionReuseHistory {
  std::vector<ADDR> region_history;
  std::vector<int> region_reuse_hist;
  RegionReuseHistory() {
    region_reuse_hist.clear();
    for (int i = 0; i < MAX_REGION_REUSE_DISTANCE + 1; i++)
      region_reuse_hist.push_back(0);
  }
  void print_hist() {
    for (int i = 0; i < MAX_REGION_REUSE_DISTANCE + 1; i++) {
      if (region_reuse_hist[i]) {
        printf("reuse dis: %d cnt: %d\n", i, region_reuse_hist[i]);
      }
    }
  }
  void access(ADDR addr) {
    // get region
    ADDR region_id = get_region_id(addr);
    int reuse_distance = MAX_REGION_REUSE_DISTANCE;

    for (int pos = 0; pos < region_history.size(); pos++) {
      if (region_history[pos] == region_id) {
        reuse_distance = region_history.size() - pos;
      }
    }
    // remove curr addr from history and insert it to the back
    region_history.erase(
        std::remove(region_history.begin(), region_history.end(), region_id),
        region_history.end());
    region_history.push_back(region_id);
    // update reuse histogram
    region_reuse_hist[reuse_distance]++;
    // remove the oldest history
    if (region_history.size() > MAX_REGION_REUSE_DISTANCE) {
      region_history.erase(region_history.begin());
    }
  }
} RRH;
// update statistical information for new address
void mem_access(ADDR addr) {
  // update inter-region reuse information
  RRH.access(addr);
  // update intra-region stride information
  RHT.access(addr);
  // RHT.print_RegionHistoryTable();
  // print_CSTs();
}

/*
input: file1: memory trace for physical address
           file2: generated proxy memory trace
*/
int main(int argc, char *argv[]) {
  // input: memory trace
  ifstream tfile;
  tfile.open(argv[1]);

  ADDR addr;
  while (!tfile.eof()) {
    string buffer;
    getline(tfile, buffer);

    if (buffer.find("Addr") == string::npos)
      continue;
    int pos0 = buffer.find(": ");
    stringstream field(buffer.substr(pos0 + 2, buffer.length() - pos0 + 2));

    field >> dec >> addr;
    mem_access(addr);
  }

  tfile.close();
  // ofs.close();
  return 0;
}
