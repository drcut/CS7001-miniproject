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
  std::map<string, int> stride_cnt; // record the number of each stride
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
      stride_cnt.insert(std::pair<string, int>(previous_strides, 0));
    }
    // update the corresponding entry
    auto curr_map = entries.find(previous_strides)->second;
    if (curr_map->find(curr_strides) == curr_map->end()) {
      curr_map->insert(std::pair<ADDR, int>(curr_strides, 0));
    }
    curr_map->find(curr_strides)->second += 1;
    stride_cnt.find(previous_strides)->second += 1;
  }
  void clear() { entries.clear(); }
  ADDR smapling_stride() {
    // this function should only be used for the first CST (as we only sample a
    // single stride)
    int sum = 0;
    for (auto e : stride_cnt) {
      sum += e.second;
    }
    int r = rand() % sum;
    int acc = 0;
    for (auto e : stride_cnt) {
      acc += e.second;
      if (acc >= r) {
        // translate string to stride
        return std::stoi(e.first);
      }
    }
  }
  bool find_pattern(std::string history_pattern) {
    return entries.find(history_pattern) != entries.end();
  }
  ADDR sample_stride(std::string history_pattern) {
    if (entries.find(history_pattern) == entries.end()) {
      printf("ERROR\n");
      exit(1);
    }
    int r = rand() % stride_cnt.find(history_pattern)->second;
    int acc = 0;
    for (auto e : *entries.find(history_pattern)->second) {
      acc += e.second;
      if (acc >= r)
        return e.first;
    }
    printf("ERROR\n");
    exit(1);
  }
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
  int region_visit;
  RegionReuseHistory() {
    region_reuse_hist.clear();

    region_visit = 0;
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
    region_visit++;
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
  int sampling_reuse_distance() {
    int r = rand() % region_visit;
    int acc = 0;
    for (int i = 0; i < MAX_REGION_REUSE_DISTANCE + 1; i++) {
      acc += region_reuse_hist[i];
      if (acc >= r)
        return i;
    }
  }
} RRH;

// use another structure, similar like RHT,
// to record the history during generation
struct GenerateHistory {
  std::map<ADDR, ADDR> last_access;
  std::map<ADDR, std::vector<ADDR>> past_stride;
  ADDR get_last_addr(ADDR region_id) {
    if (last_access.find(region_id) == last_access.end()) {
      return -1;
    } else {
      return last_access.find(region_id)->second;
    }
  }
  std::vector<ADDR> get_past_strides(ADDR region_id) {
    if (past_stride.find(region_id) == past_stride.end()) {
      return std::vector<ADDR>();
    } else {
      return past_stride.find(region_id)->second;
    }
  }
  void insert_new_access(ADDR region_id, ADDR access_addr, ADDR new_stride) {
    if (last_access.find(region_id) == last_access.end()) {
      last_access.insert(std::pair<ADDR, ADDR>(region_id, access_addr));
    } else {
      last_access.find(region_id)->second = access_addr;
    }
    if (past_stride.find(region_id) == past_stride.end()) {
      past_stride.insert(
          std::pair<ADDR, std::vector<ADDR>>(region_id, std::vector<ADDR>()));
    }
    std::vector<ADDR> stride = past_stride.find(region_id)->second;
    stride.push_back(new_stride);
    if (stride.size() > MAX_CASSCADING_DEGREE + 1) {
      stride.erase(stride.begin());
    }
    past_stride.find(region_id)->second = stride;
  }
};

void generate_proxy(int inst_num) {
  int curr_region_id = 0;
  GenerateHistory proxy_history;
  std::vector<ADDR> region_history;
  for (int i = 0; i < inst_num; i++) {
    // select region
    int region_id;
    // get reuse distance
    int reuse_distance = RRH.sampling_reuse_distance();
    if (reuse_distance == MAX_REGION_REUSE_DISTANCE ||
        reuse_distance > region_history.size()) {
      // use a new region
      region_id = curr_region_id++;
    } else {
      // reuse the previous region
      region_id = region_history[region_history.size() - reuse_distance];
      // remove curr addr from history
      region_history.erase(
          std::remove(region_history.begin(), region_history.end(), region_id),
          region_history.end());
    }
    region_history.push_back(region_id);
    // remove the oldest history
    if (region_history.size() > MAX_REGION_REUSE_DISTANCE) {
      region_history.erase(region_history.begin());
    }
    // select the intral-region stride
    ADDR last_addr = proxy_history.get_last_addr(region_id);
    ADDR generated_mem_access;
    ADDR stride;

    bool find_history_pattern = false;
    if (last_addr != -1) {
      // use stride history to find the top match
      std::vector<ADDR> past_strides =
          proxy_history.get_past_strides(region_id);
      for (int dis = past_strides.size(); dis > 0; dis--) {
        // generate string of the history
        std::string history_pattern = "";
        for (int j = 0; j < dis; j++) {
          history_pattern +=
              std::to_string(past_strides[past_strides.size() - dis + j]);
          history_pattern += ',';
        }
        // find history pattern in CST
        if (CSTs[dis].find_pattern(history_pattern)) {
          find_history_pattern = true;
          stride = CSTs[dis].sample_stride(history_pattern);
          generated_mem_access = last_addr + stride;
        }
      }
    }
    if (!find_history_pattern) {
      // first time visit this region
      // sampling strides
      stride = CSTs[1].smapling_stride();
      generated_mem_access = (region_id) << MACROBLOCK_SIZE_BITS;
    }
    printf("ADDR: %d\n", generated_mem_access);
    proxy_history.insert_new_access(region_id, generated_mem_access, stride);
  }
}

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
  generate_proxy(10);
  tfile.close();
  // ofs.close();
  return 0;
}
