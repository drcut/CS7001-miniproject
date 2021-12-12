#include <algorithm>
#include <assert.h>
#include <cstdlib>
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
const int base_region_offset = 3086126;
const int LINESIZE = 64;
const int MAX_REGION_REUSE_DISTANCE = 4096;
const int MAX_CASSCADING_DEGREE = 8;
const int LINE_IN_MEM = (MEM_SIZE_MB * 1024 * 1024) / LINESIZE;

const int MACROBLOCK_SIZE_BITS = 6; // minus LINESIZE
const int REGION_SIZE = (1 << 6);
int MACROBLOCK_CNT = 0;
const int Region_count = ((MEM_SIZE_MB << 20) >> 6) / 64;

ADDR get_region_id(ADDR addr) { return (addr >> MACROBLOCK_SIZE_BITS); }

struct CascadedStrideTable {
  std::map<string, std::map<ADDR, double> *> entries;
  std::map<string, double> stride_cnt; // record the number of each stride
  CascadedStrideTable(){};
  void print() {
    for (auto entry : entries) {
      printf("history: %s\n", entry.first.c_str());
      for (auto e : *entry.second) {
        printf("next stride: %d cnt: %lf\n", e.first, e.second);
      }
    }
  }
  // normalize counter to probabilistic
  void normalize() {
    // calculate stride cnt
    for (auto entry : entries) {
      double sum = 0.0;
      std::map<ADDR, double>::iterator itr;
      for (itr = entry.second->begin(); itr != entry.second->end(); itr++) {
        sum += itr->second;
      }
      for (itr = entry.second->begin(); itr != entry.second->end(); itr++) {
        itr->second /= sum;
      }
    }
    double sum = 0.0;
    for (auto entry : stride_cnt) {
      sum += entry.second;
    }
    for (auto entry : stride_cnt) {
      entry.second /= sum;
    }
  }
  void insert(string previous_strides, ADDR curr_strides,
              bool need_stride_cnt = false) {
    std::map<ADDR, double> *curr_map = NULL;

    auto find_entry = entries.find(previous_strides);
    // this previous_pattern is first shown
    if (find_entry == entries.end()) {
      curr_map = new std::map<ADDR, double>;
      entries.insert(std::pair<string, std::map<ADDR, double> *>(
          previous_strides, curr_map));
      if (need_stride_cnt) {
        stride_cnt.insert(std::pair<string, double>(previous_strides, 0));
      }
      curr_map->insert(std::pair<ADDR, double>(curr_strides, 1));
      return;
    } else {
      curr_map = find_entry->second;
    }
    // update the corresponding entry
    auto it = curr_map->find(curr_strides);
    if (it == curr_map->end()) {
      curr_map->insert(std::pair<ADDR, double>(curr_strides, 1));
    } else {
      it->second += 1;
    }
    if (need_stride_cnt) {
      stride_cnt.find(previous_strides)->second += 1.0;
    }
  }
  void clear() { entries.clear(); }
  ADDR sampling_first_stride() {
    // this function should only be used for the first CST (as we only sample
    // a single stride)
    double r = ((float)rand()) / (double)RAND_MAX;
    double acc = 0;
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
    double r = ((float)rand()) / (double)RAND_MAX;
    double acc = 0;
    for (auto e : *entries.find(history_pattern)->second) {
      acc += e.second;
      if (acc >= r) {
        return e.first;
      }
    }
    printf("ERROR\n");
    exit(1);
  }
};

struct InterStrideTable {
  CascadedStrideTable CST[MAX_CASSCADING_DEGREE + 1];
  void normalize_CST() {
    for (int i = 1; i < MAX_CASSCADING_DEGREE + 1; i++) {
      CST[i].normalize();
    }
  }
  // entry inside the table
  std::vector<ADDR> past_strides;

  void insert(ADDR stride) {
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
      CST[dis].insert(key, stride, dis == 1);
    }
    past_strides.push_back(stride);
    if (past_strides.size() > MAX_CASSCADING_DEGREE + 1) {
      past_strides.erase(past_strides.begin());
    }
  }
} Inter_region;

struct IntraStrideTable {
  CascadedStrideTable CST[MAX_CASSCADING_DEGREE + 1];
  void normalize_CST() {
    for (int i = 1; i < MAX_CASSCADING_DEGREE + 1; i++) {
      CST[i].normalize();
    }
  }
  // entry inside the table
  std::map<ADDR, std::vector<ADDR>> past_strides;

  void insert(ADDR region_id, ADDR stride) {
    if (past_strides.find(region_id) == past_strides.end()) {
      std::vector<ADDR> *tmp = new std::vector<ADDR>(stride);
      past_strides.insert(std::pair<ADDR, std::vector<ADDR>>(region_id, *tmp));
      return;
    }
    // TODO: update the cascaded stride tables
    auto strides = past_strides.find(region_id)->second;
    for (int dis = 1; dis < MAX_CASSCADING_DEGREE + 1; dis++) {
      if (dis > strides.size())
        break;
      // generate string for previous access pattern
      std::string key = "";
      for (int j = 0; j < dis; j++) {
        key += std::to_string(strides[strides.size() - dis + j]);
        key += ',';
      }
      CST[dis].insert(key, stride, dis == 1);
    }
    strides.push_back(stride);
    if (strides.size() > MAX_CASSCADING_DEGREE + 1) {
      strides.erase(strides.begin());
    }
  }
} Intra_region;

// record the Intra/Inter region during generation
struct GenerateHistory {
  std::vector<ADDR> inter_region_stride_history;
  std::map<ADDR, std::vector<ADDR>> intra_region_stride_history;
  ADDR last_region;
  std::map<ADDR, ADDR> last_access_addr;
  ADDR get_last_region() { return last_region; }
  ADDR get_last_address(ADDR region_id) {
    auto addr = last_access_addr.find(region_id);
    if (addr == last_access_addr.end()) {
      last_access_addr.insert(std::pair<ADDR, ADDR>(
          region_id, (region_id << MACROBLOCK_SIZE_BITS) + 128));
      return (region_id << MACROBLOCK_SIZE_BITS) + 128;
    }
    return addr->second;
  }
  bool has_visit_region(ADDR region_id) {
    return intra_region_stride_history.find(region_id) !=
           intra_region_stride_history.end();
  }
  std::vector<ADDR> get_past_inter_region_strides() {
    return inter_region_stride_history;
  }
  std::vector<ADDR> get_past_intra_region_strides(ADDR region_id) {
    auto res = intra_region_stride_history.find(region_id);
    if (res == intra_region_stride_history.end()) {
      return std::vector<ADDR>();
    } else {
      return res->second;
    }
  }
  void insert_new_access(ADDR region_id, ADDR address, ADDR inter_region_stride,
                         ADDR intra_region_stride) {
    // insert new inter-region stride
    inter_region_stride_history.push_back(inter_region_stride);
    if (inter_region_stride_history.size() > MAX_CASSCADING_DEGREE) {
      inter_region_stride_history.erase(inter_region_stride_history.begin());
    }

    // insert new intra-region stride
    if (last_access_addr.find(region_id) == last_access_addr.end()) {
      last_access_addr.insert(std::pair<ADDR, ADDR>(region_id, address));
    } else {
      last_access_addr.find(region_id)->second = address;
    }
    if (intra_region_stride_history.find(region_id) ==
        intra_region_stride_history.end()) {
      intra_region_stride_history.insert(
          std::pair<ADDR, std::vector<ADDR>>(region_id, std::vector<ADDR>()));
    }
    std::vector<ADDR> stride =
        intra_region_stride_history.find(region_id)->second;
    stride.push_back(intra_region_stride);
    if (stride.size() > MAX_CASSCADING_DEGREE) {
      stride.erase(stride.begin());
    }
    intra_region_stride_history.find(region_id)->second = stride;
    last_region = region_id;
    auto it = last_access_addr.find(region_id);
    if (it == last_access_addr.end()) {
      last_access_addr.insert(std::pair<ADDR, ADDR>(region_id, address));
    } else {
      it->second = address;
    }
  }
};

void generate_proxy(int inst_num, ofstream &fout) {

  // initilization CST table
  Inter_region.normalize_CST();
  Intra_region.normalize_CST();

  GenerateHistory proxy_history;
  std::vector<ADDR> region_history;
  int inter_region_stride;
  int intra_region_stride;
  int region_id;
  for (int inst = 0; inst < inst_num; inst++) {
    // select the inter-region stride
    std::vector<ADDR> past_inter_region_strides =
        proxy_history.get_past_inter_region_strides();
    if (past_inter_region_strides.empty()) {
      region_id = 2003741; // random number
      // sampling stride
      inter_region_stride = Inter_region.CST[1].sampling_first_stride();
    } else {
      for (int dis = past_inter_region_strides.size(); dis > 0; dis--) {
        // generate string of the history
        std::string history_pattern = "";
        for (int j = 0; j < dis; j++) {
          history_pattern += std::to_string(
              past_inter_region_strides[past_inter_region_strides.size() - dis +
                                        j]);
          history_pattern += ',';
        }
        // find history pattern in CST
        if (Inter_region.CST[dis].find_pattern(history_pattern)) {
          inter_region_stride =
              Inter_region.CST[dis].sample_stride(history_pattern);
          break;
        }
      }
      region_id = proxy_history.last_region + inter_region_stride;
    }
    region_id = (region_id % Region_count + Region_count) % Region_count;
    // select the intra-region stride
    ADDR intra_region_stride;
    bool find_history_pattern = false;
    if (proxy_history.has_visit_region(region_id)) {
      // use stride history to find the top match
      std::vector<ADDR> past_strides =
          proxy_history.get_past_intra_region_strides(region_id);
      for (int dis = past_strides.size(); dis > 0; dis--) {
        // generate string of the history
        std::string history_pattern = "";
        for (int j = 0; j < dis; j++) {
          history_pattern +=
              std::to_string(past_strides[past_strides.size() - dis + j]);
          history_pattern += ',';
        }
        // find history pattern in CST
        if (Intra_region.CST[dis].find_pattern(history_pattern)) {
          find_history_pattern = true;
          intra_region_stride =
              Intra_region.CST[dis].sample_stride(history_pattern);
          break;
        }
      }
    }

    if (!find_history_pattern) {
      // first time visit this region
      // sampling strides
      intra_region_stride = Intra_region.CST[1].sampling_first_stride();
    }
    ADDR generated_mem_access =
        proxy_history.get_last_address(region_id) + intra_region_stride;
    if ((int)generated_mem_access < 0 || generated_mem_access >= LINE_IN_MEM) {
      generated_mem_access =
          (generated_mem_access % LINE_IN_MEM + LINE_IN_MEM) % LINE_IN_MEM;
    }
    fout << generated_mem_access << std::endl;
    proxy_history.insert_new_access(region_id, generated_mem_access,
                                    inter_region_stride, intra_region_stride);
  }
}

ADDR profiling_inter_region_last_access = -1;
std::map<ADDR, ADDR> profiling_intra_region_last_access;
// update statistical information for new address
void profiling_access(ADDR addr) {
  ADDR region_id = get_region_id(addr);
  // update inter-region stride information
  if (profiling_inter_region_last_access != -1) {
    Inter_region.insert(region_id - profiling_inter_region_last_access);
  }
  profiling_inter_region_last_access = region_id;
  // update intra-region stride information
  auto it = profiling_intra_region_last_access.find(region_id);
  if (it == profiling_intra_region_last_access.end()) {
    profiling_intra_region_last_access.insert(
        std::pair<ADDR, ADDR>(region_id, addr));
  } else {
    Intra_region.insert(region_id, addr - it->second);
    it->second = addr;
  }
}

/*
input: file1: memory trace for physical address
           file2: generated proxy memory trace
*/
int main(int argc, char *argv[]) {
  srand(42);
  // input: real memory trace
  ifstream tfile;
  tfile.open(argv[1]);
  // output: proxy memory trace
  ofstream proxy_file;
  proxy_file.open(argv[2]);

  ADDR addr;
  timeval begin, end;
  int addr_cnt = 0;
  gettimeofday(&begin, NULL);
  while (!tfile.eof()) {
    string buffer;
    getline(tfile, buffer);
    try {
      addr = std::stoi(buffer);
      profiling_access(addr);
    } catch (...) {
      continue;
    }
    addr_cnt += 1;
    if (addr_cnt > 100000)
      break;
  }
  gettimeofday(&end, NULL);
  double elapsed =
      (end.tv_sec - begin.tv_sec) + ((end.tv_usec - begin.tv_usec) / 1000000.0);
  // RHT.print_RegionHistoryTable();
  printf("Time to read access %lf ms\n", elapsed * 1000.0);
  // RRH.print_hist();
  gettimeofday(&begin, NULL);
  generate_proxy(100000, proxy_file);
  gettimeofday(&end, NULL);
  elapsed =
      (end.tv_sec - begin.tv_sec) + ((end.tv_usec - begin.tv_usec) / 1000000.0);

  printf("Time to generate %lf ms\n", elapsed * 1000.0);
  tfile.close();
  return 0;
}
