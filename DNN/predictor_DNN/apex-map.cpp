#include <cstring>
#include <iostream>
#include <stdlib.h>
using namespace std;
#define MAXL (1 << 20)

int pos0;
int pos1;
int pos2;
int pos3;
void initIndexArray(double reuse_rate) {
  int last_acc = -1;
  int pos;
  for (int i = 0; i < 4; i++) {
    // temporal locality
    double r = ((double)rand() / (RAND_MAX));
    if (r < reuse_rate && last_acc != -1) {
      pos = last_acc;
    } else {
      pos = rand() % MAXL;
    }
    last_acc = pos;
    if (i == 0) {
      pos0 = pos;
    } else if (i == 1) {
      pos1 = pos;
    } else if (i == 2) {
      pos2 = pos;
    } else {
      pos3 = pos;
    }
  }
  printf("pos: %d %d %d %d\n", pos0, pos1, pos2, pos3);
}

int main(int argc, char **argv) {
  srand(42);
  int *data = new int[MAXL];
  // get reuse rate
  double reuse_rate = std::atof(argv[1]);
  // get consecutive length
  int vec_len = std::atoi(argv[2]);
  // get length of generated sequence
  int gen_len = std::atoi(argv[3]);

  int tmp = 0;
  int last_acc = -1;
  for (int i = 0; i < gen_len / (4 * vec_len); i++) {
    initIndexArray(reuse_rate);

    for (int j = 0; j < vec_len; j++) {
      tmp += data[(pos0 + j) % MAXL];
      tmp += data[(pos1 + j) % MAXL];
      tmp += data[(pos2 + j) % MAXL];
      tmp += data[(pos3 + j) % MAXL];
    }
  }
  return 0;
}