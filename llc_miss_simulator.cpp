#include <stdio.h>
#include <cstdlib>
#include <x86intrin.h>
#define LINE_SIZE 4096
#define ACCURACY 1000
#define INST_CNT 50000000
double miss_rate = 0.9;
int *hit;
int *miss;
void init()
{
    srand(42);
    hit = new int;
    miss = new int;
}
void flush_from_llc(void *p)
{
    _mm_clflush(p);
    return;
}
void llc_simulator(size_t inst_num, double miss_rate)
{
    //int h = 0;
    //int m = 0;
    for (int i = 0; i < inst_num; i++)
    {
        float p = rand() % ACCURACY / (float)ACCURACY;
        if (p > miss_rate)
        {
            // hit
            (*hit)++;
            //h++;
        }
        else
        {
            // miss
            flush_from_llc(miss);
            (*miss)++;
            //m++;
        }
    }
    //printf("hit: %d miss: %d\n", h, m);
}
int main()
{
    init();
    llc_simulator(INST_CNT, miss_rate);
}