#ifndef __EXTERNS_H__
#define __EXTERNS_H__

#include "global_types.h"

extern void die_message(const char *msg);

extern uns64       USE_IMAT_TRACES;
extern uns64       LINESIZE;     
extern uns64       OS_PAGESIZE; 
extern uns64       OS_NUM_RND_TRIES;


extern uns64       L3_LATENCY;
extern uns64       L3_PERFECT;

extern uns64       MEM_SIZE_MB;
extern uns64       MEM_RSRV_MB;
extern uns64       MEM_CHANNELS;
extern uns64       MEM_BANKS;   
extern uns64       MEM_PAGESIZE;
extern uns64       MEM_T_ACT;   
extern uns64       MEM_T_RAS;   
extern uns64       MEM_T_CAS;   
extern uns64       MEM_T_RP;    
extern uns64       MEM_T_BURST; 
extern uns64       MEM_T_PHY; 
extern uns64       MEM_CLOSEPAGE;
extern uns64       DRAM_MINIMALIST_SIZE;



// these are non-param global variables
extern int         num_threads;

#endif

