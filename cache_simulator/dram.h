#ifndef DRAM_H
#define DRAM_H

#include "global_types.h"
#include "scoreboard.h"

#define MAX_DRAM_BANKS 256
#define MAX_ROWS_IN_BANK (1<<18)

typedef struct DRAM   DRAM;
typedef struct DRAM_Bank  DRAM_Bank;
typedef struct ACTinfo ACTinfo;

//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////

struct ACTinfo {
    Addr    rowID;
    Addr    bankID;
    bool    isACT;
};

struct DRAM_Bank{
    uns   id;
    Flag  valid; // row is valid 
    uns   rowbufid;
    uns64 rowbufopen_cycle; // to enforce RAS constraint
    scoreboard *sb;
};

typedef enum DRAM_RequestType_Enum {
    DRAM_REQ_RD=0,
    DRAM_REQ_WB=1,
    DRAM_REQ_RHMITIGATION=2,
    NUM_DRAM_REQTYPE=3
} DRAM_ReqType;

typedef enum DRAM_BankAcessType_Enum {
    DRAM_ROWBUF_HIT=0,
    DRAM_ROWBUF_EMPTY=1,
    DRAM_ROWBUF_CONFLICT=2,
    NUM_DRAM_ROWBUF_OUTCOME_TYPE=3
} DRAM_BankAccessType;


////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////

struct DRAM {
    uns         id; // we will have multiple dram, so id helps differentiate
    char name[256];

    DRAM_Bank   bank[MAX_DRAM_BANKS]; 

    Flag        close_page_mode; 
    
    uns64       t_ACT;
    uns64       t_CAS;
    uns64       t_RAS;
    uns64       t_RP;
    uns64       t_BURST;

    uns64       memsize;
    uns64       num_banks;
    uns64       rowbuf_size;
    uns64       num_channels;
    uns64       num_rowbufs;
  
    uns64       lines_in_rowbuf;
    uns64       rowbufs_in_mem;
    uns64       rowbufs_in_bank;
    uns64       lines_in_mem; // total lines in this DRAM module
    uns64       banks_in_channel;

    uns64       s_rowbuf_outcome[NUM_DRAM_REQTYPE][NUM_DRAM_ROWBUF_OUTCOME_TYPE];

    uns64       s_access_type[NUM_DRAM_REQTYPE];
    uns64       s_delaysum_type[NUM_DRAM_REQTYPE];
};

////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////

DRAM*   dram_new(uns64 memsize, uns64 num_channels, uns64 numbanks,  uns64 rowbuf_size, 
		 uns64 t_ACT, uns64 t_CAS, uns64 T_CAS, uns64 t_RP, uns64 t_BURST);

uns64   dram_service(DRAM *d, Addr lineaddr, DRAM_ReqType type, double num_lineburst, uns64 in_cycle, ACTinfo *act_info);
void    dram_closepage(DRAM *d, Addr lineaddr, uns64 in_cycle);
void    dram_print_stats(DRAM *d);
double  dram_calc_avgwait(DRAM *d, DRAM_ReqType type);

void    dram_parseaddr(DRAM *d, Addr lineaddr, uns64 *myrowbufid, uns64 *mybankid, uns64 *mychannelid);
void    dram_get_neighbor_lineaddr(DRAM *d, uns64 rowbufid, uns64* row_prev_lineaddr, uns64* row_next_lineaddr);
Addr    dram_get_bankid(DRAM *d, uns64 rowbufid);


////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////


#endif // DRAM_H
