#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#include "externs.h"
#include "dram.h"

#define DRAM_STRIPE_BANKS_TO_CHANNELS 1
#define DRAM_STRIPE_ROWBUFS_TO_BANKS 1

uns DRAM_ID = 0;

extern uns64 cycle;

////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////

DRAM *dram_new(uns64 memsize, uns64 num_channels, uns64 num_banks, uns64 rowbuf_size,
               uns64 t_ACT, uns64 t_CAS, uns64 t_RAS, uns64 t_RP, uns64 t_BURST)
{
  uns ii;

  DRAM *d = (DRAM *)calloc(1, sizeof(DRAM));
  d->num_banks = num_banks;
  d->t_ACT = t_ACT;
  d->t_CAS = t_CAS;
  d->t_RAS = t_RAS;
  d->t_RP = t_RP;
  d->t_BURST = t_BURST;

  d->memsize = memsize;
  d->rowbuf_size = rowbuf_size;
  d->num_channels = num_channels;
  d->num_rowbufs = memsize / rowbuf_size;
  d->lines_in_rowbuf = rowbuf_size / LINESIZE; // DRAM access granularity is linesize
  d->rowbufs_in_bank = d->num_rowbufs / d->num_banks;
  d->lines_in_mem = memsize / LINESIZE;
  d->rowbufs_in_mem = d->memsize / d->rowbuf_size;

  d->banks_in_channel = num_banks / num_channels;

  for (ii = 0; ii < num_banks; ii++)
  {
    d->bank[ii].sb = scoreboard_new(t_CAS);
  }

  printf("Banks: %llu and Channels: %llu\n", num_banks, num_channels);

  d->id = DRAM_ID++;
  sprintf(d->name, "DRAM_%02d", d->id);

  return d;
}

////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////

uns64 dram_service(DRAM *d, Addr lineaddr, DRAM_ReqType type, double num_lineburst, uns64 in_cycle, ACTinfo *act_info)
{
  Flag rowbuf_conflict = FALSE, rowbuf_empty = FALSE, rowbuf_hit = FALSE;
  uns64 delay = 0, delay_rc = 0, delay_re = 0, delay_rh = 0;
  uns64 mybankid, myrowbufid, mychannelid;

  assert(lineaddr < d->lines_in_mem); //we get physical DRAM lineid as input

  // parse address bits and get my bank
  dram_parseaddr(d, lineaddr, &myrowbufid, &mybankid, &mychannelid);
  DRAM_Bank *mybank = &d->bank[mybankid];

  // check for rowbuf conflict
  if (mybank->valid && (mybank->rowbufid != myrowbufid))
  {
    uns64 ras_wait = 0;
    uns64 ras_done = in_cycle - mybank->rowbufopen_cycle;
    if (ras_done < d->t_RAS)
    {
      ras_wait = (d->t_RAS - ras_done);
    }
    delay_rc = scoreboard_service(mybank->sb, in_cycle + delay + ras_wait, d->t_RP);
    delay += delay_rc;
    mybank->valid = FALSE;
    rowbuf_conflict = TRUE;
  }

  // check for rowbuf empty
  if (mybank->valid == FALSE)
  {
    delay_re = scoreboard_service(mybank->sb, in_cycle + delay, d->t_ACT);
    mybank->rowbufopen_cycle = in_cycle + delay;
    mybank->valid = TRUE;
    mybank->rowbufid = myrowbufid;
    rowbuf_empty = TRUE;
    delay += delay_re;
  }

  // check for rowbuf hit
  if (mybank->valid && (mybank->rowbufid == myrowbufid))
  {
    delay_rh = scoreboard_service(mybank->sb, in_cycle + delay, d->t_CAS);
    delay += delay_rh;
    rowbuf_hit = TRUE;
  }

  //wait for channel dbus
  double dbus_time = d->t_BURST * num_lineburst; // assume constant latency BUS for now
  delay += dbus_time;

  // close the page if close page policy
  if (d->close_page_mode)
  {
    dram_closepage(d, lineaddr, in_cycle + delay);
  }

  // update stats from here
  DRAM_BankAccessType rowbuf_outcome;

  if (rowbuf_conflict)
  {
    rowbuf_outcome = DRAM_ROWBUF_CONFLICT;
  }
  else
  {
    if (rowbuf_empty)
    {
      rowbuf_outcome = DRAM_ROWBUF_EMPTY;
    }
    else
    {
      rowbuf_outcome = DRAM_ROWBUF_HIT;
    }
  }

  d->s_rowbuf_outcome[type][rowbuf_outcome]++;
  d->s_access_type[type]++;
  d->s_delaysum_type[type] += delay;

  //---- Update dram_act_info for dram access ----
  if (act_info)
  {
    act_info->rowID = myrowbufid;
    act_info->bankID = mybankid;
    act_info->isACT = (rowbuf_outcome != DRAM_ROWBUF_HIT) ? true : false;
  }
  return delay;
}

////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////

void dram_closepage(DRAM *d, Addr lineaddr, uns64 in_cycle)
{
  uns64 mybankid, myrowbufid, mychannelid;

  assert(lineaddr < d->lines_in_mem); //we get physical DRAM lineid as input
  dram_parseaddr(d, lineaddr, &myrowbufid, &mybankid, &mychannelid);
  DRAM_Bank *mybank = &d->bank[mybankid];

  // enforcing row cycle window
  uns64 ras_wait = 0;
  uns64 ras_done = in_cycle - mybank->rowbufopen_cycle;

  if (ras_done < d->t_RAS)
  {
    ras_wait = (d->t_RAS - ras_done);
  }

  if (mybank->valid)
  {
    scoreboard_service(mybank->sb, in_cycle + ras_wait, d->t_RP);
    mybank->valid = FALSE;
  }
}

////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////

void dram_print_stats(DRAM *d)
{
  char header[256];
  uns64 totaccess = 0;
  uns64 totdelaysum = 0;
  uns64 avgdelaysum = 0;
  uns64 totACT = 0;

  uns64 rd_rowbuf_hits = d->s_rowbuf_outcome[DRAM_REQ_RD][DRAM_ROWBUF_HIT];
  uns64 rd_rowbuf_empty = d->s_rowbuf_outcome[DRAM_REQ_RD][DRAM_ROWBUF_EMPTY];
  uns64 rd_rowbuf_conflict = d->s_rowbuf_outcome[DRAM_REQ_RD][DRAM_ROWBUF_CONFLICT];

  double rd_rbuf_hitperc = 0, rd_rbuf_emptyperc = 0, rd_rbuf_conflictperc = 0;
  double avg_rd_delay = 0, avg_rd_wait = 0;

  if (d->s_access_type[DRAM_REQ_RD])
  {
    rd_rbuf_hitperc = 100.0 * (double)(rd_rowbuf_hits) / (double)(d->s_access_type[DRAM_REQ_RD]);
    rd_rbuf_emptyperc = 100.0 * (double)(rd_rowbuf_empty) / (double)(d->s_access_type[DRAM_REQ_RD]);
    rd_rbuf_conflictperc = 100.0 * (double)(rd_rowbuf_conflict) / (double)(d->s_access_type[DRAM_REQ_RD]);
    avg_rd_delay = (double)(d->s_delaysum_type[DRAM_REQ_RD]) / (double)(d->s_access_type[DRAM_REQ_RD]);
    avg_rd_wait = dram_calc_avgwait(d, DRAM_REQ_RD);
  }

  totACT = rd_rowbuf_empty + rd_rowbuf_conflict;

  uns ii;

  for (ii = 0; ii < NUM_DRAM_REQTYPE; ii++)
  {
    totaccess += d->s_access_type[ii];
    totdelaysum += d->s_delaysum_type[ii];
  }

  if (totaccess)
  {
    avgdelaysum = totdelaysum / totaccess;
  }

  sprintf(header, "%s", d->name);

  printf("\n%s_TOTACCESS        \t : %llu", header, totaccess);
  printf("\n%s_TOTDELAYAVG      \t : %llu", header, avgdelaysum);
  printf("\n%s_RDACCESS         \t : %llu", header, d->s_access_type[DRAM_REQ_RD]);
  printf("\n%s_RDDELAYAVG       \t : %5.2f", header, avg_rd_delay);
  printf("\n%s_RDWAITAVG        \t : %5.2f", header, avg_rd_wait);
  printf("\n%s_RD_RBUF_HITPERC  \t : %5.2f", header, rd_rbuf_hitperc);
  printf("\n%s_RD_RBUF_EMPTYPERC\t : %5.2f", header, rd_rbuf_emptyperc);
  printf("\n%s_RD_RBUF_CONFLPERC\t : %5.2f", header, rd_rbuf_conflictperc);
  printf("\n%s_TOTAL_ACT        \t : %llu", header, totACT);

  printf("\n");
}

////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////

double dram_calc_avgwait(DRAM *d, DRAM_ReqType type)
{
  uns64 sum_banktimes = 0;
  uns64 sum_bustimes = 0;
  uns64 sum_bank_bus = 0;

  //sum of bank hit
  uns64 bank_hit_time = d->t_CAS;
  sum_banktimes += (d->s_rowbuf_outcome[type][DRAM_ROWBUF_HIT] * bank_hit_time);

  //sum of bank empty
  uns64 bank_empty_time = d->t_ACT + d->t_CAS;
  sum_banktimes += (d->s_rowbuf_outcome[type][DRAM_ROWBUF_EMPTY] * bank_empty_time);

  //sum of bank conflict
  uns64 bank_conf_time = d->t_ACT + d->t_CAS; // not including d->t_RP;
  sum_banktimes += (d->s_rowbuf_outcome[type][DRAM_ROWBUF_CONFLICT] * bank_conf_time);

  //sum of bus delay
  sum_bustimes = d->s_access_type[type] * d->t_BURST;

  sum_bank_bus = sum_banktimes + sum_bustimes;

  assert(sum_bank_bus <= d->s_delaysum_type[type]);

  double avgwait = (d->s_delaysum_type[type] - sum_bank_bus) / d->s_access_type[type];

  return avgwait;
}

////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////

void dram_parseaddr(DRAM *d, Addr lineaddr, uns64 *myrowbufid, uns64 *mybankid, uns64 *mychannelid)
{
  *myrowbufid = lineaddr / d->lines_in_rowbuf;

  *mybankid = *myrowbufid / d->rowbufs_in_bank; //consecutive rowbufs are in same bank;

  if (DRAM_STRIPE_ROWBUFS_TO_BANKS)
  {
    *mybankid = *myrowbufid % d->num_banks; //consecutive rowbufs go to diff bank
  }

  *mychannelid = *mybankid / d->banks_in_channel;

  if (DRAM_STRIPE_BANKS_TO_CHANNELS)
  {
    *mychannelid = *mybankid % d->num_channels; // consecutive banks go to diff channel
  }
}

////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////

void dram_get_neighbor_lineaddr(DRAM *d, uns64 rowbufid, uns64 *row_prev_lineaddr, uns64 *row_next_lineaddr)
{
  Addr rowbufid_next, rowbufid_prev;
  if (!DRAM_STRIPE_ROWBUFS_TO_BANKS)
  {
    //consecutive rows in same bank
    rowbufid_next = (rowbufid + 1);
    rowbufid_prev = (rowbufid - 1);
    //Ignore ends of a bank.
    if ((rowbufid_next / d->rowbufs_in_bank) != (rowbufid / d->rowbufs_in_bank))
      rowbufid_next = -1;
    if ((rowbufid_prev / d->rowbufs_in_bank) != (rowbufid / d->rowbufs_in_bank))
      rowbufid_prev = -1;
  }
  else
  {
    //consecutive rowbufs go to diff banks
    Addr rowbufid_delta = 0;
    rowbufid_delta = d->num_banks;
    rowbufid_next = (rowbufid + rowbufid_delta);
    rowbufid_prev = (rowbufid - rowbufid_delta);
    //Ignore ends of a bank.
    if ((rowbufid_next / d->rowbufs_in_bank) != (rowbufid / d->rowbufs_in_bank))
      rowbufid_next = -1;
    if ((rowbufid_prev / d->rowbufs_in_bank) != (rowbufid / d->rowbufs_in_bank))
      rowbufid_prev = -1;
  }

  //Update lineaddrs
  if (rowbufid_prev != (Addr)-1)
    *row_prev_lineaddr = rowbufid_prev * d->lines_in_rowbuf;
  else
    *row_prev_lineaddr = -1;

  if (rowbufid_next != (Addr)-1)
    *row_next_lineaddr = rowbufid_next * d->lines_in_rowbuf;
  else
    *row_next_lineaddr = -1;
}

////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////

Addr dram_get_bankid(DRAM *d, uns64 rowbufid)
{
  Addr bankid = rowbufid / d->rowbufs_in_bank; //consecutive rowbufs are in same bank;
  if (DRAM_STRIPE_ROWBUFS_TO_BANKS)
  {
    bankid = rowbufid % d->num_banks; //consecutive rowbufs go to diff bank
  }
  return bankid;
}

////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////
