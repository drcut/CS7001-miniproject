#ifndef SCOREBOARD_C
#define SCOREBOARD_C

#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

#include "scoreboard.h"

#define SCOREBOARD_RESET_INTERVAL 0

extern uns64 cycle;

uns SCOREBOARD_ID=0;


//////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

scoreboard *scoreboard_new(uns service_cycle){
    scoreboard *sb = (scoreboard *) calloc (1, sizeof (scoreboard));
    sb->service_cycle = service_cycle;
    sb->next_ready_cycle = 0;
    sb->id = SCOREBOARD_ID++;
    sb->reset_interval=service_cycle*SCOREBOARD_RESET_INTERVAL;
    return sb;
}

//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////

uns64     scoreboard_service_reg (scoreboard *s, uns64 cycle){
  return scoreboard_service(s,cycle,s->service_cycle);
}

//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////

uns64     scoreboard_service(scoreboard *s, uns64 cycle, uns service_time){
  uns64 service_start_cycle = s->next_ready_cycle;
  int64 delta;
  

    // if object was ready in past, service starts this cycle
    if(cycle > s->next_ready_cycle){
	service_start_cycle = cycle;
    }

    s->next_ready_cycle = service_start_cycle + (uns64)service_time;
    delta = (int64)s->next_ready_cycle - (int64)cycle; 

    // resetting scoreboard infrequently
    if(SCOREBOARD_RESET_INTERVAL){
      if(cycle - s->last_reset_cycle > s->reset_interval){
	s->next_ready_cycle=cycle;
	s->last_reset_cycle=cycle;
      }
    }

    return (uns64)delta;
}

//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////

uns64     scoreboard_get_waitcycles(scoreboard *s, uns64 cycle){
  uns64 wait;
  
  // if object was ready in past, service starts this cycle
  if(cycle > s->next_ready_cycle){
    wait=0;
  }else{
    wait= s->next_ready_cycle-cycle;
  }

  return wait;
}

//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////

#endif // SCOREBOARD_C
