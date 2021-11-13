#ifndef __PARAMS_H__
#define __PARAMS_H__

#include "global_types.h"
#include <string.h>

#define PRINT_DOTS 1
#define DOT_INTERVAL 20000000
#define MAX_THREADS 64

uns64 USE_IMAT_TRACES = 1;					  // second field is 4byte iaddr
uns64 TRACE_LIMIT = (2 * 1000 * 1000 * 1000); // Max 2B memory access
uns64 NUM_THREADS = 0;
uns64 MT_APP_THREADS = 0;
uns64 LINESIZE = 64;
uns64 OS_PAGESIZE = 4096;

uns64 L3_SIZE_KB = 8192;
uns64 L3_ASSOC = 16;
uns64 L3_LATENCY = 24; // cycles
uns64 L3_REPL = 0;	   //0:LRU 1:RND 2:SRRIP
uns64 L3_PERFECT = 0;  //simulate 100% hit rate for L3

uns64 MEM_SIZE_MB = 16384;
uns64 MEM_CHANNELS = 1;
uns64 MEM_BANKS = 16;				   // Total banks in memory, not  per channel (16x2 ranks here)
uns64 MEM_PAGESIZE = 8192;			   //Size of a DRAM Row
uns64 MEM_T_ACT = 44;				   // Time to activate a row (Row to Column Delay or tRCD in DDR4)
uns64 MEM_T_CAS = 44;				   // Column Access Strobe Time (CAS Latency or tCL in DDR4)
uns64 MEM_T_RP = 44;				   // Row Precharge Tiem (tRP in DDR4)
uns64 MEM_T_RAS = 102;				   // Minimum delay bet. Activate and Precharge commands (atleast tRCD + tRP in DDR4)
uns64 MEM_T_RC = MEM_T_RAS + MEM_T_RP; // RowCycle time or ACT-to-ACT delay (min time bet ACT to different rows in a bank).
uns64 MEM_T_BURST = 8;				   // proc cycles to burst out a line
uns64 MEM_CLOSEPAGE = 0;

//-----------------------

uns64 DRAM_MINIMALIST_SIZE = 0; // not using minimalist mapping

uns64 OS_NUM_RND_TRIES = 5; // page mapping (try X random invalids first)
uns64 RAND_SEED = 1234;

uns64 cycle;
uns64 last_printdot_cycle;
char addr_trace_filename[256][1024];
int num_threads = 0;

/***************************************************************************************
 * Functions
 ***************************************************************************************/

void die_usage()
{
	printf("Usage : sim [options] <FAT_trace_0> ... <FAT_trace_n> \n\n");
	printf("Trace driven DRAM-cache based memory-system simulator\n");

	printf("   Options\n");
	printf("               -imat                 Use IMAT traces (Default:0, but confirm)\n");
	printf("               -memphy      <num>    Set PHY to <num>cycles for main memory (Default: 0)\n");

	printf("               -l3sizekb    <num>    Set L3  Cache size to <num> KB (Default: 1MB)\n");
	printf("               -l3sizemb    <num>    Set L3  Cache size to <num> MB (Default: 1MB)\n");
	printf("               -l3assoc     <num>    Set L3  Cache assoc <num> (Default: 16)\n");
	printf("               -l3latency   <num>    Set L3  Caches latency in cycles (Default: 15)\n");
	printf("               -l3repl      <num>    Set L3  Caches replacement policy [0:LRU 1:RND 2:SRRIP] (Default:2)\n");
	printf("               -l3perfect            Set L3  to 100 percent hit rate(Default:off)\n");

	printf("               -os_numrndtries <num>    OS randomly tries X times to find invalid page (Default: 5)\n");

	printf("               -rand_seed <num>         Seed for Random PRNG (Default: 1234)\n");

	printf("               -mtapp         <num>    Number of threads in rate mode (Default: 1)\n");

	printf("                   -drammop         <num>     Size of DRAM Minimalist Open Page <LinesInRowBuf> (Default: off)\n");
	exit(0);
}

/***************************************************************************************
 * Functions
 ***************************************************************************************/

void die_message(const char *msg)
{
	printf("Error! %s. Exiting...\n", msg);
	exit(1);
}

/***************************************************************************************
 * Functions
 ***************************************************************************************/

void read_params(int argc, char **argv)
{
	int ii;

	//--------------------------------------------------------------------
	// -- Get command line options
	//--------------------------------------------------------------------
	for (ii = 1; ii < argc; ii++)
	{
		if (argv[ii][0] == '-')
		{
			if (!strcmp(argv[ii], "-h") || !strcmp(argv[ii], "-help"))
			{
				die_usage();
			}
			else if (!strcmp(argv[ii], "-l3perfect"))
			{
				L3_PERFECT = 1;
			}

			else if (!strcmp(argv[ii], "-l3repl"))
			{
				if (ii < argc - 1)
				{
					L3_REPL = atoi(argv[ii + 1]);
					ii += 1;
				}
			}

			else if (!strcmp(argv[ii], "-l3sizekb"))
			{
				if (ii < argc - 1)
				{
					L3_SIZE_KB = atoi(argv[ii + 1]);
					ii += 1;
				}
			}

			else if (!strcmp(argv[ii], "-l3sizemb"))
			{
				if (ii < argc - 1)
				{
					L3_SIZE_KB = 1024 * atoi(argv[ii + 1]);
					ii += 1;
				}
			}

			else if (!strcmp(argv[ii], "-l3assoc"))
			{
				if (ii < argc - 1)
				{
					L3_ASSOC = atoi(argv[ii + 1]);
					ii += 1;
				}
			}

			else if (!strcmp(argv[ii], "-l3latency"))
			{
				if (ii < argc - 1)
				{
					L3_LATENCY = atoi(argv[ii + 1]);
					ii += 1;
				}
			}

			else if (!strcmp(argv[ii], "-imat"))
			{
				USE_IMAT_TRACES = 1;
			}

			else if (!strcmp(argv[ii], "-rand_seed"))
			{
				if (ii < argc - 1)
				{
					RAND_SEED = atoi(argv[ii + 1]);
					ii += 1;
				}
			}

			else if (!strcmp(argv[ii], "-mtapp"))
			{
				if (ii < argc - 1)
				{
					MT_APP_THREADS = atoi(argv[ii + 1]);
					ii += 1;
				}
			}

			else if (!strcmp(argv[ii], "-drammop"))
			{
				if (ii < argc - 1)
				{
					DRAM_MINIMALIST_SIZE = atoi(argv[ii + 1]);
					ii += 1;
				}
			}

			else if (!strcmp(argv[ii], "-limit"))
			{
				if (ii < argc - 1)
				{
					TRACE_LIMIT = atoi(argv[ii + 1]);
					ii += 1;
				}
			}

			else
			{
				char msg[256];
				sprintf(msg, "Invalid option %s", argv[ii]);
				die_message(msg);
			}
		}
		else if (num_threads < MAX_THREADS)
		{
			strcpy(addr_trace_filename[num_threads], argv[ii]);
			num_threads++;
			NUM_THREADS = num_threads;
		}
		else
		{
			char msg[256];
			sprintf(msg, "Invalid option %s", argv[ii]);
			die_message(msg);
		}
	}

	//--------------------------------------------------------------------
	// Error checking
	//--------------------------------------------------------------------
	if (num_threads == 0)
	{
		die_message("Must provide valid at least one addr_trace");
	}

	if (MT_APP_THREADS)
	{
		num_threads = MT_APP_THREADS;
		NUM_THREADS = MT_APP_THREADS;
		for (ii = 1; ii < num_threads; ii++)
		{
			strcpy(addr_trace_filename[ii], addr_trace_filename[0]);
		}
	}
}

#endif
