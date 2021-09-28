CC=g++
CFLAGS=-O0
all: simulator
simulator: llc_miss_simulator.cpp
	$(CC) $(CFLAGS) $^ -o $@
clean:
	rm -f *.o *~ simulator