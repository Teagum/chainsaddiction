CC = gcc
CFLAGS = -Wall
OBJ = test_core.o hmmcore.o linalg.o stats.o

test_core: $(OBJ)
	$(CC) $(CFLAGS) -o $@ $(OBJ) 

test_core.o: test_core.c
	$(CC) $(CFLAGS) -c -g $*.c

hmmcore.o: hmmcore.c hmmcore.h
	$(CC) $(CFLAGS) -c $*.c

linalg.o: linalg.c linalg.h
	$(CC) $(CFLAGS) -c $*.c

stats.o: stats.c linalg.h
	$(CC) $(CFLAGS) -c $*.c 

clean:
	rm -f *.o
	rm test_core
