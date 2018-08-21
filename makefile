CC = gcc
CFLAGS = -Wall
OBJ = test_fwbw.o fwbw.o linalg.o stats.o

test_fwbw: $(OBJ)
	$(CC) $(CFLAGS) -o $@ $(OBJ) 

test_fwbw.o: test_fwbw.c
	$(CC) $(CFLAGS) -c $*.c

fwbw.o: fwbw.c
	$(CC) $(CFLAGS) -c $*.c

linalg.o: linalg.c
	$(CC) $(CFLAGS) -c $*.c

stats.o: stats.c linalg.h
	$(CC) $(CFLAGS) -c $*.c 

clean:
	rm -f *.o
	rm test_fwbw
