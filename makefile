CC = gcc
OBJ = test_fwbw.o linalg.o stats.o

test_fwbw: $(OBJ)
	$(CC) -o $@ $(OBJ)

test_fwbw.o: test_fwbw.c
	$(CC) -c $*.c

lianlg.o: linalg.c
	$(CC) -c $*.c

stats.o: stats.c
	$(CC) -c $*.c
