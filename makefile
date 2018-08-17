CC = gcc
OBJ = test.o linalg.o

test: $(OBJ)
	$(CC) -o $@ $(OBJ)

test.o: test.c
	$(CC) -c $*.c

lianlg.o: linalg.c
	$(CC) -c linalg.c
