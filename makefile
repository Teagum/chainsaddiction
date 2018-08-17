CC = gcc
OBJ = test_linalg.o linalg.o

test_linalg: $(OBJ)
	$(CC) -o $@ $(OBJ)

testi_linalg.o: test_linalg.c
	$(CC) -c $*.c

lianlg.o: linalg.c
	$(CC) -c linalg.c
