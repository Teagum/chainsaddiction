INCLUDE='-I../include/'

gcc -c $INCLUDE ../hmm/matrix.c
gcc -c $INCLUDE ../hmm/stats.c
gcc -c $INCLUDE ../hmm/fwbw.c
gcc -c $INCLUDE ../hmm/em.c
gcc -c $INCLUDE src/test_em.c
gcc -o test_em test_em.o em.o fwbw.o matrix.o stats.o
mv *.o build/
