INCLUDE='-I../include/'

gcc -c -Wall -Wsign-compare $INCLUDE ../hmm/stats.c ../hmm/fwbw.c ../hmm/em.c ../hmm/hmm.c src/test_em2.c
gcc -o test_em test_em2.o hmm.o em.o fwbw.o stats.o
mv *.o build/
