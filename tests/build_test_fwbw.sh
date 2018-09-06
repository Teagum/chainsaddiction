INCLUDE='-I../include/'

gcc -c $INCLUDE ../hmm/matrix.c
gcc -c $INCLUDE ../hmm/stats.c
gcc -c $INCLUDE ../hmm/fwbw.c
gcc -c $INCLUDE src/test_fwbw.c
gcc -o test_fwbw test_fwbw.o matrix.o stats.o fwbw.o
mv *.o build/
