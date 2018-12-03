INCLUDE='-I../include/'

gcc -c $INCLUDE ../hmm/matrix.c ../hmm/stats.c ../hmm/fwbw.c src/test_fwbw.c
gcc -o test_fwbw test_fwbw.o matrix.o stats.o fwbw.o
mv *.o build/
