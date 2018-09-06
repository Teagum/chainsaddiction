INCLUDE='-I../include/'

gcc -c $INCLUDE ../hmm/stats.c
gcc -c $INCLUDE src/test_stats.c
gcc -o test_stats test_stats.o stats.o
mv *.o build/
