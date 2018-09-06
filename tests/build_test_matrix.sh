INCLUDE='-I../include/'

gcc -c $INCLUDE ../hmm/matrix.c
gcc -c $INCLUDE src/test_matrix.c
gcc -o test_matrix test_matrix.o matrix.o
mv *.o build/
