INCLUDE='-I../include/'

gcc -c -Wall -Wsign-compare $INCLUDE ../hmm/hmm.c src/test_hmm.c
gcc -o test_hmm hmm.o test_hmm.o
mv *.o build/
