INCLUDE = -Iinclude/
CFLAGS  = -Wall -Wsign-compare

hmm.cpython-37m-darwin.so: hmm/*.c include/*.h #hmm_module.c hmm/em.c hmm/fwbw.c
	python3 setup.py build_ext --inplace

clean:
	rm -rf build/
	rm hmm.cpython-37m-darwin.so

test:
	python3 test_hmm.py

test_em:
	gcc -c $(CFLAGS) $(INCLUDE) hmm/stats.c hmm/fwbw.c hmm/em.c hmm/hmm.c tests/src/test_em.c
	gcc -o tests/test_em test_em.o hmm.o em.o fwbw.o stats.o
	rm *.o
	tests/test_em

test_fwbw:
	gcc -c $(CFLAGS) $(INCLUDE) hmm/stats.c hmm/fwbw.c tests/src/test_fwbw.c
	gcc -o tests/test_fwbw test_fwbw.o fwbw.o stats.o
	rm *.o
	tests/test_fwbw -b
