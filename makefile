INCLUDE = -Iinclude/
TEST_INCLUDE = -Iinclude/ -Itests/src/
CFLAGS  = -Wall -Wsign-compare
MACROS = -DDebug -Dwarn_nan 

hmm.cpython-37m-darwin.so: hmm/*.c include/*.h
	python3 setup.py build_ext --inplace

clean:
	rm -rf build/
	rm hmm.cpython-37m-darwin.so

test:
	python3 test_hmm.py

em:
	if [ ! -d "tests/bin/" ]; then mkdir tests/bin/; fi
	gcc -c $(CFLAGS) $(INCLUDE) hmm/stats.c hmm/fwbw.c hmm/em.c hmm/utilities.c hmm/hmm.c tests/src/test_em.c
	gcc -o tests/bin/em.test stats.o fwbw.o em.o utilities.o hmm.o test_em.o
	rm *.o

fwbw:
	if [ ! -d "tests/bin/" ]; then mkdir tests/bin/; fi
	gcc -c $(CFLAGS) $(INCLUDE) $(MACROS) hmm/stats.c hmm/fwbw.c hmm/utilities.c hmm/hmm.c tests/src/test_fwbw.c
	gcc -o tests/bin/fwbw.test stats.o fwbw.o utilities.o hmm.o test_fwbw.o
	rm *.o

test_fwbw:
	tests/bin/fwbw.test -a 3 tests/data/params_3s < tests/data/earthquakes

test_em:
	tests/bin/em.test 3 < tests/data/earthquakes
