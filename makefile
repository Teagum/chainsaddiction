CC = cc
INCLUDES = -I./include/ -I./tests/include/
WARNINGS = -Wall -Wextra -Wfloat-equal -Werror=vla
OPTIMIZE = -O3
CFLAGS = $(INCLUDES) $(WARNINGS) $(OPTIMIZE) 
TEST_BUILD_DIR = ./tests/build/
BUILD_DIR = ./build/

vpath %.h ./include/
vpath %.c ./src/chainsaddiction/
vpath %.h ./tests/include/
vpath %.c ./tests/src/

.PHONY:
all : fwbw.test stats.test utilities.test vmath.test

bw.test : run_test_bw.o test_bw.o bw.o fwbw.o hmm.o stats.o utilities.o vmath.o stats.o
	$(CC) -o $(TEST_BUILD_DIR)$@ $?

fwbw.test : test_fwbw.o fwbw.o stats.o utilities.o vmath.o
	$(CC) -o $(TEST_BUILD_DIR)$@ $?

hmm.test : run_test_hmm.o test_hmm.o hmm.o vmath.o
	$(CC) -o $(TEST_BUILD_DIR)$@ $?

rnd.test : run_test_rnd.o test_rnd.o rnd.o
	$(CC) -o $(TEST_BUILD_DIR)$@ $?

stats.test : test_stats.o stats.o utilities.o
	$(CC) -o $(TEST_BUILD_DIR)$@ $?

utilities.test : test_utilities.o utilities.o
	$(CC) -o $(TEST_BUILD_DIR)$@ $?

vmath.test : run_test_vmath.o test_vmath.o vmath.o rnd.o utilities.o
	$(CC) -o $(TEST_BUILD_DIR)$@ $?

run_test_bw.o : test_bw.h 
run_test_hmm.o : test_hmm.h unittest.h 
run_test_vmath.o : vmath.h test_vmath.h unittest.h
run_test_rnd.o : rnd.h test_rnd.h unittest.h

test_bw.o : test_bw.h bw.h hmm.h unittest.h stats.h
test_fwbw.o : fwbw.h utilities.h
test_hmm.o : hmm.h rnd.h unittest.h
test_rnd.o : test_rnd.h unittest.h
test_stats.o : stats.h
test_utilities.o : utilities.h 
test_vmath.o : rnd.h vmath.h unittest.h

bw.o : bw.h
fwbw.o : restrict.h scalar.h stats.h vmath.h
hmm.o : restrict.h scalar.h utilities.h vmath.h utilities.c
rnd.o : restrict.h rnd.h scalar.h
stats.o : restrict.h scalar.h stats.h
utilities.o : utilities.h
vmath.o : restrict.h scalar.h utilities.h vmath.h


.PHONY:
clean :
	rm *.o
