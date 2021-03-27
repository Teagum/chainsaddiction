CC = cc
WARNINGS = -Wall -Wextra -Wfloat-equal -Werror=vla -pedantic
OPTIMIZE = -O3
INCLUDES = -Isrc/chainsaddiction/
CFLAGS = $(INCLUDES) $(WARNINGS) $(OPTIMIZE) -std=c17
BIN_PATH = ./bin/
BUILD_PATH = ./build/
TEST_BIN_PATH = ./tests/bin/
TEST_BUILD_PATH = ./tests/build/

vpath %.h ./src/chainsaddiction/
vpath %.c ./src/chainsaddiction/
vpath %.h ./tests/src/
vpath %.c ./tests/src/
vpath %.o $(TEST_BUILD_PATH)
vpath %.o $(BUILD_PATH)


.PHONY:
all : fwbw.test stats.test dataset.test vmath.test

bw.test : run_test_bw.o test_bw.o bw.o fwbw.o hmm.o stats.o dataset.o vmath.o stats.o
	$(CC) -o $(TEST_BIN_PATH)$@ $?

fwbw.test : test_fwbw.o fwbw.o stats.o dataset.o vmath.o
	$(CC) -o $(TEST_BIN_PATH)$@ $?

hmm.test : run_test_hmm.o test_hmm.o hmm.o vmath.o
	$(CC) -o $(TEST_BIN_PATH)$@ $?

rnd.test : run_test_rnd.o test_rnd.o rnd.o
	$(CC) -o $(TEST_BIN_PATH)$@ $?

stats.test : run_test_stats.o test_stats.o stats.o dataset.o
	$(CC) -o $(TEST_BIN_PATH)$@ $?

dataset.test : test_dataset.o dataset.o
	$(CC) -o $(TEST_BIN_PATH)$@ $?

vmath.test : run_test_vmath.o test_vmath.o vmath.o rnd.o dataset.o libma.o
	$(CC) -o $(TEST_BIN_PATH)$@ $?

run_test_bw.o : test_bw.h 
run_test_hmm.o : test_hmm.h unittest.h 
run_test_vmath.o : vmath.h test_vmath.h unittest.h
run_test_rnd.o : rnd.h test_rnd.h unittest.h
run_test_stats.o : test_stats.h

test_bw.o : test_bw.h bw.h hmm.h unittest.h stats.h
test_fwbw.o : fwbw.h dataset.h
test_hmm.o : hmm.h rnd.h unittest.h
test_rnd.o : test_rnd.h unittest.h
test_stats.o : stats.h dataset.h unittest.h
test_dataset.o : dataset.h 
test_vmath.o : rnd.h vmath.h unittest.h

bw.o : bw.h hmm.h
fwbw.o : restrict.h scalar.h stats.h vmath.h
hmm.o : restrict.h scalar.h dataset.h vmath.h dataset.c
libma.o : libma.h
rnd.o : restrict.h rnd.h scalar.h
stats.o : restrict.h scalar.h stats.h
dataset.o : dataset.h libma.h libma.c
vmath.o : restrict.h scalar.h dataset.h vmath.h


.PHONY:
clean :
	rm *.o

.PHONY:
cleantest :
	rm tests/bin/*

.PHONY:
test :
	for cmd in tests/bin/*.test; do $$cmd; done
