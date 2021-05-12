SHELL = /bin/sh

.SUFFIXES:
.SUFFIXES: .c .o .test

SRC_DIR := src/chainsaddiction
BUILD_DIR := build
BIN_DIR   := $(BUILD_DIR)/bin
OBJ_DIR   := $(BUILD_DIR)/obj
OBJS      := dataset.o libma.o rnd.o read.o stats.o vmath.o

TEST_ROOT_DIR  := tests
TEST_SRC_DIR   := $(TEST_ROOT_DIR)/src
TEST_OBJS      := test_dataset.o test_read.o test_rnd.o test_stats.o test_vmath.o
TEST_APPS      := dataset.test read.test rnd.test stats.test vmath.test

vpath
vpath %.c $(SRC_DIR) $(TEST_SRC_DIR)
vpath %.h	$(SRC_DIR) $(TEST_SRC_DIR)
vpath %.o $(OBJ_DIR)
vpath %.test $(BIN_DIR)
GPATH := $(OBJ_DIR) $(BIN_DIR)


CC = cc
WARNINGS = -Wall -Wextra -Wfloat-equal -Werror=vla -pedantic
OPTIMIZE = -O2
DEBUG    = -g
STANDARD = -std=c17
CFLAGS = $(WARNINGS) $(STANDARD) $(OPTIMIZE) $(DEBUG)
# Flags
# LD_MATH           Typedef `scalar' to `long double', otherwise `double'.
# NO_BOUNDS_CHECK   Do not check boundaries in array setters and getters.
CPPFLAGS = -D LD_MATH
INCLUDES = -I$(SRC_DIR)
TEST_INCLUDES = $(INCLUDES) -I$(TEST_SRC_DIR)


%.o: %.c
	$(CC) $(CFLAGS) $(CPPFLAGS) $(TEST_INCLUDES) -c $< -o $(OBJ_DIR)/$@

all: $(OBJS) $(TEST_OBJS)

dataset.test: test_dataset.o dataset.o libma.o read.o rnd.o
	$(CC) $(LDFLAGS) $^ -o $(BIN_DIR)/$@

read.test: test_read.o read.o
	$(CC) $(LDFLAGS) -o $(BIN_DIR)/$@ $^

rnd.test: test_rnd.o rnd.o
	$(CC) $(LDFLAGS) -o $(BIN_DIR)/$@ $^

stats.test: test_stats.o stats.o
	$(CC) $(LDFLAGS) -o $(BIN_DIR)/$@ $^

vmath.test: test_vmath.o libma.o rnd.o vmath.o
	$(CC) $(LDFLAGS) -o $(BIN_DIR)/$@ $^
						 

dataset.o: dataset.h restrict.h scalar.h libma.h
libma.o: libma.h scalar.h
rnd.o: rnd.h restrict.h scalar.h
read.o: read.h scalar.h
rnd.o: rnd.h restrict.h scalar.h
stats.o: stats.h restrict.h scalar.h
vmath.o: restrict.h scalar.h

test_dataset.o: test_dataset.h dataset.h restrict.h scalar.h rnd.h unittest.h
test_read.o: test_read.h restrict.h scalar.h rnd.h unittest.h
test_rnd.o: test_rnd.h rnd.h unittest.h
test_stats.o: test_stats.h stats.h unittest.h
test_vmath.o: libma.h restrict.h rnd.h scalar.h unittest.h vmath.h 

$(OBJS): | $(BUILD_DIR)
$(TEST_OBJS): | $(BUILD_DIR)
$(BUILD_DIR):
	mkdir $(BUILD_DIR) $(OBJ_DIR) $(BIN_DIR) 

.PHONY: clean
clean:
	rm -f ./*.o
	rm -f ./*.test
	rm -f $(SRC_DIR)/*.o
	rm -f $(BUILD_DIR)/**/*.o
	rm -f $(BUILD_DIR)/**/*.test
	rm -f $(SRC_DIR)/*.out

.PHONY: distclean
distclean: clean
	rm -rf $(BUILD_DIR)

.PHONY: test
test: $(TEST_APPS)

.PHONY: runtest
runtest:
	@for testapp in $$(ls $(BIN_DIR)/*.test); do echo "run $$testapp"; $$testapp; echo '\n'; done

.PHONY: build_env
build_env:
	@echo 'SRC_DIR:        $(SRC_DIR)'
	@echo 'BUILD_DIR:      $(BUILD_DIR)'
	@echo 'BIN_DIR:        $(BIN_DIR)'
	@echo 'OBJ_DIR:        $(OBJ_DIR)'
	@echo 'OBJS:           $(OBJS)'
	@echo 'TEST_ROOT_DIR:  $(TEST_ROOT_DIR)'
	@echo 'TEST_SRC_DIR:   $(TEST_SRC_DIR)'
