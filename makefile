SHELL = /bin/sh

.SUFFIXES:
.SUFFIXES: .c .o

SRC_DIR := src/chainsaddiction

BUILD_DIR := build
BIN_DIR   := $(BUILD_DIR)/bin
OBJ_DIR   := $(BUILD_DIR)/obj
OBJS      := $(addprefix $(OBJ_DIR)/, dataset.o libma.o, rnd.o, read.o)

TEST_ROOT_DIR  := tests
TEST_SRC_DIR   := $(TEST_ROOT_DIR)/src
TEST_BUILD_DIR := $(TEST_ROOT_DIR)/build
TEST_BIN_DIR   := $(TEST_BUILD_DIR)/bin
TEST_OBJ_DIR   := $(TEST_BUILD_DIR)/obj
TEST_OBJS      := $(addprefix $(TEST_OBJ_DIR)/, test_dataset.o, test_read.o)
TEST_APPS      := dataset.test read.test

vpath %.c $(TEST_SRC_DIR)
vpath %.h	$(TEST_SRC_DIR)
vpath %.c $(SRC_DIR)
vpath %.h $(SRC_DIR)

CC = cc
WARNINGS = -Wall -Wextra -Wfloat-equal -Werror=vla -pedantic
OPTIMIZE = -O3
STANDARD = -std=c17
CFLAGS = $(WARNINGS) $(STANDARD) $(OPTIMIZE)
INCLUDES = -I$(SRC_DIR)
TEST_INCLUDES = $(INCLUDES) -I$(TEST_SRC_DIR)


$(TEST_OBJ_DIR)/%.o : %.c
	$(CC) $(CFLAGS) $(TEST_INCLUDES) -o $@ -c $<

$(OBJ_DIR)/%.o : %.c
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ -c $<

all: $(OBJS)

test: $(TEST_OBJS) $(TEST_APPS)

dataset.test :	$(addprefix $(TEST_OBJ_DIR)/, test_dataset.o) \
								$(addprefix $(OBJ_DIR)/, dataset.o libma.o read.o rnd.o) | $(TEST_BIN_DIR) $(TEST_OBJS_DIR)
	$(CC) $(CFLAGS) -o $(TEST_BIN_DIR)/$@ $?

read.test : $(TEST_OBJ_DIR)/test_read.o $(OBJ_DIR)/read.o | $(TEST_BIN_DIR) $(TEST_OBJS_DIR)
	$(CC) $(CFLAGS) -o $(TEST_BIN_DIR)/$@ $?


$(OBJ_DIR)/dataset.o : dataset.h restrict.h scalar.h libma.h
$(OBJ_DIR)/libma.o : libma.h
$(OBJ_DIR)/rnd.o : rnd.h restrict.h scalar.h
$(OBJ_DIR)/read.o : read.h scalar.h

$(TEST_OBJ_DIR)/test_dataset.o : test_dataset.h dataset.h restrict.h scalar.h rnd.h unittest.h
$(TEST_OBJ_DIR)/test_read.o : test_read.h restrict.h scalar.h rnd.h unittest.h

$(OBJS) : | $(OBJ_DIR)

$(OBJ_DIR) : $(BUILD_DIR)
	mkdir $(OBJ_DIR)

$(BUILD_DIR) :
	mkdir $(BUILD_DIR)

$(TEST_OBJS) : | $(TEST_OBJ_DIR)

$(TEST_OBJ_DIR) : | $(TEST_BUILD_DIR)
	mkdir $(TEST_OBJ_DIR)

$(TEST_BUILD_DIR) : | $(TEST_BUILD_DIR)
	mkdir $(TEST_BUILD_DIR)

$(TEST_BIN_DIR) : | $(TEST_BUILD_DIR)
	mkdir $(TEST_BIN_DIR)

$(TEST_ROOT_DIR) :
	mkdir tests


.PHONY: build_env
build_env :
	@echo 'SRC_DIR:        $(SRC_DIR)'
	@echo 'BUILD_DIR:      $(BUILD_DIR)'
	@echo 'BIN_DIR:        $(BIN_DIR)'
	@echo 'OBJ_DIR:        $(OBJ_DIR)'
	@echo 'OBJS:           $(OBJS)'
	@echo 'TEST_ROOT_DIR:  $(TEST_ROOT_DIR)'
	@echo 'TEST_SRC_DIR:   $(TEST_SRC_DIR)'
	@echo 'TEST_BUILD_DIR: $(TEST_BUILD_DIR)'
	@echo 'TEST_BIN_DIR:   $(TEST_BIN_DIR)'
	@echo 'TEST_OBJ_DIR:   $(TEST_OBJ_DIR)'
	@echo 'TEST_OBJS:      $(TEST_OBJS)'

.PHONY: runtest
runtest:
	@for testapp in $$(ls $(TEST_BIN_DIR)/*.test); do echo "run $$testapp"; $$testapp; echo '\n'; done

.PHONY: cleantest
cleantest:
	rm -f $(TEST_OBJ_DIR)/*.o

.PHONY: clean
clean:
	rm -f $(OBJ_DIR)/*.o

.PHONY: distclean
distclean:
	rm -rf $(TEST_BUILD_DIR) $(BUILD_DIR)
