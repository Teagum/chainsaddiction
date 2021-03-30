SRC_DIR := src/chainsaddiction

BUILD_DIR := build
BIN_DIR   := $(BUILD_DIR)/bin
OBJ_DIR   := $(BUILD_DIR)/obj
OBJS      := $(addprefix $(OBJ_DIR)/, dataset.o libma.o)

TEST_ROOT_DIR  := tests
TEST_SRC_DIR   := $(TEST_ROOT_DIR)/src
TEST_BUILD_DIR := $(TEST_ROOT_DIR)/build
TEST_BIN_DIR   := $(TEST_BUILD_DIR)/bin
TEST_OBJ_DIR   := $(TEST_BUILD_DIR)/obj
TEST_OBJS      := $(addprefix $(TEST_OBJ_DIR)/, test_dataset.o)
TEST_APPS      := dataset.test

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

dataset.test : $(TEST_OBJ_DIR)/test_dataset.o $(OBJ_DIR)/dataset.o $(OBJ_DIR)/libma.o | $(TEST_BIN_DIR)
	$(CC) $(CFLAGS) -o $(TEST_BIN_DIR)/$@ $?

$(OBJS) : | $(OBJ_DIR)

$(OBJ_DIR) : $(BUILD_DIR)
	mkdir $(OBJ_DIR)

$(BUILD_DIR) :
	mkdir $(BUILD_DIR)

$(TEST_BUILD_DIR) : $(TEST_ROOT_DIR)
	mkdir $(TEST_BUILD_DIR)

$(TEST_BIN_DIR) : $(TEST_BUILD_DIR)
	mkdir $(TEST_BIN_DIR)

$(TEST_OBJ_DIR) : $(TEST_BUILD_DIR)
	mkdir $(TEST_OBJ_DIR)

$(TEST_OBJS) : | $(TEST_OBJ_DIR)

.PHONY:
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

.PHONY:
runtest:
	@for testapp in $$(ls $(TEST_BIN_DIR)/*.test); do echo "run $$testapp"; $$testapp; echo '\n'; done

.PHONY:
cleantest:
	rm -f $(TEST_OBJ_DIR)/*.o

.PHONY:
clean:
	rm -f $(OBJ_DIR)/*.o

.PHONY:
distclean:
	rm -rf $(TEST_BUILD_DIR) $(BUILD_DIR)
