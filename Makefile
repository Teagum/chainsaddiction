SHELL = /bin/sh

.SUFFIXES:
.SUFFIXES: .c .o .test .py

build_dir := build
bin_dir   := $(build_dir)/bin
obj_dir   := $(build_dir)/obj
objs      := $(addprefix $(obj_dir)/, dataset.o fwbw.o poishmm.o libma.o rnd.o read.o stats.o vmath.o)

test_root_dir  := tests
test_src_dir   := $(test_root_dir)/src
test_objs      := $(addprefix $(obj_dir)/, test_bw.o test_dataset.o test_fwbw.o test_read.o test_rnd.o test_stats.o test_vmath.o)
test_apps      := $(addprefix $(bin_dir)/, bw.test dataset.test fwbw.test read.test rnd.test stats.test vmath.test)

vpath
vpath %.c $(src_dir) $(test_src_dir)
vpath %.h	$(src_dir) $(test_src_dir)


# Compiler flags
warnings = -Wall -Wextra -Wfloat-equal -Werror=vla -pedantic
optimize = -O3
standard = -std=c17
CFLAGS = $(warnings) $(standard) $(optimize)

# Preprocessor flags
# 	_NO_LD_MATH       Disable long double math.
# 	NO_BOUNDS_CHECK   Do not check boundaries in array setters and getters.
INCLUDE = -I$(src_dir) -I$(test_src_dir)

help:
	@echo 'Usage: make <target>'
	@echo ' '
	@echo 'Targets:'
	@echo '    install    Install the package.'
	@echo '    pkg        Build the package under build/.'
	@echo '    ext        Recompile the extension module.'
	@echo '    check      Run extension module tests.'
	@echo '    vmath      Rebuild vmath.'


install:
	pip3 install .

build:
	python3 -m build

check:
	make check -C tests

vmath:
	make clean -C src/vmath
	make -C src/vmath

$(obj_dir)/%.o: %.c
	$(COMPILE.c) $(INCLUDE) $< $(OUTPUT_OPTION)

$(bin_dir)/%.test: $(obj_dir)/%.o
	$(LINK.c) $^ $(OUTPUT_OPTION)

clean:
	$(RM) **/*.o
	$(RM) **/*.out
	$(RM) **/*.test

distclean: clean

build_env:
	@echo 'SRC_DIR:        $(src_dir)'
	@echo 'BUILD_DIR:      $(build_dir)'
	@echo 'BIN_DIR:        $(bin_dir)'
	@echo 'OBJ_DIR:        $(obj_dir)'
	@echo 'OBJS:           $(objs)'
	@echo 'TEST_ROOT_DIR:  $(test_root_dir)'
	@echo 'TEST_SRC_DIR:   $(test_src_dir)'

.PHONY: build check clean distclean
