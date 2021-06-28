SHELL = /bin/sh

.SUFFIXES:
.SUFFIXES: .c .o .test

src_dir := src/chainsaddiction
build_dir := build
bin_dir   := $(build_dir)/bin
obj_dir   := $(build_dir)/obj
objs      := $(addprefix $(obj_dir)/, dataset.o fwbw.o poishmm.o libma.o rnd.o read.o stats.o vmath.o)

test_root_dir  := tests
test_src_dir   := $(test_root_dir)/src
test_objs      := $(addprefix $(obj_dir)/, test_bw.o test_dataset.o test_fwbw.o test_poishmm.o test_read.o test_rnd.o test_stats.o test_vmath.o)
test_apps      := $(addprefix $(bin_dir)/, bw.test dataset.test fwbw.test poishmm.test read.test rnd.test stats.test vmath.test)

vpath
vpath %.c $(src_dir) $(test_src_dir)
vpath %.h	$(src_dir) $(test_src_dir)


# Compiler flags
warnings = -Wall -Wextra -Wfloat-equal -Werror=vla -pedantic
optimize = -O2
debug    = -g
standard = -std=c17
CFLAGS = $(warnings) $(standard) $(optimize) $(debug)

# Preprocessor flags
# 	LD_MATH           Typedef `scalar' to `long double', otherwise `double'.
# 	NO_BOUNDS_CHECK   Do not check boundaries in array setters and getters.
CPPFLAGS = -D LD_MATH
INCLUDE = -I$(src_dir) -I$(test_src_dir)

current: $(bin_dir)/bw.test

help:
	@echo 'Usage: make <target>\n\nTargets:'
	@echo '\ttest -- build all tests.'
	@echo '\ttest -- build and run all tests.'


$(obj_dir)/%.o: %.c
	$(COMPILE.c) $(INCLUDE) $< $(OUTPUT_OPTION)

$(bin_dir)/%.test: $(obj_dir)/%.o
	$(LINK.c) $^ $(OUTPUT_OPTION)

$(bin_dir)/dataset.test: $(addprefix $(obj_dir)/, test_dataset.o dataset.o libma.o read.o rnd.o)
$(bin_dir)/bw.test: $(addprefix $(obj_dir)/, test_bw.o bw.o dataset.o fwbw.o poishmm.o libma.o read.o rnd.o stats.o vmath.o)
$(bin_dir)/fwbw.test: $(addprefix $(obj_dir)/, test_fwbw.o fwbw.o dataset.o libma.o read.o stats.o vmath.o)
$(bin_dir)/poishmm.test: $(addprefix $(obj_dir)/, test_poishmm.o poishmm.o libma.o rnd.o vmath.o)
$(bin_dir)/read.test: $(addprefix $(obj_dir)/, test_read.o read.o)
$(bin_dir)/rnd.test: $(addprefix $(obj_dir)/, test_rnd.o rnd.o)
$(bin_dir)/stats.test: $(addprefix $(obj_dir)/, test_stats.o stats.o)
$(bin_dir)/vmath.test: $(addprefix $(obj_dir)/, test_vmath.o libma.o rnd.o vmath.o)

$(obj_dir)/dataset.o: dataset.h restrict.h read.h scalar.h libma.h
$(obj_dir)/bw.o: bw.h restrict.h scalar.h vmath.h
$(obj_dir)/fwbw.o: fwbw.h dataset.h libma.h read.h restrict.h scalar.h stats.h vmath.h
$(obj_dir)/poishmm.o: poishmm.h libma.h rnd.h vmath.h
$(obj_dir)/libma.o: libma.h scalar.h
$(obj_dir)/rnd.o: rnd.h restrict.h scalar.h
$(obj_dir)/read.o: read.h scalar.h
$(obj_dir)/stats.o: stats.h restrict.h scalar.h
$(obj_dir)/vmath.o: restrict.h scalar.h

$(obj_dir)/test_dataset.o: test_dataset.h dataset.h restrict.h scalar.h rnd.h unittest.h
$(obj_dir)/test_bw.o: test_bw.h dataset.h restrict.h scalar.h rnd.h unittest.h
$(obj_dir)/test_fwbw.o: test_fwbw.h fwbw.h dataset.h restrict.h scalar.h stats.h unittest.h vmath.h
$(obj_dir)/test_read.o: test_read.h restrict.h scalar.h rnd.h unittest.h
$(obj_dir)/test_rnd.o: test_rnd.h rnd.h unittest.h
$(obj_dir)/test_stats.o: test_stats.h stats.h unittest.h
$(obj_dir)/test_vmath.o: libma.h restrict.h rnd.h scalar.h unittest.h vmath.h

$(objs): | $(build_dir)
$(test_objs): | $(build_dir)
$(build_dir):
	mkdir $(build_dir) $(obj_dir) $(bin_dir)

.PHONY: test
test: $(test_apps)

.PHONY: clean
clean:
	rm -f ./*.o
	rm -f ./*.test
	rm -f $(src_dir)/*.o
	rm -f $(build_dir)/**/*.o
	rm -f $(build_dir)/**/*.test
	rm -f $(src_dir)/*.out

.PHONY: distclean
distclean: clean
	rm -rf $(build_dir)

.PHONY: test runtest

runtest:
	@for testapp in $$(ls $(bin_dir)/*.test); do echo "run $$testapp"; $$testapp; echo '\n'; done

.PHONY: build_env
build_env:
	@echo 'SRC_DIR:        $(src_dir)'
	@echo 'BUILD_DIR:      $(build_dir)'
	@echo 'BIN_DIR:        $(bin_dir)'
	@echo 'OBJ_DIR:        $(obj_dir)'
	@echo 'OBJS:           $(objs)'
	@echo 'TEST_ROOT_DIR:  $(test_root_dir)'
	@echo 'TEST_SRC_DIR:   $(test_src_dir)'
