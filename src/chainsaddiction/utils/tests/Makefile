SHELL = /bin/sh
.SUFFIXES: .h .c .o

pkg_root := ../../../..
module_root := $(pkg_root)/src/chainsaddiction/utils
top_include := $(pkg_root)/include
ca_include := $(pkg_root)/src/chainsaddiction
ph_include := $(pkg_root)/src/chainsaddiction/poishmm 
ut_include := $(pkg_root)/libs/unittest/include

vpath
vpath %.h .
vpath %.h $(top_inlcude)
vpath %.h $(module_root)
vpath %.h $(ca_include)
vpath %.h $(ph_include)
vpath %.h $(ut_include)

objs = runtest.o test-utils.o ../utils.o

debug    = -g
warnings = -Wall -Wextra -Wfloat-equal -Werror=vla -pedantic
optimize = -O0
standard = -std=c17
includes = -I$(ut_include) -I$(ph_include) -I$(ca_include) -I$(module_root) -I$(top_include)

CFLAGS = $(debug) $(warnings) $(optimize) $(standard)
CPPFLAGS = $(includes)
LDFLAGS = -L$(pkg_root)/src/vmath
LDLIBS = -lvmath

all: utils.test

utils.test: $(objs) ../../read.o ../../dataset.o ../../libma.o 
	$(LINK.c) $^ -o $@ $(LDLIBS)

runtest.o: unittest.h test-utils.h
test-utils.o: test-utils.h utils.h unittest.h dataset.h read.h libma.h

make check: all
	./utils.test

clean:
	$(RM) *.o
	$(RM) *.test

.PHONY: clean env
