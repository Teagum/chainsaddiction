SHELL = /bin/sh

.SUFFIXES: .h .c .o

root := ../../..
libs := $(root)/libs
lib_include_dir := $(root)/include
unittest := $(libs)/unittest/include
vmath := $(root)/src/vmath

vpath %.h ..
vpath %.h $(lib_include_dir)
vpath %.h $(unittest)

test_runner = runtest
CPPFLAGS =  -I.. -I$(lib_include_dir) -I$(unittest)

LDFLAGS = -L$(root)/src/vmath
LDLIBS = -lvmath

all: dataset.test

dataset.test: $(addprefix ../, libma.o read.o dataset.o) test-dataset.o $(test_runner).o
	$(LINK.c) $^ -o $@ $(LDLIBS)

$(test_runner).o: chainsaddiction.h libma.h read.h 

test-dataset.o: chainsaddiction.h libma.h read.h dataset.h test-dataset.h 

check: dataset.test
	./dataset.test

clean:
	$(RM) *.o
	$(RM) *.test

.PHONY: check clean
