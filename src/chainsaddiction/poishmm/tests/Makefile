SHELL = /bin/sh

.SUFFIXES: .h .c .o

root := ../../../..
libs := $(root)/libs
lib_include_dir := $(root)/include
unittest := $(libs)/unittest/include

vpath %.h $(lib_include_dir)
vpath %.h $(unittest)

test_srcs = $(wildcard test-pois-*.c)
test_objs = $(test_srcs:.c=.o)
objs = $(addprefix ../, $(subst test-,,$(test_objs)))
modules = $(addprefix ../../, dataset libma read)

test_runner = runtest
CPPFLAGS += -I$(lib_include_dir) -I$(unittest)

LDFLAGS := -L../../../vmath
LDLIBS := -lvmath

all: pois-hmm.test

pois-hmm.test: $(test_runner).o $(test_objs) $(objs) $(addsuffix .o, $(modules) ../pois-fwbw ../../utils/utils)
	$(LINK.c) $^ -o $@ $(LDLIBS)

$(test_runner).o: test-pois-params.h test-pois-probs.h test-pois-hmm.h test-pois-em.h unittest.h libvmath.h \
	$(addprefix ../, pois-params.h pois-probs.h pois-hmm.h pois-em.h) \
	$(addsuffix .h, $(modules)) $(addprefix ../../, scalar.h restrict.h utils/utils.h)

test-pois-hmm.o: test-pois-hmm.h libvmath.h \
	$(addprefix ../, pois-params.h pois-probs.h pois-hmm.h) \
	$(addsuffix .h, $(modules))

test-pois-params.o: test-pois-params.h ../pois-params.h ../../libma.h
test-pois-probs.o: test-pois-probs.h ../pois-probs.h ../../libma.h
test-pois-em.o: test-pois-em.h libvmath.h \
	$(addprefix ../, pois-em.h pois-fwbw.h pois-hmm.h pois-params.h pois-probs.h) \
	$(addprefix ../../, dataset.h libma.h read.h restrict.h utils/utils.h)

$(test_objs): $(addprefix ../../, scalar.h restrict.h) unittest.h

check: pois-hmm.test
	./pois-hmm.test

clean:
	$(RM) *.o
	$(RM) *.test

env:
	echo $(objs)
	echo $(test_objs)

.PHONY: check clean env
