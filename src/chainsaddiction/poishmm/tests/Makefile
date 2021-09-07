SHELL = /bin/sh

.SUFFIXES: .h .c .o

root := ../../../..
libs := $(root)/libs
lib_include_dir := $(root)/include
unittest := $(libs)/unittest/include

vpath %.h $(lib_include_dir)
vpath %.h $(unittest)

test_srcs = $(wildcard test_pois_*.c)
test_objs = $(test_srcs:.c=.o)
objs = $(addprefix ../, $(subst test_,,$(test_objs)))
modules = $(addprefix ../../, dataset libma read)

test_runner = runtest
CPPFLAGS += -I$(lib_include_dir) -I$(unittest)

LDFLAGS := -L../../../vmath
LDLIBS := -lvmath

all: pois_hmm.test

pois_hmm.test: $(test_runner).o $(test_objs) $(objs) $(addsuffix .o, $(modules) ../pois_utils ../pois_fwbw)
	$(LINK.c) $^ -o $@ $(LDLIBS)

$(test_runner).o: test_pois_params.h test_pois_probs.h test_pois_hmm.h test_pois_em.h test_pois_utils.h unittest.h libvmath.h \
	$(addprefix ../, pois_params.h pois_probs.h pois_utils.h pois_hmm.h pois_em.h) \
	$(addsuffix .h, $(modules)) $(addprefix ../../, scalar.h restrict.h)

test_pois_hmm.o: test_pois_hmm.h libvmath.h \
	$(addprefix ../, pois_params.h pois_probs.h pois_utils.h pois_hmm.h) \
	$(addsuffix .h, $(modules))

test_pois_params.o: test_pois_params.h ../pois_params.h ../../libma.h
test_pois_probs.o: test_pois_probs.h ../pois_probs.h ../../libma.h
test_pois_em.o: test_pois_em.h libvmath.h \
	$(addprefix ../, pois_em.h pois_fwbw.h pois_hmm.h pois_params.h pois_probs.h pois_utils.h) \
	$(addprefix ../../, dataset.h libma.h read.h restrict.h)

test_pois_utisl.o: test_pois_utils.h libvmath.h \
	$(addprefix ../, pois_em.h pois_fwbw.h pois_hmm.h pois_params.h pois_probs.h pois_utils.h) \
	$(addsuffix .h, $(modules))


$(test_objs): $(addprefix ../../, scalar.h restrict.h) unittest.h

check: pois_hmm.test
	./pois_hmm.test

clean:
	$(RM) *.o
	$(RM) *.test

.PHONY: check clean env
