SHELL = /bin/sh

.SUFFIXES: .h .c .o

test_srcs = $(wildcard test_pois_*.c)
test_objs = $(test_srcs:.c=.o)
objs = $(addprefix ../, $(subst test_,,$(test_objs)))
libs = $(addprefix ../../, dataset libma read rnd vmath)
test_runner = run

all: pois_hmm.test

pois_hmm.test: $(test_runner).o $(test_objs) $(objs) $(addsuffix .o, $(libs) pois_utils.o)
	$(LINK.c) $^ -o $@

$(test_runner).o: test_pois_params.h test_pois_probs.h test_pois_hmm.h unittest.h scalar.h restrict.h \
                  $(addprefix ../, pois_params.h pois_probs.h pois_utils.h pois_hmm.h)   \
                  $(addsuffix .h, $(libs))

test_pois_hmm.o: test_pois_hmm.h \
                $(addprefix ../, pois_params.h pois_probs.h pois_utils.h pois_hmm.h) \
                $(addsuffix .h, $(libs))

test_pois_params.o: test_pois_params.h ../pois_params.h ../../libma.h
test_pois_probs.o: test_pois_probs.h ../pois_probs.h ../../libma.h

$(test_objs): $(addprefix ../../, scalar.h restrict.h) unittest.h

check: pois_hmm.test
	./pois_hmm.test

clean:
	$(RM) *.o
	$(RM) *.test

.PHONY: check clean
