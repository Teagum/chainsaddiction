SHELL = /bin/sh

.SUFFIXES: .h .c .o

test_srcs = $(wildcard test_pois_*.c)
test_objs = $(test_srcs:.c=.o)
objs = $(addprefix ../, $(subst test_,,$(test_objs)))
libs = $(addprefix ../, libma.o vmath.o rnd.o)
test_runner = run

all: pois_hmm.test

pois_hmm.test: $(test_runner).o $(test_objs) $(objs) $(addprefix ../, $(libs) pois_utils.o)
	$(LINK.c) $^ -o $@

$(test_runner).o: $(test_objs) $(objs) unittest.h \
                  $(addprefix ../../, dataset.h read.h rnd.h vmath.h)

test_pois_hmm.o: $(test_objs) $(objs) unittest.h \
								 $(addprefix ../../, dataset.h read.h rnd.h vmath.h)

test_pois_params.o: test_pois_params.h ../pois_params.h ../../libma.h unittest.h
test_pois_probs.o: test_pois_probs.h ../pois_probs.h ../../libma.h unittest.h

$(test_objs): $(addprefix ../../, scalar.h restrict.h)

.PHONY: check
check: pois_hmm.test
	./pois_hmm.test

.PHONY: clean
clean:
	$(RM) *.o
	$(RM) *.test
