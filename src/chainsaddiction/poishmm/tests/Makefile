SHELL = /bin/sh

.SUFFIXES: .h .c .o

test_srcs = $(wildcard test_pois_*.c)
test_objs = $(test_srcs:.c=.o)
objs = $(addprefix ../, $(subst test_,,$(test_objs)))
libs = $(addprefix ../../, dataset libma read rnd vmath)
test_runner = run
CPPFLAGS = -D LD_MATH
all: pois_hmm.test

pois_hmm.test: $(test_runner).o $(test_objs) $(objs) $(addsuffix .o, $(libs) ../pois_utils ../pois_fwbw ../../stats)
	$(LINK.c) $^ -o $@

$(test_runner).o: test_pois_params.h test_pois_probs.h test_pois_hmm.h test_pois_em.h unittest.h \
	$(addprefix ../, pois_params.h pois_probs.h pois_utils.h pois_hmm.h pois_em.h)   \
	$(addsuffix .h, $(libs)) \
	$(addprefix ../../, scalar.h restrict.h)

test_pois_hmm.o: test_pois_hmm.h \
	$(addprefix ../, pois_params.h pois_probs.h pois_utils.h pois_hmm.h) \
	$(addsuffix .h, $(libs))

test_pois_params.o: test_pois_params.h ../pois_params.h ../../libma.h
test_pois_probs.o: test_pois_probs.h ../pois_probs.h ../../libma.h
test_pois_em.o: test_pois_em.h unittest.h \
	$(addprefix ../, pois_em.h pois_fwbw.h pois_hmm.h pois_params.h pois_probs.h pois_utils.h) \
	$(addprefix ../../, dataset.h libma.h read.h restrict.h rnd.h stats.h vmath.h)


$(test_objs): $(addprefix ../../, scalar.h restrict.h) unittest.h

check: pois_hmm.test
	./pois_hmm.test

clean:
	$(RM) *.o
	$(RM) *.test

.PHONY: check clean
