SHELL = /bin/sh

.SUFFIXES:
.SUFFIXES: .c .o .test

vpath
vpath %.h ./
vpath %.h ../

objs := pois_hmm.o pois_params.o pois_probs.o

all: $(objs)

pois_hmm.o: pois_hmm.h pois_params.h pois_probs.h restrict.h scalar.h
pois_params.o: pois_params.h restrict.h scalar.h
pois_probs.o: pois_probs.h restrict.h scalar.h
pois_utils.o: pois_utils.h scalar.h


.PHONY: clean
clean:
	rm -f *.o
