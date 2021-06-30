SHELL = /bin/sh

.SUFFIXES:
.SUFFIXES: .c .o .test

vpath
vpath %.h ./
vpath %.h ../

objs := pois_params.o pois_probs.o

all: $(objs)

pois_params.o: pois_params.h restrict.h scalar.h
pois_probs.o: pois_probs.h restrict.h scalar.h


.PHONY: clean
clean:
	rm -f *.o
