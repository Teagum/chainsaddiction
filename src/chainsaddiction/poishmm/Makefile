SHELL = /bin/sh

.SUFFIXES:
.SUFFIXES: .h .c .o

vpath
vpath %.h ./
vpath %.h ../
vpath %.h ../../../include


objs := pois_em.o pois_fwbw.o pois_hmm.o pois_params.o pois_probs.o pois_utils.o

# Compiler flags
warnings = -Wall -Wextra -Wfloat-equal -Werror=vla -pedantic
optimize = -O2
debug    = -g
standard = -std=c17
includes = -I. -I.. -I../../../include

export CFLAGS = $(warnings) $(standard) $(optimize) 

# Preprocessor flags
#    _NO_LD_MATH        Disable long double math.
#    NO_BOUNDS_CHECK    Do not check boundaries in array setters and getters.
export CPPFLAGS = $(includes)

all: $(objs)

pois_em.o: pois_em.h pois_fwbw.h pois_hmm.h pois_params.h pois_probs.h \
	pois_utils.h pois_fwbw.h dataset.h libma.h read.h restrict.h   \
	scalar.h libvmath.h 

pois_fwbw.o: pois_fwbw.h dataset.h libma.h read.h restrict.h scalar.h  \
	libvmath.h

pois_hmm.o: pois_hmm.h pois_params.h pois_probs.h pois_utils.o         \
	dataset.h libma.h read.h restrict.h scalar.h libvmath.h 

pois_params.o: pois_params.h libma.h read.h restrict.h scalar.h libvmath.h
pois_probs.o: pois_probs.h libma.h restrict.h scalar.h
pois_utils.o: pois_utils.h libma.h restrict.h scalar.h libvmath.h


test: all
	@$(MAKE) all -C tests

runtest: test
	tests/pois_hmm.test

check: all runtest

clean:
	rm -f *.o
	@$(MAKE) clean -C tests/

.PHONY: test clean
