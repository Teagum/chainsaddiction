SHELL = /bin/sh

.SUFFIXES:
.SUFFIXES: .h .c .o

root := ../../..

vpath
vpath %.h .
vpath %.h ..
vpath %.h $(root)/include


objs := pois-em.o pois-fwbw.o pois-hmm.o pois-params.o pois-probs.o

# Compiler flags
warnings = -Wall -Wextra -Wfloat-equal -Werror=vla -pedantic
optimize = -O3
standard = -std=c17
includes = -I.. -I../../../include

export CFLAGS = $(warnings) $(standard) $(optimize) 

# Preprocessor flags
#    _NO_LD_MATH        Disable long double math.
#    NO_BOUNDS_CHECK    Do not check boundaries in array setters and getters.
CPPFLAGS += $(includes)


all: $(objs)

pois-em.o: pois-em.h pois-fwbw.h pois-hmm.h pois-params.h pois-probs.h \
	utils/utils.h pois-fwbw.h dataset.h libma.h read.h restrict.h   \
	scalar.h libvmath.h 

pois-fwbw.o: pois-fwbw.h dataset.h libma.h read.h restrict.h scalar.h  \
	libvmath.h

pois-hmm.o: pois-hmm.h pois-params.h pois-probs.h \
	dataset.h libma.h read.h restrict.h scalar.h libvmath.h 

pois-params.o: pois-params.h libma.h read.h restrict.h scalar.h libvmath.h
pois-probs.o: pois-probs.h libma.h restrict.h scalar.h


test: all
	@$(MAKE) all -C tests

runtest: test
	@$(MAKE) check -C tests
  	

check: all runtest

clean:
	rm -f *.o
	@$(MAKE) clean -C tests

.PHONY: test clean
