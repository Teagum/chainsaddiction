SHELL = /bin/sh

.SUFFIXES:
.SUFFIXES: .h .c .o

vpath
vpath %.h ./
vpath %.h ../

objs := pois_em.o pois_fwbw.o pois_hmm.o pois_params.o pois_probs.o pois_utils.o

# Compiler flags
warnings = -Wall -Wextra -Wfloat-equal -Werror=vla -pedantic
optimize = -O2
debug    = -g
standard = -std=c17
includes = -I./ -I../
export CFLAGS = $(warnings) $(standard) $(optimize) $(includes)

# Preprocessor flags
# 	LD_MATH           Typedef `scalar' to `long double', otherwise `double'.
# 	NO_BOUNDS_CHECK   Do not check boundaries in array setters and getters.
export CPPFLAGS = -D LD_MATH

all: $(objs)

pois_em.o: pois_em.h pois_fwbw.h pois_hmm.h pois_params.h pois_probs.h \
	pois_utils.h pois_fwbw.h dataset.h libma.h read.h restrict.h   \
	rnd.h scalar.h stats.h vmath.h

pois_fwbw.o: pois_fwbw.h dataset.h libma.h read.h restrict.h scalar.h  \
	stats.h vmath.h

pois_hmm.o: pois_hmm.h pois_params.h pois_probs.h pois_utils.o         \
	dataset.h libma.h read.h restrict.h rnd.h scalar.h vmath.h

pois_params.o: pois_params.h libma.h restrict.h rnd.h scalar.h vmath.h
pois_probs.o: pois_probs.h libma.h restrict.h scalar.h
pois_utils.o: pois_utils.h libma.h restrict.h scalar.h vmath.h


test: all
	@$(MAKE) -C tests/

check: test
	tests/pois_hmm.test

clean:
	rm -f *.o
	@$(MAKE) clean -C tests/

.PHONY: test clean
