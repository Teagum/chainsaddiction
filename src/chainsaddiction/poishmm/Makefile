SHELL = /bin/sh

.SUFFIXES:
.SUFFIXES: .c .o .test

vpath
vpath %.h ./
vpath %.h ../

objs := pois_hmm.o pois_params.o pois_probs.o

# Compiler flags
warnings = -Wall -Wextra -Wfloat-equal -Werror=vla -pedantic
optimize = -O2
debug    = -g
standard = -std=c17
CFLAGS = $(warnings) $(standard) $(optimize)

# Preprocessor flags
# 	LD_MATH           Typedef `scalar' to `long double', otherwise `double'.
# 	NO_BOUNDS_CHECK   Do not check boundaries in array setters and getters.
CPPFLAGS = -D LD_MATH

all: $(objs)

pois_hmm.o: pois_hmm.h pois_params.h pois_probs.h restrict.h scalar.h
pois_params.o: pois_params.h restrict.h scalar.h
pois_probs.o: pois_probs.h restrict.h scalar.h
pois_utils.o: pois_utils.h scalar.h


.PHONY: clean
clean:
	rm -f *.o
