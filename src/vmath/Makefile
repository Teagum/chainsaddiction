SHELL = /bin/sh

.SUFFIXES:
.SUFFIXES: .h .c .o

vpath
vpath %.h .
vpath %.h ..

objs := vmath.o rnd.o stats.o
lib := libvmath.a

warnings = -Wall -Wextra -Wfloat-equal -Werror=vla -pedantic
optimize = -O2
debug    = -g
standard = -std=c17
includes = -I. -I..
CFLAGS = $(warnings) $(standard) $(optimize) $(includes)
CPPFLAGS = -D LD_MATH

all: static 

vmath.o: vmath.h config.h 
rnd.o: rnd.h config.h
stats.o: stats.h config.h
static: $(objs)
	ar rcs $(lib) $(objs) 


clean:
	rm *.a
	rm *.o

.PHONY: clean
