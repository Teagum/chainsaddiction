SHELL = /bin/sh

pkg_root := ../../..
objs := utils.o

.SUFFIXES:
.SUFFIXES: .c .h .o

vpath
vpath %.h .
vpath %.h ..
vpath %.h $(pkg_root)/include

warnings = -Wall -Wextra -Wfloat-equal -Werror=vla -pedantic
optimize = -O3
standard = -std=c17
includes = -I.. -I$(pkg_root)/include

CFLAGS = $(warnings) $(standard) $(optimize) 
CPPFLAGS = $(includes)

all: $(objs)

utils.o: utils.h libvmath.h

clean:
	$(RM) *.o

.PHONY: clean
