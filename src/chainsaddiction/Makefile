SHELL = /bin/sh

.SUFFIXES:
.SUFFIXES: .c .h .o

objs := dataset.o libma.o read.o utils.o

CFLAGS = -Wall -Wextra -pedantic -Wvla
CPPFLAGS = -I../../include ./utils

%.o: %.c %.h
	$(COMPILE.c) -o $@ $<

all: $(objs) 

utils: libvmath.h

clean:
	$(RM) *.o

.PHONY: clean
