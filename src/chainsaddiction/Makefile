SHELL = /bin/sh

.SUFFIXES:
.SUFFIXES: .c .h .o

objs := dataset.o libma.o read.o

CFLAGS = -Wall -Wextra -pedantic -Wvla

%.o: %.c %.h
	$(COMPILE.c) -o $@ $<

all: $(objs) 

clean:
	$(RM) *.o

.PHONY: clean
