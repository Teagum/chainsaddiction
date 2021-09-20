SHELL = /bin/sh

.SUFFIXES:
.SUFFIXES: .c .h .o

objs := dataset.o libma.o read.o utils/utils.o

CFLAGS = -Wall -Wextra -pedantic -Wvla
CPPFLAGS = -I../../include

%.o: %.c %.h
	$(COMPILE.c) -o $@ $<

all: $(objs) 

clean:
	$(RM) *.o

.PHONY: clean
