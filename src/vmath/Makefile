SHELL = /bin/sh

.SUFFIXES:
.SUFFIXES: .h .c .o

vpath
vpath %.h .
vpath %.h ..

export objs := vmath.o rnd.o stats.o
lib := libvmath.a

warnings = -Wall -Wextra -Wfloat-equal -Werror=vla -pedantic
optimize = -O2
debug    = -g
standard = -std=c17
includes = -iquote../../include
export CFLAGS = $(warnings) $(standard) $(optimize) $(includes)
export CPPFLAGS = -D LD_MATH

all: $(lib)

test:
	@$(MAKE) -C tests

$(lib): $(objs)
	ar rcs $@ $^

clean: cleantest
	$(RM) *.a
	$(RM) *.o

cleantest:
	@$(MAKE) clean -C tests

check: clean $(lib) test
	@$(MAKE) runtest -C tests

vmath.o: vmath.h config.h 
rnd.o: rnd.h config.h
stats.o: stats.h config.h

.PHONY: clean cleantest check
