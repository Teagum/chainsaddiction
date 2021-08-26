SHELL = /bin/sh

.SUFFIXES:
.SUFFIXES: .h .c .o

vpath
vpath %.h .
vpath %.h ..

export objs := core.o vmath.o rnd.o stats.o
lib := libvmath.a

warnings = -Wall -Wextra -Wfloat-equal -Werror=vla -pedantic
optimize = -O2
debug    = -g
standard = -std=c17
export CFLAGS = $(warnings) $(standard) $(debug) $(optimize) $(includes)

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

check: $(lib) test
	@$(MAKE) runtest -C tests

core.o: core.h config.h
vmath.o: alloc.h config.h vmath.h
rnd.o: rnd.h config.h
stats.o: stats.h config.h

.PHONY: clean cleantest check
