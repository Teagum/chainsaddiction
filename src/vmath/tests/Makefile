SHELL = /bin/sh

.SUFFIXES:
.SUFFIXES: .h .c .o

lib_inlcude := ..
top_include := ../../../include
unittest_include := ../../../libs/unittest/include

vpath
vpath %.h ..
vpath %.h $(top_include)
vpath %.h $(unittest_include)

tobjs := test_rnd.o test_stats.o test_vmath.o
CFLAGS += -iquote$(lib_inlcude) -iquote$(top_include) -iquote$(unittest_include)
LDFLAGS := -L../ -lvmath

all: vmath.test 

test_rnd.o: test_rnd.h rnd.h unittest.h
test_stats.o: test_stats.h stats.h rnd.h unittest.h 
test_vmath.o: test_vmath.h vmath.h rnd.h unittest.h
runtest.o: runtest.c test_vmath.h test_rnd.h test_stats.h vmath.h rnd.h stats.h unittest.h config.h

vmath.test: $(tobjs) $(addprefix ../, $(objs)) runtest.o
	$(LINK.c) $^ -o $@ $(LDFLAGS)

clean:
	$(RM) *.o
	$(RM) *.test

runtest: all
	./vmath.test

.PHONY: clean
