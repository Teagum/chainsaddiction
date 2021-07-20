SHELL = /bin/sh

.SUFFIXES:
.SUFFIXES: .h .c .o


include_dir := ../../../include


vpath
vpath %.h ..
vpath %.h $(include_dir)

CFLAGS += -iquote$(include_dir) -iquote..
LDFLAGS := -L.. -lvmath

tobjs := test_rnd.o test_stats.o test_vmath.o
all: vmath.test 
test_rnd.o: test_rnd.h rnd.h unittest.h
test_stats.o: test_stats.h stats.h rnd.h unittest.h 
test_vmath.o: test_vmath.h vmath.h rnd.h unittest.h
runtest.o: runtest.c test_vmath.h test_rnd.h test_stats.h vmath.h rnd.h stats.h unittest.h config.h
vmath.test: $(tobjs) $(addprefix ../, $(objs)) runtest.o
	$(LINK.c) $^ -o $@ $(LDFLAGS)


clean:
	$(RM) *.o

runtest: all
	./vmath.test

.PHONY: clean runtest
