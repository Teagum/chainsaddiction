SHELL = /bin/sh
objs = test_pois_params test_pois_probs

.SUFFIXES: .h .c .o

all: $(objs)

test_pois_params: test_pois_params.o ../pois_params.o ../../libma.o
test_pois_probs: test_pois_probs.o ../pois_probs.o ../../libma.o
test_pois_hmm: test_pois_hmm.o ../pois_hmm ../pois_params ../pois_probs

test_pois_params.o: test_pois_params.h ../pois_params.h unittest.h
test_pois_probs.o: test_pois_probs.h ../pois_probs.h unittest.h
test_pois_hmm.o: test_pois_hmm.h ../pois_hmm.h ../pois_params.h ../pois_probs.h unittest.h

.PHONY: check
check: $(objs)
	@for app in $(objs); do ./$$app; done

.PHONY: clean
clean:
	rm -f *.o
	rm -f $(objs) 
