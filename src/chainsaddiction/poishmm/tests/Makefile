test_pois_params: test_pois_params.o ../pois_params.o ../../libma.o
test_pois_probs: test_pois_probs.o ../pois_probs.o ../../libma.o

test_pois_params.o: test_pois_params.h ../pois_params.h unittest.h
test_pois_probs.o: test_pois_probs.h ../pois_probs.h unittest.h

.PHONY: clean
clean:
	rm -f *.o
	rm -f *.test
