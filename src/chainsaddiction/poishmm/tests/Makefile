pois_params.test: test_pois_params.o ../pois_params.o ../../libma.o
	$(LINK.c) $^ $(OUTPUT_OPTION)

test_pois_params.o: test_pois_params.h ../pois_params.h unittest.h

.PHONY: clean
clean:
	rm -f *.o
	rm -f *.test
