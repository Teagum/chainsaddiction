.SUFFIXES:
.SUFFIXES: .py

test_objs = test-ext-earthquakes.py test-local-decoding.py
CC = python


check: $(test_objs)
	$(foreach f, $^, $(CC) $(f))

local_decoding: test-local-decoding.py
	$(CC) $<


global-decoding: test-global-decoding.py
	$(CC) $<
.PHONY: check
