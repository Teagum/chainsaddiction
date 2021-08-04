.SUFFIXES:
.SUFFIXES: .py

test_objs = test-ext-earthquakes.py
CC = python


check: $(test_objs)
	$(foreach f, $^, $(CC) $(f))


.PHONY: check
