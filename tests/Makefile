.SUFFIXES:
.SUFFIXES: .py

test_objs = test-decoding.py
PYTHON = python -m unittest

check: $(test_objs)
	$(PYTHON) $(test_objs)

.PHONY: check
