.SUFFIXES:
.SUFFIXES: .py

test_objs = test-utils.py
PYTHON = python -m unittest

check: $(test_objs)
	$(PYTHON) $(test_objs)

.PHONY: check
