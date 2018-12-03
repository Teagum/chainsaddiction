
hmm.cython-37m-darwin.so:
	python3 setup.py build_ext --inplace
clean:
	rm -rf build/
	rm hmm.cpython-37m-darwin.so

test:
	python3 tests/test_hmm.py
