from pathlib import Path
import sys
import unittest

from utils import load_data
import numpy as np
from chainsaddiction import utils


class TestLocalDecoding(unittest.TestCase):
    def setUp(self):
        src = 'data/earthquakes/3s'
        self.lcxpt = load_data(src, 'lcsp.txt').reshape(-1, 3)
        self.res = load_data(src, 'local-decoding.txt', 'uint64')

    def test_local_decoding(self):
        dec = utils.local_decoding(self.lcxpt)
        self.assertTrue(np.array_equal(dec, self.res))


class TestGloblaDecoding(unittest.TestCase):
    def setUp(self):
        src = 'data/earthquakes/3s'
        self.lcxpt = load_data(src, 'lcsp.txt').reshape(-1, 3)
        self.lgamma = load_data(src, 'lgamma.txt').reshape(3, 3)
        self.ldelta = load_data(src, 'ldelta.txt')
        self.res = load_data(src, 'global-decoding.txt', 'uint64')

    def test_global_decoding(self):
        dec = utils.global_decoding(self.lgamma, self.ldelta, self.lcxpt)
        self.assertTrue(np.array_equal(dec, self.res))


if __name__ == '__main__':
    unittest.main()
