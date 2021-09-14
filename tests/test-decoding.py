from pathlib import Path
import sys
import unittest

from utils import load_data
import numpy as np
from apollon.hmm import utilities as hmmutils
from chainsaddiction import poishmm


class TestLocalDecoding(unittest.TestCase):
    def setUp(self):
        src = 'data/earthquakes/3s'
        self.lcxpt = load_data(src, 'lcxpt').reshape(-1, 3)
        self.res = load_data(src, 'locald', 'uint64')

    def test_local_decoding(self):
        dec = poishmm.local_decoding(self.lcxpt)
        self.assertTrue(np.array_equal(dec, self.res))


class TestGloblaDecoding(unittest.TestCase):
    def setUp(self):
        src = 'data/earthquakes/3s'
        self.lcxpt = load_data(src, 'lcxpt').reshape(-1, 3)
        self.lgamma = load_data(src, 'lgamma_').reshape(3, 3)
        self.ldelta = load_data(src, 'ldelta_')
        self.res = load_data(src, 'globald', 'uint64')

    def test_global_decoding(self):
        dec = poishmm.global_decoding(self.lgamma, self.ldelta, self.lcxpt)
        self.assertTrue(np.array_equal(dec, self.res))


if __name__ == '__main__':
    unittest.main()
