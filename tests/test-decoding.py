from pathlib import Path
import sys
import unittest

import numpy as np
from apollon.hmm import utilities as hmmutils
from chainsaddiction import poishmm

import utils


class TestLocalDecoding(unittest.TestCase):
    def setUp(self):
        self.lcxpt = np.fromfile('data/earthquakes/3s/lcxpt', dtype='float128')
        self.res = np.fromfile('data/earthquakes/3s/locald', dtype='float128')

    def test_local_decoding(self):
        dec = poishmm.local_decoding(self.lcxpt)
        self.assertEqual(dec, self.res)


class TestGloblaDecoding(unittest.TestCase):
    def setUp(self):
        self.lcxpt = np.fromfile('data/earthquakes/3s/lcxpt', dtype='float128')
        self.lgamma = np.fromfile('data/earthquakes/3s/lgamma_', dtype='float128')
        self.ldelta = np.fromfile('data/earthquakes/3s/ldelta_', dtype='float128')
        self.res = np.formfile('data/earthquakes/3s/globald_', dtype='float128')

    def test_global_decoding(self):
        dec = poishmm.global_decoding(self.lgamma, self.ldelta, self.lcxpt)
        self.assertEqual(dec, self.res)


if __name__ == '__main__':
    unittest.main()
