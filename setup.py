#/usr/bin/env python3

from setuptools import setup, Extension
from numpy.distutils.misc_util import get_numpy_include_dirs

setup(
        name            = 'chains_addiction',
        version         = '0.1',
        description     = 'Discrete time, finit state space, stationary Hidden Markov Model.',
        include_dirs    = get_numpy_include_dirs(),
        ext_modules     =   [
                                Extension(
                                    'chains_addiction',
                                    sources     = [ 'hmm/stats.c',
                                                    'hmm/fwbw.c',
                                                    'hmm/em.c',
                                                    'hmm/hmm.c',
                                                    'hmm/hmm_module.c'],

                                include_dirs    = ['./include/'])
                            ]
      )

