#/usr/bin/env python3

from setuptools import setup, Extension
from numpy.distutils.misc_util import get_numpy_include_dirs

setup(
        name            = 'hmm',
        version         = '0.1',
        description     = 'PoissonHMM for Python3',
        include_dirs    = get_numpy_include_dirs(),
        ext_modules     =   [
                                Extension(
                                    'hmm',
                                    sources     = [ 'hmm/stats.c',
                                                    'hmm/fwbw.c',
                                                    'hmm/em.c',
                                                    'hmm/hmm.c',
                                                    'hmm/hmm_module.c'],

                                include_dirs    = ['./include/'])
                            ]
      )

