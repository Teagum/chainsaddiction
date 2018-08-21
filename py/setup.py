#!python3

from setuptools import setup, Extension
from numpy.distutils.misc_util import get_numpy_include_dirs


# create the extension module
mod = Extension('hmm', sources = ['hmm/hmm_utilities.c']) 

# common setup
setup(name = 'hmm',
      version = '0.1',
      description = 'hmm algorithms',
      ext_modules = [mod],
      include_dirs = get_numpy_include_dirs()
      )
