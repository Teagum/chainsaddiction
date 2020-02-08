#/usr/bin/env python3
from pathlib import Path
from setuptools import setup, Extension
from setuptools.config import read_configuration


class GetNumpyInclude:
    def __str__(self):
        import numpy
        return numpy.get_include()

src_path = Path('chains_addiction')
config = read_configuration('setup.cfg')
ext = Extension('chains_addiction',
        sources = [f'{srcf!s}' for srcf in src_path.glob('*.c')],
        include_dirs = ['include/', GetNumpyInclude()])
setup(ext_modules = [ext])
