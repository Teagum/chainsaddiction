#/usr/bin/env python3
from pathlib import Path
from setuptools import setup, Extension
from setuptools.config import read_configuration
import numpy as np

src_path = Path('chains_addiction')
config = read_configuration('setup.cfg')

ext = Extension('chains_addiction',
        sources = [f'{srcf!s}' for srcf in src_path.glob('*.c')],
        include_dirs = ['include/', np.get_include()],
        # extra_compile_args = ['-Werror=vla'],    # not supported by MSVC
        language = 'c')

setup(ext_modules = [ext])
