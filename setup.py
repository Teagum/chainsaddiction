#/usr/bin/env python3
from pathlib import Path
from setuptools import setup, Extension
from setuptools.config import read_configuration
import numpy as np

src_path = Path('src/chainsaddiction')
config = read_configuration('setup.cfg')

ext = Extension('chainsaddiction',
        sources = [f'{srcf!s}' for srcf in src_path.glob('*.c')],
        include_dirs = ['src/chainsaddiction', np.get_include()],
        # extra_compile_args = ['-Werror=vla'],    # not supported by MSVC
        language = 'c')

setup(ext_modules = [ext])
