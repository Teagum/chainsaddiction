#/usr/bin/env python3

"""ChainsAddiction setup"""

import itertools
from pathlib import Path
from typing import Generator, Tuple
from setuptools import setup, Extension
import numpy as np


def cglob(path: str) -> Generator:
    """Generate all .c files in ``path``."""
    return (f'{src!s}' for src in Path(path).glob('*.c'))

def list_source_files (paths: Tuple[str, ...]) -> list:
    """Generate a list of .c files in found in all of ``paths``."""
    return list(itertools.chain.from_iterable(cglob(path) for path in paths))


poishmm_src = (
    'src/vmath',
    'src/chainsaddiction',
    'src/chainsaddiction/poishmm',
)


poishmm = Extension('chainsaddiction.poishmm',
        sources = list_source_files(poishmm_src),
        include_dirs =  [
            'include',
            'src/vmath',
            'src/chainsaddiction',
            'src/chainsaddiction/poishmm',
            np.get_include()
        ],
        extra_compile_args = ['-Wall', '-Wextra'],
        language = 'c')


setup(ext_modules = [poishmm])
