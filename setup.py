import itertools
from pathlib import Path
from typing import Generator

import numpy as np
from setuptools import setup, Extension


def cglob(path: str) -> Generator[str, None, None]:
    """Generate all .c files in ``path``."""
    return (f'{src!s}' for src in Path(path).glob('*.c'))

def list_source_files (paths: tuple[str, ...]) -> list[str]:
    """Generate a list of .c files in found in all of ``paths``."""
    return list(itertools.chain.from_iterable(cglob(path) for path in paths))


utils_src = (
    'src/vmath',
    'src/chainsaddiction/utils',
)

poishmm_src = (
    'src/vmath',
    'src/chainsaddiction',
    'src/chainsaddiction/utils',
    'src/chainsaddiction/poishmm',
)


utils = Extension('chainsaddiction.utils',
        sources = list_source_files(utils_src),
        include_dirs = [
            'include',
            'src/vmath',
            'src/chainsaddiction',
            'src/chainsaddiction/utils',
            np.get_include(),
        ],
        language = 'c')

poishmm = Extension('chainsaddiction.poishmm',
        sources = list_source_files(poishmm_src),
        include_dirs =  [
            'include',
            'src/vmath',
            'src/chainsaddiction',
            'src/chainsaddiction/poishmm',
            np.get_include()
        ],
        language = 'c')


setup(ext_modules = [utils, poishmm])
