#/usr/bin/env python3
import itertools
from pathlib import Path
import sys
from setuptools import setup, Extension
import numpy as np


def cglob(path: str):
    """Generate all .c files in ``path``."""
    return (f'{src!s}' for src in Path(path).glob('*.c'))

def list_source_files (paths: list[str]) -> list:
    """Generate a list of .c files in found in all of ``paths``."""
    return list(itertools.chain.from_iterable(cglob(path) for path in paths))


def main(argv=None) -> int:
    """Setup entry point

    List all directory that include C source files in the appropriate tuples
    below.
    """
    if argv is None:
        argv = sys.argv

    c_src_dirs = (
        'src/vmath',
        'src/chainsaddiction',
        'src/chainsaddiction/poishmm',
    )

    c_include_dirs = (
        'include',
        'src/chainsaddiction/',
        'src/chainsaddiction/poishmm',
        np.get_include()
    )

    ext = Extension('chainsaddiction',
        sources = list_source_files(c_src_dirs),
        include_dirs = c_include_dirs,
        extra_compile_args = ['-Wall', '-Wextra'],
        language = 'c')

    return setup(ext_modules = [ext])


if __name__ == '__main__':
    sys.exit(main())
