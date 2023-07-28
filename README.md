[![build](https://github.com/Teagum/chainsaddiction/actions/workflows/build.yml/badge.svg)](
https://github.com/Teagum/chainsaddiction/actions/workflows/build.yml)

# ChainsAddiction

ChainsAddiction is an easy to use tool for time series analysis using
discrete-time Hidden Markov Models. It is written in `C` as a `numpy`-based
Python extension module.


## Installation
### Install from PyPi

We currently provide wheels for macOS and Windows AMD 64, which you can install from PyPI via:

    python3 -m pip install chainsaddiction

Linux users have to build from source until we get that manylinux thing running.


### Install from source

Before attemting to build ChainsAddiction from source, make sure you have

- Python >= 3.9
- pip, setuptools
- C compiler

installed and ready to go.

Then, clone the source code by typing the following command in your terminal app.
Replace `path/to/ca` with the path to where ChainsAddiction should be cloned:

    git clone https://github.com/teagum/chainsaddiction path/to/ca

Second, change to the root directory of your freshly cloned code repository:

    cd path/to/ca

Third, instruct Python to build and install ChainsAddiction:

    python3 -m pip install .

---

## Notes
Currently only Poisson-distributed HMM are implemented.
