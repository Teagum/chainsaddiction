# ChainsAddiction

ChainsAddiction is an easy to use tool for time series analysis using
discrete-time Hidden Markov Models. It is written in `C` as a `numpy`-based
Python extension module.


## Installation
### Prerequisites

The installation of ChainsAddiction requires to following tools to be installed
on your system:

- Python >= 3.9
- pip, setuptools
- C compiler


### Install from PyPi

You can install chainsaddiction from PyPi with:

    python3 -m pip install chainsaddiction

Please note that ChainsAddiction is a CPython extension module. You have to
have set up a C compiler in order to install. Currently we provide wheels for
macOS. So, if you are using this OS you do not need a compiler.


### Install from source

First, clone the source code by typing the following command in your terminal app.
Replace `path/to/ca` with the path to where ChainsAddiction should be cloned:

    git clone https://github.com/teagum/chainsaddiction path/to/ca

Second, change to the root directory of your freshly cloned code repository:

    cd path/to/ca

Third, instruct Python to build and install ChainsAddiction:

    python3 -m pip install .

---

## Notes
Currently only Poisson-distributed HMM are implemented.
