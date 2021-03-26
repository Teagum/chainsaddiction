# ChainsAddiction

ChainsAddiction is an easy to use tool for time series analysis with
discrete-time Hidden Markov Models. It is written in `C` as a `numpy`-based
Python extension module.


## Installation
### Prerequisites
The installation of ChainsAddiction requires to following tools to be installed
on your system:
- Python >= 3.7
- pip
- C compiler

### Install with pip
You can install chainsaddiction from PyPi with:

    pip install chainsaddiction

Please note that ChainsAddiction is a CPython Extension module. You have to
have set up a C compiler in order to install. Currently we provide wheel for
macOS. So, if you are using this OS you do not need a compiler.


### Install from source
First, clone the source code by typing the following command in your terminal app.
Replace `path/to/ca` with the directory in which ChainsAddiction should be cloned.

    git clone https://github.com/teagum/chainsaddiction path/to/ca

Second, change to root directory of your freshly cloned code repository:

    cd path/to/ca

Third, instruct Python to build ad install ChainsAddiction:

    pip install .

DONE.


## Working with the Python interpreter
Calling chainsaddiction from `Python` is simple as pie. You just need to import
it:

    import chainsaddiction as ca
    ca.hmm_poisson_fit_em(x, m, init_means, init_tpm, int_sd, max_iter=1000, tol=1e-5)

## Notes
Currently only Poisson-distributed HMM are implemented.
