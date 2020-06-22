# ChainsAddiction

ChainsAddiction is an easy to use  tool for time series analysis with
discrete-time Hidden Markov Models. It is written in `C` as a  `numpy`-based
Python extension module.


## Installation
### Prerequisites
The installation of chainsaddiction requires to following tools:
- terminal emulator
- git
- Python >= 3.7
- pip
- C Compiler
Hence, users with BSD und Unix-like systems should be fine.

## How to install
First, clone this repository by copying the following command in your terminal app.
Replace `path/to/ca/` with the directory in which the chainsaddiction should be cloned.

    git clone https://gitlab.rrz.uni-hamburg.de/bal7668/chainsaddiction.git path/to/ca/

Second, change to root directory of your freshly clone chainsaddiction repo:

    cd path/to/ca

Third, instruct Python to install chainsaddiction:

    pip install .

DONE.


## Working with the Python interpreter
Calling chainsaddiction from `Python` is simple as pie. You just need to import
it:

    import chainsaddiction as ca
    ca.hmm_poisson_fit_em(x, m, init_means, init_tpm, int_sd, max_iter=1000, tol=1e-5)

## Notes
- Currently only Poisson-distributed HMM are implemented.
