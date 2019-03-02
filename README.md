# ChainsAddiction

ChainsAddiction is a tool for simple training discrete-time Hidden Markov
Models. It is written in `C` and features a `numpy`-based Python extension
module.

## Installation
Clone this repository, change to its root directory and issue

    pip install .

## Working with the C API

## Working with the Python interpreter
Calling Chains_addiction from `Python` is simple as pie. You just need to import
it:

    import chains_addiction as ca
    ca.hmm_poisson_fit_em(x, m, init_means, init_tpm, int_sd, max_iter=1000, tol=1e-5)

## Notes
- Currently only Poisson-distributed HMM are implemented.
- ChainsAddiction does not support Python 2. Specifically, it requires `Python >= 3.5` and `numpy >= 1.16`.