# ChainsAddiction

ChainsAddiction is an easy to use tool for time series analysis using
discrete-time Hidden Markov Models. It is written in `C` as a `numpy`-based
extension module for CPython.



## Installation
### Prerequisites
ChainsAddiction requires at least CPython 3.7. Installation is possible via
pip, or by building from source.

The pip setup procedure installs wheels. If you don't want to use wheels, pip
will try to install from source. In this case you have to have a ready to use C
compiler installed.



### Install with pip
You can install chainsaddiction from PyPi with:

    pip install chainsaddiction

lease note that ChainsAddiction is a CPython Extension module. Currently we
provide wheels for macOS. GNU/Linux and Windows users have to build from
source.



### Install from source
First, clone the source code by typing the following command in your terminal app.
Replace `path/to/ca` with the path to the directory into which ChainsAddiction should be cloned.

    git clone https://github.com/teagum/chainsaddiction path/to/ca

Second, change to root directory of your freshly cloned code repository:

    cd path/to/ca

Third, instruct Python to build ad install ChainsAddiction:

    pip install .



## Notes
Currently only Poisson-distributed HMM are implemented.
