*******************************************************************************
Getting started
*******************************************************************************

Prerequisites
===============================================================================

ChainsAddiction provides a Python extension module that needs to be compiled
before you can use it. Therefore, your system has to meet the following
requirements:

* Python >= 3.9
* pip and setuptools installed
* C compiler (Windows and GNU/Linux only)

No matter which operating system you use, the first step is to check the
currently installed Python version. To do so, open a terminal and execute the
following command:

.. code-block:: sh

    python --version

.. note::

   The Python interpreter may be available under different names, such as ``python3``, or
   ``python3.9``, etc, depending on your operating system and virtual
   environment setup.


Installation
===============================================================================

ChainsAddiction is available on `PyPi`_. That means you can download and
install it with the following command:

.. code-block:: sh

   pip3 install chainsaddiction

ChainsAddiction provides pre-built wheels only for macOS. On all other
platforms, pip will build ChainsAddiction from the source distribution. In this
case you have to make sure that there is a C compiler available on your system.

.. _PyPi: https://pypi.org/project/chainsaddiction/


Building from source
===============================================================================
Additionally, the `source code`_ with the latest changes is available on github. You can download it
from there. If you have git installed, you may clone the repository with the
following command:

.. code-block:: sh

   git clone https://github.com/teagum/chainsaddiction.git

In order to build ChainsAddiction, change to the repository`s root directory and execute

.. code-block:: sh

   make install

You can also use the make command to run the test suite, rebuild the extension
module, and build the documentation. Use make without a target to get a list of
all possible options.

.. _source code: https://github.com/teagum/chainsaddiction


