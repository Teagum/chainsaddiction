*******************************************************************************
Installation
*******************************************************************************

Download
===============================================================================

ChainsAddiction is available on `PyPi`_. That means you can download and install
it with the common ``pip`` command explained below. Note that ``pip`` only installs the Python bindings.

Additionally, the `source code`_ is available on github. You can download it from
there. If you have git installed, you may just want clone the repository with
the following command:

.. code-block:: sh

   git clone https://github.com/teagum/chainsaddiction.git


.. _PyPi: https://pypi.org

.. _source code: https://github.com/teagum/chainsaddiction


Installation
===============================================================================

ChainsAddiction provides a Python extension module that needs to be compiled
before you can use it. Therefore, your system has to meet the following
requierements:

* (at least) Python 3.7
* pip / setuptools / wheel (optional)
* C compiler

No matter which operationg system you use, the first step is to check the
currently installed Python version. To do so, open a terminal and execute the
following command:

.. code-block:: sh

    python --version

.. note::

   The Python interpreter may be available under different names, such as ``python3``, or
   ``python3.7``, etc, depending on your operating system and virtual
   environment setup.
