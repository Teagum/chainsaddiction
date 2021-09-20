#ifndef chainsaddiction_module_h
#define chainsaddiction_module_h

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_22_API_VERSION
#include <numpy/arrayobject.h>

#include "chainsaddiction.h"
#include "utils.h"


#define local_decoding_doc                                  \
    "local_decoding(lcsp, /)\n\n"                           \
    "Compute the most likely state for each observation.\n"

#define global_decoding_doc                                 \
    "global_decoding(lgamma, ldelta, lcsp, /)\n\n"          \
    "Compute the most likely sequence of states\n"


static PyObject *
local_decoding_impl (PyObject *self, PyObject *args);

static PyObject *
global_decoding_impl (PyObject *self, PyObject *args);


#endif  /* chainsaddiction_module_h */
