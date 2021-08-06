#ifndef poishmm_module_h
#define poishmm_module_h

#define NPY_NO_DEPRECATED_API NPY_1_8_API_VERSION
#define PY_SSIZE_T_CLEAN

#include <Python.h>
#include "structmember.h"
#include <numpy/arrayobject.h>
#include "pois_hmm.h"
#include "pois_params.h"


#define UNUSED(obj) obj = NULL;

#define PyArray_NEW_LD(py_obj) \
    (PyArrayObject *) PyArray_FROM_OTF (py_obj, NPY_LONGDOUBLE, NPY_ARRAY_IN_ARRAY);


#define poishmm_fit_em_doc                                              \
    "Estimate the HMM parameters given an observation time series by"   \
    "Expectation Maximization"

#define read_params_doc \
    "read_params(path, /)\n\nRead HMM parameters from file located in ``path``."


static PyObject
*poishmm_fit_em (PyObject* self, PyObject* args);

static PyObject *
read_params (PyObject *self, PyObject *args);


#endif  /* poishmm_module_h */
