#ifndef poishmm_module_h
#define poishmm_module_h

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_22_API_VERSION
#include <numpy/arrayobject.h>
#include "structmember.h"

#include "pois_hmm.h"
#include "pois_params.h"
#include "poishmm_object.h"


#define UNUSED(obj) obj = NULL;

#define PyArray_NEW_LD(py_obj) \
    (PyArrayObject *) PyArray_FROM_OTF (py_obj, NPY_LONGDOUBLE, NPY_ARRAY_IN_ARRAY);


enum PyCh_ArrayDimensions {
    PyCh_VECTOR = 1,
    PyCh_MATRIX = 2,
    PyCh_DATA   = 2,
};


#define poishmm_fit_doc                                                 \
    "Estimate HMM parameters given an observation time series by "      \
    "Expectation Maximization\n\n"                                      \
    "fit_em(n_obs: int, m_states: int, max_iter: int, tol: float, "     \
    "sdm: array-like, tpm: array-like, distr: array-like, "             \
    "data: array-like) -> Fit\n"


#define read_params_doc                                     \
    "read_params(path: str, /)\n\n"                         \
    "Read HMM parameters from file located in ``path``."

#define global_decoding_doc                                 \
    "global_decoding(lgamma, ldelta, lcxpt, /)\n\n"         \
    "Compute the most likely sequence of states\n"

#define local_decoding_doc                                  \
    "local_decoding(lcxpt, /)\n\n"                          \
    "Compute the most likely state for each observation.\n"


static PyObject *
poishmm_fit (PyObject* self, PyObject* args);

static PyObject *
local_decoding_impl (PyObject *self, PyObject *args);

static PyObject *
global_decoding_impl (PyObject *self, PyObject *args);

static PyObject *
read_params (PyObject *self, PyObject *args);


#endif  /* poishmm_module_h */
