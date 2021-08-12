#ifndef poishmm_module_h
#define poishmm_module_h

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_22_API_VERSION
#include <numpy/arrayobject.h>
#include "structmember.h"

#include "pois_hmm.h"
#include "pois_params.h"
#include "fitobject.h"



#define UNUSED(obj) obj = NULL;

#define PyArray_NEW_LD(py_obj) \
    (PyArrayObject *) PyArray_FROM_OTF (py_obj, NPY_LONGDOUBLE, NPY_ARRAY_IN_ARRAY);


#define poishmm_fit_em_doc                                              \
    "Estimate HMM parameters given an observation time series by "      \
    "Expectation Maximization\n\n"                                      \
    "fit_em(n_obs: int, m_states: int, max_iter: int, tol: float, "     \
    "sdm: array-like, tpm: array-like, distr: array-like, "             \
    "data: array-like) -> Fit\n"


#define read_params_doc \
    "read_params(path, /)\n\nRead HMM parameters from file located in ``path``."

#define global_decoding_doc \
    "global_decoding(n_obs, m_states, lgamma, ldelta, lsdp, /)\n\n"    \
    "Compute the most likely sequence of states\n"

static PyObject
*poishmm_fit_em (PyObject* self, PyObject* args);

static PyObject *
read_params (PyObject *self, PyObject *args);


#endif  /* poishmm_module_h */
