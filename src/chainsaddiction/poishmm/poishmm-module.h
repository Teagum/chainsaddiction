#ifndef poishmm_module_h
#define poishmm_module_h

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_22_API_VERSION
#include <numpy/arrayobject.h>
#include "structmember.h"

#include "chainsaddiction.h"
#include "poishmm/pois-hmm.h"
#include "poishmm/pois-params.h"
#include "poishmm/poishmm-object.h"

#define poishmm_fit_doc                                                 \
    "Estimate HMM parameters given an observation time series by "      \
    "Expectation Maximization\n\n"                                      \
    "fit_em(n_obs: int, m_states: int, max_iter: int, tol: float, "     \
    "sdm: array-like, tpm: array-like, distr: array-like, "             \
    "data: array-like) -> Fit\n"

#define poishmm_read_params_doc                                     \
    "read_params(path: str, /)\n\n"                                 \
    "Read HMM parameters from file located in ``path``."


static PyObject *
poishmm_fit (PyObject* self, PyObject* args);

static PyObject *
poishmm_read_params (PyObject *self, PyObject *args);


#endif  /* poishmm_module_h */
