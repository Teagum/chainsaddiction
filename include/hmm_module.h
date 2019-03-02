#ifndef hmm_module_h
#define hmm_module_h

#include <Python.h>
#include <numpy/arrayobject.h>
#include "hmm.h"
#include "em.h"
#include "fwbw.h"
#include "scalar.h"


#define Apollon_NewPyArray1d(shape)                     \
        ((PyArrayObject *)                              \
        PyArray_NewFromDescr (&PyArray_Type,            \
                PyArray_DescrFromType (NPY_LONGDOUBLE), \
                1, shape, NULL, NULL, 0, NULL));


#define Apollon_NewPyArray2d(shape)                     \
        ((PyArrayObject *)                              \
        PyArray_NewFromDescr (&PyArray_Type,            \
                PyArray_DescrFromType (NPY_LONGDOUBLE), \
                2, shape, NULL, NULL, 0, NULL));


/** Perform EM for given parameters. Return only fitted params. */
static PyObject *hmm_poisson_em (PyObject* self, PyObject* args);


/** Fit HMM for given parameters using EM. Return all params 
 * and quality measures. 
 */
static PyObject *hmm_poisson_fit_em (PyObject* self, PyObject* args);


/** Return forward backward, and state-dependent probabilites. */
static PyObject *hmm_poisson_fwbw (PyObject *self, PyObject *args);


/** Calculate viterbi path given HMM. */
static PyObject *hmm_poisson_viterbi (PyObject *self, PyObject *args);


#endif  /* hmm_module_h */
