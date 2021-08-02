#ifndef hmm_module_h
#define hmm_module_h

#include <Python.h>
#include <numpy/arrayobject.h>
#include "pois_hmm.h"


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


#define PyArray_NEW_LD(py_obj) \
    (PyArrayObject *) PyArray_FROM_OTF (py_obj, NPY_LONGDOUBLE, NPY_ARRAY_IN_ARRAY);

/** Fit HMM for given parameters using EM. Return all params
 * and quality measures.
 */
static PyObject *hmm_poisson_fit_em (PyObject* self, PyObject* args);


static PyObject *
read_params (PyObject *self, PyObject *args);
#endif  /* hmm_module_h */
