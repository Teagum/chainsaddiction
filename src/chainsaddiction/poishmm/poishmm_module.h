#ifndef poishmm_module_h
#define poishmm_module_h

#include <Python.h>
#include <numpy/arrayobject.h>
#include "pois_hmm.h"
#include "pois_params.h"


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
