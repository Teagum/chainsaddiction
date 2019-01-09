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


#endif  /* hmm_module_h */
