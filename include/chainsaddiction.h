#ifndef chainsaddiction_h
#define chainsaddiction_h

#include "../src/chainsaddiction/err.h"
#include "../src/chainsaddiction/scalar.h"
#include "../src/chainsaddiction/restrict.h"


#define UNUSED(obj) (void)(obj)

#define PyArray_NEW_LD(py_obj) \
    (PyArrayObject *) PyArray_FROM_OTF (py_obj, NPY_LONGDOUBLE, NPY_ARRAY_IN_ARRAY);


enum PyCh_ArrayDimensions {
    PyCh_VECTOR = 1,
    PyCh_MATRIX = 2,
    PyCh_DATA   = 2,
};


#endif  /* chainsaddiction_h */
