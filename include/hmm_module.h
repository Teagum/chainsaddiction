#ifndef hmm_module_h
#define hmm_module_h

#include <Python.h>
#include <numpy/arrayobject.h>
#include "hmm.h"
#include "em.h"
#include "fwbw.h"
#include "scalar.h"

#define Apollon_NewPyArray1d(dims)							\
	 (PyArrayObject*) PyArray_New(&PyArray_Type,			\
								  1, dims, NPY_LONGDOUBLE,	\
								  NULL, NULL, 0, 			\
								  NPY_ARRAY_CARRAY, NULL)	\

#define Apollon_NewPyArray2d(dims)							\
	 (PyArrayObject*) PyArray_New(&PyArray_Type,			\
								  2, dims, NPY_LONGDOUBLE,	\
								  NULL, NULL, 0, 			\
								  NPY_ARRAY_CARRAY, NULL)	\

#endif	/* hmm_module_h */
