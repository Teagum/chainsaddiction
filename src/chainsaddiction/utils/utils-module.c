#include "utils-module.h"


static PyObject *
local_decoding_impl (PyObject *self, PyObject *args)
{
    UNUSED (self);

    npy_intp      *shape        = NULL;
    PyObject      *pyo_lcsp    = NULL;
    PyObject      *pyo_decoding = NULL;
    PyArrayObject *arr_lcsp    = NULL;
    PyArrayObject *arr_decoding = NULL;

    if (!PyArg_ParseTuple (args, "O", &pyo_lcsp))
    {
        const char msg[] = "local_decoding: Could not parse args.";
        PyErr_SetString (PyExc_TypeError, msg);
        goto fail;
    }

    arr_lcsp = (PyArrayObject *) PyArray_FROM_OTF (pyo_lcsp, NPY_LONGDOUBLE,
                                                    NPY_ARRAY_IN_ARRAY);
    if (arr_lcsp == NULL)
    {
        const char msg[] = "local_decoding: Could not convert input array.";
        PyErr_SetString (PyExc_MemoryError, msg);
        goto fail;
    }

    if (PyArray_NDIM (arr_lcsp) != 2)
    {
        const char msg[] = "local_decoding: Number of dimension must be 2.";
        PyErr_SetString (PyExc_TypeError, msg);
        goto fail;
    }

    shape = PyArray_SHAPE (arr_lcsp);

    pyo_decoding = PyArray_SimpleNew (PyCh_VECTOR, shape, NPY_ULONG);
    if (pyo_decoding == NULL)
    {
        const char msg[] = "local_decoding: Could not allocate return object.";
        PyErr_SetString (PyExc_TypeError, msg);
        goto fail;
    }
    arr_decoding = (PyArrayObject *) pyo_decoding;

    local_decoding ((size_t) shape[0], (size_t) shape[1],
            (long double *) PyArray_DATA (arr_lcsp),
            (size_t *) PyArray_DATA (arr_decoding));

    Py_DECREF (arr_lcsp);
    Py_INCREF (arr_decoding);
    return pyo_decoding;


fail:
    Py_XDECREF (arr_lcsp);
    Py_XDECREF (arr_decoding);
    return NULL;
}



static PyObject *
global_decoding_impl (PyObject *self, PyObject *args)
{
    UNUSED (self);

    npy_intp *shape_lcsp = NULL;
    PyObject *pyo_lgamma  = NULL;
    PyObject *pyo_ldelta  = NULL;
    PyObject *pyo_lcsp   = NULL;
    PyObject *pyo_states  = NULL;

    PyArrayObject *arr_lgamma  = NULL;
    PyArrayObject *arr_ldelta  = NULL;
    PyArrayObject *arr_lcsp   = NULL;
    PyArrayObject *arr_states = NULL;

    if (!PyArg_ParseTuple (args, "OOO", &pyo_lgamma, &pyo_ldelta, &pyo_lcsp))
    {
        const char msg[] = "global_decoding: Could not parse args.";
        PyErr_SetString (PyExc_TypeError, msg);
        goto fail;
    }

    arr_lgamma = PyArray_NEW_LD (pyo_lgamma);
    if (arr_lgamma == NULL)
    {
        const char msg[] = "Could not convert lgamma.";
        PyErr_SetString (PyExc_MemoryError, msg);
        goto fail;
    }
    if (PyArray_NDIM (arr_lgamma) != PyCh_MATRIX)
    {
        const char msg[] = "Number of dimension of lgamma does not equal two.";
        PyErr_SetString (PyExc_TypeError, msg);
    }

    arr_ldelta = PyArray_NEW_LD (pyo_ldelta);
    if (arr_ldelta == NULL)
    {
        const char msg[] = "poishmm.global_decoding: Could not allocate ldelta.";
        PyErr_SetString (PyExc_MemoryError, msg);
        goto fail;
    }

    arr_lcsp = PyArray_NEW_LD (pyo_lcsp);
    if (arr_lcsp == NULL)
    {
        const char msg[] = "poishmm.global_decoding: Could not allocate lcsp copy.";
        PyErr_SetString (PyExc_MemoryError, msg);
        goto fail;
    }

    if (PyArray_NDIM (arr_lcsp) != PyCh_DATA)
    {
        const char msg[] = "lcsp has dimension != 2.";
        PyErr_SetString (PyExc_TypeError, msg);
        goto fail;
    }

    shape_lcsp = PyArray_SHAPE (arr_lcsp);
    pyo_states = PyArray_SimpleNew (PyCh_VECTOR, shape_lcsp, NPY_ULONG);
    if (pyo_states == NULL)
    {
        const char msg[] = "global_decoding: Could not allocate states object.";
        PyErr_SetString (PyExc_TypeError, msg);
        goto fail;
    }
    arr_states = (PyArrayObject *) pyo_states;

    global_decoding ((size_t) shape_lcsp[0], (size_t) shape_lcsp[1],
            (long double *) PyArray_DATA (arr_lgamma),
            (long double *) PyArray_DATA (arr_ldelta),
            (long double *) PyArray_DATA (arr_lcsp),
            (size_t *) PyArray_DATA (arr_states));

    Py_DECREF (arr_lgamma);
    Py_DECREF (arr_ldelta);
    Py_DECREF (arr_lcsp);
    Py_INCREF (arr_states);
    return pyo_states;

fail:
    Py_XDECREF (arr_lgamma);
    Py_XDECREF (arr_ldelta);
    Py_XDECREF (arr_lcsp);
    Py_XDECREF (arr_states);
    return NULL;
}


static PyMethodDef
utils_methods[] = {
    {"global_decoding", global_decoding_impl, METH_VARARGS, global_decoding_doc},
    {"local_decoding", local_decoding_impl, METH_VARARGS, local_decoding_doc},
    {NULL, NULL, 0, NULL}
};


static struct PyModuleDef
utils_module = {
    PyModuleDef_HEAD_INIT,
    .m_name    = "utils",
    .m_doc     = "ChainsAddiction HMM utilities",
    .m_size    = -1,
    .m_methods = utils_methods
};


PyMODINIT_FUNC
PyInit_utils (void)
{
    import_array ();
    return PyModule_Create (&utils_module);
}
