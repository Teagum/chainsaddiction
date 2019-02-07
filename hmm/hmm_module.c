#define NPY_NO_DEPRECATED_API NPY_1_8_API_VERSION

#include "hmm_module.h"


static PyObject *
hmm_poisson_fit(PyObject* self, PyObject* args)
{
    /* 1. alloc alpha, beta, pprob
     * 2. call fwbw
     * 3. wrap alpha, beta, pprob in arrays
     * 4. return the arrays  */
    Py_RETURN_NONE;
}


static PyObject *
hmm_poisson_EM(PyObject* self, PyObject* args)
{
    PyObject *py_X      = NULL;
    PyObject *py_lambda = NULL;
    PyObject *py_gamma  = NULL;
    PyObject *py_delta  = NULL;
    npy_intp max_iter   = 0;
    double   tol        = 0.0;

    PyArrayObject *X       = NULL;
    PyArrayObject *_lambda = NULL;
    PyArrayObject *_gamma  = NULL;
    PyArrayObject *_delta  = NULL;

    if (!PyArg_ParseTuple(args, "OOOOld", &py_X, &py_lambda, &py_gamma,
        &py_delta, &max_iter, &tol)) return NULL;

    X       = (PyArrayObject *) PyArray_FROM_OTF(py_X, NPY_LONG, NPY_ARRAY_IN_ARRAY);
    _lambda = (PyArrayObject *) PyArray_FROM_OTF(py_lambda, NPY_LONGDOUBLE, NPY_ARRAY_IN_ARRAY);
    _gamma  = (PyArrayObject *) PyArray_FROM_OTF(py_gamma, NPY_LONGDOUBLE, NPY_ARRAY_IN_ARRAY);
    _delta  = (PyArrayObject *) PyArray_FROM_OTF(py_delta, NPY_LONGDOUBLE, NPY_ARRAY_IN_ARRAY);

    if (X == NULL || _lambda == NULL || _gamma == NULL || _delta == NULL)
    {
        Py_XDECREF (X);
        Py_XDECREF (_lambda);
        Py_XDECREF (_gamma);
        Py_XDECREF (_delta);
        PyErr_SetString (PyExc_MemoryError, "Memory Error");
        Py_RETURN_NONE;
    }

    npy_intp m = PyArray_SIZE (_lambda);

    PoisHmm *ph = PoisHmm_FromData ((size_t) m,
                            PyArray_DATA (_lambda),
                            PyArray_DATA (_gamma),
                            PyArray_DATA (_delta),
                            (size_t) max_iter, (scalar) tol);
    if (ph == NULL)
    {
        Py_XDECREF (X);
        Py_XDECREF (_lambda);
        Py_XDECREF (_gamma);
        Py_XDECREF (_delta);
        PyErr_SetString (PyExc_MemoryError, "Could not allocate HMM.");
        Py_RETURN_NONE;
    }

    DataSet X_train = {PyArray_DATA (X), (size_t) PyArray_SIZE (X)};

    int success = PoisHmm_EM (&X_train, ph); 

    {
        npy_intp dims_1d[] = { m };
        npy_intp dims_2d[] = { m, m };
        size_t   vector_s  = (size_t) m * sizeof (scalar);
        size_t   matrix_s  = (size_t) m * vector_s;

        PyArrayObject *lambda_ = Apollon_NewPyArray1d (dims_1d);
        PyArrayObject *gamma_  = Apollon_NewPyArray2d (dims_2d);
        PyArrayObject *delta_  = Apollon_NewPyArray1d (dims_1d);

        memcpy (PyArray_DATA (lambda_), ph->params->lambda, vector_s);
        memcpy (PyArray_DATA (gamma_),  ph->params->gamma,  matrix_s);
        memcpy (PyArray_DATA (delta_),  ph->params->delta,  vector_s);

        ph->aic = compute_aic(ph->nll, ph->m);
        ph->bic = compute_bic(ph->nll, ph->m,  X_train.size);

        PyObject *out = NULL;
        out = Py_BuildValue("iNNNdddk", success, lambda_, gamma_, delta_, 
                            (double) ph->aic, (double) ph->bic,
                            (double) ph->nll, ph->n_iter);

        Py_DECREF (X);
        Py_DECREF (_lambda);
        Py_DECREF (_gamma);
        Py_DECREF (_delta);
        PoisHmm_DeleteHmm (ph);

        return out;
    }
}


static PyObject *
hmm_poisson_fwbw(PyObject *self, PyObject *args)
{
    PyArrayObject *x        = NULL;
    PyArrayObject *lambda_  = NULL;
    PyArrayObject *gamma_   = NULL;
    PyArrayObject *delta_   = NULL;

    if (!PyArg_ParseTuple(args, "OOOO", &x, &lambda_, &gamma_, &delta_))
        return NULL;

    npy_intp    n       = PyArray_SIZE (x);
    npy_intp    m       = PyArray_SIZE (lambda_);
    npy_intp    dims[]  = { n, m };

    PyArrayObject *alpha    = Apollon_NewPyArray2d (dims);
    PyArrayObject *beta     = Apollon_NewPyArray2d (dims);
    PyArrayObject *pois_pr  = Apollon_NewPyArray2d (dims);
    
    /*
    if (success == 1)
    {
        PyObject *out = PyTuple_Pack(3, alpha, beta, lp_prob);
        Py_INCREF(out);
        return out;
    }
    else
    {
        PyErr_SetString(PyExc_RuntimeError, "Training failed.");
        Py_RETURN_NONE;
    }
    */
    Py_RETURN_NONE;
}


static PyObject *
hmm_poisson_viterbi(PyObject* self, PyObject* args)
{
    Py_RETURN_NONE;
}

static PyMethodDef
HMM_Methods[] = {
    {"hmm_poisson_fit", hmm_poisson_fit, METH_VARARGS, 
     "hmm_poisson_fit(x, m, _lambda, _gamma, _delta, max_iter, tol)"},
    
    {"hmm_poisson_EM", hmm_poisson_EM, METH_VARARGS,
     "docstring"},

    {"hmm_poisson_fwbw", hmm_poisson_fwbw, METH_VARARGS,
     "docstring"},

    {"hmm_poisson_viterbi", hmm_poisson_viterbi, METH_VARARGS,
     "docstring"},

    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef
hmm_module = {
    PyModuleDef_HEAD_INIT,
    "hmm",
    NULL,
    -1,
    HMM_Methods
};

PyMODINIT_FUNC
PyInit_hmm(void)
{
    import_array();
    return PyModule_Create (&hmm_module);
}

