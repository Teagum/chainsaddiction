#define NPY_NO_DEPRECATED_API NPY_1_8_API_VERSION

#include "hmm_module.h"


static PyObject *
hmm_poisson_fit_em (PyObject *self, PyObject *args)
{
    PyObject *py_Xtrain = NULL;
    PyObject *py_lambda = NULL;
    PyObject *py_gamma  = NULL;
    PyObject *py_delta  = NULL;
    npy_intp m_states   = 0;
    npy_intp max_iter   = 0;
    double   tol        = 0.0;

    PyArrayObject *X_t     = NULL;
    PyArrayObject *_lambda = NULL;
    PyArrayObject *_gamma  = NULL;
    PyArrayObject *_delta  = NULL;

    if (!PyArg_ParseTuple(args, "OlOOOld", &py_Xtrain, &m_states, &py_lambda, &py_gamma,
        &py_delta, &max_iter, &tol)) return NULL;

    X_t     = (PyArrayObject *) PyArray_FROM_OTF(py_Xtrain, NPY_LONG,       NPY_ARRAY_IN_ARRAY);
    _lambda = (PyArrayObject *) PyArray_FROM_OTF(py_lambda, NPY_LONGDOUBLE, NPY_ARRAY_IN_ARRAY);
    _gamma  = (PyArrayObject *) PyArray_FROM_OTF(py_gamma,  NPY_LONGDOUBLE, NPY_ARRAY_IN_ARRAY);
    _delta  = (PyArrayObject *) PyArray_FROM_OTF(py_delta,  NPY_LONGDOUBLE, NPY_ARRAY_IN_ARRAY);

    if (X_t == NULL || _lambda == NULL || _gamma == NULL || _delta == NULL)
    {
        Py_XDECREF (X_t);
        Py_XDECREF (_lambda);
        Py_XDECREF (_gamma);
        Py_XDECREF (_delta);
        PyErr_SetString (PyExc_MemoryError,
            "Could not allocate memory for input array.");
        Py_RETURN_NONE;
    }

    PoisHmm *ph = PoisHmm_FromData ((size_t) m_states,
                            PyArray_DATA (_lambda),
                            PyArray_DATA (_gamma),
                            PyArray_DATA (_delta),
                            (size_t) max_iter, (scalar) tol);

    if (ph == NULL)
    {
        Py_XDECREF (X_t);
        Py_XDECREF (_lambda);
        Py_XDECREF (_gamma);
        Py_XDECREF (_delta);
        PyErr_SetString (PyExc_MemoryError, "Could not allocate HMM.");
        Py_RETURN_NONE;
    }

    DataSet X_train = {PyArray_DATA (X_t), (size_t) PyArray_SIZE (X_t)};
    int success = PoisHmm_EM (&X_train, ph);
    {
        npy_intp dims_1d[] = { m_states };
        npy_intp dims_2d[] = { m_states, m_states };
        size_t   vector_s  = (size_t) m_states * sizeof (scalar);
        size_t   matrix_s  = (size_t) m_states * vector_s;

        PyObject *py_success = (success == 0) ? Py_True : Py_False;
        Py_INCREF (py_success);

        PyArrayObject *lambda_ = Apollon_NewPyArray1d (dims_1d);
        PyArrayObject *gamma_  = Apollon_NewPyArray2d (dims_2d);
        PyArrayObject *delta_  = Apollon_NewPyArray1d (dims_1d);

        memcpy (PyArray_DATA (lambda_), ph->params->lambda, vector_s);
        memcpy (PyArray_DATA (gamma_),  ph->params->gamma,  matrix_s);
        memcpy (PyArray_DATA (delta_),  ph->params->delta,  vector_s);

        ph->aic = compute_aic(ph->nll, ph->m);
        ph->bic = compute_bic(ph->nll, ph->m, X_train.size);

        PyObject *out = NULL;
        out = Py_BuildValue("ONNNdddk", py_success, lambda_, gamma_, delta_,
                (double) ph->aic, (double) ph->bic,
                (double) ph->nll, ph->n_iter);

        Py_DECREF (X_t);
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
    PyErr_SetString (PyExc_NotImplementedError, "");
    Py_RETURN_NONE;
}


static PyObject *
hmm_poisson_em (PyObject *self, PyObject *args)
{
    PyErr_SetString (PyExc_NotImplementedError, "");
    Py_RETURN_NONE;
}


static PyObject *
hmm_poisson_viterbi (PyObject *self, PyObject *args)
{
    PyErr_SetString (PyExc_NotImplementedError, "");
    Py_RETURN_NONE;
}


static PyMethodDef
CA_Methods[] = {
    {"hmm_poisson_fit_em", hmm_poisson_fit_em, METH_VARARGS,
     "hmm_poisson_fit_em (x, m, _lambda, _gamma, _delta, max_iter, tol)"},

    {"hmm_poisson_fwbw", hmm_poisson_fwbw, METH_VARARGS,
     "docstring"},

    {"hmm_poisson_em", hmm_poisson_em, METH_VARARGS,
     "docstring"},

    {"hmm_poisson_viterbi", hmm_poisson_viterbi, METH_VARARGS,
     "docstring"},

    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef
chainsaddiction_module = {
    PyModuleDef_HEAD_INIT,
    "chainsaddiction",
    NULL,
    -1,
    CA_Methods
};

PyMODINIT_FUNC
PyInit_chainsaddiction (void)
{
    import_array ();
    return PyModule_Create (&chainsaddiction_module);
}
