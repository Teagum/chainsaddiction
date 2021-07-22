#define NPY_NO_DEPRECATED_API NPY_1_8_API_VERSION

#include "hmm_module.h"


static PyObject *
hmm_poisson_fit_em (PyObject *self, PyObject *args)
{
    PyObject *arg_lambda = NULL;
    PyObject *arg_gamma  = NULL;
    PyObject *arg_delta  = NULL;

    PyArrayObject *arr_lambda = NULL;
    PyArrayObject *arr_gamma  = NULL;
    PyArrayObject *arr_delta  = NULL;

    PoisHmm hmm = { 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, NULL, NULL, NULL };
    PoisParams *init = NULL;
    PoisParams *working = NULL;
    PoisProbs *probs = NULL;

    if (!PyArg_ParseTuple (args, "llldOOO",
                (npy_intp *) &hmm.n_obs,
                (npy_intp *) &hmm.m_states,
                (npy_intp *) &hmm.max_iter,
                (double *) &hmm.tol,
                &arg_lambda, &arg_gamma, &arg_delta))
    {
        return NULL;
    }

    arr_lambda = (PyArrayObject *) PyArray_FROM_OTF (arg_lambda, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    arr_gamma  = (PyArrayObject *) PyArray_FROM_OTF (arg_gamma,  NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    arr_delta  = (PyArrayObject *) PyArray_FROM_OTF (arg_delta,  NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (arr_lambda == NULL || arr_gamma == NULL || arr_delta == NULL)
    {
        PyErr_SetString (PyExc_MemoryError, "Something went wrong during array creation.");
        goto fail;
    }

    init = PoisParams_New (hmm.m_states);
    working = PoisParams_New (hmm.m_states);
    probs = PoisProbs_New (hmm.n_obs, hmm.m_states);
    if (init == NULL || working == NULL || probs == NULL)
    {
        PyErr_SetString (PyExc_MemoryError, "Error during HMM init.");
        goto fail;
    }

    PoisParams_SetLambda (init, PyArray_DATA (arr_lambda));
    PoisParams_SetGamma  (init, PyArray_DATA (arr_gamma));
    PoisParams_SetDelta  (init, PyArray_DATA (arr_delta));
    PoisParams_CopyLog (init, working);

    hmm.init = init;
    hmm.params = working;
    hmm.probs = probs;

    printf ("hmm.n_obs: %zu, hmm.m_states: %zu\ninit.m_states: %zu, working.m_states: %zu\nprobs.n_obs: %zu, probs.m_states: %zu\n",
            hmm.n_obs, hmm.m_states, hmm.init->m_states, hmm.params->m_states, hmm.probs->n_obs, hmm.probs->m_states);

    PoisParams_Print (hmm.init);
    PoisParams_Print (hmm.params);
    /*
    printf ("n_obs: %zu\nm_states: %zu\nmax_iter: %zu\ntol: %2.15f\n",
            hmm.n_obs, hmm.m_states, hmm.max_iter, hmm.tol);
            */

fail:
    PoisParams_Delete (init);
    PoisParams_Delete (working);
    PoisProbs_Delete (probs);
    Py_XDECREF (arr_lambda);
    Py_XDECREF (arr_gamma);
    Py_XDECREF (arr_delta);
    Py_RETURN_NONE;
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
