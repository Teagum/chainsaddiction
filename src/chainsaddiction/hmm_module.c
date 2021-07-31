#define NPY_NO_DEPRECATED_API NPY_1_8_API_VERSION

#include "hmm_module.h"


static PyObject *
hmm_poisson_fit_em (PyObject *self, PyObject *args)
{
    PyObject *arg_lambda = NULL;
    PyObject *arg_gamma  = NULL;
    PyObject *arg_delta  = NULL;
    PyObject *arg_inp    = NULL;

    PyArrayObject *arr_lambda = NULL;
    PyArrayObject *arr_gamma  = NULL;
    PyArrayObject *arr_delta  = NULL;
    PyArrayObject *arr_inp    = NULL;

    DataSet inp = { NULL, 0, false };
    PoisHmm hmm = { 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, NULL, NULL, NULL };
    PoisParams *init = NULL;
    PoisParams *working = NULL;
    PoisProbs *probs = NULL;

    double tol_buffer = 0;
    if (!PyArg_ParseTuple (args, "llldOOOO",
                &hmm.n_obs,
                &hmm.m_states,
                &hmm.max_iter,
                &tol_buffer,
                &arg_lambda, &arg_gamma, &arg_delta, &arg_inp))
    {
        return NULL;
    }
    hmm.tol = (long double) tol_buffer;

    arr_lambda = PyArray_NEW_LD (arg_lambda);
    arr_gamma  = PyArray_NEW_LD (arg_gamma);
    arr_delta  = PyArray_NEW_LD (arg_delta);
    arr_inp    = PyArray_NEW_LD (arg_inp);

    if (arr_lambda == NULL || arr_gamma == NULL || arr_delta == NULL || arr_inp == NULL)
    {
        PyErr_SetString (PyExc_MemoryError, "Something went wrong during array creation.");
        goto exit;
    }

    init = PoisParams_New (hmm.m_states);
    working = PoisParams_New (hmm.m_states);
    probs = PoisProbs_New (hmm.n_obs, hmm.m_states);
    if (init == NULL || working == NULL || probs == NULL)
    {
        PyErr_SetString (PyExc_MemoryError, "Error during HMM init.");
        goto exit;
    }

    PoisParams_SetLambda (init, PyArray_DATA (arr_lambda));
    PoisParams_SetGamma  (init, PyArray_DATA (arr_gamma));
    PoisParams_SetDelta  (init, PyArray_DATA (arr_delta));
    PoisParams_CopyLog (init, working);

    hmm.init = init;
    hmm.params = working;
    hmm.probs = probs;
    inp.size = PyArray_SIZE (arr_inp);
    inp.data = PyArray_DATA (arr_inp);

    PoisHmm_EstimateParams (&hmm, &inp);
    PoisHmm_PrintParams (&hmm);

exit:
    PoisParams_Delete (init);
    PoisParams_Delete (working);
    PoisProbs_Delete (probs);
    Py_XDECREF (arr_lambda);
    Py_XDECREF (arr_gamma);
    Py_XDECREF (arr_delta);
    Py_XDECREF (arr_inp);
    Py_RETURN_NONE;
}


static PyMethodDef
CA_Methods[] = {
    {"hmm_poisson_fit_em", hmm_poisson_fit_em, METH_VARARGS,
     "hmm_poisson_fit_em (x, m, _lambda, _gamma, _delta, max_iter, tol)"},

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
