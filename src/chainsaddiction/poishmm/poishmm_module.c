#define NPY_NO_DEPRECATED_API NPY_1_8_API_VERSION

#include "poishmm_module.h"
#include "structmember.h"

typedef struct {
    PyObject_HEAD
    int err;
    size_t n_iter;
    double llk;
    double aic;
    double bic;
    size_t m_states;
    PyObject *lambda;
    PyObject *gamma;
    PyObject *delta;
} PoisHmmFit;


static PyMemberDef PoisHmmFit_members[] = {
    {"err", T_INT, offsetof (PoisHmmFit, err), 0, "Error number"},
    {"n_iter", T_ULONG, offsetof (PoisHmmFit, n_iter), 0, "Number of iterations"},
    {"llk", T_DOUBLE, offsetof (PoisHmmFit, llk), 0, "Log likelihood"},
    {"aic", T_DOUBLE, offsetof (PoisHmmFit, aic), 0, "Akaike information criterion"},
    {"bic", T_DOUBLE, offsetof (PoisHmmFit, bic), 0, "Bayesian information criterion"},
    {"m_states", T_ULONG, offsetof (PoisHmmFit, m_states), 0, "Number of states"},
    {"lambda_", T_OBJECT, offsetof (PoisHmmFit, lambda), 0, "State-dependent means"},
    {"gamma_", T_OBJECT, offsetof (PoisHmmFit, gamma), 0, "Transition probability matrix"},
    {"delta_", T_OBJECT, offsetof (PoisHmmFit, delta), 0, "Initial distribution"},
    {NULL}  /* Sentinel */
};


static void
PoisHmmFit_Delete (PoisHmmFit *self)
{
    Py_XDECREF (self->lambda);
    Py_XDECREF (self->gamma);
    Py_XDECREF (self->delta);
    Py_TYPE(self)->tp_free((PyObject *) self);
}


static PyObject *
PoisHmmFit_New(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    PoisHmmFit *self = NULL;
    self = (PoisHmmFit *) type->tp_alloc (type, 0);
    if (self != NULL)
    {
        self->err = 1;
        self->n_iter = 0;
        self->llk = 0.0;
        self->aic = 0.0;
        self->bic = 0.0;
        self->m_states = 0;
        self->lambda = NULL;
        self->delta = NULL;
        self->gamma = NULL;
    }
    return (PyObject *) self;
}


static PyTypeObject PoisHmmFit_Type = {
    PyVarObject_HEAD_INIT (NULL, 0)
    .tp_name = "poishmm.Fit",
    .tp_doc = "Stuff",
    .tp_basicsize = sizeof (PoisHmmFit),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PoisHmmFit_New,
    .tp_dealloc = (destructor) PoisHmmFit_Delete,
    .tp_members = PoisHmmFit_members,
};


static PyObject *
poishmm_fit_em (PyObject *self, PyObject *args)
{
    PyObject *arg_lambda = NULL;
    PyObject *arg_gamma  = NULL;
    PyObject *arg_delta  = NULL;
    PyObject *arg_inp    = NULL;
    PyObject *out        = NULL;

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

    out = PoisHmmFit_New (&PoisHmmFit_Type, NULL, NULL);
    ((PoisHmmFit *) out)->err = 0;
    ((PoisHmmFit *) out)->n_iter = hmm.n_iter;
    ((PoisHmmFit *) out)->llk = (double) hmm.llh;
    ((PoisHmmFit *) out)->aic = (double) hmm.aic;
    ((PoisHmmFit *) out)->bic = (double) hmm.bic;

    ((PoisHmmFit *) out)->lambda = PyArray_SimpleNew (PyArray_NDIM (arr_lambda),
                                    PyArray_SHAPE (arr_lambda), NPY_DOUBLE);
    ((PoisHmmFit *) out)->gamma = PyArray_SimpleNew (PyArray_NDIM (arr_gamma),
                                    PyArray_SHAPE (arr_gamma), NPY_DOUBLE);
    ((PoisHmmFit *) out)->delta = PyArray_SimpleNew (PyArray_NDIM (arr_delta),
                                    PyArray_SHAPE (arr_delta), NPY_DOUBLE);

    double *out_ptr = NULL;
    out_ptr = (double *) PyArray_DATA ((PyArrayObject *) (((PoisHmmFit *) out)->lambda));
    for (size_t i = 0; i < hmm.m_states; i++)
    {
        out_ptr[i] = (double) hmm.params->lambda[i];
    }
    out_ptr = (double *) PyArray_DATA ((PyArrayObject *) (((PoisHmmFit *) out)->gamma));
    for (size_t i = 0; i < hmm.m_states * hmm.m_states; i++)
    {
        out_ptr[i] = (double) expl (hmm.params->gamma[i]);
    }
    out_ptr = (double *) PyArray_DATA ((PyArrayObject *) (((PoisHmmFit *) out)->delta));
    for (size_t i = 0; i < hmm.m_states; i++)
    {
        out_ptr[i] = (double) expl (hmm.params->delta[i]);
    }


exit:
    PoisParams_Delete (init);
    PoisParams_Delete (working);
    PoisProbs_Delete (probs);
    Py_XDECREF (arr_inp);
    Py_INCREF (out);
    return out;
}


static PyObject *
read_params (PyObject *self, PyObject *args)
{
    char *path = NULL;
    PoisParams *params = NULL;
    PyObject *arr_lambda = NULL;
    PyObject *arr_delta  = NULL;
    PyObject *arr_gamma  = NULL;
    PyObject *out_lambda = NULL;
    PyObject *out_delta  = NULL;
    PyObject *out_gamma  = NULL;
    PyObject *out_states = NULL;
    PyObject *out = NULL;

    if (!PyArg_ParseTuple (args, "s", &path))
    {
        PyErr_SetString (PyExc_TypeError, "read_params: Could not parse argument.");
        return NULL;
    }

    params = PoisParams_NewFromFile (path);
    const npy_intp shape[2] = {
        (npy_intp) params->m_states,
        (npy_intp) params->m_states
    };

    out_states = PyLong_FromUnsignedLong (params->m_states);
    arr_lambda = PyArray_SimpleNewFromData (1, shape, NPY_LONGDOUBLE, (void *) params->lambda);
    arr_delta  = PyArray_SimpleNewFromData (1, shape, NPY_LONGDOUBLE, (void *) params->delta);
    arr_gamma  = PyArray_SimpleNewFromData (2, shape, NPY_LONGDOUBLE, (void *) params->gamma);
    out_lambda = PyArray_SimpleNew (1, shape, NPY_DOUBLE);
    out_delta  = PyArray_SimpleNew (1, shape, NPY_DOUBLE);
    out_gamma  = PyArray_SimpleNew (2, shape, NPY_DOUBLE);
    PyArray_CopyInto ((PyArrayObject *) out_lambda, (PyArrayObject *) arr_lambda);
    PyArray_CopyInto ((PyArrayObject *) out_delta,  (PyArrayObject *) arr_delta);
    PyArray_CopyInto ((PyArrayObject *) out_gamma,  (PyArrayObject *) arr_gamma);

    out = PyDict_New ();
    if (out == NULL)
    {
        PyErr_SetString (PyExc_TypeError, "read_params: Could not create output values.");
        PoisParams_Delete (params);
        return NULL;
    }

    PyDict_SetItemString (out, "m_states", out_states);
    PyDict_SetItemString (out, "lambda", out_lambda);
    PyDict_SetItemString (out, "delta", out_delta);
    PyDict_SetItemString (out, "gamma", out_gamma);

    Py_INCREF (out_lambda);
    Py_INCREF (out_delta);
    Py_INCREF (out_gamma);
    PoisParams_Delete (params);

    return out;
}


static PyMethodDef
poishmm_methods[] = {
    {"fit_em", poishmm_fit_em, METH_VARARGS, poishmm_fit_em_doc},
    {"read_params", read_params, METH_VARARGS, read_params_doc},
    {NULL, NULL, 0, NULL}
};


static struct PyModuleDef
poishmm_module = {
    PyModuleDef_HEAD_INIT,
    "poishmm",
    "HMM with Poisson-distributed latent variables.",
    -1,
    poishmm_methods
};


PyMODINIT_FUNC
PyInit_poishmm (void)
{
    int err = 0;
    PyObject *module = NULL;

    import_array ();

    module = PyModule_Create (&poishmm_module);
    if (module == NULL) return NULL;

    err = PyType_Ready (&PoisHmmFit_Type);
    if (err < 0) return NULL;

    Py_INCREF (&PoisHmmFit_Type);
    err = PyModule_AddObject (module, "PoisHmmFit", (PyObject *) &PoisHmmFit_Type);
    if (err < 0)
    {
        Py_DECREF (&PoisHmmFit_Type);
        Py_DECREF (module);
        return NULL;
    }
    return module;
}
