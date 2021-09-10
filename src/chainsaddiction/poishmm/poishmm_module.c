#include "poishmm_module.h"


static void
PyCh_PoisHmm_Delete (PyCh_PoisHmm *self)
{
    Py_XDECREF (self->lambda);
    Py_XDECREF (self->gamma);
    Py_XDECREF (self->delta);
    Py_XDECREF (self->lalpha);
    Py_XDECREF (self->lbeta);
    Py_XDECREF (self->lcxpt);
    Py_TYPE(self)->tp_free((PyObject *) self);
}


static PyObject *
PyCh_PoisHmm_New(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    UNUSED (args);
    UNUSED (kwds);

    PyCh_PoisHmm *self = NULL;
    self = (PyCh_PoisHmm *) type->tp_alloc (type, 0);
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
        self->lalpha = NULL;
        self->lbeta = NULL;
        self->lcxpt = NULL;
    }
    return (PyObject *) self;
}


static int
PyCh_PoisHmm_CInit (PyCh_PoisHmm *self, const size_t n_obs, const size_t m_states)
{
    const npy_intp dims_vector[]  = { (npy_intp) m_states };
    const npy_intp dims_matrix[]  = { (npy_intp) m_states, (npy_intp) m_states };
    const npy_intp dims_data[]    = { (npy_intp) n_obs,    (npy_intp) m_states };

    self->lambda = PyArray_SimpleNew (PyCh_VECTOR, dims_vector, NPY_LONGDOUBLE);
    self->gamma  = PyArray_SimpleNew (PyCh_MATRIX, dims_matrix, NPY_LONGDOUBLE);
    self->delta  = PyArray_SimpleNew (PyCh_VECTOR, dims_vector, NPY_LONGDOUBLE);
    self->lalpha = PyArray_SimpleNew (PyCh_DATA,   dims_data,   NPY_LONGDOUBLE);
    self->lbeta  = PyArray_SimpleNew (PyCh_DATA,   dims_data,   NPY_LONGDOUBLE);
    self->lcxpt  = PyArray_SimpleNew (PyCh_DATA,   dims_data,   NPY_LONGDOUBLE);
    return 0;
}


static void
PyCh_PoisHmm_Set (PyCh_PoisHmm *out, PoisHmm *hmm)
{
    const npy_intp dims_data[] = { (npy_intp) hmm->n_obs, (npy_intp) hmm->m_states };

    long double *lambda_out = (long double *) PyArray_DATA ((PyArrayObject *) out->lambda);
    long double *gamma_out  = (long double *) PyArray_DATA ((PyArrayObject *) out->gamma);
    long double *delta_out  = (long double *) PyArray_DATA ((PyArrayObject *) out->delta);
    long double *lambda_est = hmm->params->lambda;
    long double *gamma_est  = hmm->params->gamma;
    long double *delta_est  = hmm->params->delta;

    PyObject *wrap_lalpha = PyArray_SimpleNewFromData (PyCh_DATA, dims_data,
                                NPY_LONGDOUBLE, (void *) hmm->probs->lalpha);
    PyObject *wrap_lbeta  = PyArray_SimpleNewFromData (PyCh_DATA, dims_data,
                                NPY_LONGDOUBLE, (void *) hmm->probs->lbeta);
    PyObject *wrap_lcxpt  = PyArray_SimpleNewFromData (PyCh_DATA, dims_data,
                                NPY_LONGDOUBLE, (void *) hmm->probs->lcxpt);

    out->err = 0;
    out->n_iter = hmm->n_iter;
    out->llk = (double) hmm->llh;
    out->aic = (double) hmm->aic;
    out->bic = (double) hmm->bic;

    for (size_t i = 0; i < hmm->m_states; i++)
    {
        *lambda_out++ = *lambda_est++;
        *delta_out++  = expl (*delta_est++);
        for (size_t j = 0; j < hmm->m_states; j++)
        {
            *gamma_out++ = expl (*gamma_est++);
        }
    }

    PyArray_CopyInto ((PyArrayObject *) out->lalpha, (PyArrayObject *) wrap_lalpha);
    PyArray_CopyInto ((PyArrayObject *) out->lbeta,  (PyArrayObject *) wrap_lbeta);
    PyArray_CopyInto ((PyArrayObject *) out->lcxpt,  (PyArrayObject *) wrap_lcxpt);
}


static PyObject *
poishmm_fit (PyObject *self, PyObject *args)
{
    UNUSED (self);
    PyObject *arg_lambda = NULL;
    PyObject *arg_gamma  = NULL;
    PyObject *arg_delta  = NULL;
    PyObject *arg_inp    = NULL;

    PyArrayObject *arr_lambda = NULL;
    PyArrayObject *arr_gamma  = NULL;
    PyArrayObject *arr_delta  = NULL;
    PyArrayObject *arr_inp    = NULL;


    DataSet inp = { NULL, 0, false };
    PoisHmm hmm = { true, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, NULL, NULL, NULL };
    PoisParams *init = NULL;
    PoisParams *working = NULL;
    PoisProbs *probs = NULL;
    PyCh_PoisHmm *out = NULL;

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

    if ((arr_lambda == NULL) || (arr_gamma == NULL) || (arr_delta == NULL) || (arr_inp == NULL))
    {
        const char msg[] = "poishmm.fit: Could not convert input arrays.";
        PyErr_SetString (PyExc_MemoryError, msg);
        goto fail;
    }

    init = PoisParams_New (hmm.m_states);
    working = PoisParams_New (hmm.m_states);
    probs = PoisProbs_New (hmm.n_obs, hmm.m_states);

    if ((init == NULL) || (working == NULL) || (probs == NULL))
    {
        const char msg[] = "poishmm.fit: Could not create PoisHmm members.";
        PyErr_SetString (PyExc_MemoryError, msg);
        goto fail;
    }

    PoisParams_SetLambda (init, PyArray_DATA (arr_lambda));
    PoisParams_SetGamma  (init, PyArray_DATA (arr_gamma));
    PoisParams_SetDelta  (init, PyArray_DATA (arr_delta));
    PoisParams_CopyLog   (init, working);

    hmm.init = init;
    hmm.params = working;
    hmm.probs = probs;
    inp.size = PyArray_SIZE (arr_inp);
    inp.data = PyArray_DATA (arr_inp);

    PoisHmm_EstimateParams (&hmm, &inp);
    if (hmm.err)
    {
        PyErr_WarnEx (PyExc_Warning, "No convergence.", 1);
    }

    out = (PyCh_PoisHmm *) PyCh_PoisHmm_New (&PyCh_PoisHmm_Type, NULL, NULL);
    PyCh_PoisHmm_CInit (out, hmm.n_obs, hmm.m_states);
    PyCh_PoisHmm_Set (out, &hmm);

    PoisParams_Delete (init);
    PoisParams_Delete (working);
    PoisProbs_Delete (probs);
    Py_XDECREF (arr_lambda);
    Py_XDECREF (arr_gamma);
    Py_XDECREF (arr_delta);
    Py_XDECREF (arr_inp);
    Py_INCREF (out);
    return (PyObject *) out;

fail:
    PoisParams_Delete (init);
    PoisParams_Delete (working);
    PoisProbs_Delete (probs);
    Py_XDECREF (arr_lambda);
    Py_XDECREF (arr_gamma);
    Py_XDECREF (arr_delta);
    Py_XDECREF (arr_inp);
    Py_RETURN_NONE;
}


static PyObject *
read_params (PyObject *self, PyObject *args)
{
    UNUSED (self);
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
    const npy_intp dims_vector[] = { (npy_intp) params->m_states };
    const npy_intp dims_matrix[] = {
        (npy_intp) params->m_states,
        (npy_intp) params->m_states
    };

    out_states = PyLong_FromUnsignedLong (params->m_states);
    arr_lambda = PyArray_SimpleNewFromData (PyCh_VECTOR, dims_vector,
                        NPY_LONGDOUBLE, (void *) params->lambda);
    arr_delta  = PyArray_SimpleNewFromData (PyCh_VECTOR, dims_vector,
                        NPY_LONGDOUBLE, (void *) params->delta);
    arr_gamma  = PyArray_SimpleNewFromData (PyCh_MATRIX, dims_matrix,
                        NPY_LONGDOUBLE, (void *) params->gamma);
    out_lambda = PyArray_SimpleNew (PyCh_VECTOR, dims_vector, NPY_LONGDOUBLE);
    out_delta  = PyArray_SimpleNew (PyCh_VECTOR, dims_vector, NPY_LONGDOUBLE);
    out_gamma  = PyArray_SimpleNew (PyCh_MATRIX, dims_matrix, NPY_LONGDOUBLE);

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


static PyObject *
global_decoding_impl (PyObject *self, PyObject *args)
{
    UNUSED (self);

    npy_intp n_obs       = 0;
    npy_intp m_states    = 0;
    PyObject *arg_lgamma = NULL;
    PyObject *arg_ldelta = NULL;
    PyObject *arg_lcxpt  = NULL;
    PyObject *arr_states = NULL;

    PyObject *arr_lgamma = NULL;
    PyObject *arr_ldelta = NULL;
    PyObject *arr_lcxpt  = NULL;

    if (!PyArg_ParseTuple (args, "llOOO", &n_obs, &m_states, &arg_lgamma,
                            &arg_ldelta, &arg_lcxpt))
    {
        PyErr_SetString (PyExc_TypeError, "global_decoding: Could not parse args.");
        return NULL;
    }

    const npy_intp dims_vector[] = { m_states };
    const npy_intp dims_matrix[] = { m_states, m_states };
    const npy_intp dims_data[]   = { n_obs, m_states };

    arr_lgamma = PyArray_SimpleNew (PyCh_MATRIX, dims_matrix, NPY_LONGDOUBLE);
    if (arr_lgamma == NULL)
    {

        PyErr_SetString (PyExc_MemoryError,
                "poishmm.global_decoding: Could not allocate lgamma.");
        return NULL;
    }

    arr_ldelta = PyArray_SimpleNew (PyCh_VECTOR, dims_vector, NPY_LONGDOUBLE);
    if (arr_ldelta == NULL)
    {
        PyErr_SetString (PyExc_MemoryError,
                "poishmm.global_decoding: Could not allocate ldelta.");
        return NULL;
    }

    arr_lcxpt = PyArray_SimpleNew (PyCh_DATA, dims_data, NPY_LONGDOUBLE);
    if (arr_lcxpt == NULL)
    {
        PyErr_SetString (PyExc_MemoryError,
                "poishmm.global_decoding: Could not allocate lcxpt copy.");
        return NULL;
    }

    arr_states = PyArray_SimpleNew (PyCh_VECTOR, &n_obs, NPY_ULONG);
    if (arr_states == NULL)
    {
        PyErr_SetString (PyExc_TypeError, "global_decoding: Could not allocate states object.");
        return NULL;
    }

    PyArray_CopyInto ((PyArrayObject *) arr_lgamma, (PyArrayObject *) arg_lgamma);
    PyArray_CopyInto ((PyArrayObject *) arr_ldelta, (PyArrayObject *) arg_ldelta);
    PyArray_CopyInto ((PyArrayObject *) arr_lcxpt,  (PyArrayObject *) arg_lcxpt);

    global_decoding ((size_t) n_obs, (size_t) m_states,
            (long double *)((PyArrayObject *) arr_lgamma)->data,
            (long double *)((PyArrayObject *) arr_ldelta)->data,
            (long double *)((PyArrayObject *) arr_lcxpt)->data,
            (size_t *)((PyArrayObject *) arr_states)->data);

    Py_DECREF (arr_lgamma);
    Py_DECREF (arr_ldelta);
    Py_DECREF (arr_lcxpt);
    Py_INCREF (arr_states);
    return arr_states;
}


static PyObject *
local_decoding_impl (PyObject *self, PyObject *args)
{
    UNUSED (self);

    npy_intp n_obs       = 0;
    npy_intp m_states    = 0;
    PyObject *arg_lcxpt   = NULL;
    PyObject *arr_states = NULL;
    PyObject *arr_lcxpt   = NULL;

    if (!PyArg_ParseTuple (args, "llO", &n_obs, &m_states, &arg_lcxpt))
    {
        PyErr_SetString (PyExc_TypeError, "local_decoding: Could not parse args.");
        return NULL;
    }

    const npy_intp dims_vector[] = { m_states };
    const npy_intp dims_data[]   = { n_obs, m_states };

    arr_lcxpt = PyArray_SimpleNew (PyCh_DATA, dims_data, NPY_LONGDOUBLE);
    if (arr_lcxpt == NULL)
    {
        PyErr_SetString (PyExc_TypeError, "local_decoding: Could not allocate lsdp copy.");
        return NULL;
    }
    PyArray_CopyInto ((PyArrayObject *) arr_lcxpt, (PyArrayObject *) arg_lcxpt);

    arr_states = PyArray_SimpleNew (PyCh_VECTOR, dims_vector, NPY_ULONG);
    if (arr_states == NULL)
    {
        PyErr_SetString (PyExc_TypeError, "local_decoding: Could not allocate states object.");
        return NULL;
    }


    local_decoding ((size_t) n_obs, (size_t) m_states,
            (long double *)((PyArrayObject *) arr_lcxpt)->data,
            (size_t *)((PyArrayObject *) arr_states)->data);

    Py_INCREF (arr_states);
    return arr_states;
}


static PyMethodDef
poishmm_methods[] = {
    {"fit", poishmm_fit, METH_VARARGS, poishmm_fit_doc},
    {"read_params", read_params, METH_VARARGS, read_params_doc},
    {"global_decoding", global_decoding_impl, METH_VARARGS, global_decoding_doc},
    {"local_decoding", local_decoding_impl, METH_VARARGS, global_decoding_doc},
    {NULL, NULL, 0, NULL}
};


static struct PyModuleDef
poishmm_module = {
    PyModuleDef_HEAD_INIT,
    .m_name    = "poishmm",
    .m_doc     = "HMM with Poisson-distributed latent variables.",
    .m_size    = -1,
    .m_methods = poishmm_methods
};


PyMODINIT_FUNC
PyInit_poishmm (void)
{
    int err = 0;
    PyObject *module = NULL;

    import_array ();

    module = PyModule_Create (&poishmm_module);
    if (module == NULL) return NULL;

    err = PyType_Ready (&PyCh_PoisHmm_Type);
    if (err < 0) return NULL;

    Py_INCREF (&PyCh_PoisHmm_Type);
    err = PyModule_AddObject (module, "PoisHmm", (PyObject *) &PyCh_PoisHmm_Type);
    if (err < 0)
    {
        Py_DECREF (&PyCh_PoisHmm_Type);
        Py_DECREF (module);
        return NULL;
    }
    return module;
}
