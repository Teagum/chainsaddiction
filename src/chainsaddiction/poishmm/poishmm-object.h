#ifndef poishmmfit_h
#define poishmmfit_h

#define PY_SSIZE_T_CLEAN
#include <Python.h>


typedef struct _PyCh_PoisHmm {
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
    PyObject *lalpha;
    PyObject *lbeta;
    PyObject *lcsp;
} PyCh_PoisHmm;


static void
PyCh_PoisHmm_Delete (PyCh_PoisHmm *self);

static PyObject *
PyCh_PoisHmm_New(PyTypeObject *type, PyObject *args, PyObject *kwds);

static int
PyCh_PoisHmm_CInit (PyCh_PoisHmm *self, const size_t n_states, const size_t m_states);

static void
PyCh_PoisHmm_Set (PyCh_PoisHmm *out, PoisHmm *hmm);


static PyMemberDef PyCh_PoisHmm_members[] = {
    {"err",         T_INT,      offsetof (PyCh_PoisHmm, err),       0, "Error number"},
    {"n_iter",      T_ULONG,    offsetof (PyCh_PoisHmm, n_iter),    0, "Number of iterations"},
    {"llk",         T_DOUBLE,   offsetof (PyCh_PoisHmm, llk),       0, "Log likelihood"},
    {"aic",         T_DOUBLE,   offsetof (PyCh_PoisHmm, aic),       0, "Akaike information criterion"},
    {"bic",         T_DOUBLE,   offsetof (PyCh_PoisHmm, bic),       0, "Bayesian information criterion"},
    {"m_states",    T_ULONG,    offsetof (PyCh_PoisHmm, m_states),  0, "Number of states"},
    {"lambda_",     T_OBJECT,   offsetof (PyCh_PoisHmm, lambda),    0, "State-dependent means"},
    {"gamma_",      T_OBJECT,   offsetof (PyCh_PoisHmm, gamma),     0, "Transition probability matrix"},
    {"delta_",      T_OBJECT,   offsetof (PyCh_PoisHmm, delta),     0, "Initial distribution"},
    {"lalpha",      T_OBJECT,   offsetof (PyCh_PoisHmm, lalpha),    0, "Forward probabilities"},
    {"lbeta",       T_OBJECT,   offsetof (PyCh_PoisHmm, lbeta),     0, "Backward probabilities"},
    {"lcsp",        T_OBJECT,   offsetof (PyCh_PoisHmm, lcsp),      0, "Log of the conditional expectations"},
    {NULL, -1, 0, 0, NULL}  /* Sentinel */
};


static PyTypeObject PyCh_PoisHmm_Type = {
    PyVarObject_HEAD_INIT (NULL, 0)
    .tp_name = "poishmm.PoisHmm",
    .tp_doc = "Hidden Markov Model with Poisson-distributed latent variables.",
    .tp_basicsize = sizeof (PyCh_PoisHmm),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PyCh_PoisHmm_New,
    .tp_dealloc = (destructor) PyCh_PoisHmm_Delete,
    .tp_members = PyCh_PoisHmm_members,
};


#endif  /* poishmmfit_h */
