#ifndef poishmmfit_h
#define poishmmfit_h

#define PY_SSIZE_T_CLEAN
#include <Python.h>


typedef struct _PoisHmmFit {
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


static void
PoisHmmFit_Delete (PoisHmmFit *self);

static PyObject *
PoisHmmFit_New(PyTypeObject *type, PyObject *args, PyObject *kwds);

static int
PoisHmmFit_CInit (PoisHmmFit *self, const size_t m_states);

static void
PoisHmmFit_Set (PoisHmmFit *out, PoisHmm *hmm);


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
    {NULL, -1, 0, 0, NULL}  /* Sentinel */
};


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


#endif  /* poishmmfit_h */
