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
	PyArrayObject	*x		= NULL;
	PyArrayObject	*_lambda= NULL;
	PyArrayObject	*_gamma	= NULL;
	PyArrayObject	*_delta	= NULL;
	npy_intp		max_iter= 0;
	double			tol		= 0;

	if (!PyArg_ParseTuple(args, "OOOOld", &x, &_lambda, &_gamma,
		&_delta, &max_iter, &tol)) return NULL;

	npy_intp n = PyArray_SIZE(x);
	npy_intp m = PyArray_SIZE(_lambda);

	PoissonHMM *hmm	= NewPoissonHMM( m,
						PyArray_DATA(_lambda),
						PyArray_DATA(_gamma),
						PyArray_DATA(_delta),
						(size_t) max_iter, (scalar) tol);

	int success = poisson_expectation_maximization(PyArray_DATA(x),
												   (size_t) n, hmm);

	if (success == 1)
	{
		npy_intp	dims_1d[]	= { m };
		npy_intp	dims_2d[]	= { m, m };
		size_t		v_size		= (size_t) m * sizeof(scalar);
		size_t		m_size		= (size_t) m * v_size;

		PyArrayObject *lambda_	= Apollon_NewPyArray1d(dims_1d);
		PyArrayObject *gamma_	= Apollon_NewPyArray2d(dims_2d);
		PyArrayObject *delta_	= Apollon_NewPyArray1d(dims_1d);

		scalar *data = PyArray_DATA(lambda_);
		memcpy (PyArray_DATA(lambda_), hmm->lambda_, v_size);	
		memcpy (PyArray_DATA(gamma_), hmm->gamma_,  m_size);	
		memcpy (PyArray_DATA(delta_), hmm->delta_, v_size);

		for (npy_intp i = 0; i < m; i++)
			fprintf(stdout, "%Lf\t", data[i]);

		hmm->aic = compute_aic(hmm->nll, m,  n);
		hmm->bic = compute_bic(hmm->nll, m,  n);

		PyObject *out = PyTuple_Pack(7, lambda_, gamma_, delta_, 
						   PyFloat_FromDouble(hmm->aic), PyFloat_FromDouble(hmm->bic),
						   PyFloat_FromDouble(hmm->nll), PyLong_FromSize_t(hmm->n_iter));

		Py_INCREF(out);
		DeletePoissonHMM(hmm);
		return out;
	}
	else
	{
		DeletePoissonHMM(hmm);
		PyErr_SetString(PyExc_ValueError, "Training failed.");
		Py_RETURN_NONE;
	}
}


static PyObject *
hmm_poisson_fwbw(PyObject* self, PyObject* args)
{
	PyArrayObject	*x		= NULL;
	PyArrayObject	*lambda_= NULL;
	PyArrayObject	*gamma_	= NULL;
	PyArrayObject	*delta_	= NULL;

	if (!PyArg_ParseTuple(args, "OOOO", &x, &lambda_, &gamma_, &delta_))
		return NULL;

	npy_intp 	n 		= PyArray_SIZE(x);
	npy_intp	m 		= PyArray_SIZE(lambda_);
	npy_intp	dims[]	= { n, m };

	PyArrayObject *alpha	= Apollon_NewPyArray2d(dims);
	PyArrayObject *beta		= Apollon_NewPyArray2d(dims);
	PyArrayObject *lp_prob	= Apollon_NewPyArray2d(dims);
							  
	int success = log_poisson_forward_backward (PyArray_DATA(x),
												(size_t) n, (size_t) m,
						    					PyArray_DATA(lambda_),
												PyArray_DATA(gamma_),
												PyArray_DATA(delta_),
						    					PyArray_DATA(alpha),
												PyArray_DATA(beta),
												PyArray_DATA(lp_prob));

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

