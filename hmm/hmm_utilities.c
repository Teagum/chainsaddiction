#define NPY_NO_DEPRECATED_API NPY_1_8_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL hmm_NP_ARRAY_API


#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdbool.h>
#include <math.h>


#define value1(arr, i)         ( *(double *) PyArray_GETPTR1(arr, i) )  
#define value2(arr, i, j)      ( *(double *) PyArray_GETPTR2(arr, i, j) )
#define value3(arr, i, j, k)   ( *(double *) PyArray_GETPTR3(arr, i, j, k) )  


double poisson_pmf(long x, double k)
{
	return exp(x * log(k) - lgamma(x+1) - k);
}


PyArrayObject *ppmf(PyArrayObject *x, PyArrayObject *k)
{
	npy_intp dims[] = { PyArray_SIZE(x), PyArray_SIZE(k) };
	PyArrayObject *probs = (PyArrayObject *) PyArray_SimpleNew(2, dims, NPY_DOUBLE);

	for (npy_intp i = 0; i < dims[0]; i++)
	{
		for (npy_intp j = 0; j < dims[1]; j++)
		{
			*(double *) PyArray_GETPTR2(probs, i, j) = 
					poisson_pmf( *(long *)PyArray_GETPTR1(x, i),
						         *(double *)PyArray_GETPTR1(k, j));
		}
	}
	Py_INCREF(probs);
	return probs;
}


static PyObject *
poisson_fwbw(PyObject* self, PyObject* args) {
	/* Compute the forward/backward probabilites of a Poisson HMM.
	 *
	 * Params:
	 * 		x		observations
	 * 		m		number of stats
	 * 		lambda	poission distribution parameters
	 * 		gamma 	transition prob matrix
	 * 		delta 	initial distribution
	 * 	Return 2d array
	 */

	// Local params
    PyArrayObject* x = NULL;
	PyArrayObject* lambda_ = NULL;
	PyArrayObject* gamma_ = NULL;
	PyArrayObject* delta_ = NULL;
	npy_intp m = 0;

    if (!PyArg_ParseTuple(args, "OiOOO", &x, &m, &lambda_, &gamma_, &delta_))
   	{
        /* Just return NULL here since PyArg_ParseTuple already raises an exception on failure */
		return NULL;
    }

	npy_intp n = PyArray_SIZE(x);
	npy_intp ab_dims[] = { 2, n, m };
	PyArrayObject *ab = (PyArrayObject *) PyArray_ZEROS(3, ab_dims, NPY_DOUBLE, 0); 
	PyArrayObject *probs = ppmf(x, lambda);

	double row_sum = 0.;
	double a0 = 0.;
	for (npy_intp i = 0; i < m; i++)
	{
		a0 = value1(delta, i) * value2(probs, 0, i);
		value3(ab, 0, 0, i) = a0;
		row_sum += a0;
	}	
	double lscale = log1p(row_sum);
	
	for (npy_intp i = 0; i < m; i++)
	{
		value3(ab, 0, 0, i) /= row_sum;
		value3(ab, 0, 0, i) = log1p(value3(ab, 0, 0, i)) + lscale;
	}

 	/* forward loop */
	PyObject a_t = NULL;
	for (npy_intp i = 1; i < n; i++)
	{
		a_t = PyArray_MatrixProduct(
	}	
	Py_INCREF(ab);
	return (PyObject *) ab;
}

/* Compute pearson correlation coefficient
 * Parama:
 * 		arr: pointer to input array
 * 		off_x
 * 		off_y 
 * 		n: how many values to process
 */

static PyMethodDef
HmmUtilities[] = {
    {"poisson_fwbw", poisson_fwbw, METH_VARARGS, "forward backward"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef
hmm_module = {
    PyModuleDef_HEAD_INIT,
    "hmm",
    NULL,
    -1,
	HmmUtilities
};

PyMODINIT_FUNC
PyInit_hmm(void) {
    import_array();
    return PyModule_Create(&hmm_module);
}

