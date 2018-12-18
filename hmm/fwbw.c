#include "fwbw.h"
#include "stdio.h"

int log_poisson_forward_backward(
		const long	 *x,
		const size_t n,
		const size_t m,
		const scalar *lambda_, 
		const scalar *gamma_,
		const scalar *delta_,
		scalar *alpha,
		scalar *beta,
		scalar *pprob)
{
	
	scalar	sum_buff	= 0;		/* sum prob */
	scalar	lsf			= 0;		/* logl scale factor */
	int 	success		= 1;

	scalar *_pxt = NULL;
	scalar *_buff = NULL;
	scalar *_eggs = NULL;

	_pxt = malloc (m * sizeof(*_pxt));		/* probabilities at time `t` */
	if (_pxt == NULL) { success=0; goto fail; } 

	_buff = malloc (m * sizeof(*_buff));	/* calculation buffer */
	if (_buff == NULL) { success=0; goto fail; } 

	_eggs = malloc (m * sizeof(*_eggs));	/* calculation buffer */
	if (_eggs == NULL) { success=0; goto fail; }


	/*
	 * Forward pass 
	 */

	/* Initial step t = 0 */
	for (size_t j = 0; j < m; j++)
	{
		pprob[j] = poisson_pmf (lambda_[j], x[0]);
		_pxt[j] = pprob[j] * delta_[j];
		sum_buff += _pxt[j];
	}
	lsf = logl (sum_buff);
	
	for (size_t j = 0; j < m; j++)
	{
		_pxt[j] /= sum_buff;
		alpha[j] = logl (_pxt[j]) + lsf;
	}
	
	/* remaining forward steps */
	for (size_t i = 1; i < n; i++)
	{
		sum_buff = 0;
		for (size_t j = 0; j < m; j++)
		{
			_buff[j] = 0;
			for (size_t k = 0; k < m; k++)
			{
				_buff[j] += _pxt[k] * gamma_[k*m+j];
			}
			pprob[i*m+j] = poisson_pmf (lambda_[j], x[i]);
			_buff[j] *= pprob[i*m+j];
			sum_buff += _buff[j];
		}
		lsf += logl (sum_buff);

		for (size_t j = 0; j < m; j++)
		{
			_pxt[j] = _buff[j] / sum_buff;
			alpha[i*m+j] = logl (_pxt[j]) + lsf;
		}
	}

	/*
	 * Backward pass
	 */

	/* Initial step */
	for (size_t j = 0; j < m; j++)
	{
		_pxt[j] = 1.0L / (scalar) m;
        beta[(n-1)*m+j] = 0.0L;
	}
	lsf = logl(m);

	/* remaining backward steps */
	for (size_t i = n-1; i > 0; i--)
	{
		for (size_t j = 0; j < m; j++)
		{
			_eggs[j] = pprob[i*m+j] * _pxt[j];
		}
		
		for (size_t j = 0; j < m; j++)
		{
			sum_buff = 0;
			for (size_t k = 0; k < m; k++)
			{
				_buff[j] += gamma_[j*m+k] * _eggs[k];
			}
			sum_buff += _buff[j];
		}

		lsf += logl (sum_buff);
		for (size_t j = 0; j < m; j++)
		{
			_pxt[j] = _buff[j] / sum_buff;
			_buff[j] = 0;
			beta[(i-1)*m+j] = logl (_pxt[j]) + lsf;
		}
	}

fail:
	free(_pxt);
	free(_buff);
	free(_eggs);
	return success;
}	

