#include "fwbw.h"


static inline void cleanUp()
{
	for (size_t i = 0; i < 3; i++)
	{
		if (memory_guard[0] != NULL)
		{
			free(memory_guard[0]);
		}
	}
}


int log_poisson_forward_backward(
		const long	 *x,
		const size_t n,
		const size_t m,
		const scalar *lambda_, 
		const scalar *gamma_,
		const scalar *delta_,
		scalar **alpha,
		scalar **beta,
		scalar **pprob)
{
	
	scalar	sum_buff	= 0;		/* sum prob */
	scalar	lsf			= 0;		/* log scale factor */

	scalar *_pxt = malloc( m * sizeof(scalar) );	 /* probabilities at time `t` */
	if (_pxt != NULL) memory_guard[0] = _pxt;
    else goto fail;

	scalar *_buff = malloc( m * sizeof(scalar) );     /* calculation buffer */
	if (_buff != NULL) memory_guard[1] = _buff;
    else goto fail;

	scalar *_eggs = malloc( m * sizeof(scalar) );     /* calculation buffer */
	if (_eggs != NULL) memory_guard[2] = _eggs; 
	else goto fail;


	/*
	 * Forward pass 
	 */

	/* Initial step t = 0*/
	for (size_t j = 0; j < m; j++)
	{
		pprob[0][j] = poisson_pmf( lambda_[j], x[0] );
		_pxt[j] = pprob[0][j] * delta_[j];
		sum_buff += _pxt[j];
	}
	lsf = log(sum_buff);

	for (size_t j = 0; j < m; j++)
	{
		_pxt[j] /= sum_buff;
		alpha[0][j] = log( _pxt[j] ) + lsf;
	}

	/* remaining forward steps */
	for (size_t i = 1; i < n; i++)
	{
		sum_buff = 0;
		for (size_t j = 0; j < m; j++)
		{
			for (size_t k = 0; k < m; k++)
			{
				_buff[j] += _pxt[k] * gamma_[k*m+j];
			}
			pprob[i][j] = poisson_pmf( lambda_[j], x[i] );
			_buff[j] *= pprob[i][j];
			sum_buff += _buff[j];
		}
		lsf += log( sum_buff );

		for (size_t j = 0; j < m; j++)
		{
			_pxt[j] = _buff[j] / sum_buff;
			alpha[i][j] = log( _pxt[j] ) + lsf;
			_buff[j] = 0;
		}
	}

	/*
	 * Backward pass
	 */

	/* Initial step */
	for (size_t j = 0; j < m; j++)
	{
		_pxt[j] = 1L / (scalar) m;
	}
	lsf = log(m);

	/* remaining backward steps */
	for (size_t i = n-1; i > 0; i--)
	{
		for (size_t j = 0; j < m; j++)
		{
			_eggs[j] = pprob[i][j] * _pxt[j];
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

		lsf += log(sum_buff);
		for (size_t j = 0; j < m; j++)
		{
			_pxt[j] = _buff[j] / sum_buff;
			_buff[j] = 0;
			beta[i-1][j] = log( _pxt[j] ) + lsf;
		}
	}

	free(_pxt);
	free(_buff);
	free(_eggs);
	return 1;

fail:
	cleanUp();
	return 0;
}	
