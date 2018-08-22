#include "fwbw.h"


AB_probs *fwbw(Scalar *x, size_t n, size_t m, Scalar *_lambda, Scalar **_gamma, Scalar *_delta)
{
	
	Scalar sum_buff = 0;		/* sum prob */
	Scalar lsf = 0;				/* log scale factor */

	Scalar *prob = malloc( m * sizeof(Scalar) );	 /* probabilities at t */
	Scalar *buff = malloc( m * sizeof(Scalar) );     /* calculation buffer */
	Scalar *eggs = malloc( m * sizeof(Scalar) );     /* calculation buffer */
	Matrix *alpha = NewEmptyMatrix(n, m);			
	Matrix *beta  = NewMatrix(n, m, 0L);
	/*
	 * Forward 
	 */

	/* Initial step t = 0*/
	for (size_t j = 0; j < m; j++)
	{
		prob[j] = poisson_pmf(_lambda[j], x[0]) * _delta[j];
		sum_buff += prob[j];
	}
	lsf = log(sum_buff);

	for (size_t j = 0; j < m; j++)
	{
		prob[j] = prob[j] / sum_buff;
		alpha->data[0][j] = log( prob[j] ) + lsf;
	}
	
	/* remaining forward steps */
	for (size_t i = 1; i < n; i++)
	{
		sum_buff = 0;
		for (size_t j = 0; j < m; j++)
		{
			for (size_t k = 0; k < m; k++)
			{
				buff[j] += prob[k] * _gamma[k][j];
			}
			buff[j] *= poisson_pmf( _lambda[j], x[i] );
			sum_buff += buff[j];
		}
		lsf += log( sum_buff );
		for (size_t j = 0; j < m; j++)
		{
			prob[j] = buff[j] / sum_buff;
			buff[j] = 0;
			alpha->data[i][j] = log( prob[j] ) + lsf;
		}
	}
	
	/*
	 * Backward pass
	 */

	/* Initial step */
	for (size_t j = 0; j < m; j++)
	{
		prob[j] = 1L / (Scalar) m;
	}
	lsf = log(m);

	/* remaining backward steps */
	for (size_t i = n-1; i > 0; i--)
	{
		for (size_t j = 0; j < m; j++)
		{
			Scalar aa =poisson_pmf(_lambda[j], x[i]);  
			eggs[j] = aa * prob[j];
		}
		
		for (size_t j = 0; j < m; j++)
		{
			sum_buff = 0;
			for (size_t k = 0; k < m; k++)
			{
				buff[j] += _gamma[j][k] * eggs[k];
			}
			sum_buff += buff[j];
		}

		lsf += log(sum_buff);
		for (size_t j = 0; j < m; j++)
		{
			prob[j] = buff[j] / sum_buff;
			buff[j] = 0;
			beta->data[i-1][j] = log( prob[j] ) + lsf;
		}
	}

	/* Params theta;
	 * return theta;
	 */
	
	AB_probs *ab = malloc( sizeof(AB_probs) );
	ab->alpha = alpha;
	ab->beta = beta;

	free(prob);
	free(buff);
	free(eggs);

	return ab;
}	
