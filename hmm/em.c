#include <string.h>
#include "em.h"




int poisson_expectation_maximization(
		const	long	*x,
		const	size_t	n,
		const	size_t	m,
		const	size_t	max_iter,
		const	scalar	tol,
		const	scalar	*init_lambda,
		const 	scalar	*init_gamma,
		const	scalar	*init_delta,
				scalar	*lambda_,
				scalar	*gamma_,
				scalar	*delta_,
				scalar	*llk,
				size_t	*n_iter)
{
	/* use these arrays to construct the return struct */
	size_t v_size = m * sizeof(scalar);
	size_t m_size = m * v_size;

	/* implement proper exception handling */
	scalar	*next_lambda = malloc( v_size );
	if (next_lambda == NULL) return 0;
	
	scalar	*next_gamma	= malloc( m_size );
	if (next_gamma == NULL) return 0;
	
	scalar	*next_delta	= malloc( v_size );
	if (next_delta == NULL) return 0;

	memcpy(lambda_, init_lambda, v_size);
	memcpy(gamma_, init_gamma, m_size);
	memcpy(delta_, init_delta, v_size);

	/* TODO: check abort on failed alloc */
	scalar	**alpha	=	alloc_matrix(n, m);
	scalar	**beta	=	alloc_matrix(n, m);
	scalar	**pprob	=	alloc_matrix(n, m);

	int		fwbw		= 0;
	scalar	acc			= 0;
	scalar 	bcc			= 0;
	scalar	c			= 0;
	scalar  crit		= 0;
	scalar 	rs_gamma 	= 0;
	scalar 	rs_delta 	= 0;
	scalar 	s			= 0;

	for (*n_iter = 0; *n_iter < max_iter; (*n_iter)++)
	{	
		/* E Step */
		fwbw = log_poisson_forward_backward(
							x, n, m, 
							lambda_, gamma_, delta_,
							alpha, beta, pprob);
		if (fwbw == 0)
		{
			fprintf(stderr, "Forward/Backward algorithm failed \
							 (n_iter = %zu).\n", *n_iter);
			return 0;
		}

		c = alpha[n-1][0];
		for (size_t i = 1; i < m; i++)
		{
			if ( alpha[n-1][i] > c )
			{
				c = alpha[n-1][i];
			}
		}

		*llk = 0;
		for (size_t i = 0; i < m; i++)
		{
			*llk += expl( alpha[n-1][i] - c );
		}
		*llk = logl(*llk) + c;
		
		/* M Step */
		crit 		= 0;
		rs_delta 	= 0;
		for (size_t i = 0; i < m; i++)
		{	
			/* Lambda */
			acc = 0;
			bcc = 0;
			for (size_t j = 0; j < n; j++)
			{
				s = expl( alpha[j][i] + beta[j][i] - (*llk)	); 
				bcc += s;
				acc += s * x[j];
			}
			next_lambda[i] = acc / bcc;
			crit += fabsl( next_lambda[i] - lambda_[i] );

			/* Gamma */
			rs_gamma = 0;
			for (size_t j = 0; j < m; j++)
			{
				acc = 0;
				for (size_t k = 0; k < (n-1); k++)
				{
					acc +=  expl(	alpha[k][i] 
								  +	beta[k+1][j] 
								  + logl( pprob[k+1][j] )
								  - (*llk) ); 
				}
				next_gamma[i*m+j] = gamma_[i*m+j] * acc;
				rs_gamma += next_gamma[i*m+j];
			}

			for (size_t j = 0; j < m; j++)
			{
				next_gamma[i*m+j] /= rs_gamma;
				crit += fabsl( next_gamma[i*m+j] - gamma_[i*m+j] );
			}

			/* Delta */
			next_delta[i] = expl( alpha[0][i] + beta[0][i] - *llk );
			rs_delta += next_delta[i];
		}

		for (size_t i = 0; i < m; i++)
		{
			next_delta[i] /= rs_delta;
			crit += fabsl( next_delta[i] - delta_[i] ); 
		}

		if (crit >= tol)		/* no convergence yet -> copy and reiterate */
		{
			memcpy( lambda_, next_lambda, v_size );
			memcpy( gamma_,  next_gamma,  m_size );
			memcpy( delta_,  next_delta,  v_size );
		}
		else	/* convergence */
		{
			free(next_lambda);
			free(next_gamma);
			free(next_delta);

			free_matrix(alpha, n);
			free_matrix(beta, n);
			free_matrix(pprob, n);

			return 1;
		}
	}

	/* No convergence after max_iter*/
	free(next_lambda);
	free(next_gamma);
	free(next_delta);

	free_matrix(alpha, n);
	free_matrix(beta, n);
	free_matrix(pprob, n);

	return 0;	
}
