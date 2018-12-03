#include <string.h>
#include "em.h"
#include "hmm.h"

int poisson_expectation_maximization(
		const	long		*x,
		const	size_t		n,
				PoissonHMM	*hmm)
{
	scalar	acc			= 0.0L;
	scalar 	bcc			= 0.0L;
	scalar	c			= 0.0L;
	scalar  crit		= 0.0L;
	scalar 	rs_delta 	= 0.0L;
	scalar 	rs_gamma 	= 0.0L;
	scalar 	s			= 0.0L;
	int		success		= 0;
	size_t	m			= hmm->m;	/* just for convenience */
	
	size_t vector_s = m * sizeof(scalar);
	size_t matrix_s	= m * vector_s;
	size_t buffer_s = n * vector_s;

	scalar *alpha 		= NULL;
	scalar *beta  		= NULL;
	scalar *pprob 		= NULL;
	scalar *next_lambda	= NULL;
	scalar *next_gamma	= NULL;
	scalar *next_delta	= NULL;

	alpha = malloc (buffer_s);
	if (alpha == NULL) goto fail;

	beta = malloc (buffer_s);
	if (beta == NULL) goto fail;

	pprob = malloc (buffer_s);
	if (pprob == NULL) goto fail;

	next_lambda	= malloc (vector_s);
	if (next_lambda == NULL) goto fail;
	
	next_gamma = malloc (matrix_s);
	if (next_gamma == NULL) goto fail;

	next_delta = malloc (vector_s);
	if (next_delta == NULL) goto fail;


	for (hmm->n_iter = 0; hmm->n_iter < hmm->max_iter; (hmm->n_iter)++)
	{	
		/* E Step */
		int fwbw_ret = log_poisson_forward_backward(
							x, (size_t) n, (size_t) m, 
							hmm->lambda_, hmm->gamma_, hmm->delta_,
							alpha, beta, pprob);
		if (fwbw_ret == 0)
		{
			fprintf(stderr, "Forward/Backward algorithm failed \
							 (n_iter = %zu).\n", hmm->n_iter);
			goto fail;
		}

		c = alpha[(n-1)*m];
		for (size_t j = 1; j < m; j++)
		{
			if ( alpha[(n-1)*m+j] > c )
			{
				c = alpha[(n-1)*m+j];
			}
		}

		hmm->nll = 0;
		for (size_t j = 0; j < m; j++)
		{
			hmm->nll += expl (alpha[(n-1)*m+j] - c);
		}
		hmm->nll = logl(hmm->nll) + c;
		
		/* M Step */
		crit 		= 0;
		rs_delta 	= 0;
		for (size_t j = 0; j < m; j++)
		{	
			/* Lambda */
			acc = 0;
			bcc = 0;
			for (size_t i = 0; i < n; i++)
			{
				s = expl (alpha[i*m+j] + beta[i*m+j] - (hmm->nll)); 
				bcc += s;
				acc += s * x[i];
			}
			next_lambda[j] = acc / bcc;
			crit += fabsl (next_lambda[j] - hmm->lambda_[j] );

			/* Gamma */
			rs_gamma = 0;
			for (size_t i = 0; i < m; i++)
			{
				acc = 0;
				for (size_t k = 0; k < (n-1); k++)
				{
					acc +=  expl(	alpha[k*m+j] 
								  +	beta[(k+1)*m+i]
								  + logl (pprob[(k+1)*m+i])
								  - hmm->nll); 
				}
				next_gamma[j*m+i] = hmm->gamma_[j*m+i] * acc;
				rs_gamma += next_gamma[j*m+i];
			}

			for (size_t i = 0; i < m; i++)
			{
				next_gamma[j*m+i] /= rs_gamma;
				crit += fabsl (next_gamma[j*m+i] - hmm->gamma_[j*m+i]);
			}

			/* Delta */
			next_delta[j] = expl (alpha[j] + beta[j] - hmm->nll);
			rs_delta += next_delta[j];
		}

		for (size_t j = 0; j < m; j++)
		{
			next_delta[j] /= rs_delta;
			crit += fabsl (next_delta[j] - hmm->delta_[j]); 
		}

		/* no convergence yet -> copy and reiterate */
		if (crit >= hmm->tol)
		{
			memcpy (hmm->lambda_, next_lambda, vector_s);
			memcpy (hmm->gamma_,  next_gamma,  matrix_s);
			memcpy (hmm->delta_,  next_delta,  vector_s);
		}
		else	/* convergence */
		{
			success = 1;
			free (next_lambda);
			free (next_gamma);
			free (next_delta);

			free (alpha);
			free (beta);
			free (pprob);
			
			hmm->aic = compute_aic (hmm->nll, hmm->m, n);
			hmm->bic = compute_bic (hmm->nll, hmm->m, n);
			fprintf(stderr, "-----EM fine----\n");
			return success;
		}
	}

fail:
	/* No convergence after max_iter*/
	fprintf(stderr, "------EM failed----\n");
	free (next_lambda);
	free (next_gamma);
	free (next_delta);

	free (alpha);
	free (beta);
	free (pprob);

	return success;	
}

