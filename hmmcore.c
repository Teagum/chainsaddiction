#include "hmmcore.h"


/* 
 * Compute the log-forward/backward probabilitiesi
 */
Lfbp *lfwbw(const Scalar *x, const size_t n, const size_t m,
		const Scalar *_lambda, Scalar **_gamma, const Scalar *_delta)
{
	
	Scalar sum_buff = 0;		/* sum prob */
	Scalar lsf = 0;			/* log scale factor */

	Scalar *memory_guard[3] = { NULL, NULL, NULL };

	Scalar *prob = malloc( sizeof(Scalar) * m );	 /* probabilities at t */
	if (prob != NULL) memory_guard[0] = prob;
       	else goto fail;

	Scalar *buff = malloc( m * sizeof(Scalar) );     /* calculation buffer */
	if (buff != NULL) memory_guard[1] = buff;
       	else goto fail;

	Scalar *eggs = malloc( m * sizeof(Scalar) );     /* calculation buffer */
	if (eggs != NULL) memory_guard[2] = eggs; 
	else goto fail;

	Matrix *alpha = NewEmptyMatrix(n, m);			
	Matrix *beta  = NewMatrix(n, m, 0L);
	Matrix *poisson_probs = NewEmptyMatrix(n, m);

	/*
	 * Forward 
	 */

	/* Initial step t = 0*/
	for (size_t j = 0; j < m; j++)
	{
		poisson_probs->data[0][j] = poisson_pmf(_lambda[j], x[0]);
		prob[j] = poisson_probs->data[0][j] * _delta[j];
		sum_buff += prob[j];
	}
	lsf = log(sum_buff);

	for (size_t j = 0; j < m; j++)
	{
		prob[j] /= sum_buff;
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

	free(prob);
	free(buff);
	free(eggs);

	Lfbp *ab = malloc( sizeof(Lfbp*) );
	ab->alpha = alpha;
	ab->beta = beta;
	ab->prob = poisson_probs;

	return ab;

fail:
	for (size_t i = 0; i < 3; i++)
	{
		if (memory_guard[0] != NULL)
		{
			free(memory_guard[0]);
		}
	}
	return NULL;
}	


HmmParams *EM(const Scalar *x, const size_t n, const size_t m,
	      	const size_t max_iter, const Scalar tol,	
		const Scalar *__lambda, Scalar **__gamma, const Scalar *__delta)
{

	Vector *this_lambda = NewEmptyVector(m);
	Matrix *this_gamma  = NewEmptyMatrix(m, m);
	Vector *this_delta  = NewEmptyVector(m);

	Scalar *_lambda = v_dptr(this_lambda);
	Scalar **_gamma = m_dptr(this_gamma);
	Scalar *_delta  = v_dptr(this_delta);

	for (size_t i = 0; i < m; i++)
	{
		_lambda[i] = __lambda[i];
		_delta[i] = __delta[i];
		for (size_t j = 0; j < m; j++)
		{
			_gamma[i][j] = __gamma[i][j];
		}
	}

	Vector *new_lambda = NewEmptyVector(m);
	Matrix *new_gamma  = NewEmptyMatrix(m, m);
	Vector *new_delta  = NewEmptyVector(m);

	Scalar *lambda_ = v_dptr(new_lambda);
	Scalar **gamma_ = m_dptr(new_gamma);
	Scalar *delta_  = v_dptr(new_delta);

	Lfbp 	*lab	= NULL;
	Scalar  **alpha	= NULL;
	Scalar  **beta	= NULL;
	Scalar  **probs	= NULL;
	Scalar	acc	= 0;
	Scalar 	bcc	= 0;
	Scalar	c	= 0;
	Scalar  crit	= 0;
	Scalar 	llk	= 0;
	Scalar 	rs_gamma = 0;
	Scalar 	rs_delta = 0;

	for (size_t n_iter = 0; n_iter < max_iter; n_iter++)
	{	
		printf("Iter: %zu\n", n_iter);
		/* E Step */
		lab 	= lfwbw(x, n, m, _lambda, _gamma, _delta);
		alpha 	= lab->alpha->data;
		beta 	= lab->beta->data;
		probs 	= lab->prob->data;

		c = alpha[n-1][0];
		for (size_t i = 1; i < m; i++)
			c = ( (alpha[n-1][0]) > c )  ?  alpha[n-1][0]  :  c;

		llk = 0;
		for (size_t i = 0; i < m; i++)
			llk += expl(alpha[n-1][i] - c);
		llk = logl(llk) + c;

		/* M Step */
		crit = 0;
		for (size_t i = 0; i < m; i++)
		{	
			/* Lambda */
			acc = 0;
			bcc = 0;
			for (size_t j = 0; j < n; j++)
			{
				acc += alpha[j][i] + beta[j][i] - llk;
				bcc += acc * x[j];
			}
			lambda_[i] = bcc / acc;
			crit += fabsl(_lambda[i] - lambda_[i]);

			/* Gamma */
			rs_gamma = 0;
			for (size_t j = 0; j < m; j++)
			{
				acc = 0;
				for (size_t k = 0; k < (n-1); k++)
				{
					acc += expl(alpha[k][i] + beta[k+1][j] + logl(probs[k+1][j]) + llk);
				}
				gamma_[i][j] = _gamma[i][j] * acc;
				rs_gamma += gamma_[i][j];
			}	

			/* Delta && normalize Gamma */		
			rs_delta = 0;
			for (size_t j = 0; j < m; j++)
			{
				gamma_[i][j] /= rs_gamma;
				delta_[i] = expl(alpha[0][j] + beta[0][j] - llk);
				rs_delta += delta_[i];

				crit += fabsl(_gamma[i][j] - gamma_[i][j]);
			}
			delta_[i] /= rs_delta;
			crit += fabsl(_delta[i] - delta_[i]); 
		}

		if (crit < tol)		/* on convergence */
		{
			v_free(new_lambda);
			m_free(new_gamma);
			v_free(new_delta);

			HmmParams *theta = malloc( sizeof(HmmParams) );
		       	theta->lambda_ = this_lambda;
			theta->gamma_ = this_gamma;
			theta->delta_ = this_delta;

			return theta;
		}
		else	/* copy data and re-iterate */
		{
			for (size_t i = 0; i < m; i++)
			{
				_lambda[i] = lambda_[i];
				_delta[i] = delta_[i];
				for (size_t j = 0; j < m; j++)
				{
					_gamma[i][j] = gamma_[i][j];
				}
			}
		}
	}	

	/* No convergence after max_iter*/
	v_free(this_lambda);
	m_free(this_gamma);
	v_free(this_delta);

	v_free(new_lambda);
	m_free(new_gamma);
	v_free(new_delta);

	return NULL;	
}
















