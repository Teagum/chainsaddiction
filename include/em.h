#ifndef EM_H
#define EM_H

#include <math.h>
#include "fwbw.h"
#include "matrix.h"

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
				size_t	*n_iter);

#endif	/* EM_H */
