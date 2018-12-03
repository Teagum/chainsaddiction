#ifndef HMM_H
#define HMM_H

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "hmm.h"
#include "scalar.h"


typedef struct {
	size_t	m;

	scalar	*init_lambda;
	scalar	*init_gamma;
	scalar	*init_delta;

	size_t	n_iter;
	size_t	max_iter;
	scalar	tol;

	scalar	*lambda_;
	scalar	*gamma_;
	scalar	*delta_;	

	scalar	aic;	
	scalar	bic;
	scalar	nll;
} PoissonHMM;

PoissonHMM*
NewPoissonHMM (size_t  m,
			   scalar *init_lambda,
			   scalar *init_gamma,
			   scalar *init_delta,
			   size_t max_iter,
			   scalar tol);

void
DeletePoissonHMM (PoissonHMM *phmm);

scalar
compute_aic(scalar nll, size_t m, size_t n);

scalar
compute_bic(scalar nll, size_t m, size_t n);

#endif	/* HMM_H */
