#ifndef FWBW_H
#define FWBW_H

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include "stats.h"
#include "scalar.h"


int log_poisson_forward_backward(
		const long	 *x,
		const size_t n,
		const size_t m,
		const scalar *lambda_, 
		const scalar *gamma_,
		const scalar *delta_,
		scalar *alpha,
		scalar *beta,
		scalar *pprob);

#endif	/* FWBW_H */
