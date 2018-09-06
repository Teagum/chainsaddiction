#ifndef FWBW_H
#define FWBW_H

#include <math.h>
#include "matrix.h"
#include "stats.h"


static scalar *memory_guard[3] = { NULL, NULL, NULL };


static inline void cleanUp();

int log_poisson_forward_backward(
		const long	 *x,
		const size_t n,
		const size_t m,
		const scalar *lambda_, 
		const scalar *gamma_,
		const scalar *delta_,
		scalar **alpha,
		scalar **beta,
		scalar **pprob);

#endif	/* FWBW_H */
