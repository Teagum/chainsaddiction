#ifndef STATS_H
#define STATS_H

#include <math.h>
#include "matrix.h"

scalar	poisson_pmf		(scalar lambda, long x);
scalar	poisson_log_pmf	(scalar lambda, long x);
void	ppmf			(scalar *lambda, size_t m, long x, scalar *out);


#endif    /* STATS_H */
