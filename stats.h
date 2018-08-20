#ifndef STATS_H
#define STATS_H

#include "linalg.h"


Scalar poisson_pmf(Scalar lambda, long x);
Scalar poisson_log_pmf(Scalar lambda, long x);
Vector *ppmf(Vector *lambda, long x);


#endif    /* STATS_H */
