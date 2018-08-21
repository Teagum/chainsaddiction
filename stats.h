#ifndef STATS_H
#define STATS_H

#include "linalg.h"
#include "stats.h"

Scalar poisson_pmf(Scalar lambda, long x);
Scalar poisson_log_pmf(Scalar lambda, long x);
void ppmf(Vector *lambda, long x, Vector *out);


#endif    /* STATS_H */
