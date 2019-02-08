#ifndef STATS_H
#define STATS_H

#include <stdlib.h>
#include <math.h>
#include "scalar.h"

#ifdef warn_nan
#include <stdio.h>
#endif

scalar  poisson_pmf     (scalar lambda, long x);
scalar  poisson_log_pmf (scalar lambda, long x);
void    ppmf            (scalar *lambda, size_t m, long x, scalar *out);

#endif    /* STATS_H */
