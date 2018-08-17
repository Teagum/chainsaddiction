#ifndef STATS_H
#define STATS_H


typedef long double scalar;


scalar poisson_pmf(scalar lambda, long x);
scalar poisson_log_pmf(scalar lambda, long x);

#endif    /* STATS_H */
