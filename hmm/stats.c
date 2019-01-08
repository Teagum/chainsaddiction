#include "stats.h"


scalar
poisson_log_pmf (scalar lambda, long x)
{
    return (scalar) x * logl (lambda) - lgamma ((scalar) x + 1) - lambda;
}


scalar
poisson_pmf (scalar lambda, long x)
{
    return expl (poisson_log_pmf (lambda, x));
}


void
ppmf (scalar *lambda, size_t m,  long x, scalar *out)
{
    for (size_t i = 0; i < m; i++)
    {
        out[i] = poisson_pmf (lambda[i], x);
    }
}
