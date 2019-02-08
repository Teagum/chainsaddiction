#include "stats.h"


scalar
poisson_log_pmf (scalar lambda, long x)
{
    return (scalar) x * logl (lambda) - lgamma ((scalar) x + 1) - lambda;
}


scalar
poisson_pmf (scalar lambda, long x)
{
    scalar out = expl (poisson_log_pmf (lambda, x));
#ifdef warn_nan
    if (out != out)
    {
        fprintf (stderr, "poisson_pmf produced NaN for input (%Lf, %ld).\n", lambda, x);
    }

    if (isinf (out))
    {
        fprintf (stderr, "poisson_pmf produced infinite value for input (%Lf, %ld).\n", lambda, x);
    }
#endif
    return out;
}


void
ppmf (scalar *lambda, size_t m,  long x, scalar *out)
{
    for (size_t i = 0; i < m; i++)
    {
        out[i] = poisson_pmf (lambda[i], x);
    }
}
