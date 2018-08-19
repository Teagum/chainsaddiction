#include <math.h>
#include "stats.h"


Scalar poisson_log_pmf(Scalar lambda, long x)
{
    return (Scalar) x * logl(lambda) - lgamma((Scalar) x + 1) - lambda;
}


Scalar poisson_pmf(scalar lambda, long x)
{
    return expl(poisson_log_pmf(lambda, x));
}
