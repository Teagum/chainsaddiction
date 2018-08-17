#include <math.h>
#include "stats.h"


scalar poisson_log_pmf(scalar lambda, long x)
{
    return (scalar) x * logl(lambda) - lgamma((scalar) x + 1) - lambda;
}


scalar poisson_pmf(scalar lambda, long x)
{
    return expl(poisson_log_pmf(lambda, x));
}
