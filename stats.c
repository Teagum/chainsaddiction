#include <math.h>
#include "stats.h"
#include "linalg.h"


Scalar poisson_log_pmf(Scalar lambda, long x)
{
    return (Scalar) x * logl(lambda) - lgamma((Scalar) x + 1) - lambda;
}


Scalar poisson_pmf(Scalar lambda, long x)
{
    return expl(poisson_log_pmf(lambda, x));
}


Vector *ppmf(Vector *lambda, long x)
{
    Vector *out = NewEmptyVector(lambda->n);
    for (size_t i = 0; i < lambda->n; i++)
    {
        out->data[i] = poisson_pmf(lambda->data[i], x);
    }
    return out;
}
