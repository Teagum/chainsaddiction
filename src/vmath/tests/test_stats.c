#include "test_stats.h"


bool
test_poisson_pmf (void)
{
    for (size_t oom = 0; oom < OOM; oom++)
    {
        scalar lambda = (scalar) pow (10.0, oom);
        for (size_t i = 0; i < OOM; i++)
        {
            long variate = (long) pow (10.0, i);
            scalar res = poisson_pmf (lambda, variate);
            /*printf ("Lambda: %10.Lf\t variate: %10.ld\t pmf: %20.15Lg\n", lambda, variate, res);*/
            if (isinf (res) || isnan (res))
            {
                return UT_FAILURE;
            }
        }
    }
    return UT_SUCCESS;
}


bool
test_poisson_logpmf (void)
{
    for (size_t oom = 0; oom < OOM; oom++)
    {
        scalar lambda = (scalar) pow (10.0, oom);
        for (size_t i = 0; i < OOM; i++)
        {
            long variate = (long) pow (10.0, i);
            scalar res = poisson_logpmf (lambda, variate);
            /*printf ("Lambda: %10.Lf\t variate: %10.ld\t log_pmf: %20.15Lf\n", lambda, variate, res);*/
            if (isinf (res) || isnan (res))
            {
                return UT_FAILURE;
            }
        }
    }
    return UT_SUCCESS;
}
