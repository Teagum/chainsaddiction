#include <assert.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "stats.h"
#include "utilities.h"

#define OOM 6


void test_poisson_pmf (void);
void test_poisson_logpmf (void);
void test_v_poisson_logpmf (DataSet *data);

int main (int argc, char *argv[])
{
    DataSet *data = read_dataset ();
    test_poisson_pmf ();
    test_poisson_logpmf();

    printf ("---\n");

    test_v_poisson_logpmf (data);

    return EXIT_SUCCESS;
}


void test_poisson_pmf (void)
{
    for (size_t oom = 0; oom < OOM; oom++)
    {
        scalar lambda = (scalar) pow (10.0, oom);
        for (size_t i = 0; i < OOM; i++)
        {
            long variate = (long) pow (10.0, i);
            scalar res = poisson_pmf (lambda, variate);
            printf ("Lambda: %10.Lf\t variate: %10.ld\t pmf: %20.15Lg\n",
                    lambda, variate, res);
            assert (!isinf (res));
            assert (!isnan (res));
        }
    }
}


void test_poisson_logpmf (void)
{
    for (size_t oom = 0; oom < OOM; oom++)
    {
        scalar lambda = (scalar) pow (10.0, oom);
        for (size_t i = 0; i < OOM; i++)
        {
            long variate = (long) pow (10.0, i);
            scalar res = poisson_logpmf (lambda, variate);
            printf ("Lambda: %10.Lf\t variate: %10.ld\t log_pmf: %20.15Lf\n",
                    lambda, variate, res);
            assert (!isinf (res));
            assert (!isnan (res));
        }
    }
}


void test_v_poisson_logpmf (DataSet *data)
{
    for (size_t i = 0; i > data->size; i++)
    {
        printf ("%ld\n", data->data[i]);
    }
}
