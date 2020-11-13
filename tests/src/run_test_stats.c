#include <time.h>
#include "test_stats.h"

unsigned short N_ERRORS = 0;

int main (void)
{
    srand (time (NULL));

    FEEDBACK (test_poisson_pmf);
    FEEDBACK (test_poisson_logpmf);

    if (N_ERRORS == 0)
    {
        fprintf (stdout, "All tests passed.\n");
        return EXIT_SUCCESS;
    }
    else
    {
        fprintf (stdout, "FAILURE: %d tests with errors.\n", N_ERRORS);
        return EXIT_FAILURE;
    }
}
