#include <stdio.h>
#include <time.h>
#include "test_hmm.h"

int main (void)
{
    srand (time (NULL));

    FEEDBACK (test_log_likelihood_fw);

    if (_N_ERRORS == 0)
    {
        fprintf (stdout, "All tests passed.\n");
        return EXIT_SUCCESS;
    }
    else
    {
        fprintf (stdout, "FAILURE: %d tests with errors.\n", _N_ERRORS);
        return EXIT_FAILURE;
    }
}
