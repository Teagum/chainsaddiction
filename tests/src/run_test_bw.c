#include <stdio.h>
#include <time.h>
#include "test_bw.h"


int main (void)
{
    srand (time (NULL));

    FEEDBACK (test_ca_bw_pois_e_step);
    FEEDBACK (test_update_lambda);

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
