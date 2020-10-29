#include <stdio.h>
#include <time.h>
#include "vmath.h"
#include "test_vmath.h"

int main (void)
{
    srand (time (NULL));

    FEEDBACK (test_strided_max);
    FEEDBACK (test_v_max);
    FEEDBACK (test_m_max);
    FEEDBACK (test_m_row_max);
    FEEDBACK (test_m_col_max);

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
