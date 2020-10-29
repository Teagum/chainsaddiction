#include "test_rnd.h"


int main (void)
{
    srand ((unsigned int) time (NULL));

    FEEDBACK (test_rnd);
    FEEDBACK (test_rnd_int);
    FEEDBACK (test_v_rnd);
    FEEDBACK (test_v_rnd_int);

    return EXIT_SUCCESS;
}
