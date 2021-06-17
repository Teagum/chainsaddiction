#include "test_hmm.h"


int
main (void)
{
    SETUP;

    RUN_TEST (test_log_lieklihood_fw);

    EVALUATE;
}


bool
test_log_likelihood_fw (void)
{
    const scalar expected = 11.4076059644443803L;
    scalar a[] = {0, 1, 2, 3, 4, 5, 6, 7, 8 ,9, 10, 11};
    scalar res = log_likelihood_fw (a, 4, 3);
    return !ASSERT_EQUAL (res, expected);
}
