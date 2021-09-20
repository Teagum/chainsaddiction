#include "unittest.h"
#include "test-utils.h"

int
main (void)
{
    SETUP;

    RUN_TEST (test__local_decoding);
    RUN_TEST (test__global_decoding);

    EVALUATE;
}
