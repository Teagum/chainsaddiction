#include "test_dataset.h"


unsigned short N_ERRORS = 0;

int main (void)
{
    srand (time (NULL));

    FEEDBACK (test_Ca_NewDataSet);

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


bool
test_Ca_NewDataSet (void)
{
    scalar acc = 0;
    DataSet *inp = Ca_NewDataSet ();
    for (size_t i=0; i<inp->size; i++) {
        acc += inp->data[i];
    }
    return ASSERT_EQUAL (0, acc);
}
