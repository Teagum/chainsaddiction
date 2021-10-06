#include "unittest.h"
#include "test-dataset.h"


int main (void)
{
    SETUP;

    RUN_TEST (test_ds_NewEmpty);
    //RUN_TEST (test_ds_New);
    RUN_TEST (test_ds_set_error_on_idx_out_of_bounds);
    RUN_TEST (test_ds_set_values);
    RUN_TEST (test_ds_get_error_on_idx_out_of_bounds);
    RUN_TEST (test_ds_get_values);

    EVALUATE;
}
