#include "test_dataset.h"


unsigned short N_ERRORS = 0;

int main (void)
{
    srand (time (NULL));

    FEEDBACK (test_Ca_NewDataSet);
    FEEDBACK (test_ds_set_error_on_idx_out_of_bounds);
    FEEDBACK (test_ds_set_values);
    FEEDBACK (test_ds_get_error_on_idx_out_of_bounds);
    FEEDBACK (test_ds_get_values);

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
    bool cond = false;

    DataSet *inp = Ca_NewDataSet ();
    for (size_t i=0; i<inp->size; i++) {
        acc += inp->data[i];
    }
    cond = ASSERT_EQUAL (0, acc);
    CA_FREE_DATASET (inp);
    return cond;
}




bool
test_ds_set_error_on_idx_out_of_bounds (void)
{
    enum { n_iter=1000, val=11 };
    DataSet *inp = Ca_NewDataSet ();

    for (size_t i=0; i<n_iter; i++)
    {
        size_t idx = (size_t) rnd_int (DATASET_INIT_SIZE, INT_MAX);

        ds_set (inp, idx, val);
        if (!inp->err) {
            CA_FREE_DATASET (inp);
            return false;
        }
    }

    CA_FREE_DATASET (inp);
    return true;
}


bool
test_ds_set_values (void)
{
    enum { n_iter=1000 };
    DataSet *inp = Ca_NewDataSet ();

    for (size_t i=0; i<n_iter; i++)
    {
        size_t idx = (size_t) rnd_int (0, DATASET_INIT_SIZE);
        scalar val = rnd ();

        ds_set (inp, idx, val);
        if (inp->err || !ASSERT_EQUAL (inp->data[idx], val)) {
            CA_FREE_DATASET (inp);
            return false;
        }
    }
    CA_FREE_DATASET (inp);
    return true;
}


bool
test_ds_get_error_on_idx_out_of_bounds (void)
{
    enum { n_iter=1000 };
    DataSet *inp = Ca_NewDataSet ();

    for (size_t i=0; i<n_iter; i++)
    {
        scalar val = 0;
        size_t idx = (size_t) rnd_int (DATASET_INIT_SIZE, INT_MAX);

        ds_get (inp, idx, &val);
        if (!inp->err) {
            CA_FREE_DATASET (inp);
            return false;
        }
    }

    CA_FREE_DATASET (inp);
    return true;
}


bool
test_ds_get_values (void)
{
    enum { n_iter=1000 };
    DataSet *inp = Ca_NewDataSet ();

    for (size_t i=0; i<n_iter; i++)
    {
        scalar val = rnd ();
        size_t idx = (size_t) rnd_int (0, DATASET_INIT_SIZE);
        scalar out = 0;

        ds_set (inp, idx, val);
        ds_get (inp, idx, &out);
        if (inp->err || !ASSERT_EQUAL (val, out)) {
           CA_FREE_DATASET (inp);
           return false;
        }
    }

    CA_FREE_DATASET (inp);
    return true;
}


