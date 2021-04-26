#include "test_dataset.h"


int main (void)
{
    SETUP;

    RUN_TEST (test_Ca_NewDataSet);
    RUN_TEST (test_Ca_DataSetFromFile);
    RUN_TEST (test_ds_set_error_on_idx_out_of_bounds);
    RUN_TEST (test_ds_set_values);
    RUN_TEST (test_ds_get_error_on_idx_out_of_bounds);
    RUN_TEST (test_ds_get_values);

    EVALUATE;
}


bool
test_Ca_NewDataSet (void)
{
    scalar acc = 0l;
    bool err = true;

    DataSet *inp = Ca_NewDataSet ();
    for (size_t i=0; i<inp->size; i++) {
        acc += inp->data[i];
    }
    if (ASSERT_EQUAL (0, acc)) {
        err = false;
    }

    CA_FREE_DATASET (inp);
    return err;
}


bool
test_Ca_DataSetFromFile (void)
{
    const char path[] = "tests/data/centroids";
    DataSet *pds = Ca_DataSetFromFile (path);

    ds_print (pds);

    CA_FREE_DATASET (pds);
    return false;
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
            return true;
        }
    }

    CA_FREE_DATASET (inp);
    return false;
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
            return true;
        }
    }
    CA_FREE_DATASET (inp);
    return false;
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
            return true;
        }
    }
    CA_FREE_DATASET (inp);
    return false;
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
           return true;
        }
    }

    CA_FREE_DATASET (inp);
    return false;
}
