#include "test_dataset.h"


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


bool
test_ds_NewEmpty (void)
{
    scalar acc = 0l;
    bool err = true;

    DataSet *inp = ds_New (DS_TEST_INIT_SIZE);
    for (size_t i = 0; i < inp->size; i++) {
        acc += inp->data[i];
    }
    if (ASSERT_EQUAL (0, acc)) {
        err = false;
    }

    ds_FREE (inp);
    return err;
}


bool
test_ds_NewFromFile (void)
{
    const char path[] = "tests/data/centroids";
    DataSet *pds = ds_NewFromFile (path);

    ds_FREE (pds);
    return false;
}


bool
test_ds_set_error_on_idx_out_of_bounds (void)
{
    DataSet *inp = ds_New (DS_TEST_INIT_SIZE);

    for (size_t i = 0; i < DS_TEST_N_ITER; i++)
    {
        size_t idx = (size_t) rnd_int (DS_TEST_INIT_SIZE, INT_MAX);
        scalar val = rnd ();

        ds_set (inp, idx, val);
        if (!inp->err) {
            ds_FREE (inp);
            return true;
        }
    }

    ds_FREE (inp);
    return false;
}


bool
test_ds_set_values (void)
{
    DataSet *inp = ds_New (DS_TEST_INIT_SIZE);

    for (size_t i = 0; i < DS_TEST_N_ITER; i++)
    {
        size_t idx = (size_t) rnd_int (0, DS_TEST_INIT_SIZE);
        scalar val = rnd ();

        ds_set (inp, idx, val);
        if (inp->err || !ASSERT_EQUAL (inp->data[idx], val)) {
            printf ("ds: %Lf\n", inp->data[idx]);
            ds_FREE (inp);
            return true;
        }
    }
    ds_FREE (inp);
    return false;
}


bool
test_ds_get_error_on_idx_out_of_bounds (void)
{
    DataSet *inp = ds_New (DS_TEST_INIT_SIZE);

    for (size_t i = 0; i < DS_TEST_N_ITER; i++)
    {
        size_t idx = (size_t) rnd_int (DS_TEST_INIT_SIZE, INT_MAX);
        scalar val = 0;

        ds_get (inp, idx, &val);
        if (!inp->err) {
            ds_FREE (inp);
            return true;
        }
    }
    ds_FREE (inp);
    return false;
}


bool
test_ds_get_values (void)
{
    DataSet *inp = ds_New (DS_TEST_INIT_SIZE);

    for (size_t i = 0; i < DS_TEST_N_ITER; i++)
    {
        scalar val = rnd ();
        size_t idx = (size_t) rnd_int (0, DS_TEST_INIT_SIZE);
        scalar out = 0;

        ds_set (inp, idx, val);
        ds_get (inp, idx, &out);
        if (inp->err || !ASSERT_EQUAL (val, out)) {
           ds_FREE (inp);
           return true;
        }
    }

    ds_FREE (inp);
    return false;
}
