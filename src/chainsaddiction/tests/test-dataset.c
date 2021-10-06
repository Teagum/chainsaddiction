#include "test-dataset.h"


bool
test_ds_NewEmpty (void)
{
    scalar acc = 0.0L;
    bool err = true;

    DataSet *inp = ds_New (DS_TEST_INIT_SIZE);
    for (size_t i = 0; i < inp->size; i++) {
        acc += inp->data[i];
    }
    if (ASSERT_EQUAL (0.0L, acc)) {
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
    return UT_SUCCESS;
}


bool
test_ds_set_error_on_idx_out_of_bounds (void)
{
    DataSet *inp = ds_New (DS_TEST_INIT_SIZE);

    for (size_t i = 0; i < DS_TEST_N_ITER; i++)
    {
        size_t idx = rnd_size (DS_TEST_INIT_SIZE, INT_MAX);
        scalar val = rnd_sample ();

        ds_set (inp, idx, val);
        if (!inp->err) {
            ds_FREE (inp);
            return UT_FAILURE;
        }
    }

    ds_FREE (inp);
    return UT_SUCCESS;
}


bool
test_ds_set_values (void)
{
    DataSet *inp = ds_New (DS_TEST_INIT_SIZE);

    for (size_t i = 0; i < DS_TEST_N_ITER; i++)
    {
        size_t idx = rnd_size (0, DS_TEST_INIT_SIZE);
        scalar val = rnd_sample ();

        ds_set (inp, idx, val);
        if (inp->err || !ASSERT_EQUAL (inp->data[idx], val)) {
            ds_FREE (inp);
            return UT_FAILURE;
        }
    }
    ds_FREE (inp);
    return UT_SUCCESS;
}


bool
test_ds_get_error_on_idx_out_of_bounds (void)
{
    DataSet *inp = ds_New (DS_TEST_INIT_SIZE);

    for (size_t i = 0; i < DS_TEST_N_ITER; i++)
    {
        size_t idx = rnd_size (DS_TEST_INIT_SIZE, INT_MAX);
        scalar val = 0;

        ds_get (inp, idx, &val);
        if (!inp->err) {
            ds_FREE (inp);
            return UT_FAILURE;
        }
    }
    ds_FREE (inp);
    return UT_SUCCESS;
}


bool
test_ds_get_values (void)
{
    DataSet *inp = ds_New (DS_TEST_INIT_SIZE);

    for (size_t i = 0; i < DS_TEST_N_ITER; i++)
    {
        scalar val = rnd_sample ();
        size_t idx = rnd_size (0, DS_TEST_INIT_SIZE);
        scalar out = 0;

        ds_set (inp, idx, val);
        ds_get (inp, idx, &out);
        if (inp->err || !ASSERT_EQUAL (val, out)) {
           ds_FREE (inp);
           return UT_FAILURE;
        }
    }

    ds_FREE (inp);
    return UT_SUCCESS;
}
