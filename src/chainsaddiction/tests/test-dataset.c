#include "test-dataset.h"


bool
test_DataSet_NewEmpty (void)
{
    scalar acc = 0.0L;
    bool err = true;

    DataSet *inp = DataSet_New (DS_TEST_INIT_SIZE);
    for (size_t i = 0; i < inp->size; i++) {
        acc += inp->data[i];
    }
    if (ASSERT_EQUAL (0.0L, acc)) {
        err = false;
    }

    DataSet_Delete (inp);
    return err;
}


bool
test_DataSet_NewFromFile (void)
{
    const char path[] = "tests/data/centroids";
    DataSet *pds = DataSet_NewFromFile (path);

    DataSet_Delete (pds);
    return UT_SUCCESS;
}


bool
test_DataSet_SetValue_error_on_idx_out_of_bounds (void)
{
    DataSet *inp = DataSet_New (DS_TEST_INIT_SIZE);

    for (size_t i = 0; i < DS_TEST_N_ITER; i++)
    {
        size_t idx = rnd_size (DS_TEST_INIT_SIZE, INT_MAX);
        scalar val = rnd_sample ();

        DataSet_SetValue (inp, idx, val);
        if (!inp->err) {
            DataSet_Delete (inp);
            return UT_FAILURE;
        }
    }

    DataSet_Delete (inp);
    return UT_SUCCESS;
}


bool
test_DataSet_SetValue (void)
{
    DataSet *inp = DataSet_New (DS_TEST_INIT_SIZE);

    for (size_t i = 0; i < DS_TEST_N_ITER; i++)
    {
        size_t idx = rnd_size (0, DS_TEST_INIT_SIZE);
        scalar val = rnd_sample ();

        DataSet_SetValue (inp, idx, val);
        if (inp->err || !ASSERT_EQUAL (inp->data[idx], val)) {
            DataSet_Delete (inp);
            return UT_FAILURE;
        }
    }
    DataSet_Delete (inp);
    return UT_SUCCESS;
}


bool
test_DataSet_GetValue_error_on_idx_out_of_bounds (void)
{
    DataSet *inp = DataSet_New (DS_TEST_INIT_SIZE);

    for (size_t i = 0; i < DS_TEST_N_ITER; i++)
    {
        size_t idx = rnd_size (DS_TEST_INIT_SIZE, INT_MAX);
        scalar val = 0;

        DataSet_GetValue (inp, idx, &val);
        if (!inp->err) {
            DataSet_Delete (inp);
            return UT_FAILURE;
        }
    }
    DataSet_Delete (inp);
    return UT_SUCCESS;
}


bool
test_DataSet_GetValue (void)
{
    DataSet *inp = DataSet_New (DS_TEST_INIT_SIZE);

    for (size_t i = 0; i < DS_TEST_N_ITER; i++)
    {
        scalar val = rnd_sample ();
        size_t idx = rnd_size (0, DS_TEST_INIT_SIZE);
        scalar out = 0;

        DataSet_SetValue (inp, idx, val);
        DataSet_GetValue (inp, idx, &out);
        if (inp->err || !ASSERT_EQUAL (val, out)) {
           DataSet_Delete (inp);
           return UT_FAILURE;
        }
    }

    DataSet_Delete (inp);
    return UT_SUCCESS;
}
