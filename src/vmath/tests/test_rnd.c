#include "test_rnd.h"


bool
test__rnd (void)
{
    LOOP {
        if (!ASSERT_IN_RANGE (rnd (), 0, 1))
        {
            return true;
        }
    }
    return false;
}


bool
test__r_rnd (void)
{
    enum setup {
        SR_LOWER = -100,
        SR_UPPER =  100  };

    LOOP {
        if (!ASSERT_IN_RANGE (r_rnd (SR_LOWER, SR_UPPER), SR_LOWER, SR_UPPER))
        {
            return true;
        }
    }
    return false;
}


bool
test__vr_rnd (void)
{
    enum setup {
        N_ELEM   =  200,
        SR_LOWER = -100,
        SR_UPPER =  100, };

    scalar arr[N_ELEM] = { 0.0L };

    vr_rnd (N_ELEM, SR_LOWER, SR_UPPER, arr);
    for (size_t i = 0; i < N_ELEM; i++)
    {
        if (!ASSERT_IN_RANGE (arr[i], SR_LOWER, SR_UPPER) || !isfinite (arr[i]))
        {
            return true;
        }
    }
    return false;
}


bool
test_rnd_int (void)
{
    LOOP {
        int low = rand () % 1000;
        int high = 1000 + rand () % 1000;
        if (!ASSERT_IN_RANGE (rnd_int (low, high), low, high))
        {
            return true;
        }
    }
    return false;
}


bool
test__v_rnd (void)
{
    scalar vals[N];
    v_rnd (N, vals);
    LOOP {
        if (!ASSERT_IN_RANGE (vals[i], 0, 1))
        {
            return true;
        }
    }
    return false;
}


bool
test_v_rnd_int (void)
{
    int low = rand () % 1000;
    int high = 1000 + rand () % 1000;
    int vals[N];
    v_rnd_int (low, high, N, vals);
    LOOP {
        if (!ASSERT_IN_RANGE (vals[i], low, high))
        {
            return true;
        }
    }
    return false;
}
