#include "test_rnd.h"


bool
test__rnd_int (void)
{
    enum setup {
        N_ITER =  100u,
        SR_LB  = -100,
        SR_UB  =  100,
    };

    for (size_t i = 0; i < N_ITER; i++)
    {
        if (!ASSERT_IN_RANGE (rnd_int (SR_LB, SR_UB), SR_LB, SR_UB))
        {
            return UT_FAILURE;
        }
    }
    return UT_SUCCESS;
}


bool
test__v_rnd_int (void)
{
    enum setup {
        N_ITER =  100u,
        N_ELEM =  200u,
        SR_LB  = -100,
        SR_UB  =  100,
        IVAL   =    0
    };

    for (size_t n = 0; n < N_ITER; n++)
    {
        int arr[N_ELEM] = { IVAL };
        v_rnd_int (N_ELEM, SR_LB, SR_UB, arr);

        for (size_t i = 0; i < N_ELEM; i++)
        {
            if (!ASSERT_IN_RANGE (arr[i], SR_LB, SR_UB) || !isfinite (arr[i]))
            {
                return UT_FAILURE;
            }
        }
    }
    return UT_SUCCESS;
}


bool
test__rnd_scalar (void)
{
    enum setup { N_ITER = 200u };
    const scalar SR_LB = -100.0L;
    const scalar SR_UB =  100.0L;

    for (size_t i = 0; i < N_ITER; i++)
    {
        if (!ASSERT_IN_RANGE (rnd_scalar (SR_LB, SR_UB), SR_LB, SR_UB))
        {
            return UT_FAILURE;
        }
    }
    return UT_SUCCESS;
}


bool
test__v_rnd_scalar (void)
{
    enum setup { N_ELEM = 200u };
    const scalar SR_LB  = -100.0L;
    const scalar SR_UB  =  233.0L;
    const scalar IVAL   =    0.0L;

    scalar arr[N_ELEM] = { IVAL };
    v_rnd_scalar (N_ELEM, SR_LB, SR_UB, arr);

    for (size_t i = 0; i < N_ELEM; i++)
    {
        if (!ASSERT_IN_RANGE (arr[i], SR_LB, SR_UB) || !isfinite (arr[i]))
        {
            return UT_FAILURE;
        }
    }
    return UT_SUCCESS;
}


bool
test__sample (void)
{
    enum setup { N_ELEM = 200u };
    const scalar SR_LB = 0.0L;
    const scalar SR_UB = 1.0L;

    for (size_t i = 0; i < N_ELEM; i++)
    {
        scalar val = rnd_sample ();
        if (!ASSERT_IN_RANGE (val, SR_LB, SR_UB) || !isfinite (val))
        {
            return UT_FAILURE;
        }
    }
    return UT_SUCCESS;
}


bool
test__v_sample (void)
{
    enum setup { N_ELEM = 200u, N_ITER =  10u, };
    const scalar SR_LB = 0.0L;
    const scalar SR_UB = 1.0L;
    const scalar IVAL  = 0.0L;

    for (size_t n = 0; n < N_ITER; n++)
    {
        scalar arr[N_ELEM] = { IVAL };
        v_rnd_sample (N_ELEM, arr);

        for (size_t i = 0; i < N_ELEM; i++)
        {
            if (!ASSERT_IN_RANGE (arr[i], SR_LB, SR_UB) || !isfinite (arr[i]))
            {
                return UT_FAILURE;
            }
        }
    }
    return UT_SUCCESS;
}
