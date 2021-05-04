#include "test_rnd.h"

int main (void)
{
    SETUP;

    RUN_TEST (test_rnd);
    RUN_TEST (test_rnd_int);
    RUN_TEST (test_v_rnd);
    RUN_TEST (test_v_rnd_int);

    EVALUATE;
}


bool
test_rnd (void)
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
test_v_rnd (void)
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
