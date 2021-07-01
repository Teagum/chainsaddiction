#ifndef UNITTEST_H
#define UNITTEST_H

#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

unsigned short n_fails;

#define N 100
#define LOOP for (size_t i = 0; i < N; i++)

#define EPSILON 1e-10
#define FP_EQUAL(_x_, _y_) \
    ((fabsl ((long double) _x_ - (long double) _y_) < EPSILON) \
     ? true : false)

#define RAND_INT(lower, upper) (lower) + rand () / ((RAND_MAX + 1u) / (upper))

/*
 * Assertions
 */
#define ASSERT_EQUAL(_val_, _comp_) (FP_EQUAL (_val_, _comp_))
#define ASSERT_GREATER(_val_, _comp_) (_val_ > _comp_)
#define ASSERT_GREATER_EQ(_val_, _comp_) (_val_ >= _comp_)
#define ASSERT_LESS(_val_, _comp) (_val_ < _comp_)
#define ASSERT_LESS_EQ(_val_, _comp_) (_val_ <= _comp_)
#define ASSERT_IN_RANGE(_val_, _lb_, _hb_) \
    (ASSERT_GREATER_EQ (_val_, _lb_) && ASSERT_LESS_EQ(_val_, _hb_))

#define STV(_str_) #_str_

#define SETUP do {          \
    srand (time (NULL));    \
    n_fails = 0;            \
} while (0)


#define CHECK_ERROR(_res_)      \
    if (_res_ == false)         \
    {                           \
        printf ("OK\n");        \
    }                           \
    else                        \
    {                           \
        n_fails += 1;           \
        printf ("FAILED\n");    \
    }

#define RUN_TEST(_func_)                        \
    do {                                        \
        printf ("\t%-50s ... ", STV(_func_));   \
        CHECK_ERROR(_func_ ())                  \
    } while (0)


#define EVALUATE do {                                                       \
    if (n_fails == 0)                                                       \
    {                                                                       \
        puts ("All tests passed");                                          \
        return EXIT_SUCCESS;                                                \
    }                                                                       \
    else                                                                    \
    {                                                                       \
        fprintf (stdout, "FAILURE: %d tests with errors.\n", n_fails);      \
        return EXIT_FAILURE;                                                \
    }                                                                       \
} while (0)


#endif  /* UNITTEST_H */
