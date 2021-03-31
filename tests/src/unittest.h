#ifndef UNITTEST_H
#define UNITTEST_H

#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

unsigned short _N_ERRORS;

#define N 100
#define LOOP for (size_t i = 0; i < N; i++)

#define EPSILON 1e-10
#define FP_EQUAL(_x_, _y_) \
    ((fabsl ((long double) _x_ - (long double) _y_) < EPSILON) \
     ? true : false)

#define ASSERT_EQUAL(_val_, _comp_) (FP_EQUAL (_val_, _comp_))
#define ASSERT_GREATER(_val_, _comp_) (_val_ > _comp_)
#define ASSERT_GREATER_EQ(_val_, _comp_) (_val_ >= _comp_)
#define ASSERT_LESS(_val_, _comp) (_val_ < _comp_)
#define ASSERT_LESS_EQ(_val_, _comp_) (_val_ <= _comp_)

#define ASSERT_IN_RANGE(_val_, _lb_, _hb_) \
    (ASSERT_GREATER_EQ (_val_, _lb_) && ASSERT_LESS_EQ(_val_, _hb_))

#define STV(_str_) #_str_

#define OK(_res_)               \
    if (_res_ == true)          \
    {                           \
        printf ("OK\n");        \
    }                           \
    else                        \
    {                           \
        _N_ERRORS += 1;         \
        printf ("FAILED\n");    \
    }


#define FEEDBACK(_func_)  \
    do {                                    \
        printf ("\t%-50s ... ", STV(_func_));    \
        OK (_func_ ())                      \
    } while (0)


#endif  /* UNITTEST_H */
