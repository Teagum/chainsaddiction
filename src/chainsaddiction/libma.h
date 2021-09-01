#ifndef libma_h
#define libma_h

#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include "scalar.h"


enum ma_types
{
    MA_SHORT,
    MA_INT,
    MA_FLOAT,
    MA_DOUBLE,
    MA_SCALAR
};


#define MA_INIT_ARRAY(arr, init_val)    \
do {                                    \
    for (size_t i = 0; i < n_elem; ++i) \
    {                                   \
        arr[i] = (init_val);            \
    }                                   \
} while (0)


#define MA_ASSERT_ALLOC(buffer, msg)    \
do {                                    \
    if (buffer == NULL)                 \
    {                                   \
        fprintf (stderr, "%s\n", msg);  \
        exit (1);                       \
    }                                   \
} while (0)


#define MA_FREE(ptr)                    \
do {                                    \
    free (ptr);                         \
    ptr = NULL;                         \
} while (0)


size_t
Ma_TypeSize (enum ma_types type);


void *
Ma_ArrayMemAlloc (const size_t n_elem, enum ma_types type, bool init);


#define MA_INT_EMPTY(n_elem) \
    (int *) Ma_ArrayMemAlloc (n_elem, MA_INT, false)

#define MA_INT_ZEROS(n_elem) \
    (int *) Ma_ArrayMemAlloc (n_elem, MA_INT, true)

#define MA_FLOAT_EMPTY(n_elem) \
    (float *) Ma_ArrayMemAlloc (n_elem, MA_FLOAT, false)

#define MA_FLOAT_ZEROS(n_elem) \
    (float *) Ma_ArrayMemAlloc (n_elem, MA_FLOAT, true)

#define MA_DOUBLE_EMPTY(n_elem) \
    (double *) Ma_ArrayMemAlloc (n_elem, MA_DOUBLE, false)

#define MA_DOUBLE_ZEROS(n_elem) \
    (double *) Ma_ArrayMemAlloc (n_elem, MA_DOUBLE, true)

#define MA_SCALAR_EMPTY(n_elem) \
    (scalar *) Ma_ArrayMemAlloc (n_elem, MA_SCALAR, false)

#define MA_SCALAR_ZEROS(n_elem) \
    (scalar *) Ma_ArrayMemAlloc (n_elem, MA_SCALAR, true)


#endif  /* libma_h */
