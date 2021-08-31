#ifndef vmath_alloc_h
#define vmath_alloc_h

#include <stdlib.h>
#include "config.h"


#define VA_SCALAR_EMPTY(n_elem) malloc ((n_elem) * sizeof (scalar));
#define VA_SCALAR_ZEROS(n_elem) calloc (n_elem, sizeof (scalar))

#define VA_INT_EMPTY(n_elem) malloc ((n_elem) * sizeof (int))
#define VA_INT_ZEROS(n_elem) calloc (n_elem, sizeof (int))

#define VA_SIZE_EMPTY(n_elem) malloc ((n_elem) * sizeof (size_t));
#define VA_SIZE_ZEROS(n_elem) calloc (n_elem, sizeof (size_t));

#define FREE(buff) do { \
    free (buff);        \
    buff = NULL;        \
} while (0)

#define ASSERT_ALLOC(buff) if (buff == NULL) {          \
    fputs ("Could not allocate buffer.\n", stderr);     \
    return 1;                                           \
}


#endif  /* vmath_alloc_h */
