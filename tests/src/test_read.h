#ifndef test_read_h
#define test_read_h

#include <time.h>
#include <stdio.h>
#include "unittest.h"
#include "scalar.h"
#include "read.h"


#define N_EQ 107

#define S_MALLOC_RAW(n_elem__)                                      \
    (scalar *) malloc (sizeof (scalar) * (n_elem__))

#define CHECK_MALLOC_RAW(ptr__) do {                                \
    if (!(ptr__))                                                   \
    {                                                               \
        fprintf (stderr, "Could not allocate memory for tests.\n"); \
        exit (EXIT_FAILURE);                                        \
    }                                                               \
} while (false)

#define S_MALLOC(ptr__, n_elem__) do {                              \
    ptr__ = S_MALLOC_RAW (n_elem__);                                \
    CHECK_MALLOC_RAW (ptr__);                                       \
} while (false)


bool
test_Ca_ReadDataFile_full_file (void);

bool
test_Ca_ReadDataFile_n_lines (void);

bool
test_Ca_CountLines_earthquakes (void);

bool
test_Ca_CountLines_empty (void);

bool
test_Ca_CountLines_wrong_format (void);


#endif	/* test_read_h */
