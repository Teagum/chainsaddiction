#ifndef TEST_DATASET_H
#define TEST_DATASET_H

#include <limits.h>
#include <stdio.h>
#include <time.h>
#include "unittest.h"
#include "chainsaddiction.h"
#include "dataset.h"
#include "libvmath.h"

#define DS_TEST_INIT_SIZE 1000
#define DS_TEST_N_ITER 1000


bool
test_DataSet_NewEmpty (void);

bool
test_DataSet_New (void);

bool
test_DataSet_NewFromFile (void);

bool
test_DataSet_read (void);

bool
test_DataSet_SetValue_error_on_idx_out_of_bounds (void);

bool
test_DataSet_SetValue (void);

bool
test_DataSet_GetValue_error_on_idx_out_of_bounds (void);

bool
test_DataSet_GetValue (void);


#endif  /* TEST_DATASET_H */
