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
test_ds_NewEmpty (void);

bool
test_ds_New (void);

bool
test_ds_NewFromFile (void);

bool
test_ds_read (void);

bool
test_ds_set_error_on_idx_out_of_bounds (void);

bool
test_ds_set_values (void);

bool
test_ds_get_error_on_idx_out_of_bounds (void);

bool
test_ds_get_values (void);


#endif  /* TEST_DATASET_H */
