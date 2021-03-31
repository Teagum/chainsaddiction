#ifndef TEST_DATASET_H
#define TEST_DATASET_H

#include <limits.h>
#include <stdio.h>
#include <time.h>
#include "unittest.h"
#include "scalar.h"
#include "rnd.h"
#include "restrict.h"
#include "dataset.h"


bool
test_Ca_NewDataSet (void);

bool
test_ds_set_error_on_idx_out_of_bounds (void);

bool
test_ds_set_values (void);

bool
test_ds_get_error_on_idx_out_of_bounds (void);

bool
test_ds_get_values (void);

#endif  /* TEST_DATASET_H */
