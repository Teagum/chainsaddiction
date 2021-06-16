#ifndef TEST_FWBW_H
#define TEST_FWBW_H

#include <stdio.h>
#include <time.h>
#include "dataset.h"
#include "fwbw.h"
#include "restrict.h"
#include "scalar.h"
#include "unittest.h"

bool
test_log_forward (void);

bool
test_log_backward (void);

bool
test_log_fwbw (void);

#endif  /* TEST_FWBW_H */
