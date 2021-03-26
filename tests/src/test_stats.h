#ifndef TEST_STATS_H
#define TEST_STATS_H
#include <math.h>
#include "unittest.h"
#include "scalar.h"
#include "restrict.h"
#include "rnd.h"
#include "stats.h"


#define OOM 6

bool test_poisson_pmf (void);
bool test_poisson_logpmf (void);

#endif  /* TEST_STATS_H */
