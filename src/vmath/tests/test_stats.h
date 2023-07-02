#ifndef test_stats_h
#define test_stats_h

#include <math.h>
#include <stdbool.h>
#include "config.h"
#include "assertations.h"
#include "rnd.h"
#include "stats.h"


#define OOM 6

bool test_poisson_pmf (void);
bool test_poisson_logpmf (void);


#endif  /* test_stats_h */
