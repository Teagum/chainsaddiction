#ifndef TEST_HMM_H
#define TEST_HMM_H

#include <math.h>
#include "unittest.h"
#include "rnd.h"
#include "poishmm.h"

bool test__PoisHmm_NewProbs (void);
bool test__PoisHmm_NewParams (void);
bool test__PoisHmm_New (void);
bool test_ca_ph_InitParams (void);
bool test_ca_log_likelihood (void);

#endif  /* TEST_HMM_H */
