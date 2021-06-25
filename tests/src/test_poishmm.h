#ifndef TEST_HMM_H
#define TEST_HMM_H

#include <math.h>
#include "unittest.h"
#include "rnd.h"
#include "hmm.h"

bool test_ca_ph_NewProbs (void);
bool test_ca_ph_NewParams (void);
bool test_ca_ph_NewHmm (void);
bool test_ca_ph_InitParams (void);
bool test_ca_log_likelihood (void);

#endif  /* TEST_HMM_H */
