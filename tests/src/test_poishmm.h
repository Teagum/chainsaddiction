#ifndef TEST_HMM_H
#define TEST_HMM_H

#include <math.h>
#include <string.h>
#include "unittest.h"
#include "rnd.h"
#include "poishmm.h"

bool test__PoisHmm_NewProbs (void);
bool test__PoisHmm_NewParams (void);
bool test__PoisHmm_New (void);
bool test__PoisHmm_Init (void);
bool test__PoisHmm_InitRandom (void);
bool test__PoisHmm_LogLikelihood (void);

#endif  /* TEST_HMM_H */
