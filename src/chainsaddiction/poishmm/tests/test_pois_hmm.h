#ifndef test_pois_hmm_h
#define test_pois_hmm_h

#include <math.h>
#include <string.h>
#include "unittest.h"
#include "../../rnd.h"
#include "pois_hmm.h"

bool test__PoisHmm_New (void);
bool test__PoisHmm_Init (void);
bool test__PoisHmm_InitRandom (void);
bool test__PoisHmm_LogLikelihood (void);

#endif  /* test_pois_hmm_h */
