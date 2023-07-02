#ifndef test_pois_hmm_h
#define test_pois_hmm_h

#include <math.h>
#include <string.h>
#include "assertations.h"
#include "../pois-hmm.h"

bool test__PoisHmm_New (void);
bool test__PoisHmm_Init (void);
bool test__PoisHmm_InitRandom (void);
bool test__PoisHmm_InitRandom_sorted_lambda (void);
bool test__PoisHmm_LogLikelihood (void);
bool test__PoisHmm_EstimateParams (void);
bool test__PoisHmm_ForwardProbabilities (void);
bool test__PoisHmm_BackwardProbabilities (void);
bool test__PoisHmm_ForwardBackward (void);

#endif  /* test_pois_hmm_h */
