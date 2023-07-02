#ifndef test_bw_h
#define test_bw_h

#include "assertations.h"
#include "../pois-em.h"


bool
test__pois_e_step (void);

bool
test__pois_m_step_lambda (void);

bool
test__pois_m_step_gamma (void);

bool
test__pois_m_step_delta (void);

bool
test__score_update (void);


#endif  /* test_bw_h */
