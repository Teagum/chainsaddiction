#ifndef test_bw_h
#define test_bw_h

#include "unittest.h"
#include "../pois_em.h"

#define SET_EARTHQUAKES_SHORT const scalar input[] = \
        { 13, 14, 8, 10, 16, 26, 32, 27, 18, 32, 36, 24, 22, 23, 22 }

#define SET_LAMBDA const scalar lambda[] = { 10, 20, 30 }
#define SET_DELTA  const scalar delta[]  = { .5, .3, .2 }
#define SET_GAMMA  const scalar gamma[]  = { .7, .2, .1,      \
                                             .1, .7, .2,      \
                                             .2, .1, .7 }


#define SET_LOG_DELTA                   \
    SET_DELTA;                          \
    scalar ldelta[m_states];            \
    v_log (delta, m_states, ldelta)     \


#define SET_LOG_GAMMA                           \
    SET_GAMMA;                                  \
    scalar lgamma[m_states*m_states];           \
    v_log (gamma, m_states*m_states, lgamma)    \


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
