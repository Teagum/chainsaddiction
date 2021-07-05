#ifndef pois_em_h
#define pois_em_h

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "../dataset.h"
#include "../restrict.h"
#include "../scalar.h"
#include "fwbw.h"
#include "pois_hmm.h"


void
pois_e_step (
    const DataSet *const restrict inp,
    PoisHmm *const restrict phmm);


void
pois_m_step (
    const DataSet *const restrict inp,
    const HmmProbs *const restrict probs,
    const scalar llh);


void
pois_m_step_lambda (
    const DataSet *const restrict inp,
    const scalar *const restrict lstate_pr,
    const size_t m_states,
    scalar *restrict out);


#endif  /* pois_em */
