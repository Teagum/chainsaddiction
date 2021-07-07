#ifndef pois_em_h
#define pois_em_h

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "../dataset.h"
#include "../restrict.h"
#include "../scalar.h"
#include "pois_hmm.h"
#include "pois_fwbw.h"


void
pois_e_step (
    const size_t n_obs,
    const size_t m_states,
    const scalar *const restrict input_data,
    const scalar *const restrict lambda,
    const scalar *const restrict lgamma,
    const scalar *const restrict ldelta,
    scalar *const restrict lsdp,
    scalar *const restrict lalpha,
    scalar *const restrict lbeta,
    scalar *const restrict lcxpt,
    scalar *const restrict llh);

/*
void
pois_e_step (
    const DataSet *const restrict inp,
    PoisHmm *const restrict phmm);
*/

void
pois_m_step (
    const DataSet *const restrict inp,
    const PoisProbs *const restrict probs,
    const scalar llh);


void
pois_m_step_lambda (
    const DataSet *const restrict inp,
    const scalar *const restrict lstate_pr,
    const size_t m_states,
    scalar *restrict out);


#endif  /* pois_em */
