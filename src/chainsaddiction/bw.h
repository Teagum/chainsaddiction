#ifndef BW_H
#define BW_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "restrict.h"
#include "scalar.h"
#include "fwbw.h"
#include "hmm.h"
#include "dataset.h"


void
ph_bw_e_step (const DataSet *const restrict inp, PoisHmm *const restrict phmm);


void
ph_bw_m_step (
    const DataSet *const restrict inp,
    const HmmProbs *const restrict probs,
    const scalar llh);


void ph_bw_m_step_lambda (
    const DataSet *const restrict inp,
    const scalar *const restrict lstate_pr,
    const size_t m_states,
    scalar *restrict out);


#endif  /* BW_H */
