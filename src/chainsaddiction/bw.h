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
PoisHmm_BaumWelch_EStep (
    const DataSet *const restrict inp,
    PoisHmm *const restrict phmm);


void
ca_bw_pois_m_step ();


void
ca_bw_pois (
    const DataSet *restrict inp,
    PoisHmm *restrict hmm);


void ph_bw_update_lambda (
    const DataSet *const restrict inp,
    const HmmProbs *const restrict probs,
    const scalar llh,
    scalar *restrict buffer,
    scalar *restrict lambda_update);


#endif  /* BW_H */
