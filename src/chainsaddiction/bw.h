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
ph_bw_m_step ();


void
PoisHmm_BaumWelch (
    const DataSet *const restrict inp,
    PoisHmm *restrict hmm);


void ph_bw_update_lambda (
    const DataSet *const restrict inp,
    const HmmProbs *const restrict probs,
    const scalar llh,
    scalar *restrict buffer,
    scalar *restrict lambda_update);


#endif  /* BW_H */
