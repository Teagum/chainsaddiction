#ifndef BW_H
#define BW_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "restrict.h"
#include "fwbw.h"
#include "scalar.h"
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

void update_lambda (
    const DataSet *restrict inp,
    const scalar *restrict lalpha,
    const scalar *restrict lbeta,
    const size_t m_states,
    const scalar llh,
    scalar *buffer,
    scalar *lambda_update);

#endif  /* BW_H */
