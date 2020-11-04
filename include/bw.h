#ifndef BW_H
#define BW_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "restrict.h"
#include "fwbw.h"
#include "scalar.h"
#include "hmm.h"
#include "utilities.h"

void
ca_bw_pois_e_step (
    const DataSet *restrict inp,
    PoisHmm *restrict hmm,
    HmmProbs *restrict probs);

void
ca_bw_pois_m_step ();

void
ca_bw_pois (
    const DataSet *restrict inp,
    PoisHmm *restrict init);

#endif  /* BW_H */
