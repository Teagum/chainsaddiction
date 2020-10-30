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

int
ca_bw_pois_e_step (
    const DataSet *restrict _inp,
    const PoisParams *restrict _params,
    PoisHmmProbs *restrict probs);

int
ca_bw_pois_m_step ();

int
ca_bw_pois (
    const DataSet *restrict _inp,
    PoisHmm *restrict _init);

#endif  /* BW_H */
