#ifndef FWBW_H
#define FWBW_H

#include <float.h>
#include <math.h>
#include <stdlib.h>
#ifdef DEBUG
#include <stdio.h>
#endif
#include "stats.h"
#include "scalar.h"
#include "hmm.h"


/** Compute the forward and backward probabilies in log domain.
 */
int PoisHmm_FwBw(
        const long *restrict x,
        const size_t n,
        const size_t m,
        PoisParams *restrict params,
        scalar *restrict alpha,
        scalar *restrict beta,
        scalar *restrict pois_pr);

#endif  /* FWBW_H */
