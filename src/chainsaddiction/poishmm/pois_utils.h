#ifndef pois_utils_h
#define pois_utils_h

#include <math.h>
#include <stdlib.h>
#include "../scalar.h"


/* Compute Akaine Information criterion. */
scalar
compute_aic (
    scalar llh,
    size_t m_states);


/* Compute Bayes Information criterion. */
scalar
compute_bic (
    scalar llh,
    size_t n_obs,
    size_t m_states);


#endif  /* pois_utils.h */
