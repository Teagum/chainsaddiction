#ifndef pois_utils_h
#define pois_utils_h

#include <math.h>
#include <stdlib.h>
#include "../scalar.h"
#include "../vmath.h"


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


/** Estimate log-likelihood given forward probabilities.
 *
 * \parma lalpha    Logarithm of the forward probabilities.
 * \param n_obs     Number of observations in the data set.
 * \param n_states  Number of HMM states.
 *
 * \return  Model log-likelihood.
 */
scalar
compute_log_likelihood (
    scalar *lalpha,
    size_t n_obs,
    size_t m_states);


#endif  /* pois_utils.h */
