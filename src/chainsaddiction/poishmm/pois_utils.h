#ifndef pois_utils_h
#define pois_utils_h

#include <math.h>
#include <stdlib.h>
#include "../scalar.h"
#include "libvmath.h"


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


/** Compute the conditional expectations. */
extern void
log_cond_expect (
    const size_t n_obs,
    const size_t m_states,
    const scalar *const restrict lalpha,
    const scalar *const restrict lbeta,
    const scalar llh,
    scalar *lcexpt);


/** Normalize buffer inplace in log domain. */
extern void
vi_log_normalize (
    size_t n_elem,
    scalar *const restrict buffer);


/** Normalize buffer in log domain. */
extern void
v_log_normalize (
    size_t n_elem,
    const scalar *const restrict src,
    scalar *const restrict dest);


#endif  /* pois_utils.h */