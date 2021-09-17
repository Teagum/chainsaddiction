#ifndef ca_utils_h
#define ca_utils_h

#include <math.h>
#include <stdlib.h>
#include "chainsaddiction.h"
#include "libvmath.h"


/* Compute Akaine Information criterion. */
scalar
compute_aic (
    const size_t m_states,
    const scalar llh);


/* Compute Bayes Information criterion. */
scalar
compute_bic (
    const size_t n_obs,
    const size_t m_states,
    const scalar llh);


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
    const size_t n_obs,
    const size_t m_states,
    const scalar *const restrict lalpha);


/** Compute the conditional expectations. */
extern void
log_csprobs (
    const size_t n_obs,
    const size_t m_states,
    const scalar llh,
    const scalar *const restrict lalpha,
    const scalar *const restrict lbeta,
          scalar *const restrict lcsp);


/** Local decoding */
extern int
local_decoding (
    const size_t n_obs,
    const size_t m_states,
    const scalar *lcsp,
    size_t *states);


/** Global decoding */
extern int
global_decoding (
    const size_t n_obs,
    const size_t m_states,
    const scalar *const restrict lgamma,
    const scalar *const restrict ldelta,
    const scalar *restrict lcsp,
    size_t *restrict states);


#endif  /* ca_utils.h */
