#ifndef fwbw_h
#define fwbw_h

#include <math.h>
#include <stdlib.h>
#include "../dataset.h"
#include "../libma.h"
#include "../restrict.h"
#include "../scalar.h"
#include "libvmath.h"

/** Forward algorithm in log domain.
 *
 * Compute the foward probabilities alpha of HMM.
 *
 * \param lprobs   - Logarithm of state dependent probabilities.
 * \param lgamma   - Logarithm of the transition probability matrix.
 * \param ldelta   - Logarithm of the initial distribution.
 * \param m_states - Number of HMM states.
 * \param n_obs    - Number of observations.
 * \param alpha    - Output buffer of size n_vars * m_states.
 */
void
log_forward (
    const scalar *restrict lprobs,
    const scalar *const restrict lgamma,
    const scalar *const restrict ldelta,
    const size_t m_states,
    const size_t n_obs,
    scalar *lalpha);


/** Backward algorithm in log domain.
 *
 * Compute the backward probabilities beta of an HMM.
 *
 * \param lprobs   - Logarithm of state dependent probabilities.
 * \param lgamma   - Logarithm of the transition probability matrix.
 * \param m_states - Number of HMM states.
 * \param n_obs    - Number of observations.
 * \param lbeta    - Output buffer of size n_obs * m_states.
 */
void
log_backward (
    const scalar *restrict lprobs,
    const scalar *restrict lgamma,
    const size_t m_states,
    const size_t n_obs,
    scalar *lbeta);


/** Forward-Backward algorithm in log domain.
 *
 * Compute the forward and backward probabilities of an HMM.
 *
 * \param lprobs   - Logarithm of state dependent probabilities.
 * \param lgamma   - Logarithm of the transition probability matrix.
 * \param ldelta   - Logarithm of the initial distribution.
 * \param m_states - Number of HMM states.
 * \param n_obs    - Number of observations.
 * \param lalpha   - Output buffer of size n_obs * m_states.
 * \param lbeta    - Output buffer of size n_obs * m_states.
 */
void
log_fwbw (
    const scalar *restrict lprobs,
    const scalar *const restrict lgamma,
    const scalar *const restrict ldelta,
    const size_t m_states,
    const size_t n_obs,
    scalar *lalpha,
    scalar *lbeta);


#endif  /* fwbw_h */
