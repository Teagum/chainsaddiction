#ifndef FWBW_H
#define FWBW_H

#include <math.h>
#include <stdlib.h>
#include "restrict.h"
#include "scalar.h"
#include "stats.h"
#include "utilities.h"
#include "vmath.h"


/** Forward algorithm in log domain.
 *
 * Compute the foward probabilities alpha of HMM.
 *
 * @param lprobs   - Logarithm of state dependent probabilities.
 * @param lgamma   - Logarithm of the transition probability matrix.
 * @param ldelta   - Logarithm of the initial distribution.
 * @param n_vars   - Number of observations.
 * @param m_states - Number of HMM states.
 * @param alpha    - Output buffer of size n_vars * m_states.
 */
void 
log_forward (
    const scalar *restrict lprobs,
    const scalar *restrict lgamma,
    const scalar *restrict ldelta,
    const size_t n_vars,
    const size_t m_states,
    scalar *alpha);


/** Backward algorithm in log domain.
 *
 * Compute the backward probabilities beta of an HMM.
 *
 * @param lprobs   - Logarithm of state dependent probabilities.
 * @param lgamma   - Logarithm of the transition probability matrix.
 * @param m_states - Number of HMM states.
 * @param n_obs    - Number of observations.
 * @param beta     - Output buffer of size n_vars * m_states.
 */
void
log_backward (
    const scalar *restrict lprobs,
    const scalar *restrict lgamma,
    const size_t m_states,
    const size_t n_obs,
    scalar *beta);


/** Forward/Backward algorithm in log domain.
 *
 * Compute the forward and backward probabilities of an HMM.
 *
 * @param lprobs   - Logarithm of state dependent probabilities.
 * @param lgamma   - Logarithm of the transition probability matrix.
 * @param ldelta   - Logarithm of the initial distribution.
 * @param m_states - Number of HMM states.
 * @param n_obs    - Number of observations.
 * @param alpha    - Output buffer of size n_vars * m_states.
 * @param beta     - Output buffer of size n_vars * m_states.
 */
void 
log_forward_backward (
    const scalar *restrict lprobs,
    const scalar *restrict lgamma,
    const scalar *restrict ldelta,
    const size_t m_states,
    const size_t n_obs,
    scalar *alpha,
    scalar *beta);


#endif  /* FWBW_H */
