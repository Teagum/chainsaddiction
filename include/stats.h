#ifndef STATS_H
#define STATS_H

#include <stdlib.h>
#include <math.h>
#include "scalar.h"
#include "restrict.h"

/** Log probability mass function of the Poisson distribution.
 *
 * @param qnt    - Qunatile.
 * @param lambda - Expected value. 
 */
scalar
poisson_logpmf (
    const scalar qnt,
    const scalar lambda);


/** Probability mass function of the Poisson distribution.
 *
 * @param qnt    - Qunatile.
 * @param lambda - Expected value. 
 */
scalar
poisson_pmf (
    const scalar qnt,
    const scalar lambda);


/** Vectorized log probability mass function of the Poisson distribution.
 *
 * @param qunatiles   - Quantiles.
 * @param n_qnts      - Number of qnts.
 * @param means       - Expected values
 * @param m_means     - Number of means.
 * @param log_probs   - Output buffer.
 */
void
v_poisson_logpmf (
    const scalar *restrict qnts,
    const size_t n_qnts,
    const scalar *restrict means,
    const size_t m_means,
    scalar *restrict log_probs);

#endif    /* STATS_H */
