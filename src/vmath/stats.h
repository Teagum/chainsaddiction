#ifndef stats_h
#define stats_h

#include <stdlib.h>
#include <math.h>
#include "config.h"


/** Log probability mass function of the Poisson distribution.
 *
 * \param qnt       Qunatile.
 * \param lambda    Expected value.
 */
scalar
poisson_logpmf (
    const scalar qnt,
    const scalar lambda);


/** Probability mass function of the Poisson distribution.
 *
 * \param qnt       Qunatile.
 * \param lambda    Expected value.
 */
scalar
poisson_pmf (
    const scalar qnt,
    const scalar lambda);


/** Vectorized log probability mass function of the Poisson distribution.
 *
 * \param[in]  qnts         Quantiles.
 * \param[in]  n_qnts       Number of qnts.
 * \param[in]  means        Expected values
 * \param[in]  m_means      Number of means.
 * \param[out] log_probs    Output buffer.
 */
void
v_poisson_logpmf (
    const size_t n_qnts,
    const size_t m_means,
    const scalar *restrict qnts,
    const scalar *restrict means,
          scalar *restrict log_probs);

#endif    /* stats_h */
