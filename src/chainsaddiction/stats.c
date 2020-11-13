#include "stats.h"


scalar
poisson_logpmf (
    const scalar qnt,
    const scalar lambda)
{
    return qnt * logl (lambda) - lgamma (qnt + 1) - lambda;
}


scalar
poisson_pmf (
    const scalar qnt,
    const scalar lambda)
{
    return expl (poisson_logpmf (lambda, qnt));
}


void
v_poisson_logpmf (
    const scalar *restrict qnts,
    const size_t n_qnts,
    const scalar *restrict means,
    const size_t m_means,
    scalar *restrict log_probs)
{

    for (size_t i = 0; i < n_qnts; i++)
    {
        for (size_t j = 0; j < m_means; j++)
        {
            log_probs[i*m_means+j] = poisson_logpmf (qnts[i], means[j]);
        }
    }
}
