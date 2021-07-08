#include <string.h>
#include "pois_em.h"


scalar
score_update (
    const PoisParams *const restrict new,
    const PoisParams *const restrict old)
{
    scalar score = 0L;
    for (size_t i = 0; i < new->m_states; i++)
    {
        score += fabsl (old->lambda[i] - new->lambda[i]);
        score += fabsl (old->delta[i] - new->delta[i]);
        for (size_t j = 0; j < new->m_states; j++)
        {
            size_t idx = i * new->m_states + j;
            score += fabsl (expl (old->gamma[idx]) - expl (new->gamma[idx]));
        }
    }
    return score;
}


void
pois_e_step (
    const size_t n_obs,
    const size_t m_states,
    const scalar *const restrict input_data,
    const scalar *const restrict lambda,
    const scalar *const restrict lgamma,
    const scalar *const restrict ldelta,
    scalar *const restrict lsdp,
    scalar *const restrict lalpha,
    scalar *const restrict lbeta,
    scalar *const restrict lcxpt,
    scalar *const restrict llh)
{
    v_poisson_logpmf (input_data, n_obs, lambda, m_states, lsdp);
    log_fwbw (lsdp, lgamma, ldelta, m_states, n_obs, lalpha, lbeta);
    *llh = compute_log_likelihood (lalpha, n_obs, m_states);
    log_cond_expect (n_obs, m_states, lalpha, lbeta, *llh, lcxpt);
}


void
pois_m_step (
    const size_t n_obs,
    const size_t m_states,
    const scalar llh,
    const scalar *const restrict data,
    const scalar *const restrict lsdp,
    const scalar *const restrict lalpha,
    const scalar *const restrict lbeta,
    const scalar *const restrict lgamma,
    const scalar *const restrict lcxpt,
    scalar *const restrict new_lambda,
    scalar *const restrict new_lgamma,
    scalar *const restrict new_ldelta)
{
    pois_m_step_lambda (n_obs, m_states, data, lcxpt, new_lambda);
    pois_m_step_gamma  (n_obs, m_states, llh, lsdp, lalpha, lbeta, lgamma, new_lgamma);
    pois_m_step_delta  (m_states, lcxpt, new_ldelta);
}


void
pois_m_step_lambda (
    const size_t n_obs,
    const size_t m_states,
    const scalar *const restrict data,
    const scalar *const restrict lcxpt,
    scalar *const restrict new_lambda)
{
    m_lse_centroid_rows (lcxpt, data, n_obs, m_states, new_lambda);
}


void
pois_m_step_gamma (
    const size_t n_obs,
    const size_t m_states,
    const scalar llh,
    const scalar *const restrict lsdp,
    const scalar *const restrict lalpha,
    const scalar *const restrict lbeta,
    const scalar *const restrict lgamma,
    scalar *const restrict new_lgamma)
{
    scalar *pr_buff = MA_SCALAR_ZEROS (n_obs-1);
    for (size_t i = 0; i < m_states; i++)
    {
        for (size_t j = 0; j < m_states; j++)
        {
            size_t idx = i * m_states + j;
            for (size_t n = 0; n < n_obs-1; n++)
            {
                size_t a = n * m_states + i;
                size_t b = (n + 1) * m_states + j;
                pr_buff[n] = lalpha[a] + lbeta[b] + lsdp[b] - llh;
            }
            new_lgamma[idx] = v_lse (pr_buff, n_obs-1) + lgamma[idx];
        }
        vi_log_normalize (m_states, new_lgamma+i*m_states);
    }
    MA_FREE (pr_buff);
}


void
pois_m_step_delta (
    const size_t m_states,
    const scalar *const restrict lcxpt,
    scalar *const restrict new_ldelta)
{
    v_log_normalize (m_states, lcxpt, new_ldelta);
}
