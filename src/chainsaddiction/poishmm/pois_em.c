#include <string.h>
#include "pois_em.h"


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
/*
void
pois_e_step (const DataSet *const restrict inp, PoisHmm *const restrict phmm)
{
    const size_t m_states = phmm->m_states;
    const size_t n_obs = phmm->n_obs;
    PoisParams *params = phmm->params;
    PoisProbs *probs = phmm->probs;

    v_poisson_logpmf (inp->data, n_obs, params->lambda, m_states, probs->lsdp);

    log_fwbw (probs->lsdp, params->gamma, params->delta,
        m_states, n_obs, probs->lalpha, probs->lbeta);

    PoisHmm_LogLikelihood (phmm);
}
*/

void
pois_m_step (
    const size_t n_obs,
    const size_t m_states,
    const scalar llh,
    const scalar *const restrict input_data,
    const scalar *const restrict lalpha,
    const scalar *const restrict lbeta,
    const scalar *const restrict lgamma,
    scalar *const restrict new_lambda,
    scalar *const restrict new_lgamma,
    scalar *const restrict new_ldelta)
{

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
