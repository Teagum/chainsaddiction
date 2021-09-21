#include "pois-em.h"


int
pois_em (
    const size_t n_obs,
    const size_t m_states,
    const size_t iter_max,
    const scalar tol,
    const scalar *const restrict data,
    size_t *restrict n_iter,
    scalar *restrict llh,
    PoisProbs *const restrict probs,
    PoisParams *const restrict params)
{
    bool   err    = true;
    scalar score  = 0.0L;
    PoisParams *nlp = PoisParams_New (m_states);

    for (; *n_iter < iter_max; (*n_iter)++)
    {
        pois_e_step (n_obs, m_states, data, params->lambda, params->gamma,
                     params->delta, probs->lsdp, probs->lalpha, probs->lbeta,
                     probs->lcsp, llh);

        pois_m_step (n_obs, m_states, *llh, data, probs->lsdp, probs->lalpha,
                     probs->lbeta, probs->lcsp, params->gamma, nlp->lambda,
                     nlp->gamma, nlp->delta);

        score = score_update (nlp, params);
        if (score < tol)
        {
            err = false;
            break;
        }
        else
        {
            PoisParams_Copy (nlp, params);
        }
    }

    PoisParams_Delete (nlp);
    return err;
}


scalar
score_update (
    const PoisParams *const restrict new,
    const PoisParams *const restrict old)
{
    scalar score = 0L;
    for (size_t i = 0; i < new->m_states; i++)
    {
        score += fabsl (old->lambda[i] - new->lambda[i]);
        score += fabsl (expl (old->delta[i]) - expl (new->delta[i]));
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
    scalar *const restrict lcsp,
    scalar *const restrict llh)
{
    v_poisson_logpmf (n_obs, m_states, input_data, lambda, lsdp);
    log_fwbw (lsdp, lgamma, ldelta, m_states, n_obs, lalpha, lbeta);
    *llh = compute_log_likelihood (n_obs, m_states, lalpha);
    log_csprobs (n_obs, m_states, *llh, lalpha, lbeta, lcsp);
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
    const scalar *const restrict lcsp,
    const scalar *const restrict lgamma,
    scalar *const restrict new_lambda,
    scalar *const restrict new_lgamma,
    scalar *const restrict new_ldelta)
{
    pois_m_step_lambda (n_obs, m_states, data, lcsp, new_lambda);
    pois_m_step_gamma  (n_obs, m_states, llh, lsdp, lalpha, lbeta, lgamma, new_lgamma);
    pois_m_step_delta  (m_states, lcsp, new_ldelta);
}


void
pois_m_step_lambda (
    const size_t n_obs,
    const size_t m_states,
    const scalar *const restrict data,
    const scalar *const restrict lcsp,
    scalar *const restrict new_lambda)
{
    m_log_centroid_cols (lcsp, data, n_obs, m_states, new_lambda);
    vi_exp (m_states, new_lambda);
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
            new_lgamma[idx] = v_lse (n_obs-1, pr_buff) + lgamma[idx];
        }
        vi_log_normalize (m_states, new_lgamma+i*m_states);
    }
    MA_FREE (pr_buff);
}


void
pois_m_step_delta (
    const size_t m_states,
    const scalar *const restrict lcsp,
    scalar *const restrict new_ldelta)
{
    v_log_normalize (m_states, lcsp, new_ldelta);
}
