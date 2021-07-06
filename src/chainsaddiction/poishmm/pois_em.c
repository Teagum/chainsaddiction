#include <string.h>
#include "pois_em.h"


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


void
pois_m_step (
    const DataSet *const restrict inp,
    const PoisProbs *const restrict probs,
    const scalar llh)
{
    size_t n_elem = probs->m_states * probs->n_obs;
    scalar *lstate_pr = MA_SCALAR_EMPTY (n_elem);
}


void pois_m_step_lambda (
    const DataSet *const restrict inp,
    const scalar *const restrict lstate_pr,
    const size_t m_states,
    scalar *restrict out)
{
    m_lse_centroid_rows (lstate_pr, inp->data, inp->size, m_states, out);
}
