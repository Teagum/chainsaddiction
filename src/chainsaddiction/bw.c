#include <string.h>
#include "bw.h"

void ph_bw_m_step_lambda (
    const DataSet *const restrict inp,
    const scalar *const restrict lstate_pr,
    const size_t m_states,
    scalar *restrict out)
{
    m_lse_centroid_rows (lstate_pr, inp->data, inp->size, m_states, out);
}


void
ph_bw_e_step (const DataSet *const restrict inp, PoisHmm *const restrict phmm)
{
    const size_t m_states = phmm->m_states;
    const size_t n_obs = phmm->n_obs;
    PoisParams *params = phmm->params;
    HmmProbs *probs = phmm->probs;

    v_poisson_logpmf (inp->data, n_obs, params->lambda, m_states, probs->lsd);

    log_fwbw (probs->lsd, params->gamma, params->delta,
        m_states, n_obs, probs->lalpha, probs->lbeta);

    phmm->llh = ca_log_likelihood (probs->lalpha, n_obs, m_states);
}


void
ph_bw_m_step (
    const DataSet *const restrict inp,
    const HmmProbs *const restrict probs,
    const scalar llh)
{
    size_t n_elem = probs->m_states * probs->n_obs;
    scalar *lstate_pr = MA_SCALAR_EMPTY (n_elem);

    /* the fourth argument should probably be `-llh'. */
    mm_add_s (probs->lalpha, probs->lbeta, n_elem, llh, lstate_pr);
}



void
PoisHmm_BaumWelch (
    const DataSet *const restrict inp,
    PoisHmm *const restrict phmm)
{}


