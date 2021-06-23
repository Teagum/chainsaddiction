#include <string.h>
#include "bw.h"

void update_lambda (
    const DataSet *restrict inp,
    const scalar *restrict lalpha,
    const scalar *restrict lbeta,
    const size_t m_states,
    const scalar llh,
    scalar *buffer,
    scalar *lambda_update)
{
    mm_add_s (lalpha, lbeta, m_states * inp->size, llh, buffer);
    m_lse_centroid_rows (buffer, (scalar *) inp->data, inp->size, m_states, lambda_update);
}

void
PoisHmm_BaumWelch_EStep (
    const DataSet *restrict inp,
    PoisHmm *restrict phmm)
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
ca_bw_pois_m_step ()
{}


void
ca_bw_pois (
    const DataSet *restrict inp,
    PoisHmm *restrict hmm)
{
    HmmProbs *probs = ca_NewHmmProbs (inp->size, hmm->m_states);

    ca_bw_pois_e_step (inp, hmm, probs);
    ca_bw_pois_m_step ();

    ca_FreeHmmProbs (probs);
}


