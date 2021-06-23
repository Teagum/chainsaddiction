#include <string.h>
#include "bw.h"

void ph_bw_update_lambda (
    const DataSet *const restrict inp,
    const HmmProbs *const restrict probs,
    const scalar llh,
    scalar *restrict buffer,
    scalar *restrict lambda_update)
{
    mm_add_s (probs->lalpha, probs->lbeta, probs->m_states * probs->n_obs,
            llh, buffer);
    m_lse_centroid_rows (buffer, inp->data, inp->size, probs->m_states,
            lambda_update);
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
ph_bw_m_step ()
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


