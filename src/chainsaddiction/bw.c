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
ca_bw_pois_e_step (
    const DataSet *restrict inp,
    PoisHmm *restrict hmm,
    HmmProbs *restrict probs)
{
    v_poisson_logpmf (inp->data, inp->size, hmm->init->lambda,
        hmm->m_states, probs->lsd);

    log_forward_backward (probs->lsd, hmm->init->gamma, hmm->init->delta,
        hmm->m_states, inp->size, probs->lalpha, probs->lbeta);

    hmm->llh = log_likelihood_fw (probs->lalpha, inp->size, hmm->m_states);
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


