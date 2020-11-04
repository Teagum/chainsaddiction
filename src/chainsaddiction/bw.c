#include <string.h>
#include "bw.h"
#include "hmm.h"


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
    PoisHmm *restrict init)
{
    HmmProbs *probs = ca_NewHmmProbs (inp->size, init->m_states);

    ca_bw_pois_e_step (inp, init, probs);

    ca_FreeHmmProbs (probs);
}


