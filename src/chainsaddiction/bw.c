#include <string.h>
#include "em.h"
#include "hmm.h"

int
ca_bw_pois_e_step (
    const DataSet *restrict _inp,
    const PoisParams *restrict _params,
    PoisHmmProbs *restrict probs)
{
    v_poisson_logpmf (_inp->data, _int->size, _params->lambda,
        _params->m_states, probs->lpp);
    log_forward_backward (_probs->lpp, _params->gamma, _params->delta,
        _params->m_states, data->size, probs->alpha, probs->beta);
}

int
ca_bw_pois_m_step ()
{}

int
ca_bw_pois (
    const DataSet *restrict _inp,
    PoisHmm *restrict _init)
{}
