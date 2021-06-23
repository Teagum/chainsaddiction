#include "test_bw.h"


int
main (void)
{
    SETUP;
    RUN_TEST (test_ph_bw_update_lambda);
    EVALUATE;
}


bool
test_ph_bw_update_lambda (void)
{
    size_t m_states = 3;

    DataSet *inp = ds_NewFromFile ("tests/data/earthquakes");
    PoisHmm *phmm = ca_ph_NewHmm (inp->size, m_states);
    scalar *buffer = MA_SCALAR_ZEROS (phmm->n_obs * phmm->m_states);

    ca_ph_InitRandom (phmm);
    phmm->llh = ca_log_likelihood (phmm->probs->lalpha, inp->size, phmm->m_states);
    ph_bw_update_lambda (inp, phmm->probs, phmm->llh, buffer, phmm->params->lambda);

    ds_FREE (inp);
    ca_ph_FREE_HMM (phmm);
    MA_FREE (buffer);
    return false;
}
