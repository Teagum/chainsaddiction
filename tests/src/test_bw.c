#include "test_bw.h"


int
main (void)
{
    SETUP;

    RUN_TEST (test__ph_bw_m_step_lambda);

    EVALUATE;
}


bool
test__ph_bw_m_step_lambda (void)
{
    size_t m_states = 3;

    DataSet *inp = ds_NewFromFile ("tests/data/earthquakes");
    PoisHmm *phmm = PoisHmm_New (inp->size, m_states);
    scalar *lstate_pr = MA_SCALAR_ZEROS (phmm->n_obs * phmm->m_states);
    scalar *new_lambda = MA_SCALAR_ZEROS (phmm->m_states);

    ca_ph_InitRandom (phmm);
    phmm->llh = PoisHmm_LogLikelihood (phmm->probs->lalpha, inp->size, phmm->m_states);
    PoisHmm_LogStateProbs (phmm->probs, phmm->llh, lstate_pr);
    ph_bw_m_step_lambda (inp, lstate_pr, m_states, new_lambda);

    puts ("");
    for (size_t i = 0; i < m_states; i++)
    {
        printf ("[%3zu] %10.5Lf %10.5Lf\n", i, phmm->init->lambda[i], new_lambda[i]);
    }
    ds_FREE (inp);
    PoisHmm_Delete (phmm);
    MA_FREE (lstate_pr);
    MA_FREE (new_lambda);
    return false;
}
