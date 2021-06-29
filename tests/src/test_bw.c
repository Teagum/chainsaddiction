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
    enum { n_repeat_test = 100 };

    for (size_t n = 0; n < n_repeat_test; n++)
    {
        size_t m_states = (size_t) rnd_int (1, 100);
        DataSet *inp = ds_NewFromFile ("tests/data/earthquakes");
        PoisHmm *phmm = PoisHmm_New (inp->size, m_states);
        scalar *lstate_pr = MA_SCALAR_ZEROS (phmm->n_obs * phmm->m_states);
        scalar *new_lambda = MA_SCALAR_ZEROS (phmm->m_states);

        PoisHmm_InitRandom (phmm);
        ph_bw_e_step (inp, phmm);
        PoisHmm_LogStateProbs (phmm->probs, phmm->llh, lstate_pr);
        ph_bw_m_step_lambda (inp, lstate_pr, m_states, new_lambda);

        for (size_t i = 0; i < m_states; i++)
        {
            if (!isnormal (new_lambda[i]))
            {
                return true;
            }
        }
        ds_FREE (inp);
        PoisHmm_Delete (phmm);
        MA_FREE (lstate_pr);
        MA_FREE (new_lambda);
    }
    return false;
}
