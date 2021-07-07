#include "test_pois_em.h"


bool
test__pois_e_step (void)
{
    enum { n_repeat_test = 10, m_states = 3, n_obs = 15 };

    SET_EARTHQUAKES_SHORT;
    SET_LAMBDA;
    SET_LOG_GAMMA;
    SET_LOG_DELTA;

    for (size_t n = 0; n < n_repeat_test; n++)
    {
        //const size_t m_states = (size_t) rnd_int (1, 10);
        //const size_t n_obs    = (size_t) rnd_int(1, 500);
        const size_t n_elem   = m_states * n_obs;

        //scalar *input  = MA_SCALAR_EMPTY (n_obs);
        scalar *lalpha = MA_SCALAR_EMPTY (n_elem);
        scalar *lbeta  = MA_SCALAR_EMPTY (n_elem);
        scalar *lsdp   = MA_SCALAR_EMPTY (n_elem);
        scalar llh     = 0;

        pois_e_step (n_obs, m_states, input, lambda, lgamma, ldelta,
                lalpha, lbeta, lsdp, &llh);

        //MA_FREE (input);
        MA_FREE (lalpha);
        MA_FREE (lbeta);
        MA_FREE (lsdp);
    }
    return false;
}
/*
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
*/

