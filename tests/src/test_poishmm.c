#include "test_poishmm.h"


int
main (void)
{
    SETUP;

    RUN_TEST (test__PoisHmm_NewProbs);
    RUN_TEST (test_ca_ph_NewParams);
    RUN_TEST (test_ca_ph_NewHmm);
    RUN_TEST (test_ca_ph_InitParams);
    RUN_TEST (test_ca_log_likelihood);

    EVALUATE;
}


bool
test__PoisHmm_NewProbs (void)
{
    enum { n_repeat_test = 10 };

    for (size_t n = 0; n < n_repeat_test; n++)
    {
        size_t n_obs = (size_t) rnd_int (1, 10000);
        size_t m_states = (size_t) rnd_int (1, 100);
        size_t n_elem = n_obs * m_states;

        HmmProbs *probs = PoisHmm_NewProbs (n_obs, m_states);
        scalar *dptr[] = { probs->lsd, probs->lalpha, probs->lbeta };

        for (size_t i=0; i<3; i++) {
            for (size_t j=0; j<n_elem; j++) {
                if (fpclassify (dptr[i][j]) != FP_ZERO) {
                    PoisHmm_DeleteProbs (probs);
                    return true;
                }
            }
        }
        PoisHmm_DeleteProbs (probs);
    }
    return false;
}


bool
test_ca_ph_NewParams (void)
{
    enum { n_repeat_test = 10 };

    for (size_t n = 0; n < n_repeat_test; n++)
    {
        size_t m_states = (size_t) rnd_int (1, 100);
        PoisParams *params = ca_ph_NewParams (m_states);

        for (size_t i=0; i<m_states; i++) {
            if (fpclassify (params->lambda[i]) != FP_ZERO ||
                fpclassify (params->delta[i]) != FP_ZERO)
            {
                ca_ph_FREE_PARAMS (params);
                return true;
            }
        }

        for (size_t i=0; i<m_states*m_states; i++) {
            if (fpclassify (params->gamma[i]) != FP_ZERO)
            {
                ca_ph_FREE_PARAMS (params);
                return true;
            }
        }
        ca_ph_FREE_PARAMS (params);
    }
    return false;
}


bool
test_ca_ph_NewHmm (void)
{
    enum { n_repeat_test = 100 };

    for (size_t n = 0; n < n_repeat_test; n++)
    {
        size_t n_obs = rnd_int (1, 1000);
        size_t m_states = rnd_int (1, 200);
        PoisHmm *phmm = ca_ph_NewHmm (n_obs, m_states);

        ca_ph_FREE_HMM (phmm);
    }
    return false;
}


bool
test_ca_ph_InitParams (void)
{
    enum { n_repeat_test = 100 };
    for (size_t n = 0; n < n_repeat_test; n++)
    {
        size_t n_obs = 10; //(size_t) rnd_int (1, 1000);
        size_t m_states = 3; //(size_t) rnd_int (1, 30);
        scalar *lambda = MA_SCALAR_ZEROS (m_states);
        scalar *gamma = MA_SCALAR_ZEROS (m_states*m_states);
        scalar *delta = MA_SCALAR_ZEROS (m_states);
        PoisHmm *phmm = ca_ph_NewHmm (n_obs, m_states);

        v_rnd (m_states, lambda);
        v_rnd (m_states*m_states, gamma);
        v_rnd (m_states, delta);
        ca_ph_InitParams (phmm, lambda, gamma, delta);

        for (size_t i = 0; i < m_states; i++)
        {
            if (!isnormal(phmm->init->lambda[i]) ||
                !isnormal(phmm->init->delta[i]) ||
                !isnormal(phmm->params->lambda[i]) ||
                !isnormal(phmm->params->delta[i]))
            {
                return true;
            }

            for (size_t j = 0; j < m_states; j++)
            {
                size_t idx = i * m_states + j;
                if (!isnormal(phmm->init->gamma[idx]) ||
                    !isnormal(phmm->params->gamma[idx]))
                {
                    return true;
                }
            }
        }

        MA_FREE (lambda);
        MA_FREE (gamma);
        MA_FREE (delta);
        ca_ph_FREE_HMM (phmm);
    }
    return false;
}


bool
test_ca_log_likelihood (void)
{
    const scalar expected = 11.4076059644443803L;
    scalar a[] = {0, 1, 2, 3, 4, 5, 6, 7, 8 ,9, 10, 11};
    scalar res = ca_log_likelihood (a, 4, 3);
    return !ASSERT_EQUAL (res, expected);
}
