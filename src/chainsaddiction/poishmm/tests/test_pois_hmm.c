#include "test_pois_hmm.h"


int
main (void)
{
    SETUP;

    RUN_TEST (test__PoisHmm_New);
    RUN_TEST (test__PoisHmm_Init);
    RUN_TEST (test__PoisHmm_InitRandom);
    RUN_TEST (test__PoisHmm_LogLikelihood);

    EVALUATE;
}


bool
test__PoisHmm_New (void)
{
    enum { n_repeat_test = 100 };

    for (size_t n = 0; n < n_repeat_test; n++)
    {
        size_t n_obs = RAND_INT (1, 1000);
        size_t m_states = RAND_INT (1, 200);
        PoisHmm *phmm = PoisHmm_New (n_obs, m_states);

        PoisHmm_Delete (phmm);
    }
    return false;
}


bool
test__PoisHmm_Init (void)
{
    enum { n_repeat_test = 100 };
    for (size_t n = 0; n < n_repeat_test; n++)
    {
        size_t n_obs = (size_t) RAND_INT (1, 1000);
        size_t m_states = (size_t) RAND_INT (1, 30);
        PoisParams *params = PoisParams_New (m_states);
        PoisHmm *phmm = PoisHmm_New (n_obs, m_states);

        v_rnd (m_states, params->lambda);
        v_rnd (m_states*m_states, params->gamma);
        v_rnd (m_states, params->delta);
        PoisHmm_Init (phmm, params->lambda, params->gamma, params->delta);

        for (size_t i = 0; i < m_states; i++)
        {
            if (!isfinite(phmm->init->lambda[i]) ||
                !isfinite(phmm->init->delta[i]) ||
                !isfinite(phmm->params->lambda[i]) ||
                !isfinite(phmm->params->delta[i]))
            {
                return true;
            }

            for (size_t j = 0; j < m_states; j++)
            {
                size_t idx = i * m_states + j;
                if (!isfinite(phmm->init->gamma[idx]) ||
                    !isfinite(phmm->params->gamma[idx]))
                {
                    return true;
                }
            }
        }
        PoisParams_Delete (params);
        PoisHmm_Delete (phmm);
    }
    return false;
}


bool
test__PoisHmm_InitRandom (void)
{
    enum { n_repeat_test = 100 };
    for (size_t n = 0; n < n_repeat_test; n++)
    {
        size_t n_obs = (size_t) RAND_INT (1, 1000);
        size_t m_states = (size_t) RAND_INT (1, 30);
        PoisHmm *phmm = PoisHmm_New (n_obs, m_states);

        PoisHmm_InitRandom (phmm);
        for (size_t i = 0; i < m_states; i++)
        {
            if (!isfinite(phmm->init->lambda[i]) ||
                !isfinite(phmm->init->delta[i]) ||
                !isfinite(phmm->params->lambda[i]) ||
                !isfinite(phmm->params->delta[i]))
            {
                return true;
            }

            for (size_t j = 0; j < m_states; j++)
            {
                size_t idx = i * m_states + j;
                if (!isfinite(phmm->init->gamma[idx]) ||
                    !isfinite(phmm->params->gamma[idx]))
                {
                    return true;
                }
            }
        }
        PoisHmm_Delete (phmm);
    }
    return false;
}


bool
test__PoisHmm_LogLikelihood (void)
{
    bool err = true;
    const size_t n_obs = 4;
    const size_t m_states = 3;
    const scalar expected = 11.4076059644443803L;
    scalar lalpha[] = {0, 1, 2, 3, 4, 5, 6, 7, 8 ,9, 10, 11};

    PoisHmm *phmm = PoisHmm_New (n_obs, m_states);
    memcpy ((void *) phmm->probs->lalpha, (void *) lalpha, sizeof (scalar) * n_obs * m_states);
    PoisHmm_LogLikelihood (phmm);
    err = !ASSERT_EQUAL (phmm->llh, expected);

    PoisHmm_Delete (phmm);
    return err;
}
