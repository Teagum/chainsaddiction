#include "test_pois_hmm.h"


int
main (void)
{
    SETUP;

    RUN_TEST (test__PoisHmm_NewProbs);
    RUN_TEST (test__PoisHmm_NewParams);
    RUN_TEST (test__PoisHmm_New);
    RUN_TEST (test__PoisHmm_Init);
    RUN_TEST (test__PoisHmm_InitRandom);
    RUN_TEST (test__PoisHmm_LogLikelihood);

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
test__PoisHmm_NewParams (void)
{
    enum { n_repeat_test = 10 };

    for (size_t n = 0; n < n_repeat_test; n++)
    {
        size_t m_states = (size_t) rnd_int (1, 100);
        PoisParams *params = PoisHmm_NewParams (m_states);

        for (size_t i=0; i<m_states; i++) {
            if (fpclassify (params->lambda[i]) != FP_ZERO ||
                fpclassify (params->delta[i]) != FP_ZERO)
            {
                PoisHmm_DeleteParams (params);
                return true;
            }
        }

        for (size_t i=0; i<m_states*m_states; i++) {
            if (fpclassify (params->gamma[i]) != FP_ZERO)
            {
                PoisHmm_DeleteParams (params);
                return true;
            }
        }
        PoisHmm_DeleteParams (params);
    }
    return false;
}


bool
test__PoisHmm_New (void)
{
    enum { n_repeat_test = 100 };

    for (size_t n = 0; n < n_repeat_test; n++)
    {
        size_t n_obs = rnd_int (1, 1000);
        size_t m_states = rnd_int (1, 200);
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
        size_t n_obs = (size_t) rnd_int (1, 1000);
        size_t m_states = (size_t) rnd_int (1, 30);
        PoisParams *params = PoisHmm_NewParams (m_states);
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

        PoisHmm_DeleteParams (params);
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
        size_t n_obs = (size_t) rnd_int (1, 1000);
        size_t m_states = (size_t) rnd_int (1, 30);
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
    const size_t n_obs = 4;
    const size_t m_states = 3;
    const scalar expected = 11.4076059644443803L;
    scalar lalpha[] = {0, 1, 2, 3, 4, 5, 6, 7, 8 ,9, 10, 11};

    PoisHmm *phmm = PoisHmm_New (n_obs, m_states);
    memcpy ((void *) phmm->probs->lalpha, (void *) lalpha, sizeof (scalar) * n_obs * m_states);
    PoisHmm_LogLikelihood (phmm);

    return !ASSERT_EQUAL (phmm->llh, expected);
}
