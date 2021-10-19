#include "test-pois-hmm.h"


bool
test__PoisHmm_New (void)
{
    const size_t n_obs    = rnd_size (1, 1000);
    const size_t m_states = rnd_size (1, 200);
    PoisHmm *phmm = PoisHmm_New (n_obs, m_states);

    PoisHmm_Delete (phmm);
    return false;
}


bool
test__PoisHmm_Init (void)
{
    const size_t n_obs    = rnd_size (1, 1000);
    const size_t m_states = rnd_size (1, 30);
    PoisParams *params = PoisParams_New (m_states);
    PoisHmm *phmm = PoisHmm_New (n_obs, m_states);

    v_rnd_scalar (m_states, 1, 100, params->lambda);
    m_rnd_sample (m_states, m_states, params->gamma);
    v_rnd_sample (m_states, params->delta);
    PoisHmm_Init (phmm, params->lambda, params->gamma, params->delta);

    for (size_t i = 0; i < m_states; i++)
    {
        if (!isfinite(phmm->init->lambda[i])   ||
            !isfinite(phmm->init->delta[i])    ||
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
    return false;
}


bool
test__PoisHmm_InitRandom (void)
{
    const size_t n_obs    = rnd_size (1, 1000);
    const size_t m_states = rnd_size (1, 50);
    PoisHmm *phmm = PoisHmm_New (n_obs, m_states);

    PoisHmm_InitRandom (phmm);

    for (size_t i = 0; i < m_states; i++)
    {
        if (!isfinite(phmm->init->lambda[i])   ||
            !isfinite(phmm->init->delta[i])    ||
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
    return false;
}


bool
test__PoisHmm_InitRandom_sorted_lambda (void)
{
    const size_t n_obs    = rnd_size (1, 1000);
    const size_t m_states = rnd_size (1, 50);
    bool err = false;
    PoisHmm *phmm = PoisHmm_New (n_obs, m_states);

    PoisHmm_InitRandom (phmm);
    for (size_t i = 0; i < m_states - 1; i++)
    {
        if (phmm->init->lambda[i] > phmm->init->lambda[i+1])
        {
            err = true;
            break;
        }
    }

    PoisHmm_Delete (phmm);
    return err ? UT_FAILURE : UT_SUCCESS;
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


bool
test__PoisHmm_EstimateParams (void)
{
    const char path[] = "../../../../tests/data/earthquakes/earthquakes";
    enum { n_repeat_test = 1, m_states = 3, n_obs = 107 };
    const scalar ilambda[] = { 10L, 20L, 30L };
    const scalar igamma[]  =  {.8, .1, .1, .1, .8, .1, .1, .1, .8 };
    const scalar idelta[]  = { 1.0L/3L, 1.0L/3L, 1.0L/3L };

    DataSet *inp = DataSet_NewFromFile (path);
    if (inp == NULL) return true;

    PoisHmm *hmm = PoisHmm_New (n_obs, m_states);
    PoisHmm_Init(hmm, ilambda, igamma, idelta);

    PoisHmm_EstimateParams (hmm, inp);

    DataSet_Delete (inp);
    PoisHmm_Delete (hmm);
    return false;
}

bool
test__PoisHmm_ForwardProbabilities(void)
{
    const char data_path[] = "../../../../tests/data/earthquakes/earthquakes";
    const char params_path[] = "data/std3s.poisparams";

    DataSet *inp = DataSet_NewFromFile (data_path);
    PoisParams *params = PoisParams_NewFromFile (params_path);
    if ((inp == NULL) || (params == NULL))
    {
        DataSet_Delete (inp);
        PoisParams_Delete (params);
        return true;
    }

    PoisHmm *hmm = PoisHmm_New (inp->size, params->m_states);
    PoisHmm_Init (hmm, params->lambda, params->gamma, params->delta);
    v_poisson_logpmf (inp->size, hmm->m_states, inp->data, hmm->params->lambda,
                      hmm->probs->lsdp);

    PoisHmm_ForwardProbabilities (hmm);

    for (size_t i = 0; i < hmm->m_states * inp->size; i++)
    {
        if (!isfinite (hmm->probs->lalpha[i]))
        {
            return true;
        }
    }
    return false;
}


bool
test__PoisHmm_BackwardProbabilities (void)
{
    const char data_path[] = "../../../../tests/data/earthquakes/earthquakes";
    const char params_path[] = "data/std3s.poisparams";

    DataSet *inp = DataSet_NewFromFile (data_path);
    PoisParams *params = PoisParams_NewFromFile (params_path);
    if ((inp == NULL) || (params == NULL))
    {
        DataSet_Delete (inp);
        PoisParams_Delete (params);
        return true;
    }

    PoisHmm *hmm = PoisHmm_New (inp->size, params->m_states);
    PoisHmm_Init (hmm, params->lambda, params->gamma, params->delta);
    v_poisson_logpmf (inp->size, hmm->m_states, inp->data, hmm->params->lambda,
                      hmm->probs->lsdp);

    PoisHmm_BackwardProbabilities (hmm);

    for (size_t i = 0; i < (hmm->m_states * inp->size) - hmm->m_states; i++)
    {
        if (!isnormal (hmm->probs->lbeta[i]))
        {
            return true;
        }
    }
    return false;
}


bool
test__PoisHmm_ForwardBackward (void)
{
    const char data_path[] = "../../../../tests/data/earthquakes/earthquakes";
    const char params_path[] = "data/std3s.poisparams";

    DataSet *inp = DataSet_NewFromFile (data_path);
    PoisParams *params = PoisParams_NewFromFile (params_path);
    if ((inp == NULL) || (params == NULL))
    {
        DataSet_Delete (inp);
        PoisParams_Delete (params);
        return true;
    }

    PoisHmm *hmm = PoisHmm_New (inp->size, params->m_states);
    PoisHmm_Init (hmm, params->lambda, params->gamma, params->delta);
    v_poisson_logpmf (inp->size, hmm->m_states, inp->data, hmm->params->lambda,
                      hmm->probs->lsdp);

    PoisHmm_ForwardBackward (hmm);

    for (size_t i = 0; i < hmm->m_states * inp->size; i++)
    {
        if (!isfinite (hmm->probs->lalpha[i]) || !isfinite (hmm->probs->lbeta[i]))
        {
            return true;
        }
    }
    return false;
}
