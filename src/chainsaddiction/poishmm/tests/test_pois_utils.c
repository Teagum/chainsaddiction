#include "test_pois_utils.h"

bool test__local_decoding (void)
{
    enum {
        m_states = 3,
        n_obs    = 107
    };

    bool err = true;
    const char   path[]    = "../../../tests/data/earthquakes";
    const scalar ilambda[] = { 10L, 20L, 30L };
    const scalar igamma[]  =  {.8, .1, .1, .1, .8, .1, .1, .1, .8 };
    const scalar idelta[]  = { 1.0L/3L, 1.0L/3L, 1.0L/3L };
    const size_t expected[] = { 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1,
        1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

    DataSet *inp     = NULL;
    PoisHmm *hmm     = NULL;
    size_t *decoding = NULL;

    inp      = ds_NewFromFile (path);
    hmm      = PoisHmm_New (n_obs, m_states);
    decoding = VA_SIZE_ZEROS (hmm->n_obs);

    PoisHmm_Init(hmm, ilambda, igamma, idelta);
    PoisHmm_EstimateParams (hmm, inp);

    local_decoding (hmm->n_obs, hmm->m_states, hmm->probs->lcxpt, decoding);

    for (size_t i = 0; i < hmm->n_obs; i++)
    {
        err = (decoding[i] != expected[i]) ? true : false;
        if (err) break;
    }

    FREE (decoding);
    ds_FREE (inp);
    PoisHmm_Delete (hmm);
    return err;
}



bool test__global_decoding (void)
{
    enum {
        m_states = 3,
        n_obs    = 107
    };

    bool err = true;
    const char   path[]    = "../../../tests/data/earthquakes";
    const scalar ilambda[] = { 10L, 20L, 30L };
    const scalar igamma[]  =  {.8, .1, .1, .1, .8, .1, .1, .1, .8 };
    const scalar idelta[]  = { 1.0L/3L, 1.0L/3L, 1.0L/3L };
    const size_t expected[] = { 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1,
        1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

    DataSet *inp     = NULL;
    PoisHmm *hmm     = NULL;
    size_t *decoding = NULL;

    inp      = ds_NewFromFile (path);
    hmm      = PoisHmm_New (n_obs, m_states);
    decoding = VA_SIZE_ZEROS (hmm->n_obs);
    if (decoding ==NULL)
    {
        fprintf (stderr, "DECODING NULL\n");
        return true;
    }

    PoisHmm_Init(hmm, ilambda, igamma, idelta);
    PoisHmm_EstimateParams (hmm, inp);
    if (hmm == NULL)
    {
        fprintf (stderr, "hmm NULL\n");
        return true;
    }

    global_decoding (hmm->n_obs, hmm->m_states, hmm->params->gamma,
                     hmm->params->delta, hmm->probs->lsdp, decoding);
    print_matrix(n_obs, 1, decoding);
    free (decoding);
    PoisHmm_Delete (hmm);
    ds_FREE (inp);
    return err;
}
