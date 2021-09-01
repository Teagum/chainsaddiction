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
    const size_t xpc[] = { 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1,
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
        err = (decoding[i] != xpc[i]) ? true : false;
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

    bool err = false;
    const char   path_data[]    = "../../../tests/data/earthquakes";
    const char   path_params[]  = "tests/data/earthquakes.lprobs";
    const size_t xpc[] = {
        0, 0, 0, 0, 1, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
       0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2,
       2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    };

    DataSet     *inp      = ds_NewFromFile (path_data);
    PoisParams  *params   = PoisParams_NewFromFile (path_params);
    size_t      *decoding = VA_SIZE_ZEROS (n_obs);
    scalar      *lsdp     = VA_SCALAR_EMPTY (n_obs*m_states);
    if (decoding == NULL || lsdp == NULL)
    {
        const char fmt[] = "(%s, %d) test__global_decoding:\nMemory error.";
        fprintf (stderr, fmt, __FILE__, __LINE__);
        VM_RETURN_FAILURE;
    }

    v_poisson_logpmf (inp->data, n_obs, params->lambda, m_states, lsdp);
    global_decoding (n_obs, m_states, params->gamma, params->delta,
                     lsdp, decoding);

    for (size_t i = 0; i < n_obs; i++)
    {
        err |= (xpc[i] != decoding[i]) ? true : false;
    }

    FREE (decoding);
    PoisParams_Delete (params);
    ds_FREE (inp);
    FREE (lsdp);
    return err ? VM_FAILURE : VM_SUCCESS;
}
