#include "test-pois-utils.h"

bool test__local_decoding (void)
{
    enum {
        m_states = 3,
        n_obs    = 107
    };

    bool err = true;
    size_t *decoding = NULL;
    const char data_path[] = "../../../../tests/data/earthquakes/lcxpt";
    DataSet *lcsp = NULL;

    lcsp = ds_NewFromFile (data_path);
    if (lcsp == NULL) return UT_FAILURE;

    decoding = VA_SIZE_ZEROS (n_obs);
    if (inp == NULL)
    {
        ds_FREE (lcsp);
        return UT_FAILURE;
    }

    local_decoding (n_obs, m_states, lcsp, decoding);

    for (size_t i = 0; i < n_obs; i++)
    {
        err = (decoding[i] != lcsp->data[i]) ? true : false;
        if (err) break;
    }

    ds_FREE (lcsp);
    FREE (decoding);
    return err;
}



bool test__global_decoding (void)
{
    enum {
        m_states = 3,
        n_obs    = 107
    };

    bool err = false;
    const char   path_data[]    = "../../../../tests/data/earthquakes/dataset";
    const char   path_params[]  = "data/earthquakes.lprobs";
    const size_t xpc[] = {
        0, 0, 0, 0, 1, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
       0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2,
       2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    };

    DataSet     *inp      = ds_NewFromFile (path_data);
    if (inp == NULL) return UT_FAILURE;
    PoisParams  *params   = PoisParams_NewFromFile (path_params);
    size_t      *decoding = VA_SIZE_ZEROS (n_obs);
    scalar      *lsdp     = VA_SCALAR_EMPTY (n_obs*m_states);
    if (decoding == NULL || lsdp == NULL)
    {
        const char fmt[] = "(%s, %d) test__global_decoding:\nMemory error.";
        fprintf (stderr, fmt, __FILE__, __LINE__);
        UT_FAILURE;
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
