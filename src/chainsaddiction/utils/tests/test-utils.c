#include <stdio.h>
#include "test-utils.h"

bool test__local_decoding (void)
{
#ifdef datapath
#undef datapath
#endif
#define datapath "../../../../tests/data/earthquakes/3s/"
    enum {
        m_states = 3,
        n_obs    = 107
    };

    bool     err  = true;
    size_t  *dec  = NULL;
    DataSet *lcsp = NULL;
    DataSet *xpct = NULL;

    lcsp = DataSet_NewFromFile (datapath "lcsp.txt");
    xpct = DataSet_NewFromFile (datapath "local-decoding.txt");
    dec = VA_SIZE_EMPTY (n_obs);
    if (lcsp == NULL || xpct == NULL || dec == NULL)
    {
        DataSet_Delete (lcsp);
        DataSet_Delete (xpct);
        FREE (dec);
        return UT_FAILURE;
    }
    local_decoding (n_obs, m_states, lcsp->data, dec);

    for (size_t i = 0; i < n_obs; i++)
    {
        err = ((size_t)xpct->data[i] != dec[i]) ? true : false;
        if (err) break;
    }

    DataSet_Delete (lcsp);
    DataSet_Delete (xpct);
    free (dec);
    return err;
}


bool test__global_decoding (void)
{
#ifdef datapath
#undef datapath
#endif
#define datapath "../../../../tests/data/earthquakes/3s/"

    enum {
        m_states = 3,
        n_obs    = 107
    };
    bool    err     = false;
    size_t  *dec    = NULL;
    DataSet *lcsp   = NULL;
    DataSet *lgamma = NULL;
    DataSet *ldelta = NULL;
    DataSet *xpct   = NULL;

    lcsp   = DataSet_NewFromFile (datapath "lcsp.txt");
    lgamma = DataSet_NewFromFile (datapath "lgamma.txt");
    ldelta = DataSet_NewFromFile (datapath "ldelta.txt");
    xpct   = DataSet_NewFromFile (datapath "global-decoding.txt");
    dec    = VA_SIZE_EMPTY (n_obs);
    if (lcsp == NULL || lgamma == NULL || ldelta == NULL || 
        xpct == NULL || dec == NULL)
    {
        DataSet_Delete (lcsp);
        DataSet_Delete (lgamma);
        DataSet_Delete (ldelta);
        DataSet_Delete (xpct);
        free (dec);
        return UT_FAILURE;
    }    

    global_decoding (n_obs, m_states, lgamma->data, ldelta->data,
                     lcsp->data, dec);

    for (size_t i = 0; i < n_obs; i++)
    {
        err = ((size_t)xpct->data[i] != dec[i]) ? true : false;
        if (err) break;
    }

    DataSet_Delete (lcsp);
    DataSet_Delete (lgamma);
    DataSet_Delete (ldelta);
    DataSet_Delete (xpct);
    free (dec);
    return err;
}
