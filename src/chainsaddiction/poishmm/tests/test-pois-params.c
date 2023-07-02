#include "test-pois-params.h"


bool
test__PoisParams_New (void)
{
    const size_t m_states = rnd_size (1, 100);
    PoisParams *params = PoisParams_New (m_states);

    for (size_t i=0; i<m_states; i++) {
        if (fpclassify (params->lambda[i]) != FP_ZERO ||
            fpclassify (params->delta[i]) != FP_ZERO)
        {
            PoisParams_Delete (params);
            return UT_FAILURE;
        }
    }

    for (size_t i=0; i<m_states*m_states; i++) {
        if (fpclassify (params->gamma[i]) != FP_ZERO)
        {
            PoisParams_Delete (params);
            return UT_FAILURE;
        }
    }

    PoisParams_Delete (params);
    return UT_SUCCESS;
}


bool
test__PoisParams_NewFromFile (void)
{
    enum { FNAME_LEN_MAX = 100, N_FILES = 5};
    bool err = false;
    PoisParams *out = NULL;
    char test_files[][FNAME_LEN_MAX] = {
        "data/ppr1",
        "data/ppr2",
        "data/ppr3",
        "data/ppr4",
        "data/ppr5"
    };

    for (size_t fcnt = 0; fcnt < N_FILES; fcnt++) {
        out = PoisParams_NewFromFile (test_files[fcnt]);
        if (out == NULL) {
            fprintf (stderr, "Failed on file ``%s''.\n", test_files[fcnt]);
            err = true;
        }
    }
    return  err;
}
