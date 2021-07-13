#include "test_pois_params.h"


bool
test__PoisParams_New (void)
{
    enum { n_repeat_test = 10 };

    for (size_t n = 0; n < n_repeat_test; n++)
    {
        size_t m_states = (size_t) RAND_INT (1, 100);
        PoisParams *params = PoisParams_New (m_states);

        for (size_t i=0; i<m_states; i++) {
            if (fpclassify (params->lambda[i]) != FP_ZERO ||
                fpclassify (params->delta[i]) != FP_ZERO)
            {
                PoisParams_Delete (params);
                return true;
            }
        }

        for (size_t i=0; i<m_states*m_states; i++) {
            if (fpclassify (params->gamma[i]) != FP_ZERO)
            {
                PoisParams_Delete (params);
                return true;
            }
        }
        PoisParams_Delete (params);
    }
    return false;
}


bool
test__PoisParams_NewFromFile (void)
{
    enum { FNAME_LEN_MAX = 100, N_FILES = 5};
    bool err = false;
    PoisParams *out = NULL;
    char test_files[][FNAME_LEN_MAX] = {
        "tests/data/ppr1",
        "tests/data/ppr2",
        "tests/data/ppr3",
        "tests/data/ppr4",
        "tests/data/ppr5"
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
