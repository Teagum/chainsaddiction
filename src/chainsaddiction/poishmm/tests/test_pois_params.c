#include "test_pois_params.h"


int main (void)
{
    SETUP;

    RUN_TEST (test__PoisParams_New);

    EVALUATE;
}


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
