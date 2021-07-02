#include "test_pois_probs.h"


bool
test__PoisProbs_New (void)
{
    enum { n_repeat_test = 10 };

    for (size_t n = 0; n < n_repeat_test; n++)
    {
        size_t n_obs = (size_t) RAND_INT (1, 1000);
        size_t m_states = (size_t) RAND_INT (1, 100);
        size_t n_elem = n_obs * m_states;

        PoisProbs *probs = PoisProbs_New (n_obs, m_states);
        scalar *dptr[] = { probs->lsdp,
            probs->lalpha,
            probs->lbeta,
            probs->lcxpt
        };

        for (size_t i = 0; i < 3; i++) {
            for (size_t j = 0; j < n_elem; j++) {
                if (!IS_ZERO (dptr[i][j])) {
                    PoisProbs_Delete (probs);
                    return true;
                }
            }
        }
        PoisProbs_Delete (probs);
    }
    return false;
}
