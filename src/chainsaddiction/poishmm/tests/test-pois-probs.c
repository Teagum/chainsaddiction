#include "test-pois-probs.h"

#define IS_ZERO(val) (fpclassify (val) == FP_ZERO)

bool
test__PoisProbs_New (void)
{
    const size_t n_obs    = rnd_size (1, 1000);
    const size_t m_states = rnd_size (1, 100);
    const size_t n_elem   = n_obs * m_states;

    PoisProbs *probs = PoisProbs_New (n_obs, m_states);
    scalar *dptr[] = { probs->lsdp,
        probs->lalpha,
        probs->lbeta,
        probs->lcsp
    };

    for (size_t i = 0; i < 4; i++) {
        for (size_t j = 0; j < n_elem; j++) {
            if (!IS_ZERO (dptr[i][j])) {
                PoisProbs_Delete (probs);
                return true;
            }
        }
    }

    PoisProbs_Delete (probs);
    return false;
}
