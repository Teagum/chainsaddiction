#include "test_hmm.h"


int
main (void)
{
    SETUP;

    RUN_TEST (test_ca_NewHmmProbs);

    EVALUATE;
}


bool
test_ca_NewHmmProbs (void)
{
    enum { n_repeat_test = 10 };

    for (size_t n = 0; n < n_repeat_test; n++)
    {
        size_t n_obs = (size_t) rnd_int (1, 1000000);
        size_t m_states = (size_t) rnd_int (1, 100);
        size_t n_elem = n_obs * m_states;

        HmmProbs *probs = ca_NewHmmProbs (n_obs, m_states);
        scalar *dptr[] = { probs->lsd, probs->lalpha, probs->lbeta };

        for (size_t i=0; i<3; i++) {
            for (size_t j=0; j<n_elem; j++) {
                if (dptr[i][j] < 0L || dptr[i][j] > 0L) {
                    ca_FREE_HMM_PROBS (probs);
                    return true;
                }
            }
        }
        ca_FREE_HMM_PROBS (probs);
    }
    return false;
}


bool
test_log_likelihood_fw (void)
{
    const scalar expected = 11.4076059644443803L;
    scalar a[] = {0, 1, 2, 3, 4, 5, 6, 7, 8 ,9, 10, 11};
    scalar res = log_likelihood_fw (a, 4, 3);
    return !ASSERT_EQUAL (res, expected);
}
