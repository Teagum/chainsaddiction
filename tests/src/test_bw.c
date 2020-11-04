#include "test_bw.h"

bool
test_ca_bw_pois_e_step (void)
{
    PoisHmm hmm = {
        3, 100, 1000, 1e-6, 0.0L, 0.0L, 0.0L,
        PoisHmm_ParamsFromFile ("data/params_3s"),
        NULL
    };

    unsigned long eq[] = {13, 14, 8, 10, 16, 26, 32, 27, 18, 32, 36, 24, 22, 23, 22, 18, 25, 21, 21,
    14, 8, 11, 14, 23, 18, 17, 19, 20, 22, 19, 13, 26, 13, 14, 22, 24, 21, 22,
    26, 21, 23, 24, 27, 41, 31, 27, 35, 26, 28, 36, 39, 21, 17, 22, 17, 19, 15,
    34, 10, 15, 22, 18, 15, 20, 15, 22, 19, 16, 30, 27, 29, 23, 20, 16, 21, 21,
    25, 16, 18, 15, 18, 14, 10, 15, 8, 15, 6, 11, 8, 7, 18, 16, 13, 12, 13, 20,
    15, 16, 12, 18, 15, 16, 13, 15, 16, 11, 11 };
    DataSet inp = { eq, 107 };
    HmmProbs *probs = ca_NewHmmProbs (inp.size, hmm.m_states);

    ca_bw_pois_e_step (&inp, &hmm, probs);
    /*
    v_poisson_logpmf (inp.data, inp.size, hmm.init->lambda, hmm.m_states, probs->lsd);

    log_forward_backward (probs->lsd, hmm.init->gamma, hmm.init->delta,
        hmm.m_states, inp.size, probs->lalpha, probs->lbeta);

    hmm.llh = log_likelihood_fw (probs->lalpha, inp.size, hmm.m_states);
    */
    puts("\n");
    for (size_t i = 0; i < inp.size; i++)
    {
        printf ("[%3zu] ", i);
        for (size_t j = 0; j < hmm.m_states; j++)
        {
            printf ("%10.5Lf\t", probs->lalpha[i*hmm.m_states+j]);
        }
        printf ("\n");
    }
    printf ("llh: %Lf\n", hmm.llh);
    PoisHmm_PrintParams (hmm.init, hmm.m_states);
    PoisHmm_FreeParams (hmm.init);
    ca_FreeHmmProbs (probs);
    return false;
}

