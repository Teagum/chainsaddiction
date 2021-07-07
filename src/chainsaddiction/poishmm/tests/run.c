#include "test_pois_probs.h"
#include "test_pois_params.h"
#include "test_pois_hmm.h"
#include "test_pois_em.h"

int
main (void)
{
    SETUP;

    RUN_TEST (test__PoisParams_New);

    RUN_TEST (test__PoisProbs_New);

    RUN_TEST (test__PoisHmm_New);
    RUN_TEST (test__PoisHmm_Init);
    RUN_TEST (test__PoisHmm_InitRandom);
    RUN_TEST (test__PoisHmm_LogLikelihood);
    RUN_TEST (test__pois_e_step);
    RUN_TEST (test__pois_m_step_gamma);

    EVALUATE;
}
