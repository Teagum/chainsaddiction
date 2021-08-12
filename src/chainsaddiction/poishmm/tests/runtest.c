#include "test_pois_probs.h"
#include "test_pois_params.h"
#include "test_pois_hmm.h"
#include "test_pois_em.h"
#include "test_pois_utils.h"

int
main (void)
{
    SETUP;
/*
    RUN_TEST (test__PoisParams_New);
    RUN_TEST (test__PoisParams_NewFromFile);

    RUN_TEST (test__PoisProbs_New);

    RUN_TEST (test__score_update);
    RUN_TEST (test__pois_e_step);
    RUN_TEST (test__pois_m_step_lambda);
    RUN_TEST (test__pois_m_step_gamma);
    RUN_TEST (test__pois_m_step_delta);

    RUN_TEST (test__PoisHmm_New);
    RUN_TEST (test__PoisHmm_Init);
    RUN_TEST (test__PoisHmm_InitRandom);
    RUN_TEST (test__PoisHmm_LogLikelihood);
    RUN_TEST (test__PoisHmm_ForwardProbabilities);
    RUN_TEST (test__PoisHmm_BackwardProbabilities);
    RUN_TEST (test__PoisHmm_ForwardBackward);
    RUN_TEST (test__PoisHmm_EstimateParams);
    RUN_TEST (test__local_decoding);
*/
    RUN_TEST (test__global_decoding);

    EVALUATE;
}
