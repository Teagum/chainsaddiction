#include "test-pois-probs.h"
#include "test-pois-params.h"
#include "test-pois-hmm.h"
#include "test-pois-em.h"
#include "unittest.h"

int
main (void)
{
    SETUP;

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
    RUN_TEST (test__PoisHmm_InitRandom_sorted_lambda);
    RUN_TEST (test__PoisHmm_LogLikelihood);
    RUN_TEST (test__PoisHmm_ForwardProbabilities);
    RUN_TEST (test__PoisHmm_BackwardProbabilities);
    RUN_TEST (test__PoisHmm_ForwardBackward);
    RUN_TEST (test__PoisHmm_EstimateParams);

    EVALUATE;
}
