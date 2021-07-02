#include "test_pois_probs.h"
#include "test_pois_params.h"
#include "test_pois_hmm.h"

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

    EVALUATE;
}
