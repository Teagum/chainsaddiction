#include "test-pois-em.h"


bool
test__pois_e_step (void)
{
    const char data_path[] = "../../../../tests/data/earthquakes/earthquakes";
    const char params_path[] = "data/ppr1";

    scalar llh = 0;
    DataSet *inp = DataSet_NewFromFile (data_path);
    if (inp == NULL) return false;

    PoisParams *params = PoisParams_NewFromFile (params_path);
    if (params == NULL) return false;

    PoisParams *lparams = PoisParams_New(params->m_states);
    PoisProbs *probs = PoisProbs_New (inp->size, params->m_states);

    PoisParams_CopyLog (params, lparams);
    pois_e_step (inp->size, params->m_states, inp->data, lparams->lambda,
            lparams->gamma, lparams->delta, probs->lsdp, probs->lalpha,
            probs->lbeta, probs->lcsp, &llh);

    DataSet_Delete(inp);
    PoisParams_Delete (params);
    PoisParams_Delete (lparams);
    return false;
}


bool
test__pois_m_step_lambda (void)
{
    bool err = false;
    scalar llh = 0L;
    const char data_path[] = "../../../../tests/data/earthquakes/earthquakes";
    const char params_path[] = "data/ppr1";

    DataSet *inp = DataSet_NewFromFile (data_path);
    if (inp == NULL) return false;

    PoisParams *params = PoisParams_NewFromFile (params_path);
    if (params == NULL) return false;

    PoisParams *lparams = PoisParams_New(params->m_states);
    PoisProbs *probs = PoisProbs_New (inp->size, params->m_states);
    scalar *new_lambda = MA_SCALAR_ZEROS (params->m_states);

    PoisParams_CopyLog (params, lparams);
    pois_e_step (inp->size, params->m_states, inp->data,
            lparams->lambda, lparams->gamma, lparams->delta,
            probs->lsdp, probs->lalpha, probs->lbeta, probs->lcsp,
            &llh);

    pois_m_step_lambda (inp->size, probs->m_states, inp->data, probs->lcsp,
            new_lambda);

    for (size_t i = 0; i < params->m_states; i++) {
        if (!isnormal (new_lambda[i])) {
            err = true;
            break;

        }
    }

    DataSet_Delete (inp);
    PoisParams_Delete (params);
    PoisParams_Delete (lparams);
    PoisProbs_Delete (probs);
    MA_FREE (new_lambda);
    return err;
}


bool
test__pois_m_step_gamma (void)
{
    bool err = false;
    scalar llh = 0L;
    const char data_path[] = "../../../../tests/data/earthquakes/earthquakes";
    const char params_path[] = "data/ppr1";

    DataSet *inp = DataSet_NewFromFile (data_path);
    if (inp == NULL) return false;

    PoisParams *params = PoisParams_NewFromFile (params_path);
    if (params == NULL) return false;

    PoisParams *lparams = PoisParams_New(params->m_states);
    PoisProbs *probs = PoisProbs_New (inp->size, params->m_states);
    scalar *new_lgamma = MA_SCALAR_ZEROS (params->m_states*probs->m_states);


    PoisParams_CopyLog (params, lparams);
    pois_e_step (inp->size, params->m_states, inp->data,
            lparams->lambda, lparams->gamma, lparams->delta,
            probs->lsdp, probs->lalpha, probs->lbeta, probs->lcsp,
            &llh);

    pois_m_step_gamma (inp->size, params->m_states, llh,
            probs->lsdp, probs->lalpha, probs->lbeta,
            lparams->gamma, new_lgamma);


    DataSet_Delete (inp);
    PoisParams_Delete (params);
    PoisParams_Delete (lparams);
    PoisProbs_Delete (probs);
    MA_FREE (new_lgamma);
    return err;
}


bool
test__pois_m_step_delta(void)
{
    bool err = false;
    scalar llh = 0L;
    const char data_path[] = "../../../../tests/data/earthquakes/earthquakes";
    const char params_path[] = "data/ppr1";

    DataSet *inp = DataSet_NewFromFile (data_path);
    if (inp == NULL) return false;

    PoisParams *params = PoisParams_NewFromFile (params_path);
    if (params == NULL) return false;

    PoisParams *lparams = PoisParams_New(params->m_states);
    PoisParams_CopyLog (params, lparams);
    PoisProbs *probs = PoisProbs_New (inp->size, params->m_states);
    scalar *new_ldelta = MA_SCALAR_ZEROS (params->m_states);

    pois_e_step (inp->size, params->m_states, inp->data,
            lparams->lambda, lparams->gamma, lparams->delta,
            probs->lsdp, probs->lalpha, probs->lbeta, probs->lcsp,
            &llh);

    pois_m_step_delta (probs->m_states, probs->lcsp, new_ldelta);


    DataSet_Delete (inp);
    PoisParams_Delete (params);
    PoisParams_Delete (lparams);
    PoisProbs_Delete (probs);
    MA_FREE (new_ldelta);
    return err;
}


bool
test__score_update (void)
{
    bool err = false;

    scalar score    = 0L;
    const size_t m_states = rnd_size (1, 50);
    PoisParams *pa  = PoisParams_NewRandom (m_states);
    PoisParams *pb  = PoisParams_New (m_states);

    PoisParams_Copy (pa, pb);
    score = score_update (pa, pb);

    PoisParams_Delete (pa);
    PoisParams_Delete (pb);

    if (fpclassify (score) != FP_ZERO)
    {
        err = true;
    }

    return err;
}
