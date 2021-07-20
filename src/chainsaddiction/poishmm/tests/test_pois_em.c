#include "test_pois_em.h"


bool
test__pois_e_step (void)
{
    enum { n_repeat_test = 10 };

    const char data_path[] = "../../../tests/data/earthquakes";
    const char params_path[] = "tests/data/ppr1";

    for (size_t n = 0; n < n_repeat_test; n++)
    {
        scalar llh = 0;
        DataSet *inp = ds_NewFromFile (data_path);
        PoisParams *params = PoisParams_NewFromFile (params_path);
        PoisParams *lparams = PoisParams_New(params->m_states);
        v_log (params->gamma, params->m_states, lparams->gamma);
        v_log (params->delta, params->m_states, lparams->delta);
        PoisProbs *probs = PoisProbs_New (params->m_states, inp->size);

        pois_e_step (inp->size, params->m_states, inp->data,
                lparams->lambda, lparams->gamma, lparams->delta,
                probs->lsdp, probs->lalpha, probs->lbeta, probs->lcxpt,
                &llh);

        ds_FREE(inp);
        PoisParams_Delete (params);
        PoisParams_Delete (lparams);
    }
    return false;
}


bool
test__pois_m_step_lambda (void)
{
    bool err = false;
    scalar llh = 0L;
    const char data_path[] = "../../../tests/data/earthquakes";
    const char params_path[] = "tests/data/ppr1";
    DataSet *inp = ds_NewFromFile (data_path);
    PoisParams *params = PoisParams_NewFromFile (params_path);
    PoisParams *lparams = PoisParams_New(params->m_states);
    PoisProbs *probs = PoisProbs_New (inp->size, params->m_states);
    scalar *new_lambda = MA_SCALAR_ZEROS (params->m_states);

    PoisParams_CopyLog (params, lparams);
    pois_e_step (inp->size, params->m_states, inp->data,
            lparams->lambda, lparams->gamma, lparams->delta,
            probs->lsdp, probs->lalpha, probs->lbeta, probs->lcxpt,
            &llh);

    pois_m_step_lambda (inp->size, probs->m_states, inp->data, probs->lcxpt,
            new_lambda);

    for (size_t i = 0; i < params->m_states; i++) {
        if (!isnormal (new_lambda[i])) {
            err = true;
            break;

        }
    }

    ds_FREE (inp);
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
    const char data_path[] = "../../../tests/data/earthquakes";
    const char params_path[] = "tests/data/ppr1";
    DataSet *inp = ds_NewFromFile (data_path);
    PoisParams *params = PoisParams_NewFromFile (params_path);
    PoisParams *lparams = PoisParams_New(params->m_states);
    v_log (params->gamma, params->m_states, lparams->gamma);
    v_log (params->delta, params->m_states, lparams->delta);
    PoisProbs *probs = PoisProbs_New (params->m_states, inp->size);
    scalar *new_lgamma = MA_SCALAR_ZEROS (params->m_states*probs->m_states);


    pois_e_step (inp->size, params->m_states, inp->data,
            lparams->lambda, lparams->gamma, lparams->delta,
            probs->lsdp, probs->lalpha, probs->lbeta, probs->lcxpt,
            &llh);

    pois_m_step_gamma (inp->size, params->m_states, llh,
            probs->lsdp, probs->lalpha, probs->lbeta,
            lparams->gamma, new_lgamma);


    ds_FREE (inp);
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
    const char data_path[] = "../../../tests/data/earthquakes";
    const char params_path[] = "tests/data/ppr1";
    DataSet *inp = ds_NewFromFile (data_path);
    PoisParams *params = PoisParams_NewFromFile (params_path);
    PoisParams *lparams = PoisParams_New(params->m_states);
    v_log (params->gamma, params->m_states, lparams->gamma);
    v_log (params->delta, params->m_states, lparams->delta);
    PoisProbs *probs = PoisProbs_New (params->m_states, inp->size);
    scalar *new_ldelta = MA_SCALAR_ZEROS (params->m_states);

/*
    pois_e_step (inp->size, params->m_states, inp->data,
            lparams->lambda, lparams->gamma, lparams->delta,
            probs->lsdp, probs->lalpha, probs->lbeta, probs->lcxpt,
            &llh);

    pois_m_step_delta (probs->m_states, probs->lcxpt, new_ldelta);
*/
    ds_FREE (inp);
    PoisParams_Delete (params);
    PoisParams_Delete (lparams);
    PoisProbs_Delete (probs);
    MA_FREE (new_ldelta);
    return err;
}


bool
test__score_update (void)
{
    enum { n_repeat_test = 100 };
    bool err = false;

    for (size_t n = 0; n < n_repeat_test; n++)
    {
        scalar score    = 0L;
        size_t m_states = (size_t) rnd_int (1, 50);
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
    }
    return err;
}
