#include "test_fwbw.h"


int
main (void)
{
    SETUP;

    RUN_TEST (test_log_forward);
    RUN_TEST (test_log_backward);
    RUN_TEST (test_log_fwbw);

    EVALUATE;
}

bool
test_log_forward (void)
{
    const size_t m_states = 3;
    const scalar lambda[] = {10, 20, 30};
    const scalar gamma[] = {.8, .1, .1,
                            .1, .8, .1,
                            .1, .1, .8};
    const scalar delta[] = {0.333333, 0.333333, .333333};

    scalar lgamma[9] = { 0. };
    scalar ldelta[3] = { 0. };

    DataSet *eq = ds_NewFromFile ("tests/data/earthquakes");
    size_t n_probs = eq->size * m_states;
    DataSet *lpp = ds_New (n_probs);
    DataSet *lalpha = ds_New (n_probs);

    v_log (gamma, m_states*m_states, lgamma);
    v_log (delta, m_states, ldelta);

    v_poisson_logpmf (eq->data, eq->size, lambda, m_states, lpp->data);

    log_forward (lpp->data, lgamma, ldelta, m_states, eq->size,  lalpha->data);

    ds_FREE (lalpha);
    ds_FREE (lpp);
    ds_FREE (eq);
    return false;
}


bool
test_log_backward (void)
{
    const size_t m_states = 3;
    const scalar lambda[] = {10, 20, 30};
    const scalar gamma[] = {.8, .1, .1,
                            .1, .8, .1,
                            .1, .1, .8};
    const scalar delta[] = {0.333333, 0.333333, .333333};

    scalar lgamma[9] = { 0. };
    scalar ldelta[3] = { 0. };

    DataSet *eq = ds_NewFromFile ("tests/data/earthquakes");
    size_t n_probs = eq->size * m_states;
    DataSet *lpp = ds_New (n_probs);
    DataSet *lbeta = ds_New (n_probs);

    v_log (gamma, m_states*m_states, lgamma);
    v_log (delta, m_states, ldelta);

    v_poisson_logpmf (eq->data, eq->size, lambda, m_states, lpp->data);
    log_backward (lpp->data, lgamma, m_states, eq->size,  lbeta->data);

    ds_FREE (lbeta);
    ds_FREE (lpp);
    ds_FREE (eq);
    return false;
}


bool
test_log_fwbw (void)
{
    const size_t m_states = 3;
    const scalar lambda[] = {10, 20, 30};
    const scalar gamma[] = {.8, .1, .1,
                            .1, .8, .1,
                            .1, .1, .8};
    const scalar delta[] = {0.333333, 0.333333, .333333};

    scalar lgamma[9] = { 0. };
    scalar ldelta[3] = { 0. };

    DataSet *eq = ds_NewFromFile ("tests/data/earthquakes");
    size_t n_probs = eq->size * m_states;
    DataSet *lpp = ds_New (n_probs);
    DataSet *lalpha = ds_New (n_probs);
    DataSet *lbeta = ds_New (n_probs);

    v_log (gamma, m_states*m_states, lgamma);
    v_log (delta, m_states, ldelta);

    v_poisson_logpmf (eq->data, eq->size, lambda, m_states, lpp->data);
    log_fwbw (lpp->data, lgamma, ldelta, m_states, eq->size, lalpha->data, lbeta->data);

    ds_FREE (lalpha);
    ds_FREE (lbeta);
    ds_FREE (lpp);
    ds_FREE (eq);
    return false;
}
