#include "test_pois_em.h"


bool
test__pois_e_step (void)
{
    enum { n_repeat_test = 10, m_states = 3, n_obs = 15 };

    SET_EARTHQUAKES_SHORT;
    SET_LAMBDA;
    SET_LOG_GAMMA;
    SET_LOG_DELTA;

    for (size_t n = 0; n < n_repeat_test; n++)
    {
        //const size_t m_states = (size_t) rnd_int (1, 10);
        //const size_t n_obs    = (size_t) rnd_int(1, 500);
        const size_t n_elem   = m_states * n_obs;

        //scalar *input  = MA_SCALAR_EMPTY (n_obs);
        scalar *lsdp   = MA_SCALAR_EMPTY (n_elem);
        scalar *lalpha = MA_SCALAR_EMPTY (n_elem);
        scalar *lbeta  = MA_SCALAR_EMPTY (n_elem);
        scalar *lcxpt  = MA_SCALAR_EMPTY (n_elem);
        scalar llh     = 0;

        pois_e_step (n_obs, m_states, input, lambda, lgamma, ldelta,
                lsdp, lalpha, lbeta, lcxpt, &llh);

        //MA_FREE (input);
        MA_FREE (lsdp);
        MA_FREE (lalpha);
        MA_FREE (lbeta);
        MA_FREE (lcxpt);
    }
    return false;
}


bool
test__pois_m_step_gamma (void)
{
    enum { n_repeat_test = 10, m_states = 3, n_obs = 15, n_elem = m_states * n_obs};

    SET_EARTHQUAKES_SHORT;
    SET_LAMBDA;
    SET_LOG_GAMMA;
    SET_LOG_DELTA;

    for (size_t n = 0; n < n_repeat_test; n++)
    {
        scalar new_lgamma[m_states] = { 0L };

        //const size_t m_states = (size_t) rnd_int (1, 100);
        //const size_t n_obs    = (size_t) rnd_int (1, 500);
        //const size_t n_elem   = m_states * n_obs;

        scalar *lsdp   = MA_SCALAR_ZEROS (n_elem);
        scalar *lalpha = MA_SCALAR_ZEROS (n_elem);
        scalar *lbeta  = MA_SCALAR_ZEROS (n_elem);
        scalar *lcxpt  = MA_SCALAR_ZEROS (n_elem);
        scalar llh     = 0L;

        pois_e_step (n_obs, m_states, input, lambda, lgamma, ldelta,
                lsdp, lalpha, lbeta, lcxpt, &llh);

        pois_m_step_gamma (n_obs, m_states, llh, lsdp, lalpha, lbeta,
                lgamma, new_lgamma);

        MA_FREE (lsdp);
        MA_FREE (lalpha);
        MA_FREE (lbeta);
        MA_FREE (lcxpt);
    }
    return false;
}


bool
test__pois_m_step_delta(void)
{
    enum { n_repeat_test = 10, m_states = 3, n_obs = 15, n_elem = m_states * n_obs};

    SET_EARTHQUAKES_SHORT;
    SET_LAMBDA;
    SET_LOG_GAMMA;
    SET_LOG_DELTA;

    for (size_t n = 0; n < n_repeat_test; n++)
    {
        scalar new_ldelta[m_states] = { 0L };

        //const size_t m_states = (size_t) rnd_int (1, 100);
        //const size_t n_obs    = (size_t) rnd_int (1, 500);
        //const size_t n_elem   = m_states * n_obs;

        scalar *lsdp   = MA_SCALAR_ZEROS (n_elem);
        scalar *lalpha = MA_SCALAR_ZEROS (n_elem);
        scalar *lbeta  = MA_SCALAR_ZEROS (n_elem);
        scalar *lcxpt  = MA_SCALAR_ZEROS (n_elem);
        scalar llh     = 0L;

        pois_e_step (n_obs, m_states, input, lambda, lgamma, ldelta,
                lsdp, lalpha, lbeta, lcxpt, &llh);

        pois_m_step_delta (m_states, lcxpt, new_ldelta);

        MA_FREE (lsdp);
        MA_FREE (lalpha);
        MA_FREE (lbeta);
        MA_FREE (lcxpt);
    }
    return false;
}
