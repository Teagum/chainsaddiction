#include "pois_utils.h"


scalar
compute_aic (scalar llh, size_t m_states)
{
    return 2.0L * llh + (scalar) (2 * m_states + m_states * m_states);
}


scalar
compute_bic (scalar llh, size_t n_obs, size_t m_states)
{
    size_t n_params = 2 * m_states + m_states * m_states;
    return 2.0L * llh + logl ((scalar) n_obs) * (scalar) (n_params);
}


scalar
compute_log_likelihood (scalar *lalpha, size_t n_obs, size_t m_states)
{
    const scalar *restrict last_row = lalpha + ((n_obs-1)*m_states);
    return v_lse (last_row, m_states);
}


inline void
log_cond_expect (
    const size_t n_obs,
    const size_t m_states,
    const scalar *const restrict lalpha,
    const scalar *const restrict lbeta,
    const scalar llh,
    scalar *lcexpt)
{
    size_t n_elem = n_obs * m_states;
    mm_add_s (lalpha, lbeta, n_elem, llh, lcexpt);
}
