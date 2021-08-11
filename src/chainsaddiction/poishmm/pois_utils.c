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
    mm_add_s (lalpha, lbeta, n_elem, -llh, lcexpt);
}


inline void
vi_log_normalize (
    size_t n_elem,
    scalar *restrict buffer)
{
    scalar lsum = v_lse (buffer, n_elem);
    for (size_t i = 0; i < n_elem; i++)
    {
        buffer[i] -= lsum;
    }
}


extern void
v_log_normalize (
    size_t n_elem,
    const scalar *const restrict src,
    scalar *const restrict dest)
{
    scalar lsum = v_lse (src, n_elem);
    for (size_t i = 0; i < n_elem; i++)
    {
        dest[i] = src[i] - lsum;
    }
}


extern int
local_decoding (
    const size_t n_obs,
    const size_t m_states,
    const scalar *lcxpt,
    size_t *states)
{
    scalar *obs_probs = malloc (sizeof (scalar) * n_obs * m_states);
    if (obs_probs == NULL)
    {
        fputs ("Alloc error in global decoding.", stderr);
        return 1;
    }
    m_exp (n_obs, m_states, lcxpt, obs_probs);
    m_row_argmax (n_obs, m_states, obs_probs, states);

    free (obs_probs);
    return 0;
}
