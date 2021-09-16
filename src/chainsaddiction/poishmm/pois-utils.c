#include "pois-utils.h"


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
    return v_lse (m_states, last_row);
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
    scalar lsum = v_lse (n_elem, buffer);
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
    scalar lsum = v_lse (n_elem, src);
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
        Ca_ErrMsg ("Allocation failed.");
        return Ca_FAILURE;
    }
    m_exp (n_obs, m_states, lcxpt, obs_probs);
    m_row_argmax (n_obs, m_states, obs_probs, states);

    free (obs_probs);
    return Ca_SUCCESS;
}


extern int
global_decoding (
    const size_t n_obs,
    const size_t m_states,
    const scalar *const restrict lgamma,
    const scalar *const restrict ldelta,
    const scalar *restrict lsdp,
    size_t *restrict states)
{
    dim n_elem = n_obs * m_states;
    scalar *chi = VA_SCALAR_ZEROS (n_elem);
    scalar *vb  = VA_SCALAR_ZEROS (m_states);
    scalar *mb  = VA_SCALAR_ZEROS (m_states*m_states);
    scalar *mp  = VA_SCALAR_ZEROS (m_states);
    if (chi == NULL || vb == NULL || mb == NULL || mp == NULL)
    {
        Ca_ErrMsg ("Allocation failed.");
        return Ca_FAILURE;
    }
    scalar *chi_prev_row = chi;
    scalar *chi_this_row = chi+m_states;

    vv_add(m_states, ldelta, lsdp, chi);
    lsdp += m_states;
    for (size_t n = 1; n < n_obs; n++)
    {
        vm_add (m_states, m_states, chi_prev_row, lgamma, mb);
        m_row_max (mb, m_states, m_states, chi_this_row);
        vvi_add (m_states, lsdp, chi_this_row);

        chi_this_row+=m_states;
        chi_prev_row+=m_states;
        lsdp+=m_states;
    }
    chi_this_row = NULL;
    lsdp = NULL;

    size_t i = n_obs - 1;
    states[i] = v_argmax (m_states, chi_prev_row);
    while (i--)
    {
        const size_t row_idx = states[i+1] * m_states;
        chi_prev_row -= m_states;

        vv_add (m_states, chi_prev_row, lgamma+row_idx, vb);
        states[i] = v_argmax(m_states, vb);
    }

    return Ca_SUCCESS;
}
