#include "utils.h"


scalar
compute_aic (
    const size_t m_states,
    const scalar llh)
{
    return 2.0L * llh + (scalar) (2 * m_states + m_states * m_states);
}


scalar
compute_bic (
    const size_t n_obs,
    const size_t m_states,
    const scalar llh)
{
    if (n_obs == 0)
    {
        Ca_ErrMsg ("compute_bic: param `n_obs' probably zero.");
        return 0.0L;
    }
    const size_t n_params = 2 * m_states + m_states * m_states;
    const scalar log_nobs = logl ((scalar) n_obs);
    return 2.0L * llh + log_nobs * (scalar) (n_params);
}


scalar
compute_log_likelihood (
    const size_t n_obs,
    const size_t m_states,
    const scalar *const restrict lalpha)
{
    const scalar *restrict last_row = lalpha + ((n_obs-1)*m_states);
    return v_lse (m_states, last_row);
}


inline void
log_csprobs (
    const size_t n_obs,
    const size_t m_states,
    const scalar llh,
    const scalar *const restrict lalpha,
    const scalar *const restrict lbeta,
          scalar *const restrict lcsp)
{
    size_t n_elem = n_obs * m_states;
    mm_add_s (lalpha, lbeta, n_elem, -llh, lcsp);
}


extern int
local_decoding (
    const size_t n_obs,
    const size_t m_states,
    const scalar *lcsp,
    size_t *states)
{
    scalar *obs_probs = malloc (sizeof (scalar) * n_obs * m_states);
    if (obs_probs == NULL)
    {
        Ca_ErrMsg ("Allocation failed.");
        return Ca_FAILURE;
    }
    m_exp (n_obs, m_states, lcsp, obs_probs);
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
    const scalar *restrict lcsp,
    size_t *restrict states)
{
    const size_t n_elem = n_obs * m_states;
    scalar *chi = VA_SCALAR_ZEROS (n_elem);
    scalar *vb  = VA_SCALAR_ZEROS (m_states);
    scalar *mb  = VA_SCALAR_ZEROS (m_states*m_states);
    scalar *mp  = VA_SCALAR_ZEROS (m_states);
    if (chi == NULL || vb == NULL || mb == NULL || mp == NULL)
    {
        Ca_ErrMsg ("Allocation failed.");
        free (chi);
        free (vb);
        free (mb);
        free (mp);
        return Ca_FAILURE;
    }
    scalar *chi_prev_row = chi;
    scalar *chi_this_row = chi+m_states;

    vv_add(m_states, ldelta, lcsp, chi);
    lcsp += m_states;
    for (size_t n = 1; n < n_obs; n++)
    {
        vm_add (m_states, m_states, chi_prev_row, lgamma, mb);
        m_row_max (mb, m_states, m_states, chi_this_row);
        vvi_add (m_states, lcsp, chi_this_row);

        chi_this_row+=m_states;
        chi_prev_row+=m_states;
        lcsp+=m_states;
    }
    chi_this_row = NULL;
    lcsp = NULL;

    size_t i = n_obs - 1;
    states[i] = v_argmax (m_states, chi_prev_row);
    while (i--)
    {
        const size_t row_idx = states[i+1] * m_states;
        chi_prev_row -= m_states;

        vv_add (m_states, chi_prev_row, lgamma+row_idx, vb);
        states[i] = v_argmax(m_states, vb);
    }

    free (chi);
    free (vb);
    free (mb);
    free (mp);
    return Ca_SUCCESS;
}
