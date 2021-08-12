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
        fputs ("Alloc error in global decoding.", stderr);
        return 1;
    }
    m_exp (n_obs, m_states, lcxpt, obs_probs);
    m_row_argmax (n_obs, m_states, obs_probs, states);

    free (obs_probs);
    return 0;
}


extern int
global_decoding (
    const size_t n_obs,
    const size_t m_states,
    const scalar *lgamma,
    const scalar *ldelta,
    const scalar *lsdp,
    size_t *states)
{
    dim n_elem = n_obs * m_states;
    scalar *chi = VA_SCALAR_ZEROS (n_elem);
    scalar *vb  = VA_SCALAR_ZEROS (m_states);
    scalar *mb  = VA_SCALAR_ZEROS (m_states*m_states);
    scalar *mp  = VA_SCALAR_ZEROS (m_states);
    if (chi == NULL || vb == NULL || mb == NULL || mp == NULL)
    {
        fputs ("Alloc error in global decoding.", stderr);
        return 1;
    }

    v_add(ldelta, lsdp, m_states, chi);
    scalar *prev_row = chi;
    scalar *this_row = chi+m_states;
    for (size_t n = 1; n < n_obs; n++, this_row+=m_states, prev_row+=m_states, lsdp+=m_states)
    {
        vm_add (m_states, m_states, prev_row, lgamma, mb);
        m_row_max (mb, m_states, m_states, this_row);
        vi_add (lsdp, this_row, m_states);
    }

    states[n_obs-1] = v_argmax (m_states, this_row);
    for (size_t i = n_obs-2; i > 0; i--, prev_row-=m_states)
    {
        v_add (prev_row, lgamma+states[i+1], m_states, vb);
        states[i] = v_argmax(m_states, vb);
    }

    return 0;
}
