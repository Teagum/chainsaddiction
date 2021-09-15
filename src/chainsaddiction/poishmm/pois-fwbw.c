#include "pois-fwbw.h"


void
log_forward (
    const scalar *restrict lprobs,
    const scalar *const restrict lgamma,
    const scalar *const restrict ldelta,
    const size_t m_states,
    const size_t n_obs,
    scalar *lalpha)
{
    /* shared buffers */
    scalar *_cs = MA_SCALAR_EMPTY (m_states);

    /* Init step */
    for (size_t i = 0; i < m_states; i++)
    {
        lalpha[i] = ldelta[i] + lprobs[i];
    }

    /* remaining steps */
    for (size_t i = 1; i < n_obs; i++)
    {
        size_t c_idx = (i-1) * m_states;
        size_t n_idx = i * m_states;
        vm_multiply_log (m_states, m_states, lalpha+c_idx, lgamma, _cs, lalpha+n_idx);
        vvi_add (m_states, lprobs+=m_states, lalpha+n_idx);
    }
    MA_FREE (_cs);
}


void
log_backward (
    const scalar *restrict lprobs,
    const scalar *const restrict lgamma,
    const size_t m_states,
    const size_t n_obs,
    scalar *lbeta)
{
    /* shared buffers */
    scalar *_cs = MA_SCALAR_ZEROS (m_states);
    scalar *obs_prob = MA_SCALAR_EMPTY (m_states);

    /* init step */
    size_t c_idx = (n_obs-1) * m_states;
    lprobs += c_idx;
    lbeta += c_idx;

    for (size_t i = 0; i < m_states; i++)
    {
        lbeta[i] = 0.0L;
    }

    for (size_t i = n_obs-1; i > 0; i--)
    {
        vv_add (m_states, lprobs, lbeta, obs_prob);
        mv_multiply_log (m_states, m_states, lgamma, obs_prob, _cs, lbeta-m_states);
        lbeta -= m_states;
        lprobs -= m_states;
    }
    MA_FREE (_cs);
}


void
log_fwbw (
    const scalar *restrict lprobs,
    const scalar *const restrict lgamma,
    const scalar *const restrict ldelta,
    const size_t m_states,
    const size_t n_obs,
    scalar *lalpha,
    scalar *lbeta)
{
    log_forward (lprobs, lgamma, ldelta, m_states, n_obs, lalpha);
    log_backward (lprobs, lgamma, m_states, n_obs, lbeta);
}
