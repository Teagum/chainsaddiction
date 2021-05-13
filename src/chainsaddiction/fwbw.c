#include "fwbw.h"

void
log_forward (
    const scalar *restrict lprobs,
    const scalar *restrict lgamma,
    const scalar *restrict ldelta,
    const size_t m_states,
    const size_t n_obs,
    scalar *alpha)
{
    /* shared buffers */
    scalar *_cs = MA_SCALAR_EMPTY (m_states);
    scalar *_mb = MA_SCALAR_EMPTY (m_states*m_states);

    /* Init step */
    for (size_t i = 0; i < m_states; i++)
    {
        alpha[i] = ldelta[i] + lprobs[i];
    }

    /* remaining steps */
    for (size_t i = 1; i < n_obs; i++)
    {
        size_t c_idx = (i-1) * m_states;
        size_t n_idx = i * m_states;
        log_vmp (alpha+c_idx, lgamma, m_states, _cs, _mb, alpha+n_idx);
        vi_add (lprobs+=m_states, alpha+n_idx, m_states);
    }

    MA_FREE (_cs);
    MA_FREE (_mb);
}


void
log_backward (
    const scalar *restrict lprobs,
    const scalar *restrict lgamma,
    const size_t m_states,
    const size_t n_obs,
    scalar *beta)
{
    /* shared buffers */
    scalar *_cs = MA_SCALAR_ZEROS (m_states);
    scalar *_mb = MA_SCALAR_ZEROS (m_states*m_states);

    /* init step */
    size_t c_idx = (n_obs-1) * m_states;
    lprobs += c_idx;
    beta += c_idx;

    for (size_t i = 0; i < m_states; i++)
    {
        beta[i] = 0.0L;
    }

    for (size_t i = n_obs-1; i > 0; i--)
    {
        beta -= m_states;
        v_add (lprobs, beta+m_states, m_states, beta);
        log_mvp (lgamma, beta, m_states, _cs, _mb, beta);
        lprobs -= m_states;
    }

    MA_FREE (_cs);
    MA_FREE (_mb);
}


void
log_forward_backward (
    const scalar *restrict lprobs,
    const scalar *restrict lgamma,
    const scalar *restrict ldelta,
    const size_t m_states,
    const size_t n_obs,
    scalar *alpha,
    scalar *beta)
{
    log_forward (lprobs, lgamma, ldelta, m_states, n_obs, alpha);
    log_backward (lprobs, lgamma, m_states, n_obs, beta);
}
