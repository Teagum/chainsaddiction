#include "pois_params.h"


PoisParams *
PoisParams_New (
    const size_t m_states)
{
    PoisParams *params = malloc (sizeof *params);
    if (params == NULL)
    {
        fprintf (stderr, "Could not allocate memory for `PoisParams'.\n");
        return NULL;
    }

    params->lambda   = MA_SCALAR_ZEROS (m_states);
    params->gamma    = MA_SCALAR_ZEROS (m_states * m_states);
    params->delta    = MA_SCALAR_ZEROS (m_states);
    params->m_states = m_states;

    return params;
}


PoisParams *
PoisParams_NewRandom (
    const size_t m_states)
{
    PoisParams *this = PoisParams_New (m_states);
    PoisParams_SetLambdaRnd (this);
    PoisParams_SetGammaRnd  (this);
    PoisParams_SetDeltaRnd  (this);

    return this;
}


inline void
PoisParams_Copy (
    const PoisParams *const restrict this,
    PoisParams *const restrict other)
{
    PoisParams_SetLambda (other, this->lambda);
    PoisParams_SetGamma  (other, this->gamma);
    PoisParams_SetDelta  (other, this->delta);
}


#define PoisParams_Set (this, lambda, gamma, delta) do {    \
    PoisParams_SetLambda (this, lambda);                    \
    PoisParams_SetGamma  (this, gamma);                     \
    PoisParams_SetDelta  (this, delta);                     \
while (false);


inline void
PoisParams_SetLambda (
    PoisParams *const restrict params,
    const scalar *const restrict lambda)
{
    memcpy (params->lambda, lambda, params->m_states * sizeof (scalar));
}


inline void
PoisParams_SetGamma (
    PoisParams *const restrict params,
    const scalar *const restrict gamma)
{
    size_t size = params->m_states * params->m_states * sizeof (scalar);
    memcpy (params->gamma, gamma, size);
}


inline void
PoisParams_SetDelta (
    PoisParams *const restrict params,
    const scalar *const restrict delta)
{
    memcpy (params->delta, delta, params->m_states * sizeof (scalar));
}


inline void
PoisParams_SetLambdaRnd (
    PoisParams *const restrict this)
{
    pp_rnd_lambda (this->m_states, this->lambda);
}


inline void
PoisParams_SetGammaRnd (
    PoisParams *const restrict this)
{
    pp_rnd_gamma (this->m_states, this->gamma);
}


inline void
PoisParams_SetDeltaRnd (
    PoisParams *const restrict this)
{
    pp_rnd_delta (this->m_states, this->delta);
}


void
pp_rnd_lambda (
    const size_t m_states,
    scalar *const restrict buffer)
{
    v_rnd (m_states, buffer);
    for (size_t i = 0; i < m_states; i++)
    {
        buffer[i] += (scalar) rnd_int (1, 100);
    }
}


void
pp_rnd_gamma (
    const size_t m_states,
    scalar *const restrict buffer)
{
    const size_t g_elem = m_states * m_states;

    v_rnd (g_elem, buffer);
    for (size_t i = 0; i < m_states; i++)
    {
        vi_softmax (buffer+i*m_states, m_states);
    }
}


void
pp_rnd_delta (
    const size_t m_states,
    scalar *const restrict buffer)
{
    v_rnd (m_states, buffer);
    vi_softmax (buffer, m_states);
}
