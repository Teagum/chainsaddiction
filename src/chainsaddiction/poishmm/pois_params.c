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
