#include "pois_params.h"


void
PoisParams_New (
    const size_t m_states)
{}


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
    const scalar *const restrict lambda)
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
