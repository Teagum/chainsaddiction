#ifndef pois_params_h
#define pois_params_h

#include <stdlib.h>
#include <string.h>
#include "../restrict.h"
#include "../scalar.h"


typedef struct {
    size_t m_states;
    scalar *restrict lambda;
    scalar *restrict gamma;
    scalar *restrict delta;
} PoisParams;


#define PoisParams_Delete(params) do {      \
    free (params->lambda);                  \
    free (params->gamma);                   \
    free (params->delta);                   \
    free (params);                          \
    params = NULL;                          \
} while (false)


void
PoisParams_New (
    const size_t m_states);


extern void
PoisParams_SetLambda (
    PoisParams *const restrict params,
    const scalar *const restrict lambda);


extern void
PoisParams_SetGamma (
    PoisParams *const restrict params,
    const scalar *const restrict gamma);


extern void
PoisParams_SetDelta (
    PoisParams *const restrict params,
    const scalar *const restrict delta);


#endif  /* pois_params_h */
