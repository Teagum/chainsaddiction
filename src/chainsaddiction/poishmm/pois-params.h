#ifndef pois_params_h
#define pois_params_h

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../libma.h"
#include "../read.h"
#include "../restrict.h"
#include "../scalar.h"
#include "libvmath.h"


#define CHECK_READ_ERROR(res) \
if (res) { \
    fprintf (stderr, "Reading error.\n"); \
    return NULL;\
} while (0)


typedef struct PoisParams {
    size_t m_states;
    scalar *lambda;
    scalar *gamma;
    scalar *delta;
} PoisParams;


#define PoisParams_Delete(params)   \
if (params)                         \
{                                   \
    free (params->lambda);          \
    free (params->gamma);           \
    free (params->delta);           \
    free (params);                  \
    params = NULL;                  \
}


PoisParams *
PoisParams_New (
    const size_t m_states);


PoisParams *
PoisParams_NewFromFile (
    const char *fpath);


PoisParams *
PoisParams_NewRandom (
    const size_t m_states);


extern void
PoisParams_Copy (
    const PoisParams *const restrict this,
    PoisParams *const restrict other);


extern void
PoisParams_CopyLog (
    const PoisParams *restrict this,
    PoisParams *restrict other);


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


extern void
PoisParams_SetLambdaRnd (
    PoisParams *const restrict this);


extern void
PoisParams_SetGammaRnd (
    PoisParams *const restrict this);


extern void
PoisParams_SetDeltaRnd (
    PoisParams *const restrict this);


void
PoisParams_Print (
    const PoisParams *const restrict this);


void
pp_rnd_lambda (
    const size_t m_states,
    scalar *const restrict buffer);


void
pp_rnd_gamma (
    const size_t m_states,
    scalar *const restrict buffer);


void
pp_rnd_delta (
    const size_t m_states,
    scalar *const restrict buffer);


#endif  /* pois_params_h */
