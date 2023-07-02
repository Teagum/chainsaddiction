#include "pois-params.h"


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
PoisParams_NewFromFile (
    const char *fpath)
{
    int err = 0;
    size_t n_lines = 0;
    scalar mbuff = 0l;
    PoisParams *out = NULL;
    FILE *dfd = Ca_OpenFile (fpath, "r");
    Ca_CountLines (dfd, &n_lines);
    if (n_lines == 0)
    {
        fprintf (stderr, "Empty file: %s\n", fpath);
        Ca_CloseFile (dfd);
        return NULL;
    }

    err = Ca_ReadSectionHeader (dfd, "[states]");
    CHECK_READ_ERROR (err);

    err = Ca_ReadSectionData (dfd, 1, &mbuff);
    CHECK_READ_ERROR (err);

    out = PoisParams_New ((size_t) mbuff);

    err = Ca_ReadSectionHeader (dfd, "[lambda]");
    CHECK_READ_ERROR (err);

    err = Ca_ReadSectionData (dfd, out->m_states, out->lambda);
    CHECK_READ_ERROR (err);

    err = Ca_ReadSectionHeader (dfd, "[gamma]");
    CHECK_READ_ERROR (err);

    err = Ca_ReadSectionData (dfd, out->m_states *out->m_states, out->gamma);
    CHECK_READ_ERROR (err);

    err = Ca_ReadSectionHeader (dfd, "[delta]");
    CHECK_READ_ERROR (err);

    err = Ca_ReadSectionData (dfd, out->m_states, out->delta);
    CHECK_READ_ERROR (err);

    Ca_CloseFile (dfd);
    return out;
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


inline void
PoisParams_CopyLog (
    const PoisParams *restrict this,
    PoisParams *restrict other)
{
    PoisParams_SetLambda (other, this->lambda);
    v_logr1 (this->m_states * this->m_states, this->gamma, other->gamma);
    v_logr1 (this->m_states, this->delta, other->delta);
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
PoisParams_Print (
    const PoisParams *const restrict this)
{
    print_vector (this->m_states, this->lambda);
    print_vector_exp (this->m_states, this->delta);
    print_matrix_exp (this->m_states, this->m_states, this->gamma);
}


void
pp_rnd_lambda (
    const size_t m_states,
    scalar *const restrict buffer)
{
    const scalar SR_LB =   1.0L;
    const scalar SR_UB = 100.0L;
    v_rnd_scalar (m_states, SR_LB, SR_UB, buffer);
}


void
pp_rnd_gamma (
    const size_t m_states,
    scalar *const restrict buffer)
{
    m_rnd_sample (m_states, m_states, buffer);
    for (size_t i = 0; i < m_states; i++)
    {
        vi_softmax (m_states, buffer+i*m_states);
    }
}


void
pp_rnd_delta (
    const size_t m_states,
    scalar *const restrict buffer)
{
    v_rnd_sample (m_states, buffer);
    vi_softmax (m_states, buffer);
}
