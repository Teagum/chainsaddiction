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
PoisParams_NewFromFile (
    const char *fpath)
{
    int err = 0;
    scalar mbuff = 0l;
    PoisParams *out = NULL;
    FILE *dfd = Ca_OpenFile (fpath, "r");

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

    fclose (dfd);
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
    PoisParams *const this)
{
    enum { linewidth=100 };
    char border[] = "====================";
    char sep[] = "--------------------";

    size_t m_states = this->m_states;

    printf ("\n\n*%s%s%s*\n\n", border, border, border);
    printf ("%25s%10zu\n", "m-states:", m_states);

    printf ("%25s", "State:");
    for (size_t i = 0; i < m_states; i++)
        printf ("%10zu", i+1);
    puts ("");
    printf ("%25s", "State dependent means:");
    for (size_t i = 0; i < m_states; i++)
        printf ("%10.5Lf", this->lambda[i]);
    puts ("");
    printf ("%25s", "Start distribution:");
    for (size_t i = 0; i < m_states; i++)
        printf ("%10.5Lf", this->delta[i]);

    printf ("\n\n%s%s%s\n\n", sep, sep, sep);

    printf ("%25s", "Transition probability matrix:\n");
    printf ("%25s", " ");
    for (size_t i = 0; i < m_states; i++)
        printf ("%10zu", i+1);
    puts ("");
    for (size_t i = 0; i < m_states; i++)
    {
        printf ("%25zu", i+1);
        for (size_t j = 0; j < m_states; j++)
        {
            printf ("%10.5Lf", this->gamma[i*m_states+j]);
        }
        puts ("");
    }
    printf ("\n*%s%s%s*\n\n", border, border, border);
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
