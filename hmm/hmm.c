#include <stdio.h>
#include "hmm.h"

PoisParams *PoisHmm_NewEmptyParams (size_t m)
{
    size_t vector_s = m * sizeof (scalar);
    size_t matrix_s = m * vector_s;

    PoisParams *params = malloc (sizeof (*params));
    if (params == NULL)
    {
        goto error;
    }

    params->lambda = malloc (vector_s);
    params->gamma  = malloc (matrix_s);
    params->delta  = malloc (vector_s);

    if (params->lambda == NULL ||
        params->gamma  == NULL ||
        params->delta  == NULL)
    {
        goto error;
    }

    return params;

error:
    fprintf (stderr, "Could not allocate parameters.\n");
    PoisHmm_FreeParams (params);
    return NULL;
}

PoisParams *PoisHmm_ParamsFromFile (const char *fname)
{
    int        success  = 0;
    size_t     m_states = 0;
    FILE       *file    = NULL;
    PoisParams *params  = NULL;

    file = fopen (fname, "r");
    if (file == NULL)
    {
        fprintf (stderr, "Could not open file.\n");
        goto error;
    }

    success = fscanf (file, "%zu", &m_states);
    if (success == EOF)
    {
        fprintf (stderr, "Could not read number of states.\n");
        goto error;
    }

    params = PoisHmm_NewEmptyParams (m_states);
    if (params == 0)
    {
        fprintf (stderr, "Could not allocate Params.\n");
        goto error;
    }

    for (size_t i = 0; i < m_states; i++)
    {
        success = fscanf (file, "%Lf,", &params->lambda[i]);
        if (success == EOF)
        {
            fprintf (stderr, "Error reading lambda at (%zu).\n", i);
            goto error;
        }
    }

    for (size_t i = 0; i < m_states * m_states; i++)
    {
        success = fscanf (file, "%Lf,", &params->gamma[i]);
        if (success == EOF)
        {
            fprintf (stderr, "Error reading gamma at (%zu, %zu).\n",
                     i/m_states, i%m_states);
            goto error;
        }
    }

    for (size_t i = 0; i < m_states; i++)
    {
        success = fscanf (file, "%Lf,", &params->delta[i]);
        if (success == EOF)
        {
            fprintf (stderr, "Error reading delta at (%zu).\n", i);
            goto error;
        }
    }

    return params;

error:
    fclose (file);
    PoisHmm_FreeParams (params);
    return NULL;
}

void PoisHmm_PrintParams (PoisParams *params, size_t m_states)
{
    fprintf (stdout, "\nStates: %zu\n\n", m_states);

    fprintf (stdout, "Lambda:\n");
    for (size_t i = 0; i < m_states; i++)
        fprintf (stdout, "%Lf\t", params->lambda[i]);

    fprintf (stdout, "\n\nGamma:\n");
    for (size_t i = 0; i < m_states; i++)
    {
        for (size_t j = 0; j < m_states; j++)
        {
            fprintf (stdout, "%20.19Lf\t", params->gamma[i*m_states+j]);
        }
        fprintf (stdout, "\n");
    }

    fprintf (stdout, "\nDelta:\n");
    for (size_t i = 0; i < m_states; i++)
        fprintf (stdout, "%Lf\t", params->delta[i]);

    fprintf (stdout, "\n");
}

void PoisHmm_FreeParams (PoisParams *params)
{
        free (params->lambda);
        free (params->gamma);
        free (params->delta);
        free (params);
}

PoisHmm *
PoisHmm_FromData (size_t m_states,
             scalar *restrict init_lambda,
             scalar *restrict init_gamma,
             scalar *restrict init_delta,
             size_t max_iter,
             scalar tol)
{
    PoisHmm *ph = malloc (sizeof (*ph));
    if (ph == NULL)
    {
        fprintf (stderr, "Could not allocate PoissonHMM.\n");
        return NULL;
    }

    size_t vector_s = m_states * sizeof (scalar);
    size_t matrix_s = m_states * vector_s;

    ph->m        = m_states;
    ph->max_iter = max_iter;
    ph->tol      = tol;
    ph->n_iter   = 0L;

    ph->init   = PoisHmm_NewEmptyParams (m_states);
    ph->params = PoisHmm_NewEmptyParams (m_states);
    if (ph->init == NULL || ph->params == NULL)
    {
        fprintf (stderr, "Could not allocate parameter vectors.\n");
        PoisHmm_FreeParams (ph->init);
        PoisHmm_FreeParams (ph->params);
        return NULL;
    }

    memcpy (ph->init->lambda, init_lambda, vector_s);
    memcpy (ph->init->gamma,  init_gamma,  matrix_s);
    memcpy (ph->init->delta,  init_delta,  vector_s);

    ph->aic = 0.0L;
    ph->bic = 0.0L;
    ph->nll = 0.0L;

    return ph;
}

void
PoisHmm_DeleteHmm (PoisHmm *ph)
{
    PoisHmm_FreeParams (ph->init);
    PoisHmm_FreeParams (ph->params);
    free (ph);
}

scalar
compute_aic(scalar nll, size_t m)
{
    return 2.0L * (scalar) (nll + 2*m + m*m);
}

scalar
compute_bic(scalar nll, size_t m, size_t n)
{
    return 2.0L * nll + logl ((scalar) n) * (scalar) (2*m + m*m);
}

