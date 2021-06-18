#include <stdio.h>
#include "hmm.h"


HmmProbs *
ca_ph_NewProbs (
    const size_t n_obs,
    const size_t m_states)
{
    size_t n_elem = n_obs * m_states;
    if (n_obs == 0)
    {
        fprintf (stderr, "`n_obs' must be greater than 0.\n");
        return NULL;
    }
    if (m_states == 0)
    {
        fprintf (stderr, "`m_states' must be greater than 0.\n");
        return NULL;
    }
    if (n_elem / n_obs != m_states)
    {
        fprintf (stderr, "Integer overflow detected.");
        return NULL;
    }

    HmmProbs *probs = malloc (sizeof *probs);
    if (probs == NULL)
    {
        fprintf (stderr, "Could not allocate memory for HmmProbs.\n");
        return NULL;
    }
    probs->lsd      = MA_SCALAR_ZEROS (n_elem);
    probs->lalpha   = MA_SCALAR_ZEROS (n_elem);
    probs->lbeta    = MA_SCALAR_ZEROS (n_elem);
    probs->n_obs    = n_obs;
    probs->m_states = m_states;

    return probs;
}


PoisParams *ca_ph_NewParams (size_t m_states)
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

    params = ca_ph_NewParams (m_states);
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
    ca_ph_FREE_PARAMS (params);
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

    ph->m_states = m_states;
    ph->max_iter = max_iter;
    ph->tol      = tol;
    ph->n_iter   = 0L;

    ph->init   = ca_ph_NewParams (m_states);
    ph->params = ca_ph_NewParams (m_states);
    if (ph->init == NULL || ph->params == NULL)
    {
        fprintf (stderr, "Could not allocate parameter vectors.\n");
        ca_ph_FREE_PARAMS (ph->init);
        ca_ph_FREE_PARAMS (ph->params);
        return NULL;
    }

    memcpy (ph->init->lambda, init_lambda, vector_s);
    memcpy (ph->init->gamma,  init_gamma,  matrix_s);
    memcpy (ph->init->delta,  init_delta,  vector_s);

    ph->aic = 0.0L;
    ph->bic = 0.0L;
    ph->llh = 0.0L;

    return ph;
}

void
PoisHmm_DeleteHmm (PoisHmm *ph)
{
    ca_ph_FREE_PARAMS (ph->init);
    ca_ph_FREE_PARAMS (ph->params);
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

scalar
ca_log_likelihood (scalar *lalpha, size_t n_obs, size_t m_states)
{
    const scalar *restrict last_row = lalpha + ((n_obs-1)*m_states);
    return v_lse (last_row, m_states);
}
