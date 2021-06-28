#include "poishmm.h"


void
PoisHmm_BaumWelch (
    const DataSet *const restrict inp,
    PoisHmm *const restrict phmm)
{}


void
PoisHmm_LogStateProbs (
    const HmmProbs *const restrict probs,
    const scalar llh,
    scalar *out)
{
    size_t n_elem = probs->m_states * probs->n_obs;
    /* the fourth argument should probably be `-llh'. */
    mm_add_s (probs->lalpha, probs->lbeta, n_elem, llh, out);
}


HmmProbs *
PoisHmm_NewProbs (
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


PoisParams *PoisHmm_NewParams (size_t m_states)
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


PoisHmm *ca_ph_NewHmm (const size_t n_obs, const size_t m_states)
{
    PoisHmm *phmm = malloc (sizeof *phmm);
    if (phmm == NULL)
    {
        fprintf (stderr, "Could not allocate memory for `PoisHmm' object.\n");
        exit (1);
    }

    phmm->init   = PoisHmm_NewParams (m_states);
    phmm->params = PoisHmm_NewParams (m_states);
    phmm->probs  = PoisHmm_NewProbs (n_obs, m_states);

    phmm->n_obs    = n_obs;
    phmm->m_states = m_states;
    phmm->n_iter   = 0;
    phmm->max_iter = DEFAULT_MAX_ITER;
    phmm->tol      = DEFAULT_TOLERANCE;
    phmm->aic      = 0.0L;
    phmm->bic      = 0.0L;
    phmm->llh      = 0.0L;

    return phmm;
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

    params = PoisHmm_NewParams (m_states);
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
    PoisHmm_DeleteParams (params);
    return NULL;
}


void PoisHmm_PrintParams (const PoisHmm *const restrict phmm)
{
    enum {linewidth=100};
    char border[] = "====================";
    char sep[] = "--------------------";

    size_t m_states = phmm->m_states;
    PoisParams *params = phmm->params;

    printf ("\n\n*%s%s%s*\n\n", border, border, border);
    printf ("%25s%10zu\n", "m-states:", m_states);
    printf ("%25s%10.5Lf\n", "-log likelihood:", phmm->llh);
    printf ("%25s%10.5Lf\n", "AIC:", phmm->aic);
    printf ("%25s%10.5Lf\n", "BIC:", phmm->bic);
    printf ("\n\n%s%s%s\n\n", sep, sep, sep);

    printf ("%25s", "State:");
    for (size_t i = 0; i < m_states; i++)
        printf ("%10zu", i+1);
    puts ("");
    printf ("%25s", "State dependent means:");
    for (size_t i = 0; i < m_states; i++)
        printf ("%10.5Lf", params->lambda[i]);
    puts ("");
    printf ("%25s", "Start distribution:");
    for (size_t i = 0; i < m_states; i++)
        printf ("%10.5Lf", params->delta[i]);

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
            printf ("%10.5Lf", params->gamma[i*m_states+j]);
        }
        puts ("");
    }
    printf ("\n*%s%s%s*\n\n", border, border, border);
}


void
ca_ph_InitParams (
    const PoisHmm *const restrict phmm,
    const scalar *const restrict lambda,
    const scalar *const restrict gamma,
    const scalar *const restrict delta)
{
    size_t m_states = phmm->m_states;
    size_t n_elem_gamma = m_states * m_states;
    size_t v_size = m_states * sizeof (scalar);
    size_t m_size = m_states * v_size;

#ifdef __STDC_LIB_EXT1__
    errno_t err = 0;
    err = memcpy_s (phmm->init->lambda, v_size, lambda, v_size);
    if (err != 0) {
        perror ("Failed to copy initial values of `lambda'.");
        exit (1)
    }
    err = memcpy_s (phmm->init->gamma, m_size, gamma, m_size);
    if (err != 0) {
        perror ("Failed to copy initial values of `gamma'.");
        exit (1)
    }
    err = memcpy_s (phmm->init->delta, v_size, delta, v_size);
    if (err != 0) {
        perror ("Failed to copy initial values of `delta'.");
        exit (1)
    }
#else
    memcpy (phmm->init->lambda, lambda, v_size);
    memcpy (phmm->init->gamma, gamma, m_size);
    memcpy (phmm->init->delta, delta, v_size);
#endif

    v_log (phmm->init->lambda, phmm->m_states, phmm->params->lambda);
    v_log (phmm->init->gamma, n_elem_gamma, phmm->params->gamma);
    v_log (phmm->init->delta, phmm->m_states, phmm->params->delta);
}


void
ca_ph_InitRandom (PoisHmm *const restrict phmm)
{
    size_t m_states = phmm->m_states;
    size_t n_elem = m_states * m_states;

    v_rnd (m_states, phmm->init->lambda);
    for (size_t i = 0; i < m_states; i++)
    {
        phmm->init->lambda[i] += (scalar) rnd_int (0, 100);
    }
    v_rnd (n_elem, phmm->init->gamma);
    v_rnd (m_states, phmm->init->delta);

    for (size_t i = 0; i < m_states; i++)
    {
        vi_softmax (phmm->init->gamma+i*m_states, m_states);
    }
    vi_softmax (phmm->init->delta, m_states);

    v_log (phmm->init->lambda, m_states, phmm->params->lambda);
    v_log (phmm->init->gamma, n_elem, phmm->params->gamma);
    v_log (phmm->init->delta, m_states, phmm->params->delta);
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

    ph->init   = PoisHmm_NewParams (m_states);
    ph->params = PoisHmm_NewParams (m_states);
    if (ph->init == NULL || ph->params == NULL)
    {
        fprintf (stderr, "Could not allocate parameter vectors.\n");
        PoisHmm_DeleteParams (ph->init);
        PoisHmm_DeleteParams (ph->params);
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
