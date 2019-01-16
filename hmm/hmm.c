#include "hmm.h"
#include <stdio.h>

PoissonHMM *read_params (const char *fname)
{
    int    success  = 0;
    size_t m_states = 0;
    size_t vector_s = 0; m * sizeof (*(phmm->lambda_));
    size_t matrix_s = 0; m * vector_s;
    FILE   *file    = NULL;

    file = fopen (fname, "r");
    if (file == NULL)
    {
        fprintf (stderr, "Could not open file.\n");
        return NULL;
    }

    PoissonHMM *ph = malloc (sizeof (*ph));
    if (phmm == NULL)
    {
        fprintf (stderr, "Could not init PoissonHMM.\n");
        fclose (file);
        return NULL;
    }

    success = fscanf (file, "%zu", &ph->m);
    if (success == EOF)
    {
        fprintf (stderr, "Error reading number of states.\n");
        fclose (file);
        DeletePoissonHMM (ph);
        return NULL;
    }


}
PoissonHMM*
NewPoissonHMM (size_t m,
               scalar *init_lambda,
               scalar *init_gamma,
               scalar *init_delta,
               size_t max_iter,
               scalar tol)
{
    PoissonHMM *phmm = malloc (sizeof (*phmm));
    if (phmm == NULL) return NULL;

    size_t vector_s     = m * sizeof (*(phmm->lambda_));
    size_t matrix_s     = m * vector_s;

    phmm->m             = m;

    phmm->init_lambda   = init_lambda; 
    phmm->init_gamma    = init_gamma;
    phmm->init_delta    = init_delta;

    phmm->max_iter      = max_iter;
    phmm->tol           = tol;
    phmm->n_iter        = 0L;

    phmm->lambda_       = malloc (vector_s);
    phmm->gamma_        = malloc (matrix_s);
    phmm->delta_        = malloc (vector_s);

    if (phmm->lambda_ == NULL || phmm->gamma_ == NULL || phmm->delta_ == NULL)
        return NULL;

    // This should be done by EM
    memcpy (phmm->lambda_, init_lambda, vector_s);
    memcpy (phmm->gamma_,  init_gamma,  matrix_s);
    memcpy (phmm->delta_,  init_delta,  vector_s);

    phmm->aic           = 0.0L;
    phmm->bic           = 0.0L;
    phmm->nll           = 0.0L;

    return phmm;
}

void
DeletePoissonHMM (PoissonHMM *phmm)
{
    free (phmm->lambda_);
    free (phmm->gamma_);
    free (phmm->delta_);
    free (phmm);
}

scalar
compute_aic(scalar nll, size_t m, size_t n)
{
    return 2.0L * (scalar) (nll + 2*m + m*m);
}

scalar
compute_bic(scalar nll, size_t m, size_t n)
{
    return 2.0L * nll + logl ((scalar) n) * (scalar) (2*m + m*m);
}

