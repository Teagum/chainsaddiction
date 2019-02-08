#include <string.h>
#include "em.h"
#include "hmm.h"

int PoisHmm_EM (const DataSet *restrict x,
                      PoisHmm *restrict hmm)
{
    scalar acc      = 0.0L;
    scalar bcc      = 0.0L;
    scalar c        = 0.0L;
    scalar crit     = 0.0L;
    scalar rs_delta = 0.0L;
    scalar rs_gamma = 0.0L;
    scalar s        = 0.0L;
    int    success  = 0;
    size_t m        = hmm->m;   /* just for convenience */
    size_t n        = x->size;  /* just for convenience */

    size_t vector_s = m * sizeof (scalar);
    size_t matrix_s = m * vector_s;
    size_t buffer_s = n * vector_s;

    scalar *alpha       = malloc (buffer_s);
    scalar *beta        = malloc (buffer_s);
    scalar *pprob       = malloc (buffer_s);
    scalar *next_lambda = malloc (vector_s);
    scalar *next_gamma  = malloc (matrix_s);
    scalar *next_delta  = malloc (vector_s);


    if (alpha == NULL || beta == NULL || pprob == NULL ||
        next_lambda == NULL || next_gamma == NULL || next_delta == NULL)
    {
        goto exit_point;
    }

    memcpy (hmm->params->lambda, hmm->init->lambda, vector_s);
    memcpy (hmm->params->gamma,  hmm->init->gamma,  matrix_s);
    memcpy (hmm->params->delta,  hmm->init->delta,  vector_s);

    for (; hmm->n_iter < hmm->max_iter; (hmm->n_iter)++)
    {
        /* E Step */
        int fwbw_ret = PoisHmm_FwBw (x->data, n, m, hmm->params,
                                     alpha, beta, pprob);


        if (fwbw_ret == 0)
        {
            fprintf (stderr, "Forward/Backward algorithm failed \
                     (n_iter = %zu).\n", hmm->n_iter);
            goto exit_point;
        }

        c = alpha[(n-1)*m];
        for (size_t j = 1; j < m; j++)
        {
            if (alpha[(n-1)*m+j] > c)
            {
                c = alpha[(n-1)*m+j];
            }
        }

        hmm->nll = 0.0L;
        for (size_t j = 0; j < m; j++)
        {
            hmm->nll += expl (alpha[(n-1)*m+j] - c);
        }
        hmm->nll = logl (hmm->nll) + c;

        /* M Step */
        crit     = 0.0L;
        rs_delta = 0.0L;
        for (size_t j = 0; j < m; j++)
        {
            /* Lambda */
            acc = 0;
            bcc = 0;
            for (size_t i = 0; i < n; i++)
            {
                s = expl (alpha[i*m+j] + beta[i*m+j] - (hmm->nll));
                bcc += s;
                acc += s * x->data[i];
            }
            next_lambda[j] = acc / bcc;
            crit += fabsl (next_lambda[j] - hmm->params->lambda[j]);

            /* Gamma */
            rs_gamma = 0;
            for (size_t i = 0; i < m; i++)
            {
                acc = 0;
                for (size_t k = 0; k < (n - 1); k++)
                {
                    acc +=  expl (alpha[k*m+j]
                                  + beta[(k+1)*m+i]
                                  + logl (pprob[(k+1)*m+i])
                                  - hmm->nll);
                }
                next_gamma[j*m+i] = hmm->params->gamma[j*m+i] * acc;
                rs_gamma += next_gamma[j*m+i];
            }

            for (size_t i = 0; i < m; i++)
            {
                next_gamma[j*m+i] /= rs_gamma;
                crit += fabsl (next_gamma[j*m+i] - hmm->params->gamma[j*m+i]);
            }

            /* Delta */
            next_delta[j] = expl (alpha[j] + beta[j] - hmm->nll);
            rs_delta += next_delta[j];
        }

        for (size_t j = 0; j < m; j++)
        {
            next_delta[j] /= rs_delta;
            crit += fabsl (next_delta[j] - hmm->params->delta[j]);
        }

        /* no convergence yet -> copy and reiterate */
        if (crit >= hmm->tol)
        {
            memcpy (hmm->params->lambda, next_lambda, vector_s);
            memcpy (hmm->params->gamma,  next_gamma,  matrix_s);
            memcpy (hmm->params->delta,  next_delta,  vector_s);
        }
        else    /* convergence */
        {
            success = 1;
            goto exit_point;
        }
    }

exit_point:

    free (alpha);
    free (beta);
    free (pprob);

    free (next_lambda);
    free (next_gamma);
    free (next_delta);

    return success;
}
