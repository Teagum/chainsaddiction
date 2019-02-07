#include "fwbw.h"

int PoisHmm_FwBw(
        const long *restrict x,
        const size_t n,
        const size_t m,
        PoisParams *restrict params,
        scalar *restrict alpha,
        scalar *restrict beta,
        scalar *restrict pois_pr)
{
    scalar sum_buff = LDBL_EPSILON;     /* acccumates probabilities */
    scalar lsf      = 0.0L;             /* log scale factor */
    int    success  = 1;
    
    scalar *pr_x_t   = malloc (m * sizeof (scalar));     /* probabilities at time `t` */
    scalar *buff     = malloc (m * sizeof (scalar));     /* calculation buffer */
    scalar *prob_acc = malloc (m * sizeof (scalar));     /* probability accumulator */

    if (pr_x_t == NULL || buff == NULL || prob_acc == NULL)
    {
        success = 0;
        goto fail;
    }

    /*
     * Forward pass
     */

    /* Initial step t = 0 */
    for (size_t i = 0; i < m; i++)
    {
        pois_pr[i] = poisson_pmf (params->lambda[i], x[0]);
        pr_x_t[i]  = pois_pr[i] * params->delta[i];
        sum_buff  += pr_x_t[i];
    }

    lsf = logl (sum_buff);
    for (size_t i = 0; i < m; i++)
    {
        pr_x_t[i] /= sum_buff;
        alpha[i] = logl (pr_x_t[i]) + lsf;
    }

    /* remaining forward steps */
    for (size_t t = 1; t < n; t++)
    {
        sum_buff = LDBL_EPSILON;
        for (size_t i = 0; i < m; i++)
        {
            buff[i] = 0.0L;
            for (size_t j = 0; j < m; j++)
            {
                buff[i] += pr_x_t[j] * params->gamma[j*m+i];
            }
            pois_pr[t*m+i] = poisson_pmf (params->lambda[i], x[t]);
            buff[i] *= pois_pr[t*m+i];
            sum_buff += buff[i];
        }

        lsf += logl (sum_buff);

        for (size_t i = 0; i < m; i++)
        {
            pr_x_t[i] = buff[i] / sum_buff;
            alpha[t*m+i] = logl (pr_x_t[i]) + lsf;
        }
    }

    /*
     * Backward pass
     */

    /* Initial step */
    for (size_t i = 0; i < m; i++)
    {
        pr_x_t[i] = 1.0L / (scalar) m;
        beta[(n-1)*m+i] = 0.0L;
    }
    lsf = logl (m);

    /* remaining backward steps */
    for (size_t t = n-1; t > 0; t--)
    {
        for (size_t i = 0; i < m; i++)
        {
            prob_acc[i] = pois_pr[t*m+i] * pr_x_t[i];
        }

        sum_buff = LDBL_EPSILON;
        for (size_t i = 0; i < m; i++)
        {
            buff[i] = 0.0L;
            for (size_t j = 0; j < m; j++)
            {
                buff[i] += params->gamma[i*m+j] * prob_acc[j];
            }
            sum_buff += buff[i];
        }

        lsf += logl (sum_buff);
        for (size_t i = 0; i < m; i++)
        {
            pr_x_t[i] = buff[i] / sum_buff;
            buff[i] = 0.0L;
            beta[(t-1)*m+i] = logl (pr_x_t[i]) + lsf;
        }
    }

fail:
    free (pr_x_t);
    free (buff);
    free (prob_acc);
    return success;
}
