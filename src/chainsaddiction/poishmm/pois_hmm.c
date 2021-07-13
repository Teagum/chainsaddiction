#include "pois_hmm.h"


PoisHmm *PoisHmm_New (const size_t n_obs, const size_t m_states)
{
    PoisHmm *phmm = malloc (sizeof *phmm);
    if (phmm == NULL)
    {
        fprintf (stderr, "Could not allocate memory for `PoisHmm' object.\n");
        exit (1);
    }

    phmm->init   = PoisParams_New (m_states);
    phmm->params = PoisParams_New (m_states);
    phmm->probs  = PoisProbs_New (n_obs, m_states);

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


void
PoisHmm_Init (
    PoisHmm *const restrict phmm,
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

    memcpy (phmm->params->lambda, phmm->init->lambda, v_size);
    v_log (phmm->init->gamma, n_elem_gamma, phmm->params->gamma);
    v_log (phmm->init->delta, phmm->m_states, phmm->params->delta);
}


void
PoisHmm_InitRandom (PoisHmm *const restrict phmm)
{
    size_t m_states = phmm->m_states;
    size_t n_elem = m_states * m_states;

    v_rnd (m_states, phmm->init->lambda);
    v_rnd (m_states, phmm->init->delta);
    v_rnd (n_elem, phmm->init->gamma);

    for (size_t i = 0; i < m_states; i++)
    {
        phmm->init->lambda[i] += (scalar) rnd_int (1, 100);
        vi_softmax (phmm->init->gamma+i*m_states, m_states);
    }
    vi_softmax (phmm->init->delta, m_states);

    memcpy (phmm->params->lambda, phmm->init->lambda, m_states * sizeof (scalar));
    v_log (phmm->init->gamma, n_elem, phmm->params->gamma);
    v_log (phmm->init->delta, m_states, phmm->params->delta);
}


void
PoisHmm_LogLikelihood (PoisHmm *phmm)
{
    phmm->llh = compute_log_likelihood (
            phmm->probs->lalpha, phmm->n_obs, phmm->m_states);
}

void
PoisHmm_PrintInit (const PoisHmm *phmm)
{
    size_t m_states = phmm->m_states;
    PoisParams *p = phmm->init;

    puts ("");
    for (size_t i = 0; i < m_states; i++)
        printf ("%10.5Lf", p->lambda[i]);

    puts ("");
    for (size_t i = 0; i < m_states; i++)
        printf ("%10.5Lf", p->delta[i]);

    puts ("");
}

void PoisHmm_PrintParams (const PoisHmm *const phmm)
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
        printf ("%10.5Lf", expl (params->delta[i]));

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
            printf ("%10.5Lf", expl (params->gamma[i*m_states+j]));
        }
        puts ("");
    }
    printf ("\n*%s%s%s*\n\n", border, border, border);
}


void
PoisHmm_EstimateParams (
    PoisHmm *const restrict this,
    const DataSet *const restrict inp)
{
    pois_em (inp->size, this->m_states, this->max_iter, this->tol, inp->data,
            &this->llh, this->probs, this->params);
}
