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

    phmm->err      = false;
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
    m_log (m_states, m_states, phmm->init->gamma, phmm->params->gamma);
    v_log (m_states, phmm->init->delta, phmm->params->delta);
}


void
PoisHmm_InitRandom (PoisHmm *const restrict phmm)
{
    size_t m_states = phmm->m_states;

    v_rnd_scalar (m_states, 1, 100, phmm->init->lambda);
    v_rnd_sample (m_states, phmm->init->delta);
    m_rnd_sample (m_states, m_states, phmm->init->gamma);

    mi_row_apply (m_states, m_states, vi_softmax, phmm->init->gamma);
    vi_softmax (m_states, phmm->init->delta);

    memcpy (phmm->params->lambda, phmm->init->lambda, m_states * sizeof (scalar));
    m_log (m_states, m_states, phmm->init->gamma, phmm->params->gamma);
    v_log (m_states, phmm->init->delta, phmm->params->delta);
}


void
PoisHmm_LogLikelihood (PoisHmm *const restrict this)
{
    this->llh = compute_log_likelihood (
            this->probs->lalpha, this->n_obs, this->m_states);
}


void
PoisHmm_PrintInit (const PoisHmm *phmm)
{
    size_t m_states = phmm->m_states;
    PoisParams *p = phmm->init;

    puts ("");
    for (size_t i = 0; i < m_states; i++)
        printf (SF, p->lambda[i]);

    puts ("");
    for (size_t i = 0; i < m_states; i++)
        printf (SF, p->delta[i]);

    puts ("");
}


void
PoisHmm_PrintParams (const PoisHmm *const this)
{
    enum {
        linewidth = 120
    };
    const char border[] = "========================================================================================================================\0";
    const char sep[]    = "------------------------------------------------------------------------------------------------------------------------\0";

    size_t m_states = this->m_states;
    PoisParams *params = this->params;

    printf ("\n\n*%s*\n\n", border);
    printf ("%25s%10zu\n", "m_states:", m_states);
    printf ("%25s", "log likelihood:");
    printf (SFN, this->llh);
    printf ("%25s", "AIC:");
    printf (SFN, this->aic);
    printf ("%25s", "BIC:");
    printf (SFN, this->bic);
    printf ("%25s%5zu /%4zu\n", "n_iter:", this->n_iter, this->max_iter);
    printf ("\n%s\n\n", sep);

    printf ("%25s", "State:");
    for (size_t i = 0; i < m_states; i++)
        printf ("%15zu", i+1);
    puts ("");
    printf ("%25s", "State dependent means:");
    for (size_t i = 0; i < m_states; i++)
        printf (SF, params->lambda[i]);
    puts ("");
    printf ("%25s", "Start distribution:");
    for (size_t i = 0; i < m_states; i++)
        printf (SF, expl (params->delta[i]));

    printf ("\n\n%s\n\n", sep);

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
    printf ("\n*%s*\n\n", border);
}

int
PoisHmm_ForwardProbabilities (PoisHmm *const restrict this)
{
    log_forward (this->probs->lsdp, this->params->gamma, this->params->delta,
            this->m_states, this->probs->n_obs, this->probs->lalpha);

    return false;
}


int
PoisHmm_BackwardProbabilities (PoisHmm *const restrict this)
{
    log_backward (this->probs->lsdp, this->params->gamma, this->m_states,
            this->probs->n_obs, this->probs->lbeta);
    return false;
}


int
PoisHmm_ForwardBackward (PoisHmm *const restrict this)
{
    log_fwbw (this->probs->lsdp, this->params->gamma, this->params->delta,
            this->m_states, this->probs->n_obs, this->probs->lalpha,
            this->probs->lbeta);
    return false;
}


void
PoisHmm_EstimateParams (
    PoisHmm *const restrict this,
    const DataSet *const restrict inp)
{
    this->err = pois_em (inp->size, this->m_states, this->max_iter,
                         this->tol, inp->data, &this->n_iter, &this->llh, this->probs,
                         this->params);
    this->aic = compute_aic (this->llh, this->m_states);
    this->bic = compute_bic (this->llh, this->n_obs, this->m_states);
}
