#include "pois-hmm.h"


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
    PoisHmm *const restrict this,
    const scalar *const restrict lambda,
    const scalar *const restrict gamma,
    const scalar *const restrict delta)
{
    size_t m_states = this->m_states;
    size_t v_size = m_states * sizeof (scalar);
    size_t m_size = m_states * v_size;

#ifdef __STDC_LIB_EXT1__
    errno_t err = 0;
    err = memcpy_s (this->init->lambda, v_size, lambda, v_size);
    if (err != 0) {
        perror ("Failed to copy initial values of `lambda'.");
        exit (1)
    }
    err = memcpy_s (this->init->gamma, m_size, gamma, m_size);
    if (err != 0) {
        perror ("Failed to copy initial values of `gamma'.");
        exit (1)
    }
    err = memcpy_s (this->init->delta, v_size, delta, v_size);
    if (err != 0) {
        perror ("Failed to copy initial values of `delta'.");
        exit (1)
    }
#else
    memcpy (this->init->lambda, lambda, v_size);
    memcpy (this->init->gamma, gamma, m_size);
    memcpy (this->init->delta, delta, v_size);
#endif

    memcpy (this->params->lambda, this->init->lambda, v_size);
    m_log (m_states, m_states, this->init->gamma, this->params->gamma);
    v_log (m_states, this->init->delta, this->params->delta);
}


void
PoisHmm_InitRandom (PoisHmm *const restrict this)
{
    size_t m_states = this->m_states;

    v_rnd_scalar (m_states, 1, 100, this->init->lambda);
    qsort (this->init->lambda, m_states, sizeof (scalar), compare_scalar);
    v_rnd_sample (m_states, this->init->delta);
    m_rnd_sample (m_states, m_states, this->init->gamma);

    mi_row_apply (m_states, m_states, vi_softmax, this->init->gamma);
    vi_softmax (m_states, this->init->delta);

    memcpy (this->params->lambda, this->init->lambda, m_states * sizeof (scalar));
    m_log (m_states, m_states, this->init->gamma, this->params->gamma);
    v_log (m_states, this->init->delta, this->params->delta);
}


void
PoisHmm_LogLikelihood (PoisHmm *const restrict this)
{
    this->llh = compute_log_likelihood (this->n_obs, this->m_states,
            this->probs->lalpha);
}


void
PoisHmm_ForwardProbabilities (PoisHmm *const restrict this)
{
    log_forward (this->probs->lsdp, this->params->gamma, this->params->delta,
            this->m_states, this->probs->n_obs, this->probs->lalpha);
}


void
PoisHmm_BackwardProbabilities (PoisHmm *const restrict this)
{
    log_backward (this->probs->lsdp, this->params->gamma, this->m_states,
            this->probs->n_obs, this->probs->lbeta);
}


void
PoisHmm_ForwardBackward (PoisHmm *const restrict this)
{
    log_fwbw (this->probs->lsdp, this->params->gamma, this->params->delta,
            this->m_states, this->probs->n_obs, this->probs->lalpha,
            this->probs->lbeta);
}


void
PoisHmm_EstimateParams (
    PoisHmm *const restrict this,
    const DataSet *const restrict inp)
{
    this->err = pois_em (inp->size, this->m_states, this->max_iter,
                         this->tol, inp->data, &this->n_iter, &this->llh, this->probs,
                         this->params);
    this->aic = compute_aic (this->m_states, this->llh);
    this->bic = compute_bic (this->n_obs, this->m_states, this->llh);
}


void
PoisHmm_LogConstStateProbs (PoisHmm *const restrict this)
{
    log_csprobs (this->n_obs, this->m_states, this->llh, this->probs->lalpha,
                 this->probs->lbeta, this->probs->lcsp);
}


void
PoisHmm_Summary (const PoisHmm *const restrict this)
{
    enum { linewidth = 79 };
    char border[linewidth+1];

    for (size_t i = 0; i < linewidth; i++)
    {
        border[i] = 96;
    }
    border[linewidth] = 0;

    fprintf (stderr, "\n%s\n", border);
    PoisParams_Print (this->params);
    fprintf (stderr, "\n%s\n", border);
    fprintf (stderr, "%10s%zu\n%10s%Lf\n%10s%Lf\n%10s%Lf\n",
            "n_iter: ", this->n_iter,
            "AIC: ", this->aic,
            "BIC: ", this->bic,
            "LLH: ", this->llh);
    fprintf (stderr, "\n%s\n", border);
}


int
compare_scalar (const void *x, const void *y)
{
    const scalar a = *(scalar *) x;
    const scalar b = *(scalar *) y;

    if (a < b)
        return -1;
    else if (a > b)
        return 1;
    else
        return 0;
}
