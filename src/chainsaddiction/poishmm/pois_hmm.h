#ifndef HMM_H
#define HMM_H

#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "restrict.h"
#include "scalar.h"
#include "dataset.h"
#include "libma.h"
#include "rnd.h"
#include "vmath.h"

#define DEFAULT_MAX_ITER 1000
#define DEFAULT_TOLERANCE 1e-5


/** \struct PoisHmm
 * \brief HMM with Poisson-distributed states.
 */
typedef struct {
    size_t n_obs;
    size_t m_states;
    size_t n_iter;
    size_t max_iter;
    scalar tol;
    scalar aic;
    scalar bic;
    scalar llh;
    PoisParams *init;
    PoisParams *params;
    PoisProbs *probs;
} PoisHmm;


/** Deallocate `PoisHmm' object.
 *
 * \param phmm  Pointer to `PoisHmm' object.
 */
#define PoisHmm_Delete(phmm) do {           \
    PoisHmm_DeleteParams (phmm->init);      \
    PoisHmm_DeleteParams (phmm->params);    \
    PoisHmm_DeleteProbs (phmm->probs);      \
    MA_FREE (phmm);                         \
} while (false)


/** Allocate memory for `PoisHmm' object.
 *
 * \param n_obs     Number of observations.
 * \param m_states  Number of states.
 */
PoisHmm *
PoisHmm_New (
    const size_t n_obs,
    const size_t m_states);


void
PoisHmm_Init (
    const PoisHmm *const restrict phmm,
    const scalar *const restrict lambda,
    const scalar *const restrict gamma,
    const scalar *const restrict delta);


void
PoisHmm_InitRandom (
    PoisHmm *const restrict phmm);


void
PoisHmm_BaumWelch (
    const DataSet *const restrict inp,
    PoisHmm *restrict hmm);


void
PoisHmm_LogLikelihood (PoisHmm *phmm);


#define PoisHmm_LogConditionalExpectation(phmm) do {                        \
    log_cond_expect (phmm->n_obs, phmm->m_states, phmm->probs->lalpha,      \
            phmm->probs->lbeta, phmm->llh, phmm->probs->lcexpt);            \
} while (false)


/** Estimate log-likelihood given forward probabilities.
 *
 * \parma lalpha    Logarithm of the forward probabilities.
 * \param n_obs     Number of observations in the data set.
 * \param n_states  Number of HMM states.
 *
 * \return  Model log-likelihood.
 */
scalar
compute_log_likelihood (
    scalar *lalpha,
    size_t n_obs,
    size_t m_states);


PoisParams *PoisHmm_ParamsFromFile (const char *fname);


/** Print Poisson parameters to stdout. */
void PoisHmm_PrintParams (
    const PoisHmm *const restrict phmm);


/** Allocate new PoisHmm with init data from compile time constants. */
PoisHmm *
PoisHmm_FromData(size_t  m,
    scalar *restrict init_lambda,
    scalar *restrict init_gamma,
    scalar *restrict init_delta,
    size_t max_iter,
    scalar tol);


/* Compute Akaine Information criterion. */
scalar
compute_aic(scalar nll, size_t m);


/* Compute Bayes Information criterion. */
scalar
compute_bic(scalar nll, size_t m, size_t n);


/** Compute the conditional expectations. */
extern void
log_cond_expect (
    const size_t n_obs,
    const size_t m_states,
    const scalar *const restrict lalpha,
    const scalar *const restrict lbeta,
    const scalar llh,
    scalar *lcexpt);


#endif  /* HMM_H */
