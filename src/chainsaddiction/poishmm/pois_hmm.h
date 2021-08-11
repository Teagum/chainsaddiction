#ifndef HMM_H
#define HMM_H

#include <stdbool.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "pois_em.h"
#include "pois_params.h"
#include "pois_probs.h"
#include "pois_utils.h"
#include "../dataset.h"
#include "../restrict.h"
#include "../scalar.h"
#include "libvmath.h"

#define DEFAULT_MAX_ITER 1000
#define DEFAULT_TOLERANCE 1e-6


/** \struct PoisHmm
 * \brief HMM with Poisson-distributed states.
 */
typedef struct PoisHmm {
    bool   err;
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
#define PoisHmm_Delete(phmm) do {       \
    PoisParams_Delete (phmm->init);     \
    PoisParams_Delete (phmm->params);   \
    PoisProbs_Delete  (phmm->probs);    \
    free (phmm);                        \
    phmm = NULL;                        \
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
    PoisHmm *const restrict phmm,
    const scalar *const restrict lambda,
    const scalar *const restrict gamma,
    const scalar *const restrict delta);


void
PoisHmm_InitRandom (
    PoisHmm *const restrict phmm);


void
PoisHmm_LogLikelihood (PoisHmm *phmm);


#define PoisHmm_LogConditionalExpectation(phmm) do {                        \
    log_cond_expect (phmm->n_obs, phmm->m_states, phmm->probs->lalpha,      \
            phmm->probs->lbeta, phmm->llh, phmm->probs->lcexpt);            \
} while (false)


void
PoisHmm_PrintInit (const PoisHmm *const restrict phmm);


/** Print Poisson parameters to stdout. */
void PoisHmm_PrintParams (
    const PoisHmm *const restrict phmm);


int
PoisHmm_ForwardProbabilities (PoisHmm *const restrict this);


int
PoisHmm_BackwardProbabilities (PoisHmm *const restrict this);


int
PoisHmm_ForwardBackward (PoisHmm *const restrict this);


void
PoisHmm_EstimateParams (
    PoisHmm *const restrict this,
    const DataSet *const restrict inp);


#endif  /* HMM_H */
