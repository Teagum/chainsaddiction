#ifndef HMM_H
#define HMM_H

#include <stdbool.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "pois-em.h"
#include "pois-params.h"
#include "pois-probs.h"
#include "../utils/utils.h"
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
 * \param this  Pointer to `PoisHmm' object.
 */
#define PoisHmm_Delete(this)            \
if (this != NULL) {                     \
    PoisParams_Delete (this->init);     \
    PoisParams_Delete (this->params);   \
    PoisProbs_Delete  (this->probs);    \
    free (this);                        \
    this= NULL;                         \
}


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
    PoisHmm *const restrict this,
    const scalar *const restrict lambda,
    const scalar *const restrict gamma,
    const scalar *const restrict delta);


void
PoisHmm_InitRandom (
    PoisHmm *const restrict this);


void
PoisHmm_LogLikelihood (PoisHmm *const restrict this);


void
PoisHmm_ForwardProbabilities (PoisHmm *const restrict this);


void
PoisHmm_BackwardProbabilities (PoisHmm *const restrict this);


void
PoisHmm_ForwardBackward (PoisHmm *const restrict this);


void
PoisHmm_EstimateParams (
    PoisHmm *const restrict this,
    const DataSet *const restrict inp);


void
PoisHmm_LogCondStateProbs (PoisHmm *const restrict this);


void
PoisHmm_Summary (const PoisHmm *const restrict this);


int
compare_scalar (const void *x, const void *y);


#endif  /* HMM_H */
