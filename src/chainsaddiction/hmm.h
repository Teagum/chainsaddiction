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

typedef struct {
    size_t m_states;
    scalar *restrict lambda;
    scalar *restrict gamma;
    scalar *restrict delta;
} PoisParams;


/** Computation buffer for HMM estimation.
 *
 * Each field is a pointer to continuous memory with
 * space for n_obs * m_states values.
 */
typedef struct {
    size_t n_obs;
    size_t m_states;
    scalar *lsd;        /**< Log of the state dependent probabilities. */
    scalar *lalpha;     /**< Log forward probabilities. */
    scalar *lbeta;      /**< Log backward probabilities. */
} HmmProbs;


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
    HmmProbs *probs;
} PoisHmm;




/** Allocate memory for `HmmProbs'. The memory is guaranteed to be initialized
 * with zeros.
 *
 * \param n_obs    - Number of observations in the data set.
 * \param m_states - Number of HMM states.
 *
 * \return  Pointer to HmmProbs object if allocation did not fail; else NULL.
 */
HmmProbs *
ca_ph_NewProbs (const size_t n_obs, const size_t m_states);


/** Deallocate HmmProbs
 *
 * \param probs    - Pointer to HmmProbs object.
 */
#define ca_ph_FREE_PROBS(probs) do {     \
    MA_FREE (probs->lsd);                \
    MA_FREE (probs->lalpha);             \
    MA_FREE (probs->lbeta);              \
    MA_FREE (probs);                     \
} while (false)


/** Allocate memory for `PoisHmm' object.
 *
 * \param n_obs       Number of observations.
 * \param m_states    Number of states.
 */
PoisHmm *
ca_ph_NewHmm (const size_t n_obs, const size_t m_states);


/** Deallocate `PoisHmm' object.
 *
 * \param phmm    Pointer to `PoisHmm' object.
 */
#define ca_ph_FREE_HMM(phmm) do {        \
    ca_ph_FREE_PARAMS (phmm->init);      \
    ca_ph_FREE_PARAMS (phmm->params);    \
    ca_ph_FREE_PROBS (phmm->probs);      \
    MA_FREE (phmm);                      \
} while (false)


/** Allocate a memory for `PoisParams' object. The memory is guarateed to be
 * initialized with zeros.
 *
 * \param m_states    Number of HMM states.
 *
 * \return  Pointer to `PoisParams' or `NULL' if allocation fails.
 */
PoisParams*
ca_ph_NewParams (size_t m_states);


/** Deallocate `PoisParams' object.
 *
 * \param params    Pointer to `PoisParams' object.
 */
#define ca_ph_FREE_PARAMS(params) do {    \
    MA_FREE (params->lambda);             \
    MA_FREE (params->gamma);              \
    MA_FREE (params->delta);              \
} while (false)


void
ca_ph_InitParams (
    const PoisHmm *const restrict phmm,
    const scalar *const restrict lambda,
    const scalar *const restrict gamma,
    const scalar *const restrict delta);


PoisParams *PoisHmm_ParamsFromFile (const char *fname);

/** Print Poisson parameters to stdout. */
void PoisHmm_PrintParams (const PoisHmm *const restrict phmm);



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


/** Estimate log-likelihood of the HMM given forward probabilities.
 *
 * \parma lalpha   - Logarithm of the forward probabilities.
 * \param n_obs    - Number of observations in the data set.
 * \param n_states - Number of HMM states.
 *
 * \return  Model log-likelihood.
 */
scalar ca_log_likelihood (scalar *lalpha, size_t n_obs, size_t m_states);


#endif  /* HMM_H */
