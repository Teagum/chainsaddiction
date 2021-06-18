#ifndef HMM_H
#define HMM_H

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "restrict.h"
#include "scalar.h"
#include "dataset.h"
#include "vmath.h"
#include "libma.h"


typedef struct {
    size_t m_states;
    scalar *restrict lambda;
    scalar *restrict gamma;
    scalar *restrict delta;
} PoisParams;


/** \struct PoisHmm
 * \brief HMM with Poisson-distributed states.
 */
typedef struct {
    size_t m_states;
    size_t n_iter;
    size_t max_iter;
    scalar tol;
    scalar aic;
    scalar bic;
    scalar llh;
    PoisParams *init;
    PoisParams *params;
} PoisHmm;


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


/** Allocate memory for `HmmProbs'. The memory is guaranteed to be initialized
 * with zeros.
 *
 * \param n_obs    - Number of observations in the data set.
 * \param m_states - Number of HMM states.
 *
 * \return  Pointer to HmmProbs object if allocation did not fail; else NULL.
 */
HmmProbs *
ca_NewHmmProbs (const size_t n_obs, const size_t m_states);


/** Deallocate HmmProbs
 *
 * \param probs    - Pointer to HmmProbs object.
 */
#define ca_FREE_HMM_PROBS(probs) do {    \
    MA_FREE (probs->lsd);                \
    MA_FREE (probs->lalpha);             \
    MA_FREE (probs->lbeta);              \
    MA_FREE (probs);                     \
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


PoisParams *PoisHmm_ParamsFromFile (const char *fname);

/** Print Poisson parameters to stdout. */
void PoisHmm_PrintParams (PoisParams *params, size_t m_states);

/** Deallocate the Parms struct */
void PoisHmm_FreeParams (PoisParams *params);

/** Allocate new PoisHmm with init data from compile time constants. */
PoisHmm *
PoisHmm_FromData(size_t  m,
        scalar *restrict init_lambda,
        scalar *restrict init_gamma,
        scalar *restrict init_delta,
        size_t max_iter,
        scalar tol);


/** Deallocate a PoisHmm.*/
void
PoisHmm_DeleteHmm (PoisHmm *ph);


/* Compute Akaine Information criterion. */
scalar
compute_aic(scalar nll, size_t m);


/* Compute Bayes Information criterion. */
scalar
compute_bic(scalar nll, size_t m, size_t n);


/** Estimate log-likelihood of the HMM given forward probabilities.
 *
 * @parma: lalpha   - Logarithm of the forward probabilities.
 * @param: n_obs    - Number of observations in the data set.
 * @param: n_states - Number of HMM states.  */
scalar log_likelihood_fw (scalar *lalpha, size_t n_obs, size_t m_states);


#endif  /* HMM_H */
