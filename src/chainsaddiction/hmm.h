#ifndef HMM_H
#define HMM_H

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "restrict.h"
#include "scalar.h"
#include "dataset.h"
#include "vmath.h"

typedef struct {
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
    scalar *lsd;        /**< Log of the state dependent probabilities. */
    scalar *lalpha;     /**< Log forward probabilities. */
    scalar *lbeta;      /**< Log backward probabilities. */
} HmmProbs;


/** Allocate memory for HmmProbs.
 *
 * @param n_obs    - Number of observations in the data set.
 * @param m_states - Number of HMM states.
 */
HmmProbs *
ca_NewHmmProbs (const size_t n_obs, const size_t m_states);

void
ca_FreeHmmProbs (HmmProbs *probs);



/** Allocate a new Params struct. 
 * Elements of parameter vectors remain uninitialized.
 */
PoisParams *PoisHmm_NewEmptyParams (size_t m);

/** Allocate new PoisParams and init with parameters read from file.
 */
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
